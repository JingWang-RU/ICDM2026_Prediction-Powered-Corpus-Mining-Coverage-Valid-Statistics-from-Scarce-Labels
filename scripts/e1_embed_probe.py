"""Frozen-embedding linear probes as a stronger supervised baseline (camera-ready addition).

For each embedder, embed the 123 gold case reports (full text, truncated to the model's window),
save the embeddings (released in data/cached_gpu/ so the notebook can refit the probe without full
text), and fit a logistic-regression probe (StandardScaler + LR, C=1, class-balanced) under the SAME
5 folds and label-efficiency protocol as e1_baselines.py.  Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python e1_embed_probe.py qwen3emb8b bgem3 medgemma4b [zembed1]
"""
import os, sys, json, time
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score)
from transformers import AutoTokenizer, AutoModel
from common import load_gold, RESULTS, SEED
from e1_baselines import platt_cv, ece

REL_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "release", "data", "cached_gpu")
MODELS = {  # tag: (repo, pooling, max_len, dtype)
    "qwen3emb8b": ("Qwen/Qwen3-Embedding-8B", "last", 8192, torch.bfloat16),
    "bgem3":      ("BAAI/bge-m3", "cls", 8192, torch.float16),
    "medgemma4b": ("google/medgemma-1.5-4b-it", "mean", 8192, torch.bfloat16),
    "zembed1":    ("zeroentropy/zembed-1-embedding", "st", 8192, torch.bfloat16),   # sentence-transformers: custom module, last-token pooling
}


class STWrapper:
    """sentence-transformers path (zembed-1 ships custom modules + last-token pooling + normalisation)."""
    def __init__(self, repo, dtype, max_len):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(repo, trust_remote_code=True, model_kwargs={"torch_dtype": dtype}, device="cuda")
        self.m.max_seq_length = max_len
    def encode(self, texts):
        return self.m.encode([str(t) for t in texts], batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def load_encoder(repo, dtype):
    tok = AutoTokenizer.from_pretrained(repo)
    try:
        model = AutoModel.from_pretrained(repo, torch_dtype=dtype, trust_remote_code=True)
    except Exception as e:
        print("  AutoModel failed:", str(e)[:120], "-> trying image-text class")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(repo, torch_dtype=dtype)
    # text-only backbone for VLM checkpoints (MedGemma = Gemma3)
    for path in ("language_model", "model.language_model", "model"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            if hasattr(obj, "embed_tokens") or hasattr(obj, "layers"):
                model = obj; break
        except AttributeError:
            continue
    return tok, model.cuda().eval()


ZEMBED_PREFIX = "<|im_start|>system\ndocument<|im_end|>\n<|im_start|>user\n"   # zembed-1 document prompt (config_sentence_transformers.json)
ZEMBED_SUFFIX = "<|im_end|>\n"


@torch.no_grad()
def embed(tok, model, texts, pooling, max_len, tag):
    if pooling == "last":
        tok.padding_side = "left"
    out = []
    for i, t in enumerate(texts):
        if tag == "zembed1":   # replicate ZembedTransformer.tokenize: prompt + truncated document + suffix, last-token pooling
            ids = tok(str(t), add_special_tokens=False, truncation=True, max_length=max_len - 32)["input_ids"]
            t = ZEMBED_PREFIX + tok.decode(ids) + ZEMBED_SUFFIX
        enc = tok(str(t), truncation=True, max_length=max_len, return_tensors="pt", add_special_tokens=(tag != "zembed1")).to("cuda")
        h = model(**enc).last_hidden_state.float()           # (1, L, d)
        m = enc["attention_mask"].unsqueeze(-1).float()
        if pooling == "cls":
            v = h[:, 0]
        elif pooling == "last":
            v = h[:, -1]                                      # left padding -> last position is the last real token
        else:
            v = (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy()[0])
        if i % 40 == 0:
            print(f"  [{tag}] {i}/{len(texts)} (len {enc['input_ids'].shape[1]})", flush=True)
    return np.stack(out)


def probe(X, y, folds, fracs=(0.25, 0.5, 0.75, 1.0), C=1.0):
    def fit(Xtr, ytr):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced")).fit(Xtr, ytr)
    oof = np.zeros(len(y))
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        oof[te] = fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    rng = np.random.default_rng(SEED); le = {}
    for fr in fracs:
        f1s = []
        for f in sorted(set(folds)):
            tr_idx = np.where(folds != f)[0]; te = folds == f
            k = max(10, int(len(tr_idx) * fr)); sub = rng.choice(tr_idx, k, replace=False)
            f1s.append(f1_score(y[te], fit(X[sub], y[sub]).predict(X[te]), average="macro"))
        le[str(fr)] = round(float(np.mean(f1s)), 3)
    return oof, le


def main(tags):
    g = load_gold(); y = g.label.values; folds = g.fold.values
    os.makedirs(REL_CACHE, exist_ok=True)
    for tag in tags:
        repo, pooling, max_len, dtype = MODELS[tag]; t0 = time.time()
        print(f"== {tag}: {repo} ({pooling} pooling, max_len {max_len})", flush=True)
        if pooling == "st":
            try:
                st = STWrapper(repo, dtype, max_len); X = st.encode(list(g.content)); del st
            except Exception as e:   # old sentence-transformers cannot load the custom module -> backbone last-token pooling
                print("  sentence-transformers path failed:", str(e)[:160], "-> falling back to last-token pooling", flush=True)
                torch.cuda.empty_cache(); pooling = "last"
        if pooling != "st":
            tok, model = load_encoder(repo, dtype)
            X = embed(tok, model, list(g.content), pooling, max_len, tag)
            del model
        torch.cuda.empty_cache()
        MODELS[tag] = (repo, pooling, max_len, dtype)
        np.save(os.path.join(RESULTS, f"embeddings_gold_{tag}.npy"), X.astype(np.float32))
        np.save(os.path.join(REL_CACHE, f"gold_embeddings_{tag}.npy"), X.astype(np.float32))
        oof, le = probe(X, y, folds)
        pred = (oof >= 0.5).astype(int); cal = platt_cv(oof, y, folds)
        res = dict(model=repo, pooling=MODELS[tag][1], max_len=max_len, dim=int(X.shape[1]), protocol="StandardScaler + LogisticRegression(C=1, balanced), 5-fold CV",
                   acc=round(accuracy_score(y, pred), 3), macro_f1=round(f1_score(y, pred, average="macro"), 3),
                   prec1=round(precision_score(y, pred, zero_division=0), 3), rec1=round(recall_score(y, pred, zero_division=0), 3),
                   auroc=round(roc_auc_score(y, oof), 3), auprc=round(average_precision_score(y, oof), 3),
                   ece_raw=round(ece(y, oof), 3), ece_platt=round(ece(y, cal), 3), label_efficiency_macroF1=le, seconds=round(time.time() - t0))
        np.savez(os.path.join(RESULTS, f"e1_probe_{tag}_oof.npz"), oof=oof, cal=cal, y=y, folds=folds)
        json.dump(res, open(os.path.join(RESULTS, f"e1_probe_{tag}.json"), "w"), indent=2)
        print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen3emb8b", "bgem3", "medgemma4b"])
