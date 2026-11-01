"""Long-context control for the fine-tuned encoders (camera-ready, reviewer R1/R2).

BERT / BioBERT are fine-tuned on ALL 512-token chunks of each training document (chunk
inherits the document label) and scored on ALL chunks of each test document; chunk
probabilities are mean-pooled (and max-pooled) into one document score. Same 5 folds,
epochs, LR, batch size and class weighting as e1_baselines.py, so the ONLY change is that
the encoder now sees the whole document instead of a 512-token salient window.
Outputs results/e1_{bert,biobert}_chunked.json
"""
import os, sys, json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)
from common import load_gold, RESULTS, SEED

HF = {"bert": "bert-base-uncased", "biobert": "dmis-lab/biobert-base-cased-v1.1"}
MAXLEN, EPOCHS, LR, BS = 512, 12, 1e-5, 8
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "32"))   # 32*510 tokens ~ 16k tokens >= any doc


def chunk_doc(tok, text):
    ids = tok(str(text), add_special_tokens=False)["input_ids"]
    body = MAXLEN - 2
    chunks = [ids[i:i + body] for i in range(0, len(ids), body)][:MAX_CHUNKS]
    if not chunks:
        chunks = [[]]
    return [[tok.cls_token_id] + c + [tok.sep_token_id] for c in chunks]


def pad(batch, pad_id):
    L = max(len(b) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b)] = torch.tensor(b); am[i, :len(b)] = 1
    return ids, am


def run_model(name, g):
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(HF[name])
    docs = [chunk_doc(tok, t) for t in g.content]
    n_chunks = np.array([len(d) for d in docs])
    y = g.label.values; folds = g.fold.values
    oof_mean = np.zeros(len(g)); oof_max = np.zeros(len(g))
    t0 = time.time()
    for f in sorted(set(folds)):
        tr = np.where(folds != f)[0]; te = np.where(folds == f)[0]
        tr_chunks = [(c, int(y[i])) for i in tr for c in docs[i]]
        model = AutoModelForSequenceClassification.from_pretrained(HF[name], num_labels=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * ((len(tr_chunks) + BS - 1) // BS)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
        ytr = np.array([l for _, l in tr_chunks])
        cw = torch.tensor([1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float).to(device)
        lossf = torch.nn.CrossEntropyLoss(weight=cw)
        model.train()
        rng = np.random.default_rng(SEED + f)
        for ep in range(EPOCHS):
            perm = rng.permutation(len(tr_chunks))
            for i in range(0, len(perm), BS):
                b = [tr_chunks[j] for j in perm[i:i + BS]]
                ids, am = pad([c for c, _ in b], tok.pad_token_id)
                lab = torch.tensor([l for _, l in b])
                opt.zero_grad()
                out = model(input_ids=ids.to(device), attention_mask=am.to(device)).logits
                loss = lossf(out, lab.to(device)); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            for i in te:
                probs = []
                for k in range(0, len(docs[i]), 16):
                    ids, am = pad(docs[i][k:k + 16], tok.pad_token_id)
                    pr = torch.softmax(model(input_ids=ids.to(device), attention_mask=am.to(device)).logits, -1)[:, 1]
                    probs.append(pr.cpu().numpy())
                probs = np.concatenate(probs)
                oof_mean[i] = probs.mean(); oof_max[i] = probs.max()
        del model; torch.cuda.empty_cache()
        print(f"  [{name}] fold {f} done ({len(tr_chunks)} train chunks) {time.time()-t0:.0f}s", flush=True)

    def metrics(prob):
        pred = (prob >= 0.5).astype(int)
        return dict(acc=round(accuracy_score(y, pred), 3), macro_f1=round(f1_score(y, pred, average="macro"), 3),
                    prec1=round(precision_score(y, pred, zero_division=0), 3),
                    rec1=round(recall_score(y, pred, zero_division=0), 3),
                    auroc=round(roc_auc_score(y, prob), 3), auprc=round(average_precision_score(y, prob), 3))
    out = dict(model=HF[name], protocol="all 512-token chunks, chunk inherits doc label; 5-fold CV",
               epochs=EPOCHS, lr=LR, batch_size=BS, max_chunks=MAX_CHUNKS,
               chunks_per_doc=dict(mean=round(float(n_chunks.mean()), 2), median=int(np.median(n_chunks)),
                                   max=int(n_chunks.max()), total=int(n_chunks.sum())),
               mean_pool=metrics(oof_mean), max_pool=metrics(oof_max), seconds=round(time.time() - t0))
    np.savez(os.path.join(RESULTS, f"e1_{name}_chunked_oof.npz"), oof_mean=oof_mean, oof_max=oof_max, y=y, folds=folds)
    with open(os.path.join(RESULTS, f"e1_{name}_chunked.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    print(name, json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    g = load_gold()
    for name in (sys.argv[1:] or ["bert", "biobert"]):
        run_model(name, g)
