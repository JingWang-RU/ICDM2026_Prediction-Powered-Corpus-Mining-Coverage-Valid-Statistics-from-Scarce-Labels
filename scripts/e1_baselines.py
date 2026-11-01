import os, sys, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)
from common import load_gold, load_zeroshot_gold, RESULTS, SEED

HF = {"bert": "bert-base-uncased", "biobert": "dmis-lab/biobert-base-cased-v1.1"}
MAXLEN = 512
EPOCHS = 12
LR = 1e-5
BS = 8
import re as _re
_KW = _re.compile(r"long covid|long[- ]?haul|post[- ]?acute sequelae|\bpasc\b|post[- ]?covid|"
                  r"chronic covid|persistent symptom|post[- ]?viral|months after|"
                  r"3 months|three months|fatigue|brain fog|dyspn|breath|sequelae", _re.I)


def salient_text(content, head_words=120, win_words=320):
    """Return head context + the densest LC-keyword window (so BERT sees the signal)."""
    words = str(content).split()
    head = " ".join(words[:head_words])
    if len(words) <= head_words + win_words:
        return " ".join(words)
    best_i, best_score = 0, -1
    step = 40
    for i in range(0, len(words) - win_words, step):
        chunk = " ".join(words[i:i + win_words])
        s = len(_KW.findall(chunk))
        if s > best_score:
            best_score, best_i = s, i
    win = " ".join(words[best_i:best_i + win_words])
    return head + " ... " + win


def metrics(y, pred, prob=None):
    out = dict(acc=round(accuracy_score(y, pred), 3),
               macro_f1=round(f1_score(y, pred, average="macro"), 3),
               prec1=round(precision_score(y, pred, zero_division=0), 3),
               rec1=round(recall_score(y, pred, zero_division=0), 3))
    if prob is not None and len(set(y)) > 1:
        out["auroc"] = round(roc_auc_score(y, prob), 3)
        out["auprc"] = round(average_precision_score(y, prob), 3)
    return out


def ece(y, prob, n_bins=8):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (prob >= bins[i]) & (prob < bins[i + 1] if i < n_bins - 1 else prob <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf = prob[m].mean(); acc = (y[m] == (prob[m] >= 0.5)).mean()
        e += (m.sum() / len(y)) * abs(conf - acc)
    return float(e)


def platt_cv(oof_prob, y, folds):
    """Leakage-free Platt: fit logistic on OOF logits of 4 folds, apply to held fold."""
    eps = 1e-6
    logit = np.log(np.clip(oof_prob, eps, 1 - eps) / np.clip(1 - oof_prob, eps, 1 - eps))
    cal = np.zeros_like(oof_prob)
    for f in sorted(set(folds)):
        tr = folds != f; te = folds == f
        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(logit[tr].reshape(-1, 1), y[tr])
        cal[te] = lr.predict_proba(logit[te].reshape(-1, 1))[:, 1]
    return cal


def run_tfidf(g, fracs):
    y = g.label.values; folds = g.fold.values
    oof = np.zeros(len(g))
    for f in sorted(set(folds)):
        tr = folds != f; te = folds == f
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
                              stop_words="english")
        Xtr = vec.fit_transform(g.content[tr]); Xte = vec.transform(g.content[te])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    # label efficiency
    le = label_efficiency_tfidf(g, fracs)
    return oof, le


def label_efficiency_tfidf(g, fracs):
    y = g.label.values; folds = g.fold.values
    rng = np.random.default_rng(SEED); res = {}
    for fr in fracs:
        f1s = []
        for f in sorted(set(folds)):
            tr_idx = np.where(folds != f)[0]; te = folds == f
            k = max(10, int(len(tr_idx) * fr)); sub = rng.choice(tr_idx, k, replace=False)
            vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
                                  stop_words="english")
            Xtr = vec.fit_transform(g.content.iloc[sub]); Xte = vec.transform(g.content[te])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(Xtr, y[sub]); pred = clf.predict(Xte)
            f1s.append(f1_score(y[te], pred, average="macro"))
        res[fr] = round(float(np.mean(f1s)), 3)
    return res


def run_transformer(g, model_name, fracs):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = g.label.values; folds = g.fold.values
    tok = AutoTokenizer.from_pretrained(model_name)
    salient = g.content.map(salient_text)

    def train_eval(train_idx, test_idx):
        enc_tr = tok(list(salient.iloc[train_idx]), truncation=True, padding=True,
                     max_length=MAXLEN, return_tensors="pt")
        enc_te = tok(list(salient.iloc[test_idx]), truncation=True, padding=True,
                     max_length=MAXLEN, return_tensors="pt")
        ytr = torch.tensor(y[train_idx])
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        n = len(train_idx)
        steps = EPOCHS * ((n + BS - 1) // BS)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                    pct_start=0.1)
        cw = torch.tensor([1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float).to(device)
        lossf = torch.nn.CrossEntropyLoss(weight=cw)
        model.train()
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BS):
                b = perm[i:i + BS]
                ids = enc_tr["input_ids"][b].to(device); am = enc_tr["attention_mask"][b].to(device)
                opt.zero_grad()
                out = model(input_ids=ids, attention_mask=am).logits
                loss = lossf(out, ytr[b].to(device)); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
        model.eval(); probs = []
        with torch.no_grad():
            for i in range(0, len(test_idx), 16):
                ids = enc_te["input_ids"][i:i + 16].to(device); am = enc_te["attention_mask"][i:i + 16].to(device)
                p = torch.softmax(model(input_ids=ids, attention_mask=am).logits, -1)[:, 1]
                probs.append(p.cpu().numpy())
        del model; torch.cuda.empty_cache()
        return np.concatenate(probs)

    oof = np.zeros(len(g))
    for f in sorted(set(folds)):
        tr = np.where(folds != f)[0]; te = np.where(folds == f)[0]
        oof[te] = train_eval(tr, te)
        print(f"  fold {f} done")
    # label efficiency
    rng = np.random.default_rng(SEED); le = {}
    for fr in fracs:
        f1s = []
        for f in sorted(set(folds)):
            tr_idx = np.where(folds != f)[0]; te = np.where(folds == f)[0]
            k = max(10, int(len(tr_idx) * fr)); sub = rng.choice(tr_idx, k, replace=False)
            prob = train_eval(sub, te)
            f1s.append(f1_score(y[te], (prob >= 0.5).astype(int), average="macro"))
        le[fr] = round(float(np.mean(f1s)), 3)
        print(f"  labeleff frac={fr} f1={le[fr]}")
    return oof, le


def finalize(name, g, oof, le):
    y = g.label.values; folds = g.fold.values
    pred = (oof >= 0.5).astype(int)
    m = metrics(y, pred, oof)
    cal = platt_cv(oof, y, folds)
    m["ece_raw"] = round(ece(y, oof), 3)
    m["ece_platt"] = round(ece(y, cal), 3)
    m["label_efficiency_macroF1"] = le
    np.savez(os.path.join(RESULTS, f"e1_{name}_oof.npz"), oof=oof, cal=cal, y=y, folds=folds)
    with open(os.path.join(RESULTS, f"e1_{name}.json"), "w") as fp:
        json.dump(m, fp, indent=2)
    print(name, json.dumps(m, indent=2))
    return m


def run_zeroshot(g):
    z = load_zeroshot_gold().set_index("pmcid").reindex(g.pmcid)["pred"].values
    y = g.label.values
    m = metrics(y, z, None)
    with open(os.path.join(RESULTS, "e1_zeroshot.json"), "w") as fp:
        json.dump(m, fp, indent=2)
    print("zeroshot", json.dumps(m, indent=2))
    return m


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    g = load_gold()
    fracs = [0.25, 0.5, 0.75, 1.0]
    if which in ("tfidf", "all"):
        oof, le = run_tfidf(g, fracs); finalize("tfidf", g, oof, le)
    if which in ("zeroshot", "all"):
        run_zeroshot(g)
    if which in ("bert", "all"):
        oof, le = run_transformer(g, HF["bert"], fracs); finalize("bert", g, oof, le)
    if which in ("biobert", "all"):
        oof, le = run_transformer(g, HF["biobert"], fracs); finalize("biobert", g, oof, le)


if __name__ == "__main__":
    main()
