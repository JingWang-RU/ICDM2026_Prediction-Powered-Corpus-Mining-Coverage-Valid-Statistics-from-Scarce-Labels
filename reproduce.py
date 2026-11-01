# %% [markdown]
# # Prediction-Powered Corpus Mining — reproduction notebook
#
# This single notebook regenerates **every number, table and figure** of
# *Prediction-Powered Corpus Mining: Coverage-Valid Statistics from Scarce Labels*
# (IEEE ICDM 2026) from (i) two public benchmarks downloaded on the fly
# (Hallmarks of Cancer, ADE corpus v2) and (ii) the small derived-data bundle in `data/`
# (PMCIDs, physician labels, LLM scores, similarity scores, structured extractions).
# Article full text is **not** redistributed; the GPU steps that produced the cached
# LLM scores / encoder results are in `scripts/` and are optional.
#
# Run top-to-bottom (`Kernel > Restart & Run All`, ~15 min on a laptop CPU) or `python reproduce.py`.
# Outputs: `outputs/paper_numbers.json` (every quoted number), `outputs/tables/*.tex`
# (table bodies inlined into the paper) and `outputs/figs/*.pdf`.
#
# Section map (numbering of the final ICDM 2026 paper): §1 data card (Fig. 2, Sec. V-A) · §2 classification,
# Table I & Fig. 4 · §3 coverage on HoC and ADE, Table II & Fig. 5(a–c) · §4 robustness / ablations, Table IV ·
# §5 assumption violations (Sec. V-E) · §6 Long COVID applied (supplementary, not printed in the final paper) ·
# §7 measured clinical coverage, Table III & Fig. 5(d) · §8 retrieval operating point, Fig. 3 ·
# §9 association mining, Table V · §10 numbers quoted in the abstract and contributions · §11 export. README.md lists the few paper numbers not regenerated here.

# %% [markdown]
# ## 0. Setup

# %%
import os, sys, json, re, math, glob, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, fisher_exact
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, precision_recall_curve)
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")   # public datasets only; no token needed
# The paper's estimators (Algorithm 1 and the Sec. V-C baselines) live in ppi_corpus.py, the released algorithm.
from ppi_corpus import (ppi_plusplus, cross_ppi, crossfit_calibrate, platt_fit, platt_apply, choose_threshold,
                        union_screen, naive_ci, gold_only_ci, wald_ci, acc_ci)

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "outputs")
FIGS, TABS = os.path.join(OUT, "figs"), os.path.join(OUT, "tables")
for d in (OUT, FIGS, TABS):
    os.makedirs(d, exist_ok=True)
SEED, ALPHA = 42, 0.05
CLIP_CI = False   # the paper's tables report intervals without the [0, 1] intersection of Eq. (21); see README
R_DRAWS = int(os.environ.get("R_DRAWS", "500"))      # coverage draws on the public benchmarks
R_CLIN = int(os.environ.get("R_CLIN", "1000"))       # coverage draws on the clinical labels
NUM = {}                                             # every number quoted in the paper


def rec(key, val):
    NUM[key] = val
    return val


def save_fig(fig, name):
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight"); plt.close(fig); print("  fig  ->", name)


def save_tab(name, lines):
    with open(os.path.join(TABS, name), "w") as f:
        f.write("\n".join(lines))          # no trailing newline: the paper inlines these inside tabulars
    print("  table->", name)


plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "axes.labelsize": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold", "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8, "figure.dpi": 200, "axes.grid": True, "grid.alpha": 0.28,
                     "lines.linewidth": 1.7, "axes.linewidth": 0.9, "legend.framealpha": 0.92, "legend.edgecolor": "#cccccc"})
BLUE, RED, GREEN, GREY, PURPLE, ORANGE = "#2166ac", "#b2182b", "#1b7837", "#7f7f7f", "#762a83", "#e08214"
print("ppi_py OK; outputs ->", os.path.relpath(OUT, HERE))

# %% [markdown]
# ### Statistical helpers
# Wald interval for naive counting, Platt scaling, ECE, and the quantification-learning
# baseline **Adjusted Classify-and-Count** (ACC; Forman 2008 / Saerens et al. 2002) with a
# nonparametric bootstrap interval.

# %%
def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & ((p <= bins[i + 1]) if i == n_bins - 1 else (p < bins[i + 1]))
        if m.sum():
            e += m.sum() / len(y) * abs(p[m].mean() - y[m].mean())
    return float(e)


def reliability(y, p, nb=10):
    bins = np.linspace(0, 1, nb + 1); xs, ys = [], []
    for i in range(nb):
        m = (p >= bins[i]) & ((p <= bins[i + 1]) if i == nb - 1 else (p < bins[i + 1]))
        if m.sum():
            xs.append(float(p[m].mean())); ys.append(float(y[m].mean()))
    return xs, ys


def clf_metrics(y, pred, score=None):
    m = dict(acc=round(accuracy_score(y, pred), 3), macro_f1=round(f1_score(y, pred, average="macro"), 3),
             prec1=round(precision_score(y, pred, zero_division=0), 3), rec1=round(recall_score(y, pred, zero_division=0), 3))
    if score is not None and len(np.unique(np.round(score, 4))) > 2:
        m["auroc"] = round(roc_auc_score(y, score), 3)
    return m


def fp_ci(y_s, f_s, fbar_N, n, N, lam=None):
    """Exact finite-population interval for the mean of a KNOWN population of N units when a
    simple random sample of n is labeled and every unit is scored.

    With all N units scored, the PPI/PPI++ point estimator coincides with the survey
    difference (regression) estimator
        theta(lam) = ybar_S + lam (fbar_N - fbar_S),
    whose design variance is (1 - n/N) S_d^2 / n with d_i = y_i - lam f_i.  lam=0 is the
    gold-only estimator; lam=None uses the variance-minimising Cov(y,f)/Var(f), clipped to
    [0,1] exactly as PPI++ clips its power-tuning parameter.  The (1 - n/N) factor is the
    finite-population correction; omitting it (the superpopulation form used on the public
    benchmarks, where N is effectively infinite) makes both intervals conservative here.
    """
    y_s = np.asarray(y_s, float); f_s = np.asarray(f_s, float)
    if lam is None:
        vf = f_s.var(ddof=1)
        lam = float(np.cov(y_s, f_s, ddof=1)[0, 1] / vf) if vf > 1e-12 else 0.0
        lam = min(max(lam, 0.0), 1.0)
    theta = float(y_s.mean() + lam * (fbar_N - f_s.mean()))
    var = (1.0 - n / N) * float((y_s - lam * f_s).var(ddof=1)) / n
    h = float(norm.ppf(1 - ALPHA / 2) * np.sqrt(max(var, 0.0)))
    return theta, theta - h, theta + h, float(lam)


def coverage_study_fp(y, f, ngs, R, seed):
    """Clinical stress test done as an exact finite-population design (Table III, Fig. 5d)."""
    rng = np.random.default_rng(seed); truth = float(y.mean()); N = len(y); fbar = float(np.mean(f)); out = {}
    for ng in ngs:
        c = {"naive": 0, "gold": 0, "ppi": 0}; w = {"naive": 0.0, "gold": 0.0, "ppi": 0.0}; lams = []
        for _ in range(R):
            gi = rng.choice(N, ng, replace=False); mask = np.ones(N, bool); mask[gi] = False
            ys, fs = y[gi].astype(float), np.asarray(f, float)[gi]
            _, lo, hi, _ = fp_ci(ys, fs, fbar, ng, N, lam=0.0); c["gold"] += lo <= truth <= hi; w["gold"] += hi - lo
            _, lo, hi, lam = fp_ci(ys, fs, fbar, ng, N); c["ppi"] += lo <= truth <= hi; w["ppi"] += hi - lo; lams.append(lam)
            # naive practice: tally the labeler over the WHOLE population (no labels are used),
            # with the usual binomial interval. Its failure is bias, not the sampling design.
            lo, hi = wald_ci(fbar, N)
            c["naive"] += lo <= truth <= hi; w["naive"] += hi - lo
        out[int(ng)] = {**{f"{m}_cov": round(c[m] / R, 3) for m in c}, **{f"{m}_width": round(w[m] / R, 4) for m in w},
                        "lam_mean": round(float(np.mean(lams)), 3)}
    return truth, out


def coverage_study(y, f_cal, f_hard, ngs, R, seed, methods=("naive", "acc", "gold", "ppi"), B=200):
    """Empirical 95% coverage / mean width over R random gold draws of size ng from a
    finite population with known truth mean(y). PPI uses the continuous (calibrated) score
    with PPI++ power tuning; naive/ACC use the hard decisions; gold uses labels only."""
    rng = np.random.default_rng(seed); truth = float(y.mean()); N = len(y); out = {}
    for ng in ngs:
        cov = {m: 0 for m in methods}; wid = {m: 0.0 for m in methods}; pts = []
        for _ in range(R):
            gi = rng.choice(N, ng, replace=False); mask = np.ones(N, bool); mask[gi] = False
            Y = y[gi].astype(float)
            r = ppi_plusplus(Y, f_cal[gi], f_cal[mask], alpha=ALPHA, clip=CLIP_CI); lo, hi = r["ci"]
            cov["ppi"] += lo <= truth <= hi; wid["ppi"] += hi - lo
            pts.append(r["estimate"])
            lo, hi = gold_only_ci(Y, alpha=ALPHA, clip=CLIP_CI)["ci"]
            cov["gold"] += lo <= truth <= hi; wid["gold"] += hi - lo
            lo, hi = naive_ci(f_hard[mask], alpha=ALPHA, clip=CLIP_CI)["ci"]
            cov["naive"] += lo <= truth <= hi; wid["naive"] += hi - lo
            if "acc" in methods:
                lo, hi = acc_ci(y[gi], f_hard[gi], f_hard[mask], rng, B)
                cov["acc"] += lo <= truth <= hi; wid["acc"] += hi - lo
        out[int(ng)] = {**{f"{m}_cov": round(cov[m] / R, 3) for m in methods},
                        **{f"{m}_width": round(wid[m] / R, 4) for m in methods},
                        "ppi_point_mean": round(float(np.mean(pts)), 4)}
    return truth, out


def fixed_predictor(train_txt, ytr, pop_txt, seed=SEED):
    """TF-IDF + logistic regression fit on 80% of the disjoint train split; Platt scaling on
    the other 20%; scores for the estimation population (never seen in training)."""
    rng = np.random.default_rng(seed); n = len(train_txt); idx = rng.permutation(n); cut = int(0.8 * n)
    vec = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), sublinear_tf=True); Xtr = vec.fit_transform(train_txt)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr[idx[:cut]], ytr[idx[:cut]])
    held_raw = clf.predict_proba(Xtr[idx[cut:]])[:, 1]; held_y = ytr[idx[cut:]]
    platt = platt_fit(held_raw, held_y)
    raw = clf.predict_proba(vec.transform(pop_txt))[:, 1]; cal = platt_apply(platt, raw)
    return dict(raw=raw, cal=cal, hard=(cal >= 0.5).astype(float), held_raw=held_raw, held_y=held_y)

# %% [markdown]
# ## 1. Data card (corpus numbers used in Fig. 2 and Sec. V-A)
# Long COVID ground truth (physician-curated, 123 reports, 52/71), unlabeled pool (3,066
# screened case reports, disjoint from the ground truth), treatment–outcome subset (551
# reports with structured extractions), plus the two public truth-known benchmarks.

# %%
gold = pd.read_csv(os.path.join(DATA, "longcovid_gold.csv"))            # pmcid, label, fold
BENCH_LLM = {}          # zero-shot LLM predictions on the benchmark populations (optional arm, §3)
for _b in ("hoc", "ade"):
    _fp = os.path.join(DATA, "bench_llm", f"{_b}.csv")
    if os.path.exists(_fp):
        BENCH_LLM[_b] = pd.read_csv(_fp)
print("benchmark LLM predictions available for:", list(BENCH_LLM) or "none (TF-IDF predictor only)")
screen_gold = pd.read_csv(os.path.join(DATA, "screen_gold.csv"))
screen_pool = pd.read_csv(os.path.join(DATA, "screen_pool.csv"))
assert (screen_gold.pmcid.values == gold.pmcid.values).all()
rec("n_gold", int(len(gold))); rec("n_gold_pos", int(gold.label.sum())); rec("n_gold_neg", int((gold.label == 0).sum()))
rec("n_pool", int(len(screen_pool))); rec("gold_pool_overlap", int(len(set(gold.pmcid) & set(screen_pool.pmcid))))
rec("gold_prevalence", round(float(gold.label.mean()), 4))
print(f"gold n={NUM['n_gold']} ({NUM['n_gold_pos']}/{NUM['n_gold_neg']}), prevalence {NUM['gold_prevalence']}; "
      f"pool N={NUM['n_pool']}; overlap {NUM['gold_pool_overlap']}; folds {sorted(gold.fold.unique())}")


def load_llm(tag):
    """Released LLM scores: pmcid, pred (hard 0/1), p_yes (continuous) for gold and (optionally) pool."""
    fg = os.path.join(DATA, "llm", f"{tag}_gold.csv"); fp = os.path.join(DATA, "llm", f"{tag}_pool.csv")
    if not os.path.exists(fg):
        return None
    g = pd.read_csv(fg).set_index("pmcid").reindex(gold.pmcid)
    assert g.pred.notna().all(), tag
    p = pd.read_csv(fp).set_index("pmcid").reindex(screen_pool.pmcid) if os.path.exists(fp) else None
    if "p_yes" not in g:          # hard-label-only file (accepted-version provenance): score = decision
        g["p_yes"] = g.pred.astype(float)
    if p is not None and "p_yes" not in p:
        p["p_yes"] = p.pred.astype(float)
    return dict(tag=tag, gold_pred=g.pred.values.astype(int), gold_p=g.p_yes.values.astype(float),
                pool_pred=None if p is None else p.pred.values.astype(int),
                pool_p=None if p is None else p.p_yes.values.astype(float))


LABELERS = [("qwen38_27b", "Qwen3.8-27B"), ("qwen14b", "Qwen3-14B"), ("qwen14b4bit", "Qwen3-14B 4-bit"), ("llama8b", "Llama-3.1-8B"),
            ("qwen30b", "Qwen3-30B-A3B"), ("qwen32b", "Qwen3-32B"), ("qwen72b", "Qwen2.5-72B (AWQ)"), ("qwen14b_4bit_hard", "Qwen3-14B (accepted run)")]
MAIN = os.environ.get("MAIN_TAG", "qwen38_27b")   # default weak labeler: Qwen3.8-27B, bf16, vLLM, greedy
ACCEPTED = "qwen14b_4bit_hard"                 # hard labels of the run used in the accepted version (provenance)
if MAIN not in dict(LABELERS):
    LABELERS.insert(0, (MAIN, MAIN))
LLM = {tag: load_llm(tag) for tag, _ in LABELERS}
LLM = {k: v for k, v in LLM.items() if v is not None}
NAMES = dict(LABELERS)
rec("default_labeler", {"tag": MAIN, "name": NAMES.get(MAIN, MAIN)}); rec("labelers_available", [NAMES.get(t, t) for t in LLM])
print("labelers available:", list(LLM), "| default:", MAIN)
assert MAIN in LLM and LLM[MAIN]["pool_pred"] is not None, f"data/llm/{MAIN}_{{gold,pool}}.csv are required"

# %% [markdown]
# ## 2. Weak-supervision extraction quality (Table I, Fig. 4)
# Fine-tuned encoders and the TF-IDF baseline were trained with 5-fold CV on the article
# full text (GPU / full-text step, `scripts/e1_baselines.py`, `scripts/e1_chunked_control.py`);
# their out-of-fold outputs are cached in `data/cached_gpu/`. Zero-shot LLM rows are
# recomputed here from the released scores.

# %%
def cached(name):
    fp = os.path.join(DATA, "cached_gpu", name); return json.load(open(fp)) if os.path.exists(fp) else None


rows_t1 = []
for key, nm in [("e1_tfidf.json", "TF-IDF~\\cite{salton1988term}+LogReg"), ("e1_bert.json", "BERT~\\cite{devlin2019bert}"),
                ("e1_biobert.json", "BioBERT~\\cite{lee2020biobert}")]:
    j = cached(key)
    if j:
        rows_t1.append((nm, j)); rec(f"t1_{key[3:-5]}", {k: j[k] for k in ("acc", "macro_f1", "prec1", "rec1", "auroc")})
for key, nm in [("e1_bert_chunked.json", "BERT, full doc"), ("e1_biobert_chunked.json", "BioBERT, full doc")]:
    j = cached(key)
    if j:
        m = j["mean_pool"]; rows_t1.append((nm, m)); rec(f"t1_{key[3:-5]}", {**m, "chunks_per_doc": j["chunks_per_doc"]})
y_gold = gold.label.values
# frozen-embedding linear probes (stronger supervised baseline): embeddings of the 123 gold reports
# were computed once on GPU (scripts/e1_embed_probe.py); the probe itself is refit here under the same folds.
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
PROBES = [("qwen3emb8b", "Qwen3-Embedding-8B probe"), ("zembed1", "zembed-1 probe"), ("bgem3", "BGE-M3 probe"), ("medgemma4b", "MedGemma-4B probe")]


def linear_probe(X, y, folds, fracs=(0.25, 0.5, 0.75, 1.0)):
    fit = lambda Xtr, ytr: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")).fit(Xtr, ytr)
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


PROBE_RES = {}
for tag, nm in PROBES:
    fp = os.path.join(DATA, "cached_gpu", f"gold_embeddings_{tag}.npy")
    if not os.path.exists(fp):
        continue
    X = np.load(fp); assert X.shape[0] == len(gold)
    oof, le = linear_probe(X, y_gold, gold.fold.values)
    m = clf_metrics(y_gold, (oof >= 0.5).astype(int), oof); m["label_efficiency_macroF1"] = le; m["dim"] = int(X.shape[1])
    PROBE_RES[tag] = m; rec(f"t1_probe_{tag}", m); rows_t1.append((nm, m))
for tag, nm in LABELERS:
    if tag in LLM:
        m = clf_metrics(y_gold, LLM[tag]["gold_pred"], LLM[tag]["gold_p"]); rec(f"t1_{tag}", m)
        rec(f"{tag}_gold_yes_rate", round(float(LLM[tag]["gold_pred"].mean()), 3))
        if tag == MAIN:
            rows_t1.append((f"{NAMES.get(tag, tag)}, 0-shot~\\cite{{yang2025qwen3}}", m))
lines = []
for nm, m in rows_t1:
    au = f"{m['auroc']:.3f}" if m.get("auroc") is not None else "$-$"
    lines.append(f"{nm} & {m['acc']:.3f} & {m['macro_f1']:.3f} & {m['prec1']:.3f} & {m['rec1']:.3f} & {au} \\\\")
if PROBE_RES:
    i_probe = next(i for i, (nm, _) in enumerate(rows_t1) if nm.endswith("probe"))
    lines.insert(i_probe, "\\midrule")
    lines.insert(len(lines) - 1, "\\midrule")
    rec("probe_macro_f1_range", [min(m["macro_f1"] for m in PROBE_RES.values()), max(m["macro_f1"] for m in PROBE_RES.values())])
    rec("probe_acc_range", [min(m["acc"] for m in PROBE_RES.values()), max(m["acc"] for m in PROBE_RES.values())])
    rec("probe_best", max(PROBE_RES, key=lambda t: PROBE_RES[t]["macro_f1"]))
save_tab("tab_e1_rows.tex", lines)
print(pd.DataFrame([{"model": nm, **m} for nm, m in rows_t1]).to_string(index=False))

# %%
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.85))
for key, lab, c, mk in [("e1_tfidf.json", "TF-IDF+LR", BLUE, "o"), ("e1_bert.json", "BERT", GREEN, "s"), ("e1_biobert.json", "BioBERT", PURPLE, "^")]:
    j = cached(key)
    if j:
        le = j["label_efficiency_macroF1"]; xs = sorted(float(k) for k in le)
        ax[0].plot([x * 100 for x in xs], [le[str(x)] for x in xs], marker=mk, color=c, label=lab)
for (tag, nm), c, mk in zip(PROBES, [ORANGE, "#c51b7d", "#4d9221", "#8c510a"], ["D", "v", "P", "X"]):
    if tag in PROBE_RES:
        le = PROBE_RES[tag]["label_efficiency_macroF1"]; xs = sorted(float(k) for k in le)
        ax[0].plot([x * 100 for x in xs], [le[str(x)] for x in xs], marker=mk, color=c, ls=":", lw=1.3, ms=4, label=nm)
ax[0].axhline(NUM[f"t1_{MAIN}"]["macro_f1"], color=RED, ls="--", lw=1.5, label=f"zero-shot {NAMES.get(MAIN, MAIN)} (default)")
ax[0].set_xlabel("% of ground-truth labels used"); ax[0].set_ylabel("macro-F1 (5-fold CV)"); ax[0].set_ylim(0.38, 0.92)
ax[0].set_title("(a) Label efficiency"); ax[0].legend(fontsize=6.0, loc="upper left", handlelength=1.5, ncol=2, columnspacing=0.8, labelspacing=0.25)
labs = [(nm, NUM[f"t1_{tag}"]) for tag, nm in LABELERS if tag in LLM]; x = np.arange(len(labs))
ax[1].bar(x - 0.25, [m["acc"] for _, m in labs], 0.25, label="accuracy", color=BLUE)
ax[1].bar(x, [m["prec1"] for _, m in labs], 0.25, label="precision$_1$", color=GREEN)
ax[1].bar(x + 0.25, [m["rec1"] for _, m in labs], 0.25, label="recall$_1$", color=RED)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{n} (default)" if n == NAMES.get(MAIN, MAIN) else n for n, _ in labs], fontsize=6.6, rotation=30, ha="right", rotation_mode="anchor")
ax[1].set_ylim(0, 1.16); ax[1].set_ylabel("score on ground truth"); ax[1].set_title("(b) Zero-shot labelers (same prompt)"); ax[1].tick_params(axis="x", pad=1.5)
ax[1].legend(fontsize=7.6, ncol=3, loc="upper center", columnspacing=0.9, handlelength=1.2)
fig.tight_layout(w_pad=1.6); save_fig(fig, "fig_e1.pdf")
rec("labeler_acc_range", [min(m["acc"] for _, m in labs), max(m["acc"] for _, m in labs)])

# %% [markdown]
# ## 3. Does the interval cover the truth? Two public truth-known benchmarks (Table II, Fig. 5a–c)
# **HoC** (sentence-level; positive = a specific, non-generic cancer hallmark) and **ADE
# corpus v2** (sentence-level; positive = sentence reports an adverse drug event). A fixed
# TF-IDF classifier is trained on the disjoint train split; the held-out population is the
# estimation universe. For each gold size we repeat 500 random gold draws and compare naive
# LLM-style counting, Adjusted Classify-and-Count (quantification learning), the gold-only
# interval and PPI (calibrated scores, PPI++ power tuning).

# %%
from datasets import load_dataset

NGS = [50, 100, 150, 200, 300]


def load_hoc():
    ds = load_dataset("qanastek/HoC", revision="refs/convert/parquet")
    biny = lambda r: int(any(k != 7 for k in r["label"]))
    tr = list(ds["train"]); pop = list(ds["validation"]) + list(ds["test"])
    return ([r["text"] for r in tr], np.array([biny(r) for r in tr]), [r["text"] for r in pop], np.array([biny(r) for r in pop]))


def load_ade():
    # SetFit mirror of ADE corpus v2 (classification). The mirror's splits share sentences
    # (~14% of test also in train), so we de-duplicate and keep the population disjoint from train.
    ds = load_dataset("SetFit/ade_corpus_v2_classification")
    tr = pd.DataFrame(ds["train"]).drop_duplicates("text"); te = pd.DataFrame(ds["test"]).drop_duplicates("text")
    te = te[~te.text.isin(set(tr.text))]
    return (list(tr.text), tr.label.values.astype(int), list(te.text), te.label.values.astype(int))


BENCH = {}
for name, loader in [("hoc", load_hoc), ("ade", load_ade)]:
    t0 = time.time(); tr_txt, ytr, pop_txt, ypop = loader()
    P = fixed_predictor(tr_txt, ytr, pop_txt)
    truth, cov = coverage_study(ypop, P["cal"], P["hard"], NGS, R_DRAWS, SEED)
    info = dict(n_train=len(tr_txt), n_pop=len(ypop), true_rate=round(truth, 4),
                f_acc=round(accuracy_score(ypop, P["hard"]), 3), f_macro_f1=round(f1_score(ypop, P["hard"], average="macro"), 3),
                naive_pred_prevalence=round(float(P["hard"].mean()), 4), naive_bias=round(float(P["hard"].mean() - truth), 4),
                ece_raw=round(ece(ypop, P["raw"]), 4), ece_platt=round(ece(ypop, P["cal"]), 4), R=R_DRAWS, coverage=cov)
    rec(f"bench_{name}", info); BENCH[name] = dict(P=P, y=ypop, cov=cov, info=info)
    print(f"[{name}] N_train={info['n_train']} N_pop={info['n_pop']} truth={info['true_rate']} f_acc={info['f_acc']} "
          f"bias={info['naive_bias']:+.3f} ECE {info['ece_raw']}->{info['ece_platt']}  ({time.time()-t0:.0f}s)")
    print(pd.DataFrame(cov).T.to_string())
    # same coverage study with the paper's zero-shot LLM as the predictor, so the truth-known
    # validation uses the same kind of predictor as the clinical deployment (Sec. V-C).
    if name in BENCH_LLM:
        d = BENCH_LLM[name]; yl = d.label.values.astype(int)
        assert len(yl) == len(ypop) and abs(yl.mean() - truth) < 1e-9, "benchmark LLM rows must match the population"
        # No labeled train split exists for the LLM, so its yes/no probability p_yes is passed to PPI
        # as is (NOT recalibrated; PPI validity does not need a calibrated score), and its hard
        # decisions `pred` drive the naive and ACC estimators.
        truth_l, cov_l = coverage_study(yl, d.p_yes.values.astype(float), d.pred.values.astype(float), NGS, R_DRAWS, SEED)
        info_l = dict(n_pop=len(yl), true_rate=round(truth_l, 4), f_acc=round(accuracy_score(yl, d.pred.values), 3),
                      naive_pred_prevalence=round(float(d.pred.mean()), 4), naive_bias=round(float(d.pred.mean() - truth_l), 4),
                      auroc=round(roc_auc_score(yl, d.p_yes.values), 3), R=R_DRAWS, coverage=cov_l)
        rec(f"bench_{name}_llm", info_l); BENCH[name]["llm"] = dict(cov=cov_l, info=info_l)
        print(f"[{name}/LLM] acc={info_l['f_acc']} bias={info_l['naive_bias']:+.3f} AUROC={info_l['auroc']} "
              f"naive_cov={[cov_l[n]['naive_cov'] for n in NGS]} ppi_cov={[cov_l[n]['ppi_cov'] for n in NGS]}")

LLM_BENCH = {n: BENCH[n]["llm"] for n in BENCH if "llm" in BENCH[n]}
if LLM_BENCH:
    allc = [v["cov"][n] for v in LLM_BENCH.values() for n in NGS]
    rec("bench_llm_ppi_cov_range", [min(c["ppi_cov"] for c in allc), max(c["ppi_cov"] for c in allc)])
    rec("bench_llm_naive_cov_range", [min(c["naive_cov"] for c in allc), max(c["naive_cov"] for c in allc)])
    rec("bench_llm_ppi_vs_gold_width_ratio", round(float(np.mean([c["ppi_width"] / c["gold_width"] for c in allc])), 3))
    rec("bench_llm_bias_pts", {n: round(100 * v["info"]["naive_bias"], 1) for n, v in LLM_BENCH.items()})

# %%
lines = []
for name, lab in [("hoc", "HoC"), ("ade", "ADE")]:
    cov = BENCH[name]["cov"]; info = BENCH[name]["info"]
    lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{lab}}} (bias ${info['naive_bias']*100:+.1f}$\\,pts)}}\\\\")
    for ng in [n for n in NGS if n in (50, 100, 200, 300)]:   # n_g=150 is the Table IV setting
        c = cov[ng]
        lines.append(f"{ng} & {c['naive_cov']:.3f} & {c['acc_cov']:.3f} & {c['gold_cov']:.3f} & {c['ppi_cov']:.3f} & "
                     f"{c['acc_width']:.3f} & {c['gold_width']:.3f} & \\textbf{{{c['ppi_width']:.3f}}} \\\\")
    if name == "hoc":
        lines.append("\\midrule")
for name, lab in [("hoc", "HoC"), ("ade", "ADE")]:
    if name not in LLM_BENCH:
        continue
    v = LLM_BENCH[name]; cov = v["cov"]; info = v["info"]
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{lab}}}, zero-shot LLM predictor (bias ${info['naive_bias']*100:+.1f}$\\,pts)}}\\\\")
    for ng in [n for n in NGS if n in (50, 100, 300)]:     # compact LLM blocks
        c = cov[ng]
        lines.append(f"{ng} & {c['naive_cov']:.3f} & {c['acc_cov']:.3f} & {c['gold_cov']:.3f} & {c['ppi_cov']:.3f} & "
                     f"{c['acc_width']:.3f} & {c['gold_width']:.3f} & \\textbf{{{c['ppi_width']:.3f}}} \\\\")
save_tab("tab_cov_rows.tex", lines)
for name in BENCH:
    cov = BENCH[name]["cov"]
    rec(f"{name}_ppi_cov_range", [min(cov[n]["ppi_cov"] for n in NGS), max(cov[n]["ppi_cov"] for n in NGS)])
    rec(f"{name}_naive_cov_max", max(cov[n]["naive_cov"] for n in NGS))
    rec(f"{name}_acc_cov_range", [min(cov[n]["acc_cov"] for n in NGS), max(cov[n]["acc_cov"] for n in NGS)])
    rec(f"{name}_ppi_vs_gold_width_ratio", round(float(np.mean([cov[n]["ppi_width"] / cov[n]["gold_width"] for n in NGS])), 3))
    rec(f"{name}_ppi_vs_gold_width_ratio_range", [round(min(cov[n]["ppi_width"] / cov[n]["gold_width"] for n in NGS), 3), round(max(cov[n]["ppi_width"] / cov[n]["gold_width"] for n in NGS), 3)])
    rec(f"{name}_acc_vs_ppi_width_ratio", round(float(np.mean([cov[n]["acc_width"] / cov[n]["ppi_width"] for n in NGS])), 3))
print({k: NUM[k] for k in NUM if k.startswith(("hoc_", "ade_"))})

# %%
xr, yr = reliability(BENCH["hoc"]["y"], BENCH["hoc"]["P"]["raw"]); xc, yc = reliability(BENCH["hoc"]["y"], BENCH["hoc"]["P"]["cal"])
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="perfect")
ax.plot(xr, yr, "o-", color=RED, label=f"raw (ECE {NUM['bench_hoc']['ece_raw']:.3f})")
ax.plot(xc, yc, "s-", color=BLUE, label=f"Platt (ECE {NUM['bench_hoc']['ece_platt']:.3f})")
ax.set_xlabel("predicted probability"); ax.set_ylabel("empirical positive rate"); ax.set_title("Reliability on HoC")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=7.6); save_fig(fig, "fig_calibration.pdf")

# %% [markdown]
# ## 4. Robustness and component ablation on HoC (Table IV)
# (a) pseudo-label form, (b) PPI++ power-tuning parameter, (c) calibration on/off ×
# recall-first screen on/off. The screen threshold is chosen on the *disjoint* held-out
# calibration split to keep 99% of positives; the estimand then becomes the positive rate
# of the screened pool and PPI is asked to cover *that*.

# %%
NG_ROB, R_ROB = 150, R_DRAWS
H = BENCH["hoc"]; yH = H["y"]; PH = H["P"]


def cov_width(y, f, ng, R, seed, lam=None, truth=None):
    rng = np.random.default_rng(seed); truth = float(y.mean()) if truth is None else truth; c = w = 0.0
    for _ in range(R):
        gi = rng.choice(len(y), ng, replace=False); mask = np.ones(len(y), bool); mask[gi] = False
        lo, hi = ppi_plusplus(y[gi], f[gi], f[mask], alpha=ALPHA, lam=lam, clip=CLIP_CI)["ci"]
        c += lo <= truth <= hi; w += hi - lo
    return round(c / R, 3), round(w / R, 4)


rob = {"pseudo_label_form": {}, "lambda": {}, "component": {}}
for nm, f in [("hard", PH["hard"]), ("raw_prob", PH["raw"]), ("calibrated_prob", PH["cal"])]:
    rob["pseudo_label_form"][nm] = cov_width(yH, f, NG_ROB, R_ROB, SEED)
for lam in [0.0, 0.5, 1.0, None]:
    rob["lambda"]["auto" if lam is None else str(lam)] = cov_width(yH, PH["cal"], NG_ROB, R_ROB, SEED, lam=lam)
# recall-first screen on the raw classifier score, threshold picked on the disjoint held-out split
hr, hy = PH["held_raw"], PH["held_y"]; cand = np.unique(hr[hy == 1])
tau = float(max(t for t in cand if (hr[hy == 1] >= t).mean() >= 0.99))
keep = PH["raw"] >= tau
screen_info = dict(tau=round(tau, 4), n_kept=int(keep.sum()), frac_pruned=round(1 - keep.mean(), 3),
                   recall_on_population=round(float(yH[keep].sum() / yH.sum()), 4), truth_screened=round(float(yH[keep].mean()), 4))
for cal_on in (False, True):
    f_full = PH["cal"] if cal_on else PH["raw"]
    rob["component"][f"cal_{'on' if cal_on else 'off'}_screen_off"] = cov_width(yH, f_full, NG_ROB, R_ROB, SEED)
    rob["component"][f"cal_{'on' if cal_on else 'off'}_screen_on"] = cov_width(yH[keep], f_full[keep], NG_ROB, R_ROB, SEED)
rec("robustness", rob); rec("hoc_screen", screen_info)
print(json.dumps(rob, indent=1)); print(screen_info)
lines = [f"pseudo-label: hard $\\{{0,1\\}}$ & {rob['pseudo_label_form']['hard'][0]:.3f} & {rob['pseudo_label_form']['hard'][1]:.3f} \\\\",
         f"pseudo-label: calibrated prob & {rob['pseudo_label_form']['calibrated_prob'][0]:.3f} & {rob['pseudo_label_form']['calibrated_prob'][1]:.3f} \\\\",
         "\\midrule"]
for k, lab in [("0.0", "$\\lambda{=}0$ (gold-only)"), ("1.0", "$\\lambda{=}1$"), ("auto", "$\\lambda{=}$\\texttt{auto} (PPI++)")]:
    c, w = rob["lambda"][k]; lines.append(f"{lab} & {c:.3f} & {w:.3f} \\\\")
lines.append("\\midrule")
for k, lab in [("cal_off_screen_on", "screen on, calibration off"), ("cal_on_screen_on", "screen on, calibration on")]:   # screen-off rows = the two pseudo-label rows above
    c, w = rob["component"][k]; lines.append(f"{lab} & {c:.3f} & {w:.3f} \\\\")
save_tab("tab_rob_rows.tex", lines)

# %% [markdown]
# ## 5. When the assumption is violated (HoC; Sec. V-E)
# (a) predictor fit on the very labels used for estimation: in-sample PPI vs K=5
# cross-prediction-powered inference; (b) non-representative gold (positives oversampled).

# %%
def crossppi_regime(pop_txt_vec, y, ng=200, R=200, K=5, seed=SEED):
    rng = np.random.default_rng(seed); truth = float(y.mean()); agg = {"naive_insample": [0, 0.0], "crossppi": [0, 0.0]}
    for _ in range(R):
        gi = rng.choice(len(y), ng, replace=False); mask = np.ones(len(y), bool); mask[gi] = False
        Xg, Yg, Xu = pop_txt_vec[gi], y[gi].astype(float), pop_txt_vec[mask]
        clf = LogisticRegression(max_iter=500).fit(Xg, Yg)
        lo, hi = ppi_plusplus(Yg, clf.predict_proba(Xg)[:, 1], clf.predict_proba(Xu)[:, 1], alpha=ALPHA, clip=CLIP_CI)["ci"]
        agg["naive_insample"][0] += lo <= truth <= hi; agg["naive_insample"][1] += hi - lo
        oof = np.zeros(ng); YuK = np.zeros((Xu.shape[0], K))
        for k, (tri, tei) in enumerate(KFold(K, shuffle=True, random_state=0).split(np.arange(ng))):
            m = LogisticRegression(max_iter=500).fit(Xg[tri], Yg[tri]); oof[tei] = m.predict_proba(Xg[tei])[:, 1]; YuK[:, k] = m.predict_proba(Xu)[:, 1]
        lo, hi = cross_ppi(Yg, oof, YuK, alpha=ALPHA, clip=CLIP_CI)["ci"]
        agg["crossppi"][0] += lo <= truth <= hi; agg["crossppi"][1] += hi - lo
    return {m: dict(coverage=round(v[0] / R, 3), width=round(v[1] / R, 4)) for m, v in agg.items()}


_, _, hoc_pop_txt, _ = load_hoc()
Xpop = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True).fit_transform(hoc_pop_txt)
xppi = crossppi_regime(Xpop, yH); rec("crossppi", xppi); print("cross-prediction regime:", xppi)


def nonrep_gold(y, f, ng=150, R=300, seed=SEED):
    rng = np.random.default_rng(seed); truth = float(y.mean()); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; out = {}
    for gr in [round(truth, 3), 0.35, 0.45, 0.55, 0.65]:
        pc = pw = gc = nc = pt = 0.0
        for _ in range(R):
            npos = min(int(round(ng * gr)), len(pos))
            gi = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, ng - npos, replace=False)])
            mask = np.ones(len(y), bool); mask[gi] = False; Y = y[gi].astype(float)
            lo, hi = ppi_plusplus(Y, f[gi], f[mask], alpha=ALPHA, clip=CLIP_CI)["ci"]; pc += lo <= truth <= hi; pw += hi - lo; pt += (lo + hi) / 2
            glo, ghi = gold_only_ci(Y, alpha=ALPHA, clip=CLIP_CI)["ci"]; gc += glo <= truth <= ghi
            nlo, nhi = naive_ci(f[mask] >= 0.5, alpha=ALPHA, clip=CLIP_CI)["ci"]; nc += nlo <= truth <= nhi
        out[str(gr)] = dict(ppi_cov=round(pc / R, 3), ppi_width=round(pw / R, 4), ppi_point=round(pt / R, 4), gold_cov=round(gc / R, 3), naive_cov=round(nc / R, 3))
    return out


nonrep = nonrep_gold(yH, PH["cal"]); rec("nonrep_gold", nonrep); print(pd.DataFrame(nonrep).T.to_string())

# %% [markdown]
# ## 6. Applied on Long COVID (supplementary: Table III of the accepted version, not printed in the final paper)
# The default zero-shot labeler (`MAIN`, Qwen3.8-27B) scores every pool document; the 123 physician labels are the PPI
# rectifier. Two pseudo-label forms: the labeler's **hard** decisions (no label is consumed
# by any calibrator) and **Platt-calibrated continuous scores, K=5 cross-fitted** on the
# gold folds (cross-prediction-powered inference), so no document's pseudo-label is ever
# calibrated on its own label. The gold is curated and disjoint from the pool, so this
# estimate is *applied-under-assumption*; the gold-only vs PPI divergence is the diagnostic.

# %%
Y = y_gold.astype(float); folds = gold.fold.values
glo, ghi = gold_only_ci(Y, alpha=ALPHA, clip=CLIP_CI)["ci"]


def longcovid_block(tag):
    L = LLM[tag]; hard_g, hard_p = L["gold_pred"].astype(float), L["pool_pred"].astype(float)
    nv = naive_ci(hard_p, alpha=ALPHA, clip=CLIP_CI); naive, (nlo, nhi) = nv["estimate"], nv["ci"]
    ph = ppi_plusplus(Y, hard_g, hard_p, alpha=ALPHA, clip=CLIP_CI); ppi_h, (hlo, hhi) = ph["estimate"], ph["ci"]
    out = dict(naive=round(naive, 6), naive_ci=[round(nlo, 6), round(nhi, 6)],
               ppi_hard=round(ppi_h, 6), ppi_hard_ci=[round(hlo, 6), round(hhi, 6)],
               n_pool_pos=int(hard_p.sum()), gold_pos_rate_given_no=round(float(y_gold[L["gold_pred"] == 0].mean()), 4),
               n_gold_no=int((L["gold_pred"] == 0).sum()), n_gold_no_pos=int(y_gold[L["gold_pred"] == 0].sum()),
               labeler_on_gold=clf_metrics(y_gold, L["gold_pred"], L["gold_p"]), gold_yes_rate=round(float(hard_g.mean()), 3),
               naive_bias_on_gold_pts=round(float(hard_g.mean() - Y.mean()) * 100, 1))
    continuous = len(np.unique(np.round(L["gold_p"], 4))) > 2
    if continuous:   # K=5 cross-fitted Platt calibration of the continuous score (cross-prediction-powered inference)
        # this supplementary analysis of the accepted version fit Platt with a unit ridge (C = 1)
        cal_g, cal_pK = crossfit_calibrate(L["gold_p"], y_gold, L["pool_p"], folds, C=1.0)
        cx = cross_ppi(Y, cal_g, cal_pK, alpha=ALPHA, clip=CLIP_CI); clo, chi = cx["ci"]
        out.update(ppi_cal_crossfit=round(cx["estimate"], 6), ppi_cal_crossfit_ci=[round(clo, 6), round(chi, 6)],
                   cal_pool_mean=round(float(cal_pK.mean()), 4),
                   ece_gold_raw=round(ece(y_gold, L["gold_p"]), 3), ece_gold_crossfit_platt=round(ece(y_gold, cal_g), 3))
    out["divergence_gold_minus_ppi"] = round(float(Y.mean() - out.get("ppi_cal_crossfit", out["ppi_hard"])), 4)
    return out


lc = dict(gold_only=round(float(Y.mean()), 4), gold_ci=[round(glo, 4), round(ghi, 4)], default=longcovid_block(MAIN))
# how far the curated set is from a random draw: surface-cue rate in the curated set vs the pool
lc["cue_rate_gold"] = round(float(screen_gold.regex_hit.mean()), 3)
lc["cue_rate_gold_neg"] = round(float(screen_gold.regex_hit[y_gold == 0].mean()), 3)
lc["cue_rate_pool"] = round(float(screen_pool.regex_hit.mean()), 3)
if ACCEPTED in LLM and LLM[ACCEPTED]["pool_pred"] is not None:
    lc["accepted_run"] = longcovid_block(ACCEPTED)
rec("longcovid", lc); print(json.dumps(lc, indent=1))


def ci(v): return f"$[{max(v[0], 0):.3f},\\,{v[1]:.3f}]$"


rows = [f"Curated set, positive fraction$^{{\\ast}}$ & {lc['gold_only']:.3f} & {ci(lc['gold_ci'])} \\\\", "\\midrule",
        f"\\multicolumn{{3}}{{l}}{{\\emph{{Default labeler}} ({NAMES.get(MAIN, MAIN)}; bias ${lc['default']['naive_bias_on_gold_pts']:+.1f}$\\,pts)}}\\\\",
        f"\\quad Naive zero-shot count & {lc['default']['naive']:.3f} & {ci(lc['default']['naive_ci'])} \\\\",
        f"\\quad PPI, hard pseudo-labels & {lc['default']['ppi_hard']:.3f} & {ci(lc['default']['ppi_hard_ci'])} \\\\",
        f"\\quad PPI, calibrated (cross-fitted) & {lc['default']['ppi_cal_crossfit']:.3f} & {ci(lc['default']['ppi_cal_crossfit_ci'])} \\\\"]
if "accepted_run" in lc:
    a = lc["accepted_run"]
    rows += ["\\midrule", f"\\multicolumn{{3}}{{l}}{{\\emph{{Accepted-run labeler}} (bias ${a['naive_bias_on_gold_pts']:+.1f}$\\,pts)}}\\\\",
             f"\\quad Naive zero-shot count & {a['naive']:.3f} & {ci(a['naive_ci'])} \\\\",
             f"\\quad PPI, hard pseudo-labels & {a['ppi_hard']:.3f} & {ci(a['ppi_hard_ci'])} \\\\"]
save_tab("tab_lc_rows.tex", rows)

# %% [markdown]
# ## 7. Measured coverage on the clinical labels (Table III, Fig. 5d)
# The 123 physician labels are treated as a finite population; a representative gold
# subset is re-drawn at random (so the PPI assumption holds by construction) and coverage
# is measured over 1,000 draws with the same zero-shot labeler (hard decisions). `fig_coverage.pdf`(d) plots the
# finite-population curves reported in Table III and Sec. V-D; the superpopulation curves (`coverage_superpop`,
# no finite-population correction) are recorded alongside, and the image printed as Fig. 5(d) of the final paper
# was rendered from those.

# %%
NGC = [30, 50, 70, 90]; NG_TAB = 50
CLIN = {}
for tag, nm in LABELERS:
    if tag not in LLM:
        continue
    hg = LLM[tag]["gold_pred"].astype(float)      # hard decisions, as deployed
    truth_c, cov = coverage_study_fp(y_gold, hg, NGC, R_CLIN, SEED)                       # exact finite-population design
    _, cov_sp = coverage_study(y_gold, hg, hg, NGC, R_CLIN, SEED, methods=("naive", "gold", "ppi"))  # superpopulation (conservative)
    CLIN[tag] = dict(name=nm, bias_pts=round(float(hg.mean() - y_gold.mean()) * 100, 1), coverage=cov, coverage_superpop=cov_sp)
    print(f"[{nm}] bias {CLIN[tag]['bias_pts']:+.1f} pts | n_g={NG_TAB} FPC: naive {cov[NG_TAB]['naive_cov']:.2f} "
          f"gold {cov[NG_TAB]['gold_cov']:.3f} PPI {cov[NG_TAB]['ppi_cov']:.3f} (widths {cov[NG_TAB]['gold_width']:.3f}/{cov[NG_TAB]['ppi_width']:.3f}) "
          f"| superpop PPI {cov_sp[NG_TAB]['ppi_cov']:.3f}")
rec("clinical_coverage", dict(true_rate=round(float(y_gold.mean()), 4), R=R_CLIN, n_g_table=NG_TAB, N_population=int(len(y_gold)),
                             design="simple random sample without replacement from the 123 labeled reports; every report is scored, "
                                    "so PPI++ coincides with the survey difference estimator and its variance carries the (1-n/N) factor",
                             by_labeler=CLIN))
_allc = [v["coverage"][n] for v in CLIN.values() for n in NGC]
_allsp = [v["coverage_superpop"][n] for v in CLIN.values() for n in NGC]
rec("clin_fpc_vs_superpop", dict(
    ppi_cov_fpc=[min(c["ppi_cov"] for c in _allc), max(c["ppi_cov"] for c in _allc)],
    ppi_cov_superpop=[min(c["ppi_cov"] for c in _allsp), max(c["ppi_cov"] for c in _allsp)],
    gold_cov_fpc=[min(c["gold_cov"] for c in _allc), max(c["gold_cov"] for c in _allc)],
    gold_cov_superpop=[min(c["gold_cov"] for c in _allsp), max(c["gold_cov"] for c in _allsp)],
    ppi_vs_gold_width_ratio_fpc=round(float(np.mean([c["ppi_width"] / c["gold_width"] for c in _allc])), 3),
    ppi_vs_gold_width_ratio_superpop=round(float(np.mean([c["ppi_width"] / c["gold_width"] for c in _allsp])), 3)))
clin = CLIN[MAIN]["coverage"]
TEXNAME = {"Qwen3-14B (accepted run)": "Qwen3-14B$^\\dagger$"}
save_tab("tab_clincov_rows.tex", [f"{TEXNAME.get(v['name'], v['name'])} & ${v['bias_pts']:+.1f}$ & {v['coverage'][NG_TAB]['naive_cov']:.3f} & {v['coverage'][NG_TAB]['gold_cov']:.3f} & "
                                  f"{v['coverage'][NG_TAB]['ppi_cov']:.3f} & {v['coverage'][NG_TAB]['gold_width']:.3f} & \\textbf{{{v['coverage'][NG_TAB]['ppi_width']:.3f}}} \\\\"
                                  for t, v in CLIN.items()])
allc = [v["coverage"][n] for v in CLIN.values() for n in NGC]
rec("clin_ppi_cov_range", [min(c["ppi_cov"] for c in allc), max(c["ppi_cov"] for c in allc)])
rec("clin_naive_cov_range_biased", [min(v["coverage"][n]["naive_cov"] for v in CLIN.values() for n in NGC if abs(v["bias_pts"]) >= 5),
                                    max(v["coverage"][n]["naive_cov"] for v in CLIN.values() for n in NGC if abs(v["bias_pts"]) >= 5)])
rec("clin_ppi_narrower_than_gold_all", bool(all(c["ppi_width"] <= c["gold_width"] for c in allc)))

# %%
# fig_coverage.pdf: panel (d) = finite-population curves (Table III, Sec. V-D).
# fig_coverage_as_printed.pdf: panel (d) = superpopulation curves, the image printed as Fig. 5.
for key, fname in [("coverage", "fig_coverage.pdf"), ("coverage_superpop", "fig_coverage_as_printed.pdf")]:
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 2.15))
    for i, (name, lab) in enumerate([("hoc", "HoC"), ("ade", "ADE")]):
        cov = BENCH[name]["cov"]
        ax[i].axhline(0.95, color=GREY, ls="--", lw=1, label="nominal 0.95")
        ax[i].plot(NGS, [cov[n]["ppi_cov"] for n in NGS], "o-", color=BLUE, label="PPI")
        ax[i].plot(NGS, [cov[n]["gold_cov"] for n in NGS], "s-", color=GREEN, label="gold-only")
        ax[i].plot(NGS, [cov[n]["acc_cov"] for n in NGS], "d-", color=ORANGE, label="ACC")
        ax[i].plot(NGS, [cov[n]["naive_cov"] for n in NGS], "^-", color=RED, label="naive count")
        ax[i].set_ylim(-0.05, 1.1); ax[i].set_xlabel("# gold labels"); ax[i].set_title(f"({'ab'[i]}) {lab} coverage")
    ax[0].set_ylabel("empirical 95% coverage"); ax[0].legend(loc="center right", fontsize=6.0, handlelength=1.2, labelspacing=0.3)
    for name, ls, lab in [("hoc", "-", "HoC"), ("ade", "--", "ADE")]:
        cov = BENCH[name]["cov"]
        ax[2].plot(NGS, [cov[n]["ppi_width"] for n in NGS], "o" + ls, color=BLUE, label=f"PPI, {lab}")
        ax[2].plot(NGS, [cov[n]["gold_width"] for n in NGS], "s" + ls, color=GREEN, label=f"gold-only, {lab}")
    ax[2].set_xlabel("# gold labels"); ax[2].set_ylabel("mean 95% CI width"); ax[2].set_title("(c) CI width"); ax[2].legend(fontsize=5.9, handlelength=1.6, labelspacing=0.3, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    ax[3].axhline(0.95, color=GREY, ls="--", lw=1)
    for t, v in CLIN.items():
        ax[3].plot(NGC, [v[key][n]["ppi_cov"] for n in NGC], "o-", color=BLUE, alpha=0.85)
        ax[3].plot(NGC, [v[key][n]["naive_cov"] for n in NGC], "^-", color=RED, alpha=0.85)
    ax[3].plot(NGC, [CLIN[MAIN][key][n]["gold_cov"] for n in NGC], "s-", color=GREEN)
    ax[3].set_ylim(-0.05, 1.1); ax[3].set_xlabel("# gold labels"); ax[3].set_title(f"(d) Clinical, {len(CLIN)} labelers")
    ax[3].text(0.97, 0.5, "one curve per labeler\ncolours as in (a)", transform=ax[3].transAxes, ha="right", va="center", fontsize=6.2, color="#333")
    fig.tight_layout(w_pad=1.0); save_fig(fig, fname)

# %% [markdown]
# ## 8. Retrieval operating point (Fig. 3)
# The screen is a **union** on the raw pool: keep a document if its embedding similarity to
# the Long COVID prototype clears τ *or* a regex cue matches. τ is the largest threshold that
# keeps every gold positive (selected on the gold set, so this is by construction on
# $\mathcal{L}$); we therefore report the pruning fraction rather than a recall figure.

# %%
TAU_TARGET = 1.0
sim_g, sim_p = screen_gold.embed_sim.values, screen_pool.embed_sim.values
prec, recl, thr = precision_recall_curve(y_gold, sim_g)
pos_sims = np.sort(sim_g[y_gold == 1])
tau = choose_threshold(sim_g, y_gold, recall=TAU_TARGET)     # Algorithm 1, line 1
emb_g, rx_g = sim_g >= tau, screen_gold.regex_hit.values.astype(bool)
emb_p, rx_p = sim_p >= tau, screen_pool.regex_hit.values.astype(bool)
ret = dict(tau=tau, gold_recall_embed=round(float((emb_g & (y_gold == 1)).sum() / (y_gold == 1).sum()), 3),
           gold_recall_regex=round(float((rx_g & (y_gold == 1)).sum() / (y_gold == 1).sum()), 3),
           gold_precision_regex=round(float((rx_g & (y_gold == 1)).sum() / max(1, rx_g.sum())), 3),
           gold_recall_union=round(float(((emb_g | rx_g) & (y_gold == 1)).sum() / (y_gold == 1).sum()), 3),
           gold_precision_union=round(float(((emb_g | rx_g) & (y_gold == 1)).sum() / (emb_g | rx_g).sum()), 3),
           pool_total=int(len(sim_p)), pool_kept_embed=int(emb_p.sum()), pool_kept_regex=int(rx_p.sum()),
           pool_kept_union=int(union_screen(sim_p, rx_p, tau).sum()), pool_pruned_frac_union=round(1 - union_screen(sim_p, rx_p, tau).mean(), 3),
           pool_pruned_frac_embed=round(1 - emb_p.mean(), 3),
           embed_auprc=round(average_precision_score(y_gold, sim_g), 3), embed_auroc=round(roc_auc_score(y_gold, sim_g), 3))
rr = pd.read_csv(os.path.join(DATA, "reranker_gold.csv")).set_index("pmcid").reindex(gold.pmcid)


def opmetrics(s):
    p, r, _ = precision_recall_curve(y_gold, s); m80 = p[:-1] >= 0.8; m95 = r[:-1] >= 0.95
    return dict(auprc=round(average_precision_score(y_gold, s), 3), auroc=round(roc_auc_score(y_gold, s), 3),
                recall_at_prec80=round(float(r[:-1][m80].max()) if m80.any() else 0.0, 3), prec_at_rec95=round(float(p[:-1][m95].max()) if m95.any() else 0.0, 3))


ret["reranker"] = opmetrics(rr.reranker.values); ret["embedding_ops"] = opmetrics(sim_g); rec("retrieval", ret); print(json.dumps(ret, indent=1))

# %%
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.5))
grid = np.round(np.arange(0.30, 0.61, 0.01), 2)
ax[0].plot(grid, [(y_gold[sim_g >= t].mean() if (sim_g >= t).any() else 0) for t in grid], color=BLUE, label="precision")
ax[0].plot(grid, [(sim_g[y_gold == 1] >= t).mean() for t in grid], color=RED, label="recall")
ax[0].axvline(tau, color=GREY, ls="--", lw=1.1, label=f"τ = {tau:.2f}"); ax[0].set_ylim(-0.03, 1.13)
ax[0].set_xlabel("embedding similarity threshold"); ax[0].set_ylabel("score on gold"); ax[0].set_title("(a) Embedding screen"); ax[0].legend(loc="lower left", fontsize=7.2)
for s, col, lab in [(sim_g, BLUE, "embedding"), (rr.reranker.values, GREEN, "reranker")]:
    p, r, _ = precision_recall_curve(y_gold, s); ax[1].plot(r, p, color=col, label=lab)
ax[1].axhline(0.80, color=GREY, ls=":", lw=1.0)
ax[1].annotate(f"recall@80%-prec\n{ret['embedding_ops']['recall_at_prec80']:.2f} → {ret['reranker']['recall_at_prec80']:.2f}",
               xy=(ret["reranker"]["recall_at_prec80"], 0.80), xytext=(0.30, 0.50), fontsize=6.8,
               ha="left", va="top", zorder=6,
               bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc", lw=0.5, alpha=0.92),
               arrowprops=dict(arrowstyle="->", lw=0.8, shrinkB=1.5, connectionstyle="arc3,rad=-0.15"))
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_xlim(-0.02, 1.02); ax[1].set_ylim(0.3, 1.06); ax[1].set_title("(b) Reranking sharpens"); ax[1].legend(loc="upper right", fontsize=7.5)
labels = ["raw pool", f"similarity ≥ τ", "regex cue", "union (kept)"]
vals = [ret["pool_total"], ret["pool_kept_embed"], ret["pool_kept_regex"], ret["pool_kept_union"]]
cols = [GREY, BLUE, ORANGE, GREEN]; yy = np.arange(len(vals))[::-1]
ax[2].barh(yy, vals, color=cols, height=0.6)
for yi, v in zip(yy, vals):
    ax[2].text(v + 40, yi, f"{v:,}", va="center", fontsize=7.5)
ax[2].set_yticks(yy); ax[2].set_yticklabels(labels, fontsize=7.8); ax[2].set_xlim(0, ret["pool_total"] * 1.25)
ax[2].set_xlabel("# documents (pool)"); ax[2].set_title("(c) Union screen on the raw pool"); ax[2].grid(axis="y", alpha=0)
ax[2].text(ret["pool_total"] * 0.02, -0.75, f"union keeps {ret['pool_kept_union']:,} / {ret['pool_total']:,} (prunes {ret['pool_pruned_frac_union']*100:.1f}%)", fontsize=6.6)
ax[2].set_ylim(-1.1, 3.7)
fig.tight_layout(w_pad=1.3); save_fig(fig, "fig_retrieval.pdf")

# %% [markdown]
# ## 9. Treatment–outcome association mining: severity-stratified sensitivity analysis (Table V; `fig_confounding.pdf` is supplementary)
# Report-level co-occurrence for every treatment mentioned in ≥ 8 reports, each stratum rate with a
# Wilson 95% interval and a one-sided Fisher exact test under Benjamini–Hochberg control (q = 0.10).
# The stratified test is run TWICE. (a) *Outcome-derived severity*, as in the accepted version: 12 of
# its 14 severity markers (death, ICU, mechanical ventilation, intubation, …) are also adverse-outcome
# markers, so P(adverse | severe) = 1 **by construction** and mechanical ventilation is at once a
# tested treatment, a severity marker and an adverse-outcome marker. (b) *Leakage-controlled*:
# severity from PRE-TREATMENT `patient_info` only (≥ 2 comorbidities or age ≥ 65), and, per pair,
# outcome strings that merely restate the tested treatment do not count as its adverse outcome.
# **(b) is what the paper reports**; (a) is computed only to quantify the leakage. Note that PPI does
# NOT rectify these estimates: the 123 physician labels are Long COVID status labels, not labels on
# the (treatment, outcome) extraction, so this section is a sensitivity analysis, not a
# coverage-valid or formal confounding-by-indication result (see Sec. V-G of the paper).

# %%
POS = ["improv", "recover", "resolv", "resolution", "discharg", "remission", "normaliz", "normalis", "stable", "stabil", "cured", "cure",
       "survived", "better", "decreased symptom", "subsided", "relief", "reduction in", "negative for covid", "negative pcr", "clear"]
NEG = ["death", "died", "deceas", "expired", "fatal", "mortal", "deteriorat", "worsen", "failure", "intubat", "mechanical ventilat", "icu",
       "intensive care", "sepsis", "septic", "arrest", "complication", "respiratory failure", "shock", "readmission", "relapse", "progress",
       "exacerbat", "thrombo", "embol", "hemorrhag", "haemorrhag"]
SEVERITY = ["icu", "intensive care", "mechanical ventilat", "intubat", "respiratory failure", "septic shock", "vasopressor", "ecmo",
            "death", "died", "expired", "deceas", "fatal", "cardiac arrest"]
MIN_TREAT, SEV_LIFT_MIN, LIFT_MIN, Q_FDR = 8, 1.25, 1.1, 0.10


def norm_s(s): return re.sub(r"\s+", " ", str(s).lower().strip())
def polarity(o):
    t = norm_s(o); return "neg" if any(k in t for k in NEG) else ("pos" if any(k in t for k in POS) else "other")
def canon_treat(t):
    t = re.sub(r"\([^)]*\)", "", norm_s(t)).strip(); t = t.split(",")[0].split(" for ")[0].split(" with ")[0].strip()
    return re.sub(r"\b\d+[\d\.\-/ ]*(mg|g|mcg|ml|units?|grams?|kg)\b.*$", "", t).strip()


def load_reports(path):
    reps = []
    for line in open(path):
        j = json.loads(line); treats = sorted({canon_treat(x) for x in j["treatments"] if str(x).strip()} - {""})
        treats = [t for t in treats if len(t) >= 3]; outs = [norm_s(x) for x in j["outcomes"] if str(x).strip()]
        comorb = j.get("comorbidities") or []; sev_text = " ".join(outs) + " " + " ".join(norm_s(c) for c in comorb)
        severe = any(k in sev_text for k in SEVERITY) or (len(comorb) >= 2)          # (a) leaky, accepted version
        age = j.get("age", (j.get("patient_info", {}) or {}).get("age"))   # released JSONL flattens patient_info
        severe_base = (len(comorb) >= 2) or (isinstance(age, (int, float)) and age >= 65)   # (b) pre-treatment only
        if treats:
            reps.append(dict(pmcid=j["pmcid"], treatments=treats, outcomes=outs, pol=[polarity(o) for o in outs],
                             severe=bool(severe), severe_base=bool(severe_base)))
    return reps


def wilson(k, n): return (np.nan, np.nan) if n == 0 else tuple(map(float, proportion_confint(k, n, alpha=0.05, method="wilson")))


def adverse(rep, treat=None):
    """Report-level adverse outcome; strings that merely restate `treat` do not count as its harm."""
    for o, pol in zip(rep["outcomes"], rep["pol"]):
        if pol != "neg":
            continue
        if treat:
            head = treat.split()[0] if treat.split() else treat
            if treat in o or (len(head) >= 5 and head in o):
                continue
        return True
    return False


def assoc_table(reps, sev_key="severe", drop_treat_in_outcome=False):
    from collections import Counter
    N = len(reps); tcount = Counter(t for r in reps for t in set(r["treatments"]))
    hit = lambda r, p, t: (adverse(r, t if (drop_treat_in_outcome and p == "neg") else None) if p == "neg" else (p in r["pol"]))
    base = {p: np.mean([hit(r, p, None) for r in reps]) for p in ("pos", "neg")}
    p_sev = np.mean([r[sev_key] for r in reps]); rows = []
    for t in [t for t, c in tcount.items() if c >= MIN_TREAT]:
        rt = [r for r in reps if t in r["treatments"]]; nt = len(rt)
        sev = [r for r in rt if r[sev_key]]; non = [r for r in rt if not r[sev_key]]
        for p in ("pos", "neg"):
            k = sum(hit(r, p, t) for r in rt); ks, kn = sum(hit(r, p, t) for r in sev), sum(hit(r, p, t) for r in non); ns, nn = len(sev), len(non)
            pval = fisher_exact([[ks, ns - ks], [kn, nn - kn]], alternative="greater")[1] if (ns and nn) else np.nan
            rows.append(dict(treatment=t, polarity=p, n_reports=nt, k=k, lift=k / nt / base[p], pmi=np.log2((k / N) / ((nt / N) * base[p])) if k else np.nan,
                             severity_lift=(ns / nt) / p_sev, n_severe=ns, k_severe=ks, rate_severe=ks / ns if ns else np.nan,
                             wilson_severe=wilson(ks, ns), n_nonsevere=nn, k_nonsevere=kn, rate_nonsevere=kn / nn if nn else np.nan,
                             wilson_nonsevere=wilson(kn, nn), fisher_p=pval))
    return pd.DataFrame(rows), base, N, p_sev


def top_pairs(sub, topn=20):
    from collections import Counter
    c = Counter((t, p) for r in sub for t in set(r["treatments"]) for p in set(r["pol"]) if p in ("pos", "neg"))
    return {k for k, _ in c.most_common(topn)}


reps = load_reports(os.path.join(DATA, "extractions_primary.jsonl"))
df4, base4, N4, p_sev4 = assoc_table(reps, sev_key="severe_base", drop_treat_in_outcome=True)   # (b) reported
df4_leaky, _, _, _ = assoc_table(reps, sev_key="severe", drop_treat_in_outcome=False)            # (a) leakage check
def flagged_neg(df):
    n = df[df.polarity == "neg"].sort_values("lift", ascending=False).copy()
    n["flag_point"] = (n.severity_lift > SEV_LIFT_MIN) & (n.lift > LIFT_MIN) & (n.rate_nonsevere < n.rate_severe)
    rej, q, _, _ = multipletests(n.fisher_p.fillna(1.0).values, alpha=Q_FDR, method="fdr_bh"); n["bh_q"] = q; n["bh_reject"] = rej
    n["flag"] = n.flag_point & n.bh_reject
    return n


neg = flagged_neg(df4)                                     # (b) leakage-controlled: the reported analysis
neg_leaky = flagged_neg(df4_leaky)                         # (a) outcome-derived severity: leakage quantification
_sev_leaky = [r for r in reps if r["severe"]]
_sev_from_outcome = [r for r in reps if any(k in " ".join(r["outcomes"]) for k in SEVERITY)]
rec("assoc_leaky", dict(stratification="outcome-derived severity (accepted version)",
                        p_adverse_given_severe=round(float(np.mean([adverse(r) for r in _sev_leaky])), 3),
                        frac_reports_severe_via_outcome_text=round(float(len(_sev_from_outcome) / len(reps)), 3),
                        p_adverse_given_severity_from_outcome=round(float(np.mean([adverse(r) for r in _sev_from_outcome])), 3),
                        n_flag_point=int(neg_leaky.flag_point.sum()), n_bh_reject=int(neg_leaky.bh_reject.sum()),
                        n_flag=int(neg_leaky.flag.sum()), flagged=neg_leaky[neg_leaky.flag].treatment.tolist(),
                        severity_markers_also_adverse=sorted({m for m in SEVERITY for a in NEG if m in a or a in m})))
rng4 = np.random.default_rng(SEED); idx = rng4.permutation(len(reps)); h1, h2 = [reps[i] for i in idx[: len(reps) // 2]], [reps[i] for i in idx[len(reps) // 2:]]
a, b = top_pairs(h1), top_pairs(h2); split_half = len(a & b) / len(a | b)
reps2 = load_reports(os.path.join(DATA, "extractions_repeat.jsonl")); a2, b2 = top_pairs(reps), top_pairs(reps2); repeat_run = len(a2 & b2) / len(a2 | b2)
assoc = dict(stratification="pre-treatment baseline only (>=2 comorbidities or age >=65); no outcome or treatment text",
             adverse_rule="report-level adverse outcome, excluding outcome strings that restate the tested treatment",
             label="severity-stratified sensitivity analysis (not a formal confounding-by-indication test; not PPI-rectified)",
             n_reports=N4, n_treatments=int(df4.treatment.nunique()), n_neg_pairs=int(len(neg)), base_rate_neg=round(base4["neg"], 3), p_severe=round(float(p_sev4), 3),
             n_flag_point=int(neg.flag_point.sum()), n_bh_reject=int(neg.bh_reject.sum()), n_flag=int(neg.flag.sum()), min_nt=int(neg.n_reports.min()),
             flagged=neg[neg.flag].treatment.tolist(), flagged_point_only=neg[neg.flag_point & ~neg.flag].treatment.tolist(),
             bh_reject_not_enriched=neg[neg.bh_reject & ~neg.flag_point].treatment.tolist(),
             split_half_jaccard=round(split_half, 3), repeat_run_jaccard=round(repeat_run, 3))
rec("assoc", assoc); print(json.dumps(assoc, indent=1))
show = pd.concat([neg.head(8), neg[neg.flag & ~neg.index.isin(neg.head(8).index)]]).sort_values("lift", ascending=False)
NAMES = {"low-molecular-weight heparin": "LMW heparin"}
lines = []
for _, r in show.iterrows():
    ws, wn = r.wilson_severe, r.wilson_nonsevere; nm = NAMES.get(r.treatment, r.treatment)
    flag_s = "$\\checkmark$" if r.flag else ""
    lines.append(f"{nm} & {r.n_reports} & {r.lift:.2f} & {r.severity_lift:.2f} & {r.rate_severe:.2f} [{ws[0]:.2f},{ws[1]:.2f}] ({r.n_severe}) & "
                 f"{r.rate_nonsevere:.2f} [{wn[0]:.2f},{wn[1]:.2f}] ({r.n_nonsevere}) & {r.bh_q:.3f} & {flag_s} \\\\")
save_tab("tab_e4_rows.tex", lines)
rec("assoc_table_rows", [dict(treatment=r.treatment, n=int(r.n_reports), lift=round(r.lift, 3), sev_lift=round(r.severity_lift, 3), rate_severe=round(r.rate_severe, 3),
                             rate_nonsevere=round(r.rate_nonsevere, 3), n_severe=int(r.n_severe), n_nonsevere=int(r.n_nonsevere), q=round(r.bh_q, 4), flag=bool(r.flag), flag_point=bool(r.flag_point)) for _, r in show.iterrows()])
print(pd.DataFrame(NUM["assoc_table_rows"]).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(7.16, 3.6)); yy = np.arange(len(show))[::-1]
for off, col, key, wkey, lab in [(0.2, RED, "rate_severe", "wilson_severe", "severe stratum"), (-0.2, BLUE, "rate_nonsevere", "wilson_nonsevere", "non-severe stratum")]:
    v = show[key].values; lo = np.array([w[0] for w in show[wkey]]); hi = np.array([w[1] for w in show[wkey]])
    ax.barh(yy + off, v, height=0.38, color=col, label=lab, xerr=[v - lo, hi - v], error_kw=dict(lw=0.8, capsize=2, ecolor="#333"))
ax.set_yticks(yy); ax.set_yticklabels([NAMES.get(t, t) + ("  ✓" if f else "") for t, f in zip(show.treatment, show.flag)], fontsize=8.2)
ax.set_xlabel("P(adverse outcome | treatment), 95% Wilson interval"); ax.set_xlim(0, 1.02)
ax.set_title(f"Severity-stratified sensitivity analysis, baseline strata "
             f"(✓ = severity-enriched, attenuating, BH-FDR q ≤ {Q_FDR:.2f}; {int(neg.flag.sum())}/{len(neg)} pairs)", fontsize=9)
ax.legend(loc="lower right", fontsize=8); save_fig(fig, "fig_confounding.pdf")

# %% [markdown]
# ## 10. Numbers quoted in the abstract and contributions
# Each quantitative claim of the abstract and the contribution list, next to the regenerated values it rests on.
# Width ratios use the TF-IDF predictor (upper blocks of Table II), as in Sec. V-C and Fig. 5(c), and are averaged
# over n_g in {50, 100, 150, 200, 300}. Where the abstract and contributions quote one figure for both benchmarks,
# it is the more conservative benchmark: HoC for the width saving (15.6%; ADE 21.2%) and ADE for the ACC
# penalty (2.5x; HoC 3.0x). Sec. V-C and the Fig. 5 caption give both benchmarks.

# %%
def _rng(v): return [min(v), max(v)]


_hoc, _ade = BENCH["hoc"]["cov"], BENCH["ade"]["cov"]
_save = {b: [1 - c[n]["ppi_width"] / c[n]["gold_width"] for n in NGS] for b, c in (("hoc", _hoc), ("ade", _ade))}
ABSTRACT = {
    "expert labels (~100 reports)": dict(paper="~100 reports", benchmark_gold_sizes=NGS, clinical_gold=int(len(y_gold))),
    "PPI coverage on the two benchmarks": dict(
        paper="0.91-0.96 (abstract, contributions; Sec. V-C)",
        tfidf=_rng([c[n]["ppi_cov"] for c in (_hoc, _ade) for n in NGS]),
        llm=_rng([v["cov"][n]["ppi_cov"] for v in LLM_BENCH.values() for n in NGS])),
    "PPI narrower than expert-only (TF-IDF predictor)": dict(
        paper="approximately 15% (abstract); 16% on average (contributions); 16% HoC / 21% ADE (Fig. 5c); "
              "15-16% HoC, 20-22% ADE at every n_g (Sec. V-C)",
        hoc_mean=round(float(np.mean(_save["hoc"])), 4), ade_mean=round(float(np.mean(_save["ade"])), 4),
        hoc_range=[round(min(_save["hoc"]), 4), round(max(_save["hoc"]), 4)],
        ade_range=[round(min(_save["ade"]), 4), round(max(_save["ade"]), 4)]),
    "ACC interval width / PPI interval width (TF-IDF predictor)": dict(
        paper="2.5 times (abstract, contributions); 3.0x HoC and 2.5x ADE (Sec. V-C)",
        hoc_mean=NUM["hoc_acc_vs_ppi_width_ratio"], ade_mean=NUM["ade_acc_vs_ppi_width_ratio"]),
    "ACC coverage": dict(paper="comparable coverage (abstract); 0.94-0.97 on HoC (Sec. V-C)",
                         hoc=NUM["hoc_acc_cov_range"], ade=NUM["ade_acc_cov_range"]),
    "naive coverage under systematic bias": dict(
        paper="zero coverage (abstract); 0.00 on the biased benchmark (contributions)",
        hoc_tfidf_bias_pts=round(100 * BENCH["hoc"]["info"]["naive_bias"], 1), hoc_tfidf=_rng([_hoc[n]["naive_cov"] for n in NGS]),
        llm_both_benchmarks=NUM["bench_llm_naive_cov_range"],
        ade_tfidf_nearly_unbiased=_rng([_ade[n]["naive_cov"] for n in NGS])),
    "PPI coverage on the clinical labels": dict(paper="0.92-0.96 (contributions)", value=NUM["clin_ppi_cov_range"]),
}
rec("abstract_numbers", ABSTRACT)
for k, v in ABSTRACT.items():
    print(f"{k}\n   paper: {v['paper']}\n   this run: " + ", ".join(f"{a}={b}" for a, b in v.items() if a != "paper"))

# %% [markdown]
# ## 11. Export every number

# %%
def _py(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


with open(os.path.join(OUT, "paper_numbers.json"), "w") as f:
    json.dump(NUM, f, indent=1, default=_py)
print("wrote", os.path.relpath(os.path.join(OUT, "paper_numbers.json"), HERE), "with", len(NUM), "entries")
print(sorted(os.listdir(FIGS)), sorted(os.listdir(TABS)))
