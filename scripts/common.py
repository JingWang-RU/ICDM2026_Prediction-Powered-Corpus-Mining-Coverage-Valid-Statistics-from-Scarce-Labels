import os, json, re
import numpy as np
import pandas as pd

LC = os.environ.get("LC_DIR", "longcovid")  # source-data root (full-text corpus, splits, embeddings); see README "Source data"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
FIGS = os.path.join(os.path.dirname(HERE), "figs")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

SEED = 42


def load_gold():
    g = pd.read_csv(os.path.join(LC, "amar_longcovid_yesno.csv"))
    g = g[["pmcid", "label", "content"]].copy()
    g["pmcid"] = g["pmcid"].astype(int)
    folds = pd.read_csv(os.path.join(LC, "5fold_pmcid_splits.csv"))
    folds["pmcid"] = folds["pmcid"].astype(int)
    g = g.merge(folds, on="pmcid", how="left")
    assert g["fold"].notna().all(), "missing fold assignment"
    g["fold"] = g["fold"].astype(int)
    return g.reset_index(drop=True)


def load_pool():
    p = pd.read_csv(os.path.join(LC, "aaai_test.csv"))
    p["pmcid"] = p["pmcid"].astype(int)
    return p.reset_index(drop=True)


def load_zeroshot_pool():
    z = pd.read_csv(os.path.join(LC, "qwen14B_zeroshot_test.csv"))
    z["pmcid"] = z["pmcid"].astype(int)
    z["pred"] = (z["reason"].astype(str).str.strip().str.lower() == "yes").astype(int)
    return z[["pmcid", "pred"]]


def load_zeroshot_gold():
    z = pd.read_csv(os.path.join(LC, "qwen14B_zero_train.csv"))
    z["pmcid"] = z["pmcid"].astype(int)
    z["pred"] = (z["label"].astype(str).str.strip().str.lower() == "yes").astype(int)
    return z[["pmcid", "pred"]]


def load_embedding_sims():
    gold_sim = np.load(os.path.join(LC, "similarity_matrix.npy"))      # (2,123)
    pool_sim = np.load(os.path.join(LC, "similarity_matrix_test.npy"))  # (2,3066)
    return gold_sim, pool_sim


def load_treatment_outcomes(folder="treatment_updated_597"):
    d = os.path.join(LC, folder)
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        m = re.search(r"(\d+)", fn)
        pmcid = int(m.group(1)) if m else None
        try:
            with open(os.path.join(d, fn)) as f:
                j = json.load(f)
        except Exception:
            continue
        j["pmcid"] = pmcid
        out.append(j)
    return out


if __name__ == "__main__":
    g = load_gold()
    print("gold:", g.shape, "label dist", dict(g.label.value_counts()))
    print("fold sizes:", dict(g.fold.value_counts().sort_index()))
    p = load_pool()
    print("pool:", p.shape)
    zp = load_zeroshot_pool(); print("zeroshot pool:", zp.shape, "yes-rate %.3f" % zp.pred.mean())
    zg = load_zeroshot_gold(); print("zeroshot gold:", zg.shape, "yes-rate %.3f" % zg.pred.mean())
    gs, ps = load_embedding_sims(); print("sims:", gs.shape, ps.shape)
    to = load_treatment_outcomes(); print("treatment-outcome reports:", len(to))
