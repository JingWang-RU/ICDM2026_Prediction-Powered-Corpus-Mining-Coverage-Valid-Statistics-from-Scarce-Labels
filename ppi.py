import numpy as np
from scipy.stats import norm

Z_CRIT = 1.959964
MIN_GOLD = 3


def rectifier(R_gold):
    return float(R_gold.mean()), float(R_gold.std(ddof=1) / np.sqrt(len(R_gold)))


def pooled_interval(f_group, R_gold, alpha=0.05):
    z = norm.ppf(1 - alpha / 2)
    r, se = rectifier(R_gold)
    centre = float(f_group.mean()) + r
    return centre - z * se, centre + z * se


def group_interval(f_group, R_gold_group, alpha=0.05, min_gold=MIN_GOLD):
    if len(R_gold_group) < min_gold:
        return None
    z = norm.ppf(1 - alpha / 2)
    r, se = rectifier(R_gold_group)
    centre = float(f_group.mean()) + r
    return centre - z * se, centre + z * se


def evaluate(y, f, g, n_gold, replicates=600, alpha=0.05, seed=0):
    R = y - f
    keys = [k for k in sorted(set(g.tolist())) if (g == k).sum() >= 100]
    idx = {k: np.flatnonzero(g == k) for k in keys}
    truth = {k: y[idx[k]].mean() for k in keys}
    rng = np.random.default_rng(seed)
    cov_p = {k: 0 for k in keys}
    cov_g = {k: 0 for k in keys}
    w_p, w_g = [], {k: [] for k in keys}
    for _ in range(replicates):
        s = rng.choice(len(y), n_gold, replace=False)
        for k in keys:
            lo, hi = pooled_interval(f[idx[k]], R[s], alpha)
            w_p.append(hi - lo)
            cov_p[k] += lo <= truth[k] <= hi
            ci = group_interval(f[idx[k]], R[s[g[s] == k]], alpha)
            if ci is None:
                continue
            w_g[k].append(ci[1] - ci[0])
            cov_g[k] += ci[0] <= truth[k] <= ci[1]
    cp = np.array([cov_p[k] / replicates for k in keys])
    cg = np.array([cov_g[k] / replicates for k in keys])
    wp = float(np.mean(w_p))
    wg = float(np.mean([np.mean(w_g[k]) for k in keys if w_g[k]]))
    return {"groups": keys, "coverage_pooled": cp, "coverage_group": cg,
            "worst_pooled": float(cp.min()), "worst_group": float(cg.min()),
            "width_ratio": wg / wp}


def rectifier_gap(y, f, g):
    R = y - f
    keys = [k for k in sorted(set(g.tolist())) if (g == k).sum() >= 100]
    return {k: float(R[g == k].mean() - R.mean()) for k in keys}


def regimes(y, f, g, alpha=0.05, epsilon=0.05, min_gold=MIN_GOLD):
    from scipy.optimize import brentq
    z = norm.ppf(1 - alpha / 2)
    tau = brentq(lambda t: (norm.cdf(z - t) - norm.cdf(-z - t)) - (1 - alpha - epsilon), 1e-3, 10)
    sigma = float((y - f).std(ddof=1))
    out = {}
    for k, d in rectifier_gap(y, f, g).items():
        pi = float((g == k).mean())
        n_fail = (tau * sigma / abs(d)) ** 2 if d != 0 else np.inf
        n_feas = min_gold / pi
        out[k] = {"pi": pi, "delta": d, "n_fail": n_fail, "n_feas": n_feas,
                  "gap_regime": n_fail < n_feas}
    return out


def coverage_law(delta, sigma, n, alpha=0.05):
    z = norm.ppf(1 - alpha / 2)
    zeta = np.sqrt(n) * delta / sigma
    return norm.cdf(z - zeta) - norm.cdf(-z - zeta)


def transfer_certificate(y_gold, f_gold, f_pool, thresholds=(0.3, 0.5, 0.7)):
    out = {}
    for t in thresholds:
        hg = (f_gold > t).astype(float)
        q = float((f_pool > t).mean())
        tpr = float(hg[y_gold == 1].mean())
        fpr = float(hg[y_gold == 0].mean())
        theta = (q - fpr) / (tpr - fpr) if abs(tpr - fpr) > 1e-9 else np.nan
        out[t] = {"theta_pool": theta, "admissible": 0.0 <= theta <= 1.0}
    return out


def lambda_rules(y_gold, f_gold, f_pool):
    n, N = len(y_gold), len(f_pool)
    cov = float(np.cov(y_gold, f_gold, ddof=1)[0, 1])
    v_gold = float(f_gold.var(ddof=1))
    v_pooled = float(np.concatenate([f_gold, f_pool]).var(ddof=1))
    lam_pooled = float(np.clip(cov / (v_pooled * (1 + n / N)), 0, 1))
    lam_labeled = float(np.clip(cov / (v_gold * (1 + n / N)), 0, 1))
    return {"lambda_pooled_rule": lam_pooled, "lambda_labeled_rule": lam_labeled,
            "agree": abs(lam_pooled - lam_labeled) < 0.1}


def budget_floor(g, min_gold=MIN_GOLD):
    keys = [k for k in sorted(set(g.tolist())) if (g == k).sum() >= 100]
    return min_gold / min(float((g == k).mean()) for k in keys)
