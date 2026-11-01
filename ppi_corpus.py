"""Prediction-powered corpus estimation (Wang, Shen, Sra; IEEE ICDM 2026).

Estimate a corpus-level positive rate theta* = E[Y] from LLM pseudo-labels on a large unlabeled
pool plus a small expert-labeled (gold) set, with a confidence interval whose coverage holds when
the gold set is representative of the pool (Sec. IV-B of the paper).

    estimate_corpus_rate   Algorithm 1, estimation part: Platt calibration on an independent set + PPI++,
                           or K-fold cross-fitted calibration on the gold set + Cross-PPI; score-shift diagnostic
    hard_label_ppi         PPI++ on prespecified hard pseudo-labels h(x) = 1{p_raw(x) >= t0}
    ppi_plusplus           PPI++ mean estimate and interval (Eqs. 22, 23, 21)
    cross_ppi              Cross-PPI mean estimate and interval, lambda = 1 (Eqs. 20, 21)
    crossfit_calibrate     K-fold cross-fitted Platt calibration (Eqs. 15, 16)
    platt_fit, platt_apply Platt scaling of the raw yes-probability (Eqs. 13, 14)
    score_shift            observable score-shift diagnostic Delta_f (Eq. 26)
    choose_threshold, union_screen   candidate screen of Algorithm 1, lines 1-2 (not used by the estimator)
    naive_ci, gold_only_ci, acc_ci   the baselines of Sec. V-C

PPI++ and Cross-PPI are computed with `ppi-python` 0.2.3 (Angelopoulos et al.; Zrnic & Candes), the
library that produced every number in the paper. All intervals are two-sided at level 1 - alpha.
`clip=True` intersects an interval with [0, 1] as in Eq. (21); the paper's tables report the
unclipped intervals (see README).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

try:
    from ppi_py import classical_mean_ci, crossppi_mean_ci, ppi_mean_ci, ppi_mean_pointestimate
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install ppi-python==0.2.3") from e

EPS = 1e-6          # clipping of the raw score before the logit (epsilon in Sec. IV-A)


# ----------------------------------------------------------------------------- calibration
def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def platt_fit(scores, y, C=1e3):
    """Platt scaling f(x) = sigmoid(a * logit p(x) + c), fit by logistic regression of y on logit p.
    The default C = 1e3 is a negligible ridge, i.e. the binomial maximum-likelihood fit of Eq. (14)."""
    return LogisticRegression(C=C).fit(logit(scores).reshape(-1, 1), np.asarray(y))


def platt_apply(model, scores):
    return model.predict_proba(logit(scores).reshape(-1, 1))[:, 1]


def crossfit_calibrate(p_gold, y_gold, p_pool, folds, C=1e3):
    """K-fold cross-fitted calibration. For each fold k, Platt is fit on the gold reports outside fold k.
    Returns (f_gold_oof, f_pool_k): the out-of-fold prediction for every gold report (n,) and the
    prediction of every fold's calibrator for every pool report (N, K); the pool's cross-fitted
    pseudo-label f_cf of Eq. (16) is f_pool_k.mean(1)."""
    p_gold, y_gold, folds = np.asarray(p_gold, float), np.asarray(y_gold), np.asarray(folds)
    ks = sorted(np.unique(folds))
    f_gold_oof = np.zeros(len(y_gold))
    f_pool_k = np.zeros((len(p_pool), len(ks)))
    for j, k in enumerate(ks):
        tr, te = folds != k, folds == k
        model = platt_fit(p_gold[tr], y_gold[tr], C=C)
        f_gold_oof[te] = platt_apply(model, p_gold[te])
        f_pool_k[:, j] = platt_apply(model, p_pool)
    return f_gold_oof, f_pool_k


# ----------------------------------------------------------------------------- estimators
def _clip01(lo, hi, clip):
    return (max(lo, 0.0), min(hi, 1.0)) if clip else (lo, hi)


def ppi_lambda(y_gold, f_gold, f_pool):
    """Variance-minimising power-tuning weight of Eq. (23), clipped to [0, 1]:
    lambda = Cov(y, f) / ((1 + n/N) Var(f)), with Var(f) pooled over gold and pool predictions."""
    y, fg, fp = (np.asarray(a, float) for a in (y_gold, f_gold, f_pool))
    n, N = len(y), len(fp)
    cov = np.mean((y - y.mean()) * (fg - fg.mean()))
    var = np.var(np.r_[fg, fp], ddof=1)
    return float(np.clip(cov / ((1 + n / N) * var), 0.0, 1.0)) if var > 0 else 0.0


def ppi_plusplus(y_gold, f_gold, f_pool, alpha=0.05, lam=None, clip=True):
    """PPI++ (Eq. 22): theta_lam = mean(y) + lam * (mean f_pool - mean f_gold). lam=None estimates
    lambda by Eq. (23); lam=1 is basic PPI (Eq. 18); lam=0 is the gold-only estimate.
    The predictor must not have been fit on these gold labels.
    Returns dict(estimate, ci, lam)."""
    y = np.asarray(y_gold, float)
    fg, fp = np.asarray(f_gold, float), np.asarray(f_pool, float)
    est = float(ppi_mean_pointestimate(y, fg, fp, lam=lam))
    lo, hi = map(float, ppi_mean_ci(y, fg, fp, alpha=alpha, lam=lam))
    return dict(estimate=est, ci=_clip01(lo, hi, clip),
                lam=ppi_lambda(y, fg, fp) if lam is None else float(lam))


def cross_ppi(y_gold, f_gold_oof, f_pool_k, alpha=0.05, clip=True):
    """Cross-PPI with lambda = 1 (Eq. 20): mean of the pool's fold predictions plus the mean
    out-of-fold residual y - f^(-k)(x); its variance follows Zrnic & Candes (2024).
    f_pool_k is (N, K). Returns dict(estimate, ci, lam=1)."""
    y = np.asarray(y_gold, float)
    fo, fk = np.asarray(f_gold_oof, float), np.asarray(f_pool_k, float)
    est = float(fk.mean() + (y - fo).mean())
    lo, hi = map(float, crossppi_mean_ci(y, fo, fk, alpha=alpha))
    return dict(estimate=est, ci=_clip01(lo, hi, clip), lam=1.0)


def score_shift(f_pool, f_gold):
    """Delta_f of Eq. (26): mean pool prediction minus mean gold prediction. A large value
    flags a visible mismatch between the gold set and the pool; a small one does not prove
    representativeness."""
    return float(np.mean(f_pool) - np.mean(f_gold))


def estimate_corpus_rate(y_gold, p_gold, p_pool, calibration=None, folds=None, K=5, alpha=0.05,
                         seed=0, platt_C=1e3, clip=True):
    """Algorithm 1 (estimation part).

    y_gold : (n,) 0/1 expert labels.  p_gold : (n,) and p_pool : (N,) raw LLM yes-probabilities.
    calibration : optional (p_cal, y_cal) from an INDEPENDENT labeled set -> Platt + PPI++ (lambda estimated).
    Otherwise the gold set is split into K report-level folds (`folds`, or a random split with `seed`)
    -> cross-fitted Platt + Cross-PPI (lambda = 1).

    Returns dict(estimate, ci, lam, score_shift, method, f_gold, f_pool), where f_gold / f_pool are the
    pseudo-labels the estimator used (out-of-fold for gold, fold-averaged for the pool when cross-fitting).
    """
    y = np.asarray(y_gold, float)
    if calibration is not None:
        model = platt_fit(*calibration, C=platt_C)
        fg, fp = platt_apply(model, p_gold), platt_apply(model, p_pool)
        out = ppi_plusplus(y, fg, fp, alpha=alpha, clip=clip)
        out["method"] = "Platt (independent set) + PPI++"
    else:
        if folds is None:
            folds = np.random.default_rng(seed).permutation(len(y)) % K
        fg, fk = crossfit_calibrate(p_gold, y, p_pool, folds, C=platt_C)
        fp = fk.mean(1)
        out = cross_ppi(y, fg, fk, alpha=alpha, clip=clip)
        out["method"] = f"cross-fitted Platt (K={len(np.unique(folds))}) + Cross-PPI"
    out.update(score_shift=score_shift(fp, fg), f_gold=fg, f_pool=fp)
    return out


def hard_label_ppi(y_gold, p_gold, p_pool, t0=0.5, alpha=0.05, clip=True):
    """PPI++ on hard pseudo-labels h(x) = 1{p_raw(x) >= t0}. Needs no calibration labels, provided the
    model, prompt and t0 were fixed without looking at the gold set."""
    hg = (np.asarray(p_gold, float) >= t0).astype(float)
    hp = (np.asarray(p_pool, float) >= t0).astype(float)
    out = ppi_plusplus(y_gold, hg, hp, alpha=alpha, clip=clip)
    out.update(method=f"hard pseudo-labels (t0={t0}) + PPI++", score_shift=score_shift(hp, hg))
    return out


# ----------------------------------------------------------------------------- candidate screen
def choose_threshold(sim_dev, y_dev, recall=1.0, grid=None):
    """Largest similarity threshold on `grid` whose recall of the development positives is >= `recall`
    (Algorithm 1, line 1)."""
    grid = np.round(np.arange(0.30, 0.71, 0.05), 2) if grid is None else grid
    pos = np.asarray(sim_dev)[np.asarray(y_dev) == 1]
    return float(max(t for t in grid if (pos >= t).mean() >= recall))


def union_screen(sim, cue_hit, tau):
    """Keep a document if its prototype similarity clears tau OR a regex cue matches (Algorithm 1, line 2).
    The screen only prioritizes review; the estimator is always run on the full pool."""
    return (np.asarray(sim) >= tau) | np.asarray(cue_hit, bool)


# ----------------------------------------------------------------------------- baselines (Sec. V-C)
def wald_ci(phat, n, alpha=0.05):
    z = norm.ppf(1 - alpha / 2)
    se = math.sqrt(max(phat * (1 - phat), 1e-12) / n)
    return phat - z * se, phat + z * se


def naive_ci(h_pool, alpha=0.05, clip=True):
    """Naive count (Eq. (3)) of hard pseudo-labels on the pool, with the binomial (Wald) interval.
    It is centred on E[f(X)] = theta* + b_f, so it does not cover theta* when the labeler is biased."""
    h = np.asarray(h_pool, float)
    lo, hi = wald_ci(float(h.mean()), len(h), alpha)
    return dict(estimate=float(h.mean()), ci=_clip01(lo, hi, clip))


def gold_only_ci(y_gold, alpha=0.05, clip=True):
    """Classical interval from the expert labels alone."""
    y = np.asarray(y_gold, float)
    lo, hi = map(float, classical_mean_ci(y, alpha=alpha))
    return dict(estimate=float(y.mean()), ci=_clip01(lo, hi, clip))


def acc_estimate(y_gold, h_gold, p_unlabeled):
    """Adjusted Classify-and-Count: (p_naive - FPR) / (TPR - FPR), clipped to [0, 1]."""
    pos, neg = y_gold == 1, y_gold == 0
    if not pos.any() or not neg.any():
        return float(np.clip(p_unlabeled, 0, 1))
    tpr, fpr = h_gold[pos].mean(), h_gold[neg].mean()
    if tpr <= fpr:
        return float(np.clip(p_unlabeled, 0, 1))
    return float(np.clip((p_unlabeled - fpr) / (tpr - fpr), 0, 1))


def acc_ci(y_gold, h_gold, h_unl, rng, B=200, alpha=0.05):
    """Percentile-bootstrap interval for ACC: B resamples of the gold set (TPR, FPR) and a binomial
    redraw of the pool's naive rate. `rng` is a seed or a numpy Generator."""
    rng = np.random.default_rng(rng)
    y_gold, h_gold = np.asarray(y_gold), np.asarray(h_gold, float)
    n, N, p_unl = len(y_gold), len(h_unl), float(np.mean(h_unl))
    est = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        est[b] = acc_estimate(y_gold[idx], h_gold[idx], rng.binomial(N, p_unl) / N)
    return float(np.percentile(est, 100 * alpha / 2)), float(np.percentile(est, 100 * (1 - alpha / 2)))


__all__ = ["estimate_corpus_rate", "hard_label_ppi", "ppi_plusplus", "ppi_lambda", "cross_ppi",
           "crossfit_calibrate", "platt_fit", "platt_apply", "logit", "score_shift", "choose_threshold",
           "union_screen", "naive_ci", "gold_only_ci", "wald_ci", "acc_estimate", "acc_ci"]
