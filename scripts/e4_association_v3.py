"""E4 v3 (camera-ready correction): severity-stratified SENSITIVITY analysis with a
leakage-controlled stratification.

Why v3 exists. In v1/v2 the severity stratum was derived from the *outcome* text with markers
(death, ICU, mechanical ventilation, intubation, respiratory failure, septic shock, ...) that
also appear in the adverse-outcome lexicon, so P(adverse | severe) was 1.000 by construction,
and mechanical ventilation was simultaneously a tested treatment, a severity marker and an
adverse-outcome marker. That contrast cannot support a confounding-by-indication claim.

v3 removes the three leakage paths:
  (i)  severity is defined ONLY from pre-treatment baseline characteristics extracted into
       patient_info (comorbidity burden, age); no outcome text, no treatment enters it;
  (ii) for each tested treatment, outcome strings that mention that treatment itself are not
       counted as its adverse outcome (a ventilated patient's "mechanical ventilation" outcome
       is not evidence that ventilation caused harm);
  (iii) treatments whose own name matches the baseline-severity definition are excluded from
       the stratified test (none under a pure patient_info definition, checked at runtime).
The result is reported as a severity-stratified sensitivity analysis, NOT a formal
confounding-by-indication test: baseline severity here is a coarse proxy from LLM-extracted
fields, and no expert labels rectify the extraction.
"""
import os, json, re
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
from e4_association import load_reports, split_half_stability, cross_source_stability, POS, NEG, norm
from common import load_treatment_outcomes, RESULTS

MIN_TREAT = 8
SEV_LIFT_MIN, LIFT_MIN, Q_FDR = 1.25, 1.1, 0.10
BASELINE_COMORB, BASELINE_AGE = 2, 65


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return (float(lo), float(hi))


def baseline_severity(folder="treatment_updated_597"):
    """Pre-treatment severity proxy from patient_info ONLY (comorbidity burden or age)."""
    out = {}
    for j in load_treatment_outcomes(folder):
        info = j.get("patient_info", {}) or {}
        comorb = info.get("comorbidities", []) or []
        age = info.get("age")
        age_ok = isinstance(age, (int, float)) and age >= BASELINE_AGE
        out[j.get("pmcid")] = dict(severe=bool(len(comorb) >= BASELINE_COMORB or age_ok),
                                   n_comorb=len(comorb), age=age if isinstance(age, (int, float)) else None)
    return out


def adverse_excluding_treatment(rep, treat):
    """Adverse outcome for this pair, ignoring outcome strings that mention the treatment itself."""
    head = treat.split()[0] if treat.split() else treat
    for o, p in zip(rep["outcomes"], rep["pol"]):
        if p != "neg":
            continue
        if treat and (treat in o or (len(head) >= 5 and head in o)):
            continue          # the outcome text is restating the treatment: not evidence of harm
        return True
    return False


def build(reps, base_sev):
    for r in reps:
        b = base_sev.get(r["pmcid"], {})
        r["severe_baseline"] = bool(b.get("severe", False)); r["n_comorb"] = b.get("n_comorb", 0)
    N = len(reps)
    tcount = Counter(t for r in reps for t in set(r["treatments"]))
    treats = sorted(t for t, c in tcount.items() if c >= MIN_TREAT)
    base_rate = float(np.mean([adverse_excluding_treatment(r, "") for r in reps]))
    p_sev = float(np.mean([r["severe_baseline"] for r in reps]))
    rows = []
    for t in treats:
        rt = [r for r in reps if t in r["treatments"]]
        nt = len(rt)
        adv = [adverse_excluding_treatment(r, t) for r in rt]
        joint = float(np.mean(adv))
        # the treatment's own name must not define the stratum (guaranteed by patient_info-only severity)
        assert not any(k in t for k in ("comorbid", "age")), f"treatment {t} touches the baseline definition"
        sev = [a for a, r in zip(adv, rt) if r["severe_baseline"]]
        non = [a for a, r in zip(adv, rt) if not r["severe_baseline"]]
        n_s, k_s, n_n, k_n = len(sev), int(sum(sev)), len(non), int(sum(non))
        r_s = k_s / n_s if n_s else np.nan
        r_n = k_n / n_n if n_n else np.nan
        p = np.nan
        if n_s and n_n:
            p = float(fisher_exact([[k_s, n_s - k_s], [k_n, n_n - k_n]], alternative="greater")[1])
        lo_s, hi_s = wilson(k_s, n_s); lo_n, hi_n = wilson(k_n, n_n)
        sev_lift = float(np.mean([r["severe_baseline"] for r in rt]) / p_sev) if p_sev else np.nan
        rows.append(dict(treatment=t, n_reports=nt, adverse_rate=round(joint, 3),
                         lift=round(joint / base_rate, 3) if base_rate else np.nan,
                         severity_lift=round(sev_lift, 3),
                         n_severe=n_s, k_severe=k_s, rate_severe=None if n_s == 0 else round(r_s, 3),
                         wilson_severe_lo=round(lo_s, 3) if n_s else None, wilson_severe_hi=round(hi_s, 3) if n_s else None,
                         n_nonsevere=n_n, k_nonsevere=k_n, rate_nonsevere=None if n_n == 0 else round(r_n, 3),
                         wilson_nonsevere_lo=round(lo_n, 3) if n_n else None, wilson_nonsevere_hi=round(hi_n, 3) if n_n else None,
                         fisher_p_onesided=None if np.isnan(p) else round(p, 4)))
    df = pd.DataFrame(rows)
    ok = df.fisher_p_onesided.notna()
    df["bh_q"] = np.nan
    if ok.any():
        df.loc[ok, "bh_q"] = multipletests(df.loc[ok, "fisher_p_onesided"].values, alpha=Q_FDR, method="fdr_bh")[1].round(4)
    df["flag_point"] = ((df.severity_lift > SEV_LIFT_MIN) & (df.lift > LIFT_MIN) &
                        df.rate_nonsevere.notna() & df.rate_severe.notna() & (df.rate_nonsevere < df.rate_severe))
    df["flag"] = df.flag_point & df.bh_q.notna() & (df.bh_q <= Q_FDR)
    return df.sort_values("lift", ascending=False).reset_index(drop=True), base_rate, p_sev, N


def main():
    reps = load_reports("treatment_updated_597")
    base_sev = baseline_severity()
    df, base_rate, p_sev, N = build(reps, base_sev)
    jac, inter, topn = split_half_stability(reps)
    try:
        xjac, _ = cross_source_stability(reps, load_reports("treatmenet_updated"))
    except Exception:
        xjac = None
    df.to_csv(os.path.join(RESULTS, "e4_association_v3_pairs.csv"), index=False)
    summ = dict(
        stratification="pre-treatment baseline only (comorbidity count >= %d or age >= %d); no outcome or treatment text" % (BASELINE_COMORB, BASELINE_AGE),
        adverse_rule="report-level adverse outcome, excluding outcome strings that mention the tested treatment itself",
        analysis_label="severity-stratified sensitivity analysis (not a formal confounding-by-indication test)",
        n_reports=N, n_treatments=int(len(df)), base_rate_adverse=round(base_rate, 3), p_severe_baseline=round(p_sev, 3),
        min_support=MIN_TREAT, sev_lift_min=SEV_LIFT_MIN, lift_min=LIFT_MIN, q_fdr=Q_FDR,
        n_flag_point=int(df.flag_point.sum()), n_bh_reject=int((df.bh_q <= Q_FDR).sum()), n_flag=int(df.flag.sum()),
        flagged=df.loc[df.flag, "treatment"].tolist(),
        flagged_point_only=df.loc[df.flag_point & ~df.flag, "treatment"].tolist(),
        bh_reject_not_enriched=df.loc[(df.bh_q <= Q_FDR) & ~df.flag_point, "treatment"].tolist(),
        split_half_jaccard=jac, repeat_run_jaccard=xjac)
    json.dump(summ, open(os.path.join(RESULTS, "e4_association_v3.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1))
    cols = ["treatment", "n_reports", "lift", "severity_lift", "rate_severe", "n_severe", "rate_nonsevere", "n_nonsevere", "bh_q", "flag_point", "flag"]
    print(df[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
