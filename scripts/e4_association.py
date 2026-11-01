import os, re, json, itertools
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from common import load_treatment_outcomes, RESULTS, FIGS, SEED

rng = np.random.default_rng(SEED)

POS = ["improv", "recover", "resolv", "resolution", "discharg", "remission",
       "normaliz", "normalis", "stable", "stabil", "cured", "cure", "survived",
       "better", "decreased symptom", "subsided", "relief", "reduction in",
       "negative for covid", "negative pcr", "clear"]
NEG = ["death", "died", "deceas", "expired", "fatal", "mortal", "deteriorat",
       "worsen", "failure", "intubat", "mechanical ventilat", "icu",
       "intensive care", "sepsis", "septic", "arrest", "complication",
       "respiratory failure", "shock", "readmission", "relapse", "progress",
       "exacerbat", "thrombo", "embol", "hemorrhag", "haemorrhag"]
SEVERITY = ["icu", "intensive care", "mechanical ventilat", "intubat",
            "respiratory failure", "septic shock", "vasopressor", "ecmo",
            "death", "died", "expired", "deceas", "fatal", "cardiac arrest"]


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def polarity(o):
    t = norm(o)
    if any(k in t for k in NEG):
        return "neg"
    if any(k in t for k in POS):
        return "pos"
    return "other"


# crude treatment canonicalisation: keep the head drug/intervention token(s)
DROP_PAREN = re.compile(r"\([^)]*\)")
def canon_treat(t):
    t = norm(t)
    t = DROP_PAREN.sub("", t).strip()
    # take first comma/clause chunk
    t = t.split(",")[0].split(" for ")[0].split(" with ")[0].strip()
    # collapse dosage tails
    t = re.sub(r"\b\d+[\d\.\-/ ]*(mg|g|mcg|ml|units?|grams?|kg)\b.*$", "", t).strip()
    return t


def load_reports(folder):
    reps = []
    for j in load_treatment_outcomes(folder):
        treats = [canon_treat(x) for x in j.get("treatments", []) if str(x).strip()]
        treats = [t for t in treats if len(t) >= 3]
        outs = [norm(x) for x in j.get("outcomes", []) if str(x).strip()]
        pol = [polarity(x) for x in outs]
        info = j.get("patient_info", {}) or {}
        comorb = info.get("comorbidities", []) or []
        # severity proxy: any severity marker in outcomes OR >=2 comorbidities
        sev_text = " ".join(outs) + " " + " ".join(norm(c) for c in comorb)
        severe = any(k in sev_text for k in SEVERITY) or (len(comorb) >= 2)
        reps.append(dict(pmcid=j.get("pmcid"), treatments=sorted(set(treats)),
                         outcomes=outs, pol=pol, severe=bool(severe),
                         n_comorb=len(comorb)))
    return [r for r in reps if r["treatments"]]


def association_table(reps, min_treat=8):
    N = len(reps)
    # treatment frequency
    tcount = Counter()
    for r in reps:
        for t in set(r["treatments"]):
            tcount[t] += 1
    treats = [t for t, c in tcount.items() if c >= min_treat]
    # polarity base rates (report-level: does report contain >=1 outcome of polarity p)
    base = {p: np.mean([p in r["pol"] for r in reps]) for p in ["pos", "neg"]}
    rows = []
    for t in treats:
        rt = [r for r in reps if t in r["treatments"]]
        nt = len(rt)
        for p in ["pos", "neg"]:
            joint = np.mean([p in r["pol"] for r in rt])          # P(O_p | T)
            lift = joint / base[p] if base[p] > 0 else np.nan
            # PMI on report-level co-occurrence
            p_t = nt / N
            p_o = base[p]
            p_to = np.mean([(t in r["treatments"]) and (p in r["pol"]) for r in reps])
            pmi = np.log2(p_to / (p_t * p_o)) if p_to > 0 else np.nan
            # confounding-by-indication probe: severity enrichment of the treatment
            p_sev_t = np.mean([r["severe"] for r in rt])
            p_sev = np.mean([r["severe"] for r in reps])
            sev_lift = p_sev_t / p_sev if p_sev > 0 else np.nan
            # stratified association within severe / non-severe
            def strat_joint(sub):
                return np.mean([p in r["pol"] for r in sub]) if sub else np.nan
            joint_sev = strat_joint([r for r in rt if r["severe"]])
            joint_non = strat_joint([r for r in rt if not r["severe"]])
            rows.append(dict(treatment=t, n_reports=nt, polarity=p,
                             P_O_given_T=round(joint, 3), base_rate=round(base[p], 3),
                             lift=round(lift, 3), pmi=round(pmi, 3),
                             P_severe_given_T=round(p_sev_t, 3),
                             severity_lift=round(sev_lift, 3),
                             joint_severe=None if np.isnan(joint_sev) else round(joint_sev, 3),
                             joint_nonsevere=None if np.isnan(joint_non) else round(joint_non, 3)))
    return pd.DataFrame(rows), base, N


def flag_confounded(df, sev_thresh=1.25):
    """A (treatment, neg-outcome) pair is flagged confounded-by-indication when the
    treatment is enriched for severity (severity_lift > thresh) AND the neg association
    attenuates in the non-severe stratum vs the severe stratum."""
    flags = []
    for _, r in df[df.polarity == "neg"].iterrows():
        if r.lift is None or np.isnan(r.lift):
            continue
        confounded = (r.severity_lift is not None and not np.isnan(r.severity_lift)
                      and r.severity_lift > sev_thresh and r.lift > 1.1)
        attenuates = (r.joint_nonsevere is not None and r.joint_severe is not None
                      and r.joint_nonsevere < r.joint_severe)
        flags.append(dict(treatment=r.treatment, polarity="neg", lift=r.lift,
                          severity_lift=r.severity_lift,
                          joint_severe=r.joint_severe, joint_nonsevere=r.joint_nonsevere,
                          flagged_confounded=bool(confounded and attenuates)))
    return pd.DataFrame(flags)


def split_half_stability(reps, topn=20):
    idx = np.arange(len(reps)); rng.shuffle(idx)
    h1 = [reps[i] for i in idx[:len(reps)//2]]
    h2 = [reps[i] for i in idx[len(reps)//2:]]
    def top_pairs(sub):
        c = Counter()
        for r in sub:
            for t in set(r["treatments"]):
                for p in set(r["pol"]):
                    if p in ("pos", "neg"):
                        c[(t, p)] += 1
        return [k for k, _ in c.most_common(topn)]
    a, b = set(top_pairs(h1)), set(top_pairs(h2))
    jac = len(a & b) / len(a | b) if (a | b) else 0.0
    return round(jac, 3), len(a & b), topn


def cross_source_stability(primary, secondary, topn=20):
    def top_pairs(sub):
        c = Counter()
        for r in sub:
            for t in set(r["treatments"]):
                for p in set(r["pol"]):
                    if p in ("pos", "neg"):
                        c[(t, p)] += 1
        return [k for k, _ in c.most_common(topn)]
    a, b = set(top_pairs(primary)), set(top_pairs(secondary))
    jac = len(a & b) / len(a | b) if (a | b) else 0.0
    return round(jac, 3), len(a & b)


def main():
    reps = load_reports("treatment_updated_597")
    df, base, N = association_table(reps)
    df = df.sort_values(["polarity", "lift"], ascending=[True, False])
    flags = flag_confounded(df)
    jac, inter, topn = split_half_stability(reps)

    # cross-source: second extraction run (treatmenet_updated, 664 reports)
    try:
        reps2 = load_reports("treatmenet_updated")
        xjac, xinter = cross_source_stability(reps, reps2)
    except Exception as e:
        xjac, xinter = None, None

    df.to_csv(os.path.join(RESULTS, "e4_association_full.csv"), index=False)
    flags.to_csv(os.path.join(RESULTS, "e4_flagged_pairs.csv"), index=False)

    n_flag = int(flags.flagged_confounded.sum()) if len(flags) else 0
    top_neg = df[df.polarity == "neg"].head(8)[["treatment", "n_reports", "lift", "severity_lift", "joint_severe", "joint_nonsevere"]]
    top_pos = df[df.polarity == "pos"].head(8)[["treatment", "n_reports", "lift", "pmi"]]

    summary = dict(
        n_reports=N, base_rate_pos=round(base["pos"], 3), base_rate_neg=round(base["neg"], 3),
        n_treatments_tested=int(df.treatment.nunique()),
        n_neg_pairs=int((df.polarity == "neg").sum()),
        n_flagged_confounded=n_flag,
        split_half_jaccard_top20=jac, split_half_intersection=inter,
        cross_source_jaccard_top20=xjac, cross_source_intersection=xinter,
        top_neg_pairs=top_neg.to_dict("records"),
        top_pos_pairs=top_pos.to_dict("records"),
        flagged_examples=flags[flags.flagged_confounded].to_dict("records")[:10],
    )
    with open(os.path.join(RESULTS, "e4_association.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
