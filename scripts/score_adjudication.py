"""Score a filled adjudication sheet: field-level precision (Y / (Y+N)) with Wilson 95% CIs."""
import sys, pandas as pd
from statsmodels.stats.proportion import proportion_confint
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "../results/adjudication_sheet_50.csv")
for col in ["treatment_correct_YN", "outcome_correct_YN", "polarity_correct_YN", "severity_correct_YN"]:
    v = df[col].astype(str).str.strip().str.upper(); y = (v == "Y").sum(); n = v.isin(["Y", "N"]).sum()
    if n: lo, hi = proportion_confint(y, n, method="wilson"); print(f"{col:24s} {y}/{n} = {y/n:.3f}  [{lo:.3f}, {hi:.3f}]")
    else: print(f"{col:24s} not filled")
