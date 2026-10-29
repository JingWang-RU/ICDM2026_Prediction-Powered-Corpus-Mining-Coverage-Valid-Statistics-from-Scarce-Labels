# Prediction-Powered Inference for Clinical Subgroup Analyses — code supplement

`ppi.py`      every component of the method
`demo.ipynb`  one runnable end-to-end example (ACS Public Coverage)
`repair.pdf`  the figure the notebook produces

## Run

    pip install folktables scikit-learn matplotlib pandas scipy
    jupyter nbconvert --to notebook --execute --inplace demo.ipynb

The notebook downloads ACS Public Coverage (2018 1-Year; CA, TX, NY, FL, GA, MS)
through `folktables`. No local data files are required.

## What the notebook reproduces

| quantity | value |
|---|---|
| inference pool | 365,862 |
| retained race categories | 8 |
| budget floor (Proposition 2) | 1,704 |
| worst-group coverage, pooled rectifier (n=1000) | 0.013 |
| worst-group coverage, per-group rectifier (n=1000) | 0.180 |
| mean width ratio | 8.0 |

The first three rows of Table 1. The budget sweep in the last two cells shows the
per-group rectifier crossing into valid coverage once the gold budget clears the floor.

## `ppi.py`

| function | role |
|---|---|
| `rectifier` | rectifier mean and standard error |
| `pooled_interval` | standard PPI interval, pooled rectifier |
| `group_interval` | the repair: per-group rectifier, withheld below `MIN_GOLD` |
| `evaluate` | replicate-level coverage and width for both estimators |
| `rectifier_gap` | the group gap that drives failure |
| `regimes` | the three-regime classification |
| `coverage_law` | coverage as a function of the standardized gap |
| `transfer_certificate` | BBSE admissibility check |
| `lambda_rules` | the two PPI++ tuning rules and whether they agree |
| `budget_floor` | Proposition 2 |

@inproceedings{wang2026prediction,
  title={Prediction-Powered Corpus Mining: Coverage-Valid Statistics from Scarce Labels},
  author={Wang, Jing and Shen, Jie and Sra, Amar and Luo, Yiming and Xie, Qiaomin and Weiss, Jeremy C.},
  booktitle={Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year={2026}
}