# Prediction-Powered Corpus Mining: Coverage-Valid Statistics from Scarce Labels (ICDM 2026)

Code for the IEEE ICDM 2026 paper by Jing Wang, Jie Shen and Amar Sra (`paper_camera_ready.pdf` is the final
camera-ready version).

The method estimates how often a property occurs in a large document corpus, for example the fraction of case
reports that describe Long COVID. An LLM labels every document; a small random sample is labeled by experts; and
prediction-powered inference (PPI) uses the expert sample to remove the LLM's bias. The result is a corpus rate with
a confidence interval that covers the true rate when the expert sample is representative of the corpus.

## Contents

| file | content |
|---|---|
| `ppi_corpus.py` | **the algorithm**: Algorithm 1 of the paper and the baselines of Sec. V-C |
| `ppi_demo.ipynb` | usage demo with outputs: load data → input format → run → output format → baselines → candidate screen |
| `reproduce.ipynb` (= `reproduce.py`) | regenerates the paper's tables, figures and quoted numbers; it calls `ppi_corpus.py` for every estimate |
| `data/` | derived data (~7 MB): labels, LLM scores, similarity scores, extractions; no article full text |
| `scripts/` | optional GPU scripts that produced the cached LLM scores and encoder results in `data/` |
| `outputs/` | the outputs of `reproduce.ipynb` |

## Install

```bash
pip install -r requirements.txt     # numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, datasets,
                                    # ppi-python==0.2.3, jupyter
```

Everything runs on a CPU.

## Quick start

```python
import pandas as pd
import ppi_corpus as pc

gold = pd.read_csv("gold.csv")     # columns: id, label (0/1 expert label), p_yes (LLM yes-probability)
pool = pd.read_csv("pool.csv")     # columns: id, p_yes

res = pc.estimate_corpus_rate(gold.label.values, gold.p_yes.values, pool.p_yes.values, K=5)
print(res["estimate"], res["ci"], res["score_shift"])
```

`ppi_demo.ipynb` runs this on the Hallmarks of Cancer benchmark (`data/bench_llm/hoc.csv`), where the true rate is
known, and compares it with the baselines.

## Input format

`estimate_corpus_rate(y_gold, p_gold, p_pool, calibration=None, folds=None, K=5, alpha=0.05, seed=0, clip=True)`

| input | shape | meaning |
|---|---|---|
| `y_gold` | `(n,)` 0/1 | expert labels of the gold set |
| `p_gold` | `(n,)` float in [0, 1] | LLM yes-probability $p_{\mathrm{raw}}(x)$ of each gold document (Eq. (11)) |
| `p_pool` | `(N,)` float in [0, 1] | LLM yes-probability of each unlabeled document |
| `calibration` | `(p_cal, y_cal)` or `None` | a labeled set **independent** of the gold set, used only to fit Platt scaling |
| `folds` | `(n,)` int or `None` | report-level fold of each gold document for cross-fitting (default: random split into `K` folds) |
| `alpha` | float | the interval has level `1 - alpha` |
| `clip` | bool | intersect the interval with [0, 1], as in Eq. (21) |

The gold set must be a random sample of the same corpus as the pool; that is the condition under which the interval
covers (Sec. IV-B).

## Output format

A dict:

| key | meaning |
|---|---|
| `estimate` | estimated corpus rate $\hat\theta$ |
| `ci` | `(lower, upper)` confidence interval |
| `lam` | power-tuning weight $\lambda$ (estimated by PPI++; 1 for Cross-PPI) |
| `score_shift` | $\hat\Delta_f$ of Eq. (26): mean pool pseudo-label minus mean gold pseudo-label; a large value flags a gold set that does not look like the pool |
| `method` | which branch of Algorithm 1 ran |
| `f_gold`, `f_pool` | the pseudo-labels the estimator used (out-of-fold for the gold set when cross-fitting) |

## Functions and the paper

| function | paper |
|---|---|
| `estimate_corpus_rate` | Algorithm 1, estimation part (lines 4–19): with an independent calibration set, Platt + PPI++; otherwise K-fold cross-fitted Platt + Cross-PPI ($\lambda=1$); score-shift diagnostic |
| `ppi_plusplus` | PPI++, Eqs. (22), (23), (21); `lam=1` gives basic PPI, Eq. (18) |
| `cross_ppi` | Cross-PPI, Eqs. (20), (21) |
| `crossfit_calibrate` | out-of-fold Platt calibration, Eqs. (15), (16) |
| `platt_fit`, `platt_apply` | Platt scaling, Eqs. (13), (14) |
| `hard_label_ppi` | the hard-label variant $h(x)=\mathbf 1\{p_{\mathrm{raw}}(x)\ge t_0\}$ with PPI++ |
| `score_shift` | Eq. (26) |
| `choose_threshold`, `union_screen` | Algorithm 1, lines 1–2: the candidate screen (prioritization only; never applied to the estimator) |
| `naive_ci`, `gold_only_ci`, `acc_estimate`, `acc_ci` | baselines of Sec. V-C: naive count (Eq. (3)), expert-only, Adjusted Classify-and-Count with a 200-resample bootstrap |

PPI++ and Cross-PPI are computed with `ppi-python` 0.2.3, the library behind every number in the paper.

## Reproducing the paper

```bash
jupyter nbconvert --to notebook --execute --inplace reproduce.ipynb   # or: python reproduce.py  (a few minutes on a CPU)
```

The notebook downloads the two public benchmarks (HoC, ADE) and writes:

| output | content |
|---|---|
| `outputs/paper_numbers.json` | every number the notebook regenerates, keyed by analysis (keys per paper element below) |
| `outputs/tables/tab_*_rows.tex` | the bodies of Tables I–V (`tab_e1`, `tab_cov`, `tab_clincov`, `tab_rob`, `tab_e4`), plus `tab_lc_rows.tex`, a supplementary Long COVID table the final paper does not print |
| `outputs/figs/*.pdf` | Figs. 3–5 (`fig_retrieval`, `fig_e1`, `fig_coverage`; `fig_coverage_as_printed` is the image printed as Fig. 5, see notes) plus two supplementary figures (`fig_confounding`, `fig_calibration`) |

| paper element | notebook section | `paper_numbers.json` keys | inputs |
|---|---|---|---|
| Fig. 1 and Fig. 2 counts, Sec. V-A (123 = 52/71 gold, 3,066 pool, 0/123 overlap, 551 reports, benchmark sizes and prevalences) | §1, §3, §9 | `n_gold*`, `n_pool`, `gold_pool_overlap`, `gold_prevalence`, `bench_{hoc,ade}`, `assoc.n_reports` | `data/longcovid_gold.csv`, `data/screen_pool.csv`, public benchmarks |
| Table I, Fig. 4, Sec. V-B (classification, label efficiency, labeler comparison) | §2 | `t1_*`, `probe_*`, `labeler_acc_range` | `data/cached_gpu/*`, `data/llm/*` |
| Table II, Fig. 5(a–c), Sec. V-C (HoC + ADE; naive / ACC / gold-only / PPI; TF-IDF and zero-shot-LLM predictors; ECE) | §3 | `bench_{hoc,ade}`, `bench_{hoc,ade}_llm`, `{hoc,ade}_*`, `bench_llm_*` | public benchmarks + `data/bench_llm/*.csv` |
| Table III, Fig. 5(d), Sec. V-D (measured coverage on clinical labels, per labeler; finite-population design) | §7 | `clinical_coverage`, `clin_*` | all `data/llm/*_gold.csv` |
| Table IV, Sec. V-E (pseudo-label form, PPI++ λ, calibration × screen ablation) | §4 | `robustness`, `hoc_screen` | HoC |
| Sec. V-E, "When the assumption is violated" (cross-prediction regime, non-representative gold) | §5 | `crossppi`, `nonrep_gold` | HoC |
| Fig. 3, Sec. V-F (embedding screen, reranker, union screen) | §8 | `retrieval` | `data/screen_*.csv`, `data/reranker_gold.csv` |
| Table V, Sec. V-G (severity-stratified sensitivity analysis, Wilson CIs, one-sided Fisher + BH-FDR) | §9 | `assoc`, `assoc_leaky`, `assoc_table_rows` | `data/extractions_*.jsonl` |
| Abstract and contribution list | §10 | `abstract_numbers` | the keys above |
| Supplementary, not printed in the final paper: Long COVID applied estimate | §6 | `longcovid` | `data/llm/*_{gold,pool}.csv`, gold labels |

### Numbers in the abstract and the contribution list (§10)

Width ratios use the TF-IDF predictor (upper blocks of Table II), as in Sec. V-C and Fig. 5(c), averaged over
$n_g \in \{50, 100, 150, 200, 300\}$. Where the abstract and the contributions quote one figure for both benchmarks, it
is the more conservative benchmark: HoC for the width saving and ADE for the ACC penalty.

| claim | paper | regenerated |
|---|---|---|
| PPI coverage on the two benchmarks | 0.91–0.96 | TF-IDF predictor 0.912–0.964; LLM predictor 0.908–0.964 |
| PPI narrower than expert-only | ≈15% (abstract); 16% on average (contributions); 16% HoC, 21% ADE (Fig. 5c); 15–16% HoC, 20–22% ADE at every $n_g$ (Sec. V-C) | HoC 15.6% on average (15.0–16.1% per $n_g$); ADE 21.2% (20.1–22.4%) |
| ACC interval width / PPI width | 2.5 times (abstract, contributions); 3.0× HoC, 2.5× ADE (Sec. V-C) | HoC 2.96×; ADE 2.52× |
| ACC coverage | comparable; 0.94–0.97 on HoC | HoC 0.942–0.966; ADE 0.934–0.968 |
| naive count under systematic bias | zero coverage | 0.000 at every $n_g$ on HoC (TF-IDF, bias −4.7 pts) and on both benchmarks with the LLM predictor; with the nearly unbiased ADE TF-IDF predictor it covers (0.994–1.000) |
| PPI coverage on the clinical labels | 0.92–0.96 (contributions) | 0.924–0.957 |
| expert labels | ~100 reports | gold sizes 50–300 on the benchmarks; 123 clinical gold reports |

### Notes on matching the paper

* **Interval clipping.** Eq. (21) intersects each interval with [0, 1]. The paper's tables report the intervals
  without that intersection, so `reproduce.py` sets `CLIP_CI = False`; `ppi_corpus.py` clips by default. With
  clipping, no coverage value changes; four mean widths at ADE $n_g = 50$ shrink by at most 0.0004, which changes one
  printed cell (Table II, ADE, $n_g = 50$, PPI width 0.155 → 0.154).
* **Table I.** The notebook labels the first row `TF-IDF+LogReg`; the paper prints `TF-IDF`. The numbers are identical.
* **Table V.** `tab_e4_rows.tex` carries two extra trailing columns (BH-adjusted q, discovery flag) that the paper's
  six-column table does not print. All 45 leakage-controlled pairs have q = 1.00 (`assoc_table_rows[*].q`).
* **Fig. 5(d).** The image printed in the paper plots the superpopulation curves
  (`clinical_coverage.by_labeler.<tag>.coverage_superpop`: no finite-population correction). Table III and Sec. V-D
  use the finite-population design (`clinical_coverage.by_labeler.<tag>.coverage`), which `fig_coverage.pdf`(d)
  plots. §7 computes both and also writes `fig_coverage_as_printed.pdf`, which is the printed Fig. 5. Panels (a–c)
  are identical in both files.
* **Fig. 4.** The printed figure is the notebook's `fig_e1.pdf`. The seed-stability statistics in the caption
  (R = 15 seeded fine-tunes, D = 200 subsample draws) come from repeated GPU fine-tuning runs that are not part of
  this bundle.
* **Numbers not computed by the notebook.** The PubMed Open Access size (1,488,346 articles) and the 152,974
  probable case reports come from the cited prior work. The ~15k characters per report needs the full text, which
  is not redistributed. The ADE corpus size 23,516 is the public train + test split size (17,637 + 5,879) before
  de-duplication. The default labeler's per-class F1 (0.611 / 0.755) follows from `data/llm/qwen38_27b_gold.csv`,
  and the binomial half-width of 8.7 points is 1.96·sqrt(0.423·0.577/123).
* **Platt penalty.** `platt_fit` uses `C = 1e3`, a negligible ridge (the maximum-likelihood fit of Eq. (14)),
  everywhere except the supplementary Long COVID cross-fit of §6, which keeps the accepted version's `C = 1`.

## Data (`data/`, ~7 MB, mostly the frozen embeddings in `cached_gpu/`; no article full text is redistributed)

* `longcovid_gold.csv` — 123 PMCIDs, physician Long COVID label (52 yes / 71 no), 5-fold assignment.
* `screen_gold.csv`, `screen_pool.csv` — embedding similarity to the Long COVID prototype
  (Qwen3-Embedding) and regex-cue flag for the 123 gold and the 3,066-report pool.
* `reranker_gold.csv` — Qwen3-Reranker scores on the gold set. (`embed_sim_proto2` in the screen
  files is a second, unused prototype.)
* `llm/<tag>_{gold,pool}.csv` — zero-shot labeler outputs: hard decision `pred`, continuous
  yes/no token probability `p_yes`, and the short generated rationale. All labelers use the same
  prompt (`prompts.md`). `qwen38_27b` (Qwen3.8-27B, bf16, vLLM >= 0.17, greedy, thinking off) is the paper's
  default labeler; `qwen14b` is Qwen3-14B in bf16 and `qwen14b4bit` the same model in 4-bit NF4; `llama8b`, `qwen30b`, `qwen32b`, `qwen72b` are the
  comparison labelers (gold only where no pool file exists). `qwen14b_4bit_hard_*` are the hard-only
  outputs of the run behind the accepted version (the "accepted run" labeler: Qwen3-14B† in Table III and
  the recall-heavy labeler, precision 0.61 / recall 0.96, in Sec. V-B and Fig. 4(b)).
  Columns: `pmcid`, `label` (gold only), `pred`, `p_yes`, `parse` (how `pred` was read: `answer_tag` /
  `first_word`), `score_src` (`logprob` or `hard_fallback`), `text` (generated answer text, usually a bare Yes/No).
  `llm/<tag>_summary.json` holds run metadata (model id, truncation, gold metrics). `qwen72b` is the
  `Qwen2.5-72B-Instruct-AWQ` 4-bit checkpoint. The Qwen3.8-27B scores were produced with vLLM 0.28.0 /
  transformers 5.16.1 / torch 2.13 (CUDA 13), thinking disabled, text-only (`scripts/run_qwen38.sh`); all
  other labelers with vLLM 0.13 / transformers 4.57.
* `extractions_primary.jsonl` (570 reports) and `extractions_repeat.jsonl` (664 reports, repeat
  extraction run with resampled decoding) — structured (treatments, outcomes, comorbidities)
  extracted by Qwen3-14B; the notebook keeps the 551 reports with at least one canonical treatment.
* `adjudication_sheet_50.csv` — 50 randomly sampled (treatment, outcome) extractions with the
  short source passage around each mention and empty Y/N columns, so a clinician can measure
  field-level extraction precision (`scripts/score_adjudication.py` scores a filled sheet). These
  passages are the only article excerpts in the bundle.
* `bench_llm/{hoc,ade}.csv` — the paper's zero-shot labeler (Qwen3.8-27B) applied to every benchmark
  population sentence: gold `label`, hard `pred`, continuous `p_yes` (`scripts/bench_llm_label.py`). These drive
  the LLM-predictor blocks of Table II and the usage demo.
* `cached_gpu/` — 5-fold CV outputs of the fine-tuned encoders (BERT, BioBERT, and their chunked
  full-document variants) and of TF-IDF+LR (`scripts/e1_baselines.py`, `scripts/e1_chunked_control.py`), plus
  `gold_embeddings_<tag>.npy`: frozen full-document embeddings of the 123 gold reports from
  Qwen3-Embedding-8B (last-token pooling), zembed-1 (backbone via `transformers` with the model's own
  document prompt and last-token pooling, since its sentence-transformers loader needs a newer library),
  BGE-M3 (CLS pooling) and MedGemma-4B (`google/medgemma-1.5-4b-it`, mean-pooled hidden states), all L2-normalised, documents
  truncated to 8,192 tokens (`scripts/e1_embed_probe.py`); the notebook refits the linear probes from these. All of these need article full text and a GPU to regenerate.

Full text of the PubMed Open Access articles can be re-fetched by PMCID from PMC; the
PMCIDs are in the CSVs.

## Source data for the GPU scripts (`scripts/`)

The scripts read a source folder given by the environment variable `LC_DIR` (default `./longcovid`) that
must contain: `amar_longcovid_yesno.csv` (pmcid, label, content =
full text), `5fold_pmcid_splits.csv`, `aaai_test.csv` (pmcid, content for the 3,066-report pool),
the precomputed embedding similarities `similarity_matrix.npy` / `similarity_matrix_test.npy`, and
`text/<pmcid>.txt` plus the `treatment_updated_597/` extraction JSONs for the association study.
Rebuild these from PMC full text by PMCID; the notebook itself never needs them. They write their
outputs to `../results/` (names `llm_<tag>_gold.csv`, `llm_<tag>_pool.csv`, `e1_*.json`), which
`build_release_data.py` in the paper repository copies into `data/`.

## GPU scripts (`scripts/`, optional)

`run_qwen38.sh` + `llm_label_v2.py` (vLLM, any HF chat model, same prompt; produced the default and
comparison labelers; set `CUDA_VISIBLE_DEVICES`/`HF_HUB_OFFLINE` as needed), `llm_label_4bit.py` (4-bit NF4
replication attempt of the accepted run; `MODEL=Qwen/Qwen3-14B`), `e1_embed_probe.py` (frozen embeddings for the
linear probes), `bench_llm_label.py` (zero-shot labels for the benchmark populations),
`e4_association_v3.py` with its helper `e4_association.py` (standalone severity-stratified reanalysis; the
notebook implements the same analysis), `e1_baselines.py`,
`e1_chunked_control.py`, `e3_reranker_gold.py`. They regenerate the files in `data/llm/` and
`data/cached_gpu/` from the article full text.

## Determinism

All random draws use `numpy.random.default_rng(42)`; coverage values are Monte-Carlo estimates
over 500 (benchmarks) / 1,000 (clinical) gold draws and reproduce exactly with the pinned
`ppi-python`. Set `R_DRAWS`/`R_CLIN` to smaller values for a quick smoke test.
