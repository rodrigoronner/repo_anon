# Privacy-Preserving Personalization in Education: A Federated Recommender System for Student Performance Prediction

Reproducibility package for the manuscript of the same title (anonymized for
double-blind review). Every number, table and data figure in the manuscript is
produced by `run_all.sh`, starting from the raw ASSISTments 2009–2010
skill-builder file, which is included in `data/raw/` together with the
examples produced by the pipeline.

## What the study does

Each of the 1,159 students of ASSISTments 2009–2010 is a federated client. A
recommender network (student and skill embeddings plus three aggregate
features, followed by a two-layer perceptron) predicts whether a student will
master a skill in a future window of their own history. The network is trained
with FedAvg and FedProx and compared with a centralized network of identical
architecture, with XGBoost and with logistic regression (centralized and
federated). All models are tuned with Optuna on validation
data and evaluated over ten paired seeds and 1,000 communication rounds. An
ablation study removes the personalized components of the recommender, and a
row-wise aggregation of the embedding tables tests whether they can be learned.

## Main results (test set, mean over 10 seeds)

| Model | ROC AUC |
|---|---|
| XGBoost, centralized | 0.789 |
| Recommender, centralized | 0.788 |
| Recommender, FedAvg | 0.777 |
| Recommender, FedProx (μ = 0.1, 0.5, 1.0, tuned) | 0.775–0.778 |
| Ranking by the student's past success rate (no model) | 0.689 |

- Cost of federation: 0.011 global AUC (95% CI over seeds 0.010–0.013;
  student-cluster bootstrap 0.004–0.019), but only 0.002 in per-student AUC
  (the ordering of each student's skills; CI includes zero). Both recommenders
  rank the same skill first for 89% of the students.
- With the aggregate features only, centralized and federated training both
  reach 0.779: the cost of federation is the value of the embedding tables.
  Under dense averaging, FedAvg moves the student embeddings by 0.01 (initial
  norm 5.6) in 1,000 rounds, i.e. it does not learn them; row-wise aggregation
  learns them but does not improve test performance.
- Logistic regression on the three features: 0.780 centralized, 0.779 federated.
- FedAvg and FedProx are equivalent within ±0.005 AUC (smallest TOST margins
  0.0007–0.0047).
- Leave-one-out encoding of the aggregates leaks the label
  (validation AUC 0.928 against 0.719 on test).

The complete results are in `outputs/REPORT.md`, `outputs/REPORT_ablation.md`
and `outputs/REPORT_review.md`.

## Where each result of the manuscript comes from

Each table and figure of the manuscript is built from the versioned files below.

| Manuscript item | Produced by | Data in this repository |
|---|---|---|
| Table 1 (cohort) | `01_prepare_data.py` | `data/processed_meta.json` |
| Table 2 (test-set performance) | `20`–`22_eval_*.py`, `30_analysis.py` | `outputs/analysis.json` (`summary`, `trivial`); per seed in `outputs/evaluation/*.json` |
| Table 3 (global and per-student AUC, agreement) | `26_eval_extensions.py`, `34_review_analyses.py` | `outputs/review_analyses.json` (`models`, `agreement`), `outputs/REPORT_review.md` |
| Table 4 (ablation) | `24_ablation.py`, `33_ablation_analysis.py` | `outputs/ablation_analysis.json`, `outputs/REPORT_ablation.md`; per seed in `outputs/evaluation/ablation.json`, predictions in `outputs/evaluation/ablation_preds/` |
| Table 5 (FedAvg vs FedProx) | `30_analysis.py` | `outputs/analysis.json` (`strategy_comparisons`); smallest equivalence margins in `outputs/review_analyses.json` |
| Table 6, Appendix A (search spaces and selections) | `10`–`13_tune_*.py` | `outputs/tuning/*_best.json`, `outputs/tuning/*_trials.csv` |
| Table 7, Appendix B (cost of federation) | `30_analysis.py` | `outputs/analysis.json` (`decomposition`) |
| Table 8, Appendix B (communication budget) | `30_analysis.py` | `outputs/evaluation/fl_budget_sensitivity.csv` |
| Row-wise aggregation and logistic regression (Section 5.4) | `26_eval_extensions.py` | `outputs/evaluation/extensions.json`, `extensions_rounds.csv.gz` |
| Student-cluster bootstrap (Sections 5.3–5.4) | `34_review_analyses.py` | `outputs/review_analyses.json` (`bootstrap`) |
| Figure 3 (local optimizer in the searches) | `10`–`12_tune_*.py` | `outputs/analysis.json` (`tuning`: best value per optimizer, fANOVA importances) |
| Figure 4 (test AUC and balanced accuracy) | `20`–`22_eval_*.py` | `outputs/evaluation/*.json` (per seed) |
| Figure 5 (trajectories per round) | `22_eval_federated.py` | `outputs/evaluation/fl_rounds.csv.gz` (every round of every run) |
| Embedding norms (Section 5.4) | `25_embedding_diagnostic.py` | `outputs/embedding_diagnostic.json` |
| Leave-one-out leakage (Section 4.2) | `03_loo_leakage_diagnostic.py` | `outputs/evaluation/loo_leakage_diagnostic.json` |
| Verification against Flower (Section 5.8) | `23_flower_crosscheck.py` | `outputs/evaluation/flower_crosscheck.json`, `flower_crosscheck_seed42.csv` |

Figures 1 (feature and label windows) and 2 (data flow of the federated
recommender) are diagrams drawn in the manuscript source and contain no data.

## Setup and reproduction

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
PY=python ./run_all.sh
```

The full run takes about five
hours on a 10-core laptop (CPU only). Evaluation is deterministic: the client
sample of every round and the random state of every local update are functions
of the seed, the round and the client, so re-running a condition reproduces
its trajectory exactly. Tuning runs nine Optuna trials in parallel and is
therefore not bit-for-bit reproducible; the study journals are included so the
selected configurations can be inspected or reused.

## Repository layout

| Path | Content |
|---|---|
| `config.py` | Paths, data source and checksum, splits, seeds, budgets |
| `models.py` | Recommender network, logistic regression, optimizer and training loop |
| `fl_sim.py` | In-process FedAvg/FedProx simulator (one client per student; dense or row-wise aggregation of the embedding tables) |
| `common.py`, `tuning_utils.py` | Metrics, threshold calibration, Optuna helpers |
| `01`–`03` | Data preparation, learning-free references, leakage diagnostic |
| `10`–`13` | Optuna studies (XGBoost, centralized recommender, one per federated strategy, row-wise aggregation, logistic regression) |
| `20`–`26` | Evaluation over 10 seeds, Flower verification, ablation, embedding diagnostic, row-wise aggregation and logistic regression |
| `30`–`34` | Statistical analysis, ablation analysis, per-student and bootstrap analyses, and the plotting and table-formatting code of the manuscript |
| `outputs/tuning/` | Optuna journals, trials and selected configurations |
| `outputs/evaluation/` | Per-seed results, per-round logs, test predictions of every model |
| `data/raw/` | Original ASSISTments 2009–2010 skill-builder file (compressed) |
| `data/processed_*.csv`, `data/processed_meta.json` | Examples produced by `01_prepare_data.py` and cohort statistics |

## Design decisions that matter for interpretation

- **No feature contains its own label.** Training examples use features from
  the first 40% (or 60%) of each student's history and labels from the next
  20%; test examples use the first 80% and the last 20%.
- **The test set is never used for any choice.** Hyperparameters, checkpoints
  and decision thresholds are chosen on a validation split of the training
  examples; tuning and evaluation use disjoint seeds. Test AUC is recorded in
  every round only to describe trajectories.
- **Paired federated comparisons.** For a given seed every federated condition
  samples the same clients in every round and seeds each local update
  identically, so strategy differences are not confounded with sampling noise.
- **Architecture frozen from the centralized search.** This favors the
  centralized reference, so the measured cost of federation is conservative.
- **Data flow.** Raw interactions, answers, labels and student-level features
  stay on each client; clients exchange model weights, and the skill statistic
  requires per-skill totals that a deployment would obtain through secure
  aggregation.
- **Simulator.** Federated training runs in-process (`fl_sim.py`) for speed;
  `23_flower_crosscheck.py` reruns tuned FedAvg in Flower 1.7 (seed 42, 100
  rounds): per-round test AUC differs from the simulator by at most 7e-5.

## Data

- `data/raw/skill_builder_data_corrected_collapsed.csv.gz`: the ASSISTments
  2009–2010 skill-builder data, "corrected, collapsed" version, as publicly
  distributed by the ASSISTments project (Feng, Heffernan & Koedinger, 2009,
  *User Modeling and User-Adapted Interaction*, 19(3), 243–266). It is gzip-
  compressed only; `run_all.sh` decompresses it, and `01_prepare_data.py`
  verifies the SHA-256 of the uncompressed file
  (`162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da`). If the
  compressed copy is absent, `run_all.sh` downloads the file from the original
  distribution.
- `data/processed_train.csv`, `data/processed_test.csv`: the leakage-free
  (student, skill) examples after filtering (1,159 students, 121 skills; 19,576
  training examples, of which 1,907 form the validation split, and 9,575 test
  examples). Columns: re-indexed student and skill identifiers, the three
  aggregate features (min–max scaled on training data), the target success
  rate and its binarized value (threshold 0.70), flags for pairs already seen
  in the student's history and for cold-start students, the validation flag
  and the window.
- `data/processed_meta.json`: cohort statistics reported in Table 1.

The data are de-identified by their distributors (numeric student, class,
teacher and school identifiers) and contain no directly identifying information.

## License

Code: MIT (see `LICENSE`). The ASSISTments data remain subject to the terms of
their original distributors, who should be cited in any use.
