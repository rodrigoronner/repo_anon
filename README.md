# Privacy-Preserving Personalization in Education: A Federated Recommender System for Student Performance Prediction

Reproducibility package for the manuscript of the same title (anonymized for
double-blind review). Every number, table and data figure in the manuscript is
produced by `run_all.sh`, starting from the raw ASSISTments 2009–2010
skill-builder file.

## What the study does

Each of the 1,159 students of ASSISTments 2009–2010 is a federated client. A
recommender network (student and skill embeddings plus three aggregate
features, followed by a two-layer perceptron) predicts whether a student will
master a skill in a future window of their own history. The network is trained
with FedAvg and FedProx and compared with a centralized network of identical
architecture and with XGBoost. All models are tuned with Optuna on validation
data and evaluated over ten paired seeds and 1,000 communication rounds. An
ablation study removes the personalized components of the recommender.

## Main results (test set, mean over 10 seeds)

| Model | ROC AUC |
|---|---|
| XGBoost, centralized | 0.789 |
| Recommender, centralized | 0.788 |
| Recommender, FedAvg | 0.777 |
| Recommender, FedProx (μ = 0.1, 0.5, 1.0, tuned) | 0.775–0.778 |
| Ranking by the student's past success rate (no model) | 0.689 |

- Cost of federation: 0.011 AUC (95% CI 0.010–0.013).
- With the aggregate features only, centralized and federated training both
  reach 0.779: the cost of federation is the value of the embedding tables.
  FedAvg moves the student embeddings by 0.01 (initial norm 5.6) in 1,000
  rounds, i.e. it does not learn them.
- FedAvg and FedProx are equivalent within ±0.01 AUC (TOST p < 0.001).
- Leave-one-out encoding of the aggregates leaks the label
  (validation AUC 0.928 against 0.719 on test).

The complete results are in `outputs/REPORT.md` and `outputs/REPORT_ablation.md`.

## Where each result of the manuscript comes from

| Manuscript item | Produced by | File |
|---|---|---|
| Table 1 (cohort) | `01_prepare_data.py`, `32_latex_tables.py` | `outputs/latex/tab_cohort.tex` |
| Table 2 (test-set performance) | `20`–`22_eval_*.py`, `30_analysis.py`, `32_latex_tables.py` | `outputs/latex/tab_main.tex` |
| Table 3 (ablation) | `24_ablation.py`, `33_ablation_analysis.py` | `outputs/latex/tab_ablation.tex` |
| Table 4 (FedAvg vs FedProx) | `30_analysis.py`, `32_latex_tables.py` | `outputs/latex/tab_strategies.tex` |
| Table 5, Appendix A (search spaces) | `10`–`12_tune_*.py`, `32_latex_tables.py` | `outputs/latex/tab_search.tex` |
| Table 6, Appendix B (cost of federation) | `30_analysis.py`, `32_latex_tables.py` | `outputs/latex/tab_decomposition.tex` |
| Table 7, Appendix B (communication budget) | `30_analysis.py`, `32_latex_tables.py` | `outputs/latex/tab_budget.tex` |
| Figure 3 (local optimizer in the searches) | `31_figures.py` | `outputs/figures/fig_optuna_optimizer.pdf` |
| Figure 4 (test AUC and balanced accuracy) | `31_figures.py` | `outputs/figures/fig_forest.pdf` |
| Figure 5 (trajectories per round) | `31_figures.py` | `outputs/figures/fig_convergence.pdf` |
| Embedding norms (Section 5.4) | `25_embedding_diagnostic.py` | `outputs/embedding_diagnostic.json` |
| Leave-one-out leakage (Section 4.2) | `03_loo_leakage_diagnostic.py` | `outputs/evaluation/loo_leakage_diagnostic.json` |
| Verification against Flower (Section 5.8) | `23_flower_crosscheck.py` | `outputs/evaluation/flower_crosscheck*.{json,csv}` |

Figures 1 (feature and label windows) and 2 (data flow of the federated
recommender) are diagrams drawn in the manuscript source and contain no data.

## Setup and reproduction

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
PY=python ./run_all.sh
```

The raw file (`skill_builder_data_corrected_collapsed.csv`) is downloaded from
the ASSISTments project's public distribution and verified against the SHA-256
in `config.py`; it is not redistributed here. The full run takes about five
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
| `models.py` | Recommender network, optimizer and training loop |
| `fl_sim.py` | In-process FedAvg/FedProx simulator (one client per student) |
| `common.py`, `tuning_utils.py` | Metrics, threshold calibration, Optuna helpers |
| `01`–`03` | Data preparation, learning-free references, leakage diagnostic |
| `10`–`12` | Optuna studies (XGBoost, centralized recommender, one per federated strategy) |
| `20`–`25` | Evaluation over 10 seeds, Flower verification, ablation, embedding diagnostic |
| `30`–`33` | Statistical analysis, figures, LaTeX tables, ablation analysis |
| `outputs/tuning/` | Optuna journals, trials and selected configurations |
| `outputs/evaluation/` | Per-seed results, per-round logs, test predictions of every model |
| `outputs/figures/`, `outputs/latex/` | Figures (PDF, PNG at 300 dpi) and tables used in the manuscript |

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

## License

Code: MIT (see `LICENSE`). The ASSISTments data are distributed by their
original authors under their own terms and are not included in this repository.
