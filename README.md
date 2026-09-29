# Privacy-Preserving Personalization in Education: A Federated Recommender System for Student Performance Prediction

Reproducibility package for the manuscript of the same title (anonymized for
double-blind review). Every number, table and data figure in the manuscript is
produced by `run_all.sh`, starting from the raw ASSISTments 2009–2010
skill-builder file, which is included in `data/raw/` together with the
examples produced by the pipeline.

## In plain language

**The problem.** Online learning platforms record every exercise a student
does: what they got right, what they got wrong, how long they took. From these
records, a program can predict which topics a student has already mastered and
recommend the next exercise, much as a private tutor would. These records,
however, are personal data, often of children and teenagers, and data
protection laws such as Brazil's LGPD require special care with them. The usual
way to build such systems is to gather every student's data on a single
server, which creates a privacy risk.

**The idea we tested.** *Federated learning* reverses the flow: instead of the
data travelling to the program, the program travels to the data. Each student's
device trains its own copy of the model, and only what was *learned* (the model
update), never the student's answers, is sent back and combined with the
others. The raw data never leave the device.

**Our questions.** (1) How much prediction quality is lost by protecting the
data this way? (2) Can the system still personalize, that is, learn something
specific about each student? (3) Which of the known ways of combining the
students' updates works best?

**How we did it.** We used a public dataset of 1,159 students solving math
exercises on an online tutoring platform, treating each student as a separate
participant. We compared the privacy-preserving system with a traditional one
that pools all data, making the comparison fair: the same model on both sides,
the same tuning effort, no hidden "cheating" in how the data are prepared, and
every experiment repeated ten times.

**What we found.**

1. **The loss in quality is small.** On a scale from 0.5 (guessing) to 1
   (always right), the traditional system scored 0.788 and the
   privacy-preserving one 0.777, about one percentage point apart.
2. **For the decision that matters, there is practically no loss.** To
   recommend the next exercise, what counts is ranking each student's own
   topics. Here the two systems were almost identical: for 89% of the students
   they would put the same topic first.
3. **The "individual" part of the model is not learned, and it is barely
   missed.** The model had a component meant to represent each student
   individually. In the privacy-preserving version this component learns
   nothing, because each student contributes too little data; even when we
   fixed that with a dedicated technique, predictions did not improve, and it
   helped very little in the traditional system too. This component can
   therefore be *removed*: it costs nothing and *increases* privacy, since it
   was the part that could most identify a student.
4. **A simple model is enough.** A classic statistical formula using only each
   student's and each topic's past success rates lost *nothing* when trained
   in the privacy-preserving way.
5. **It is easy to be misled when evaluating these systems.** An earlier
   version of this study reached wrong conclusions: a common way of preparing
   the data let the answer "leak" into the model, a widely used score (F1) gave
   high marks even to a system that simply predicted "the student will succeed"
   for everyone, and a poorly configured system made privacy look thirteen
   times more costly than it really is.

**Why it matters.** It shows that schools and learning platforms can protect
students' data, in line with the LGPD, *without hurting the recommendation of
activities*; it tells developers how to design these systems (remove, or keep on
the device, the individual part of the model; tune the local training, which
matters more than the choice of combination rule); and it gives researchers a
more rigorous way of evaluating them, with common pitfalls identified. All code
and data are in this repository.

**Honest limits.** The study uses a single dataset (2009–2010, math, a platform
in the United States), so the results must be confirmed with other students and
subjects. Federated learning alone does not guarantee full privacy; additional
protections, such as secure aggregation and differential privacy, are
discussed in the manuscript.

**In one sentence:** it is possible to personalize teaching while protecting
students' data, at a minimal cost in quality, provided the system is well
designed and carefully evaluated.

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
- Robustness: with a temporal validation window (fit on the first training window,
  validate on the later one) the conclusions hold, and the untrained student
  embedding even lowers FedAvg's accuracy; run with the same local optimizer as
  FedAvg (SGD), FedProx is as stable as FedAvg.
- Leave-one-out encoding of the aggregates leaks the label
  (validation AUC 0.928 against 0.719 on test).

The complete results are in `outputs/REPORT.md`, `outputs/REPORT_ablation.md`,
`outputs/REPORT_review.md` and `outputs/REPORT_robustness.md`.

## Where each result of the manuscript comes from

Each table and figure of the manuscript is built from the versioned files below.

| Manuscript item | Produced by | Data in this repository |
|---|---|---|
| Table 1 (cohort) | `01_prepare_data.py` | `data/processed_meta.json` |
| Table 2 (test-set performance) | `20`–`22_eval_*.py`, `30_analysis.py` | `outputs/analysis.json` (`summary`, `trivial`); per seed in `outputs/evaluation/*.json` |
| Table 3 (global and per-student AUC, agreement) | `26_eval_extensions.py`, `34_review_analyses.py` | `outputs/review_analyses.json` (`models`, `agreement`), `outputs/REPORT_review.md` |
| Table 4 (ablation) | `24_ablation.py`, `33_ablation_analysis.py` | `outputs/ablation_analysis.json`, `outputs/REPORT_ablation.md`; per seed in `outputs/evaluation/ablation.json`, predictions in `outputs/evaluation/ablation_preds/` |
| Table 5 (FedAvg vs FedProx) | `30_analysis.py` | `outputs/analysis.json` (`strategy_comparisons`); smallest equivalence margins in `outputs/review_analyses.json` |
| Table 6 (robustness: temporal validation; FedProx with the local optimizer of FedAvg) | `27_robustness.py`, `35_robustness_analysis.py` | `outputs/robustness_analysis.json`, `outputs/REPORT_robustness.md`; runs in `outputs/evaluation/robustness/` |
| Table 7, Appendix A (search spaces and selections) | `10`–`13_tune_*.py` | `outputs/tuning/*_best.json`, `outputs/tuning/*_trials.csv` |
| Table 8, Appendix B (cost of federation, with Cohen's d) | `30_analysis.py` | `outputs/analysis.json` (`decomposition`) |
| Table 9, Appendix B (communication budget) | `30_analysis.py` | `outputs/evaluation/fl_budget_sensitivity.csv` |
| Row-wise aggregation and logistic regression (Section 5.4) | `26_eval_extensions.py` | `outputs/evaluation/extensions.json`, `extensions_rounds.csv.gz` |
| Table 10, Appendix B (bootstrap contrasts; embedding rows after training) | `25_embedding_diagnostic.py`, `26_eval_extensions.py`, `34_review_analyses.py` | `outputs/review_analyses.json` (`bootstrap`), `outputs/embedding_diagnostic.json`, `outputs/evaluation/extensions.json` |
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
| `20`–`27` | Evaluation over 10 seeds, Flower verification, ablation, embedding diagnostic, row-wise aggregation and logistic regression, robustness analyses |
| `30`–`35` | Statistical analysis, ablation analysis, per-student and bootstrap analyses, and the plotting and table-formatting code of the manuscript |
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
