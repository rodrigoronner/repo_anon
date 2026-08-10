# Privacy-Preserving Personalization in Education
## A Federated Recommender System for Student Performance Prediction

Source code accompanying the manuscript submitted to **Expert Systems with Applications (ESA)**.

> **Anonymized repository.** Author names, institutional affiliations, and all externally hosted
> resources that would identify the authors (dataset host account, notebook host account, preprint
> record) have been suppressed for double-anonymous peer review. They will be restored in the
> camera-ready version upon acceptance.

---

## Repository Structure

```
.
├── data/
│   ├── interactions_real_rich_scaled_processed.csv  # pre-processed cohort (see Dataset below)
│   └── .gitkeep
├── outputs/                       # Generated metrics, models, and figures
│   └── .gitkeep
├── recommender_net.py             # RecommenderNet architecture (Table 4)
├── 01_data_preparation.py         # ASSISTments preprocessing pipeline (Section 4.2)
├── 02_centralized_baseline.py     # XGBoost centralized benchmark (Section 4.3.1)
├── 03_federated_training.py       # FL simulation: FedAvg + FedProx (Section 4.3.2)
├── 04_visualization.py            # Publication figures (Figures 4, 5, 6)
├── reports/
│   └── reproducibility_report.html
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **ASSISTments Skill Builder Dataset**, a publicly available benchmark for
knowledge tracing research.

> **Dataset DOI suppressed for anonymization.** The dataset is hosted under an account that would
> identify the authors. The full DOI is provided in the manuscript's Data Availability statement and
> will be restored in this README upon acceptance. Reviewers can obtain the equivalent raw data by
> searching for the public *ASSISTments Skill Builder* dataset (file
> `skill_builder_data_corrected_collapsed.csv`).

There are two ways to get data into `data/`, depending on whether you want to reproduce the
preprocessing itself:

1. **From raw ASSISTments logs (full pipeline).** Obtain `skill_builder_data_corrected_collapsed.csv`,
   place it in `data/`, and run `01_data_preparation.py` — this reproduces the filtering, feature
   engineering, and scaling described in Section 4.2, and writes `data/processed_assistments.csv`.
2. **From the already-processed cohort (shortcut).** `data/interactions_real_rich_scaled_processed.csv`,
   included in this repository, is the exact post-processing output (1,365 students × 107 skills after
   filtering, features scaled to [0, 1]). It has `target_correct_rate` but not yet the binarized
   `target` column that `02_centralized_baseline.py` / `03_federated_training.py` expect — derive it with:
   ```python
   df['target'] = (df['target_correct_rate'] >= 0.70).astype(int)
   ```

This shortcut lets reviewers reproduce every reported result without any external download.

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10/3.11. GPU (CUDA) support is optional for training.

> **Note:** `requirements.txt` pins `flwr==1.7.0` and `ray==2.6.3` for federated simulation.
> `ray==2.6.3` has no published wheel for Python 3.12 — if you are on 3.12, install unpinned
> `flwr`/`ray` instead (`pip install flwr ray`); `flwr.simulation.start_simulation` remains
> available in newer releases.

---

## Reproducing the Experiments

Run the scripts in order:

```bash
# 1. Preprocess the ASSISTments dataset
python 01_data_preparation.py

# 2. Train and evaluate the centralized XGBoost baseline (Table 5)
python 02_centralized_baseline.py

# 3. Run all federated experiments (Table 7)
#    -- This trains FedAvg + FedProx (mu=0.1, 0.5, 1.0) for 100 rounds each
python 03_federated_training.py --all

#    -- Or run a single configuration
python 03_federated_training.py --mu 0.5

# 4. Generate all publication figures
python 04_visualization.py
```

---

## Architecture Summary (Table 4)

| Layer | Type | Input Shape | Output Shape | Activation |
|---|---|---|---|---|
| 1 | Embedding (User ID) | (batch, 1) | (batch, 10) | — |
| 2 | Embedding (Skill ID) | (batch, 1) | (batch, 10) | — |
| 3 | Input (Engineered) | (batch, 3) | (batch, 3) | — |
| 4 | Concatenation | (batch, 10+10+3) | (batch, 23) | — |
| 5 | Dense (Hidden 1) | (batch, 23) | (batch, 32) | ReLU |
| 6 | Dense (Hidden 2) | (batch, 32) | (batch, 16) | ReLU |
| 7 | Dense (Output) | (batch, 16) | (batch, 1) | Sigmoid |

---

## Key Hyperparameters (Table 3)

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999, ε=1e-8) |
| Learning rate | 1×10⁻³ |
| Loss function | Binary Cross-Entropy |
| Local epochs | 5 |
| Local batch size | 32 |
| Communication rounds | 100 |
| Clients per round (train) | 10% (≈136) |
| Clients per round (eval) | 20% (≈273) |
| min_fit_clients | 50 |
| FedProx μ (grid search) | {0.1, 0.5, 1.0} |
| Random seed | 42 |

---

## Expected Results (Table 7)

| Strategy | Best F1 | Best Round | Mean F1 | Std Dev |
|---|---|---|---|---|
| FedAvg | 0.7584 | 70 | 0.7249 | 0.0249 |
| FedProx μ=0.1 | 0.7526 | 89 | 0.7226 | 0.0242 |
| **FedProx μ=0.5** | **0.7628** | **88** | 0.7238 | 0.0205 |
| FedProx μ=1.0 | 0.7555 | 80 | 0.7280 | **0.0152** |

Centralized XGBoost baseline: **F1 = 0.8285** (round 24).

---

## Independent Reproducibility Check

This pipeline was independently re-run end-to-end (unmodified code, `01`→`04`) against the
pre-processed cohort included in `data/` — 100 communication rounds × 4 strategies, ~10h15min on a
10-core machine. A visual summary is available at
[reports/reproducibility_report.html](reports/reproducibility_report.html).

> A hosted notebook version of this reproducibility run exists but its URL has been
> **suppressed for anonymization**, as the hosting account would identify the authors. The full
> report is included in this repository under `reports/` so that no external resource is needed.

**Centralized baseline** — reproduced within 0.1–0.3pp of Table 5 on every metric
(F1 0.8274 vs. 0.8285, round 20 vs. 24).

**Federated results** — Table 7 comparison:

| Strategy | Best F1 (paper) | Best F1 (repro) | Mean F1 (paper) | Mean F1 (repro) | Std Dev (paper) | Std Dev (repro) |
|---|---|---|---|---|---|---|
| FedAvg | 0.7584 | 0.7710 | 0.7249 | 0.7279 | 0.0249 | 0.0150 |
| FedProx μ=0.1 | 0.7526 | 0.7638 | 0.7226 | 0.7275 | 0.0242 | 0.0161 |
| **FedProx μ=0.5** | **0.7628** | **0.7737** | 0.7238 | 0.7291 | 0.0205 | 0.0162 |
| FedProx μ=1.0 | 0.7555 | 0.7581 | 0.7280 | 0.7280 | **0.0152** | **0.0129** |

The paper's central claims replicate — FedProx μ=0.5 gives the best peak F1, and μ=1.0 gives the
most stable training — while absolute F1 values ran ~1–1.5pp higher and the ranking of the
*least*-stable strategy did not reproduce (FedAvg was among the most stable here, not the least).
This is consistent with run-to-run variance inherent to Flower/Ray's client-sampling and
actor-scheduling order, which a fixed `torch.manual_seed` does not fully pin down across processes.

Two environment gaps surfaced during reproduction and are reflected in this repository: `ray` was
missing from `requirements.txt` (added), and the pinned `flwr`/`ray` versions do not install on
Python 3.12 (documented above).

---

## License

This code is released under the **Apache 2.0 License**.

---

## Citation

> **Citation suppressed for anonymization.** A preprint of this work exists but citing it would
> identify the authors. The full citation will be restored upon acceptance.
