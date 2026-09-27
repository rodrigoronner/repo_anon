#!/usr/bin/env bash
# Full pipeline from the raw ASSISTments file to the manuscript tables and figures.
# Environment: Python 3.11 with requirements.txt. Wall time on a 10-core laptop: about 5 h.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONWARNINGS=ignore

# Raw ASSISTments file: shipped compressed in data/raw/; downloaded from the
# original distribution only if the compressed copy is missing. Step 01 verifies
# the SHA-256 of the uncompressed file either way.
RAW=data/raw/skill_builder_data_corrected_collapsed.csv
if [ ! -f "$RAW" ]; then
  mkdir -p data/raw
  if [ -f "$RAW.gz" ]; then
    gzip -dc "$RAW.gz" > "$RAW"
  else
    curl -sL "$($PY -c 'import config; print(config.RAW_URL)')" -o "$RAW"
  fi
fi

$PY 01_prepare_data.py            # verifies SHA-256, builds leakage-free examples
$PY 02_trivial_baselines.py
$PY 03_loo_leakage_diagnostic.py     # shows why leave-one-out encoding is not used

$PY 10_tune_xgboost.py            # Optuna, validation AUC
$PY 11_tune_centralized_dnn.py
$PY 12_tune_federated.py          # one study per strategy; architecture frozen from step 11

$PY 20_eval_xgboost.py            # 10 evaluation seeds; checkpoints and thresholds chosen on validation
$PY 21_eval_centralized_dnn.py
$PY 22_eval_federated.py
$PY 23_flower_crosscheck.py       # Flower 1.7 with matched sampling, seed 42 x 100 rounds
$PY 24_ablation.py               # input blocks and weight decay on the embedding tables
$PY 25_embedding_diagnostic.py   # embedding norms at initialization and after training

$PY 30_analysis.py
$PY 31_figures.py
$PY 32_latex_tables.py
$PY 33_ablation_analysis.py
