"""Paths, seeds and budgets shared by every step of the final pipeline."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "data", "raw", "skill_builder_data_corrected_collapsed.csv")
RAW_SHA256 = "162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da"
RAW_URL = "https://drive.usercontent.google.com/download?id=1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE&export=download&confirm=t"
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
TUNE_DIR = os.path.join(OUT_DIR, "tuning")
EVAL_DIR = os.path.join(OUT_DIR, "evaluation")
FIG_DIR = os.path.join(OUT_DIR, "figures")
for d in (OUT_DIR, TUNE_DIR, EVAL_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

# Cohort and split
MIN_STUDENT_INTERACTIONS = 50
MIN_SKILL_ATTEMPTS = 100
SUCCESS_THRESHOLD = 0.70
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
SPLIT_SEED = 42

# Seeds: tuning and evaluation never share a seed.
TUNE_SEED = 0
EVAL_SEEDS = list(range(42, 52))

# Federated protocol
N_ROUNDS = 1000
FRACTION_FIT = 0.10
MIN_FIT_CLIENTS = 50
FL_PRUNE_EVERY = 50

# Tuning budgets (trials per study)
N_TRIALS_XGB = 100
N_TRIALS_CDNN = 100
N_TRIALS_FL = 30
N_WORKERS = 9

# Architecture and training of the preliminary study, kept as an ablation.
V1_ARCH = {"emb_dim": 10, "h1": 32, "h2": 16, "dropout": 0.0}
V1_TRAIN = {"optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "batch_size": 32, "local_epochs": 5}

FEAT_COLS = ["user_mean_correct", "user_interaction_count", "skill_mean_correct"]
FL_STRATEGIES = {"FedAvg": 0.0, "FedProx_mu0.1": 0.1, "FedProx_mu0.5": 0.5,
                 "FedProx_mu1.0": 1.0, "FedProx_muTuned": None}
