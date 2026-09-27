"""Step 13 - Optuna search for centralized logistic regression on the three aggregate features.

A learnable, federable baseline without embeddings: the same model is trained
federatedly in step 12 (study "federated_LogReg"). Training uses the loop and
early stopping of the centralized DNN (step 11) with its optimizer, learning-rate,
weight-decay and batch-size ranges.
"""

import importlib

import optuna

from common import load
from config import N_TRIALS_CDNN, TUNE_SEED
from tuning_utils import run_study

tc = importlib.import_module("11_tune_centralized_dnn")
ARCH = {"model": "logreg"}


def suggest(trial):
    opt = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    lr = (trial.suggest_float("lr_adam", 1e-4, 3e-2, log=True) if opt == "adam"
          else trial.suggest_float("lr_sgd", 1e-2, 2.0, log=True))
    return {"optimizer": opt, "lr": lr,
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256])}


def params_to_config(p):
    return {"optimizer": p["optimizer"], "lr": p["lr_adam"] if p["optimizer"] == "adam" else p["lr_sgd"],
            "weight_decay": p["weight_decay"], "batch_size": p["batch_size"]}


def objective(trial):
    fit, val, _, meta = load()
    hp = suggest(trial)

    def report(step, value):
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    best, _ = tc.train_centralized(fit, val, meta, ARCH, hp, TUNE_SEED, report=report)
    if best is None:
        return 0.5
    trial.set_user_attr("best_epoch", best["epoch"])
    return best["val_auc"]


if __name__ == "__main__":
    run_study("centralized_logreg", objective, N_TRIALS_CDNN,
              pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5))
