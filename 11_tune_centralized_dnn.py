"""Step 11 - Optuna search for the centralized RecommenderNet (architecture + training).

The selected architecture is then frozen for every federated configuration, so the
architecture is tuned in the setting most favorable to it (centralized training).
"""

import numpy as np
import optuna
import torch

from common import auc, load
from config import N_TRIALS_CDNN, TUNE_SEED
from models import build, make_optimizer, predict, to_tensors, train_epochs
from tuning_utils import run_study

MAX_EPOCHS = 60
PATIENCE = 8


def suggest(trial):
    opt = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    lr = (trial.suggest_float("lr_adam", 1e-4, 3e-2, log=True) if opt == "adam"
          else trial.suggest_float("lr_sgd", 1e-2, 2.0, log=True))
    arch = {"emb_dim": trial.suggest_categorical("emb_dim", [2, 4, 8, 16, 32]),
            "h1": trial.suggest_categorical("h1", [16, 32, 64, 128, 256]),
            "h2": trial.suggest_categorical("h2", [8, 16, 32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.7, step=0.1)}
    hp = {"optimizer": opt, "lr": lr,
          "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True),
          "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256])}
    return arch, hp


def train_centralized(fit, val, meta, arch, hp, seed, test=None, report=None):
    """Early stopping on validation AUC. Returns best checkpoint predictions and the curve."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = build(meta, arch)
    opt = make_optimizer(model, hp)
    t_fit, t_val = to_tensors(fit), to_tensors(val)
    t_test = to_tensors(test) if test is not None else None
    best, curve, since = None, [], 0
    for ep in range(1, MAX_EPOCHS + 1):
        train_epochs(model, t_fit, 1, hp, opt=opt)
        p_val = predict(model, t_val)
        if not np.isfinite(p_val).all():
            break
        v = auc(val["target"], p_val)
        p_test = predict(model, t_test) if t_test is not None else None
        curve.append({"epoch": ep, "val_auc": v,
                      "test_auc": auc(test["target"], p_test) if p_test is not None else None})
        if best is None or v > best["val_auc"]:
            best, since = {"val_auc": v, "epoch": ep, "p_val": p_val, "p_test": p_test}, 0
        else:
            since += 1
        if report is not None:
            report(ep, best["val_auc"])
        if since >= PATIENCE:
            break
    return best, curve


def objective(trial):
    fit, val, _, meta = load()
    arch, hp = suggest(trial)

    def report(step, value):
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    best, _ = train_centralized(fit, val, meta, arch, hp, TUNE_SEED, report=report)
    if best is None:
        return 0.5
    trial.set_user_attr("best_epoch", best["epoch"])
    return best["val_auc"]


def params_to_config(p):
    arch = {k: p[k] for k in ("emb_dim", "h1", "h2", "dropout")}
    hp = {"optimizer": p["optimizer"], "lr": p["lr_adam"] if p["optimizer"] == "adam" else p["lr_sgd"],
          "weight_decay": p["weight_decay"], "batch_size": p["batch_size"]}
    return arch, hp


if __name__ == "__main__":
    run_study("centralized_dnn", objective, N_TRIALS_CDNN,
              pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5))
