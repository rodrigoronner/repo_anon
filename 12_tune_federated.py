"""Step 12 - Optuna search for the federated training hyperparameters, one study per strategy.

The architecture is frozen to the centralized optimum (step 11). Each strategy
gets the same search space and trial budget; FedProx_muTuned additionally
searches mu. Objective: best validation AUC within the N_ROUNDS budget, which
is also the checkpoint rule used at evaluation. Test data is not loaded.

Usage: python 12_tune_federated.py [strategy ...]   (default: all strategies)
"""

import os
import sys

import optuna

from common import load, read
from config import FL_PRUNE_EVERY, FL_STRATEGIES, N_ROUNDS, N_TRIALS_FL, TUNE_DIR, TUNE_SEED
from fl_sim import run_federated
from tuning_utils import run_study


def centralized_arch():
    p = read(os.path.join(TUNE_DIR, "centralized_dnn_best.json"))["best_params"]
    return {k: p[k] for k in ("emb_dim", "h1", "h2", "dropout")}


def suggest(trial, strategy):
    opt = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    lr = (trial.suggest_float("lr_adam", 1e-4, 1e-1, log=True) if opt == "adam"
          else trial.suggest_float("lr_sgd", 1e-2, 2.0, log=True))
    hp = {"optimizer": opt, "lr": lr,
          "local_epochs": trial.suggest_categorical("local_epochs", [1, 2, 5, 10]),
          "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
          "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True)}
    mu = FL_STRATEGIES[strategy]
    if mu is None:
        mu = trial.suggest_float("mu", 1e-3, 1.0, log=True)
    return hp, mu


def params_to_config(p, strategy):
    hp = {"optimizer": p["optimizer"], "lr": p["lr_adam"] if p["optimizer"] == "adam" else p["lr_sgd"],
          "local_epochs": p["local_epochs"], "batch_size": p["batch_size"],
          "weight_decay": p["weight_decay"]}
    mu = FL_STRATEGIES[strategy]
    return hp, (p["mu"] if mu is None else mu)


def make_objective(strategy):
    def objective(trial):
        fit, val, _, meta = load()
        hp, mu = suggest(trial, strategy)

        def report(rnd, value):
            if rnd % FL_PRUNE_EVERY == 0:
                trial.report(value, rnd)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        _, best, _ = run_federated(fit, val, None, meta, centralized_arch(), hp, mu, TUNE_SEED,
                                   N_ROUNDS, report=report, eval_test=False)
        if best is None:
            return 0.5
        trial.set_user_attr("best_round", best["round"])
        return best["val_auc"]
    return objective


class Objective:
    """Picklable wrapper so spawned workers can rebuild the closure."""

    def __init__(self, strategy):
        self.strategy = strategy

    def __call__(self, trial):
        return make_objective(self.strategy)(trial)


if __name__ == "__main__":
    strategies = sys.argv[1:] or list(FL_STRATEGIES)
    for s in strategies:
        run_study(f"federated_{s}", Objective(s), N_TRIALS_FL,
                  pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=200))
