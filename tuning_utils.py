"""Parallel Optuna studies on a shared journal file, plus export of trials and best parameters."""

import os
import warnings
from multiprocessing import get_context

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from common import dump
from config import N_WORKERS, TUNE_DIR, TUNE_SEED

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def storage(name):
    return JournalStorage(JournalFileBackend(os.path.join(TUNE_DIR, f"{name}.journal")))


def _worker(args):
    name, objective, n_trials, pruner, wid = args
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=TUNE_SEED + wid, multivariate=True, group=True,
                                         constant_liar=True, n_startup_trials=10,
                                         warn_independent_sampling=False)
    study = optuna.load_study(study_name=name, storage=storage(name), sampler=sampler, pruner=pruner)
    study.optimize(objective, callbacks=[MaxTrialsCallback(
        n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL))])


def run_study(name, objective, n_trials, pruner=None, n_workers=N_WORKERS):
    pruner = pruner or optuna.pruners.NopPruner()
    optuna.create_study(study_name=name, storage=storage(name), direction="maximize",
                        load_if_exists=True)
    with get_context("spawn").Pool(n_workers) as pool:
        pool.map(_worker, [(name, objective, n_trials, pruner, w) for w in range(n_workers)])
    study = optuna.load_study(study_name=name, storage=storage(name))
    export(study, name)
    return study


def export(study, name):
    df = study.trials_dataframe()
    df.to_csv(os.path.join(TUNE_DIR, f"{name}_trials.csv"), index=False)
    try:
        imp = {k: float(v) for k, v in optuna.importance.get_param_importances(study).items()}
    except Exception as exc:  # importance needs >1 complete trial with varying params
        imp = {"error": str(exc)}
    states = [t.state.name for t in study.trials]
    dump({"study": name, "best_value": study.best_value, "best_params": study.best_params,
          "best_user_attrs": study.best_trial.user_attrs, "best_trial": study.best_trial.number,
          "n_trials": len(study.trials),
          "n_complete": states.count("COMPLETE"), "n_pruned": states.count("PRUNED"),
          "n_failed": states.count("FAIL"), "param_importances": imp},
         os.path.join(TUNE_DIR, f"{name}_best.json"))
    print(f"[{name}] best val AUC {study.best_value:.4f} | {study.best_params}", flush=True)
