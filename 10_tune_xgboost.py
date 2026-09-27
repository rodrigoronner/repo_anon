"""Step 10 - Optuna search for the centralized XGBoost baseline (objective: validation AUC)."""

import pandas as pd
import xgboost as xgb

from common import auc, load
from config import FEAT_COLS, N_TRIALS_XGB, TUNE_SEED
from tuning_utils import run_study

ID_COLS = ["user_id_new", "skill_id_new"]


def design(df, encoding, meta):
    if encoding == "none":
        return df[FEAT_COLS]
    X = df[ID_COLS + FEAT_COLS].copy()
    if encoding == "categorical":
        X["user_id_new"] = pd.Categorical(X["user_id_new"], categories=range(meta["num_users"]))
        X["skill_id_new"] = pd.Categorical(X["skill_id_new"], categories=range(meta["num_skills"]))
    return X


def suggest(trial):
    return {
        "id_encoding": trial.suggest_categorical("id_encoding", ["numeric", "categorical", "none"]),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def fit_xgb(params, fit, val, meta, seed):
    p = dict(params)
    enc = p.pop("id_encoding")
    model = xgb.XGBClassifier(n_estimators=5000, early_stopping_rounds=100, eval_metric="auc",
                              tree_method="hist", enable_categorical=(enc == "categorical"),
                              random_state=seed, n_jobs=1, verbosity=0, **p)
    model.fit(design(fit, enc, meta), fit["target"],
              eval_set=[(design(val, enc, meta), val["target"])], verbose=False)
    return model, enc


def objective(trial):
    fit, val, _, meta = load()
    model, enc = fit_xgb(suggest(trial), fit, val, meta, TUNE_SEED)
    trial.set_user_attr("best_iteration", int(model.best_iteration))
    return auc(val["target"], model.predict_proba(design(val, enc, meta))[:, 1])


if __name__ == "__main__":
    run_study("xgboost", objective, N_TRIALS_XGB)
