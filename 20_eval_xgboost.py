"""Step 20 - Tuned XGBoost, refitted with each evaluation seed; test pool used once per seed."""

import importlib
import os

import numpy as np

from common import dump, evaluate, load, read
from config import EVAL_DIR, EVAL_SEEDS, TUNE_DIR

tx = importlib.import_module("10_tune_xgboost")

fit, val, test, meta = load()
params = read(os.path.join(TUNE_DIR, "xgboost_best.json"))["best_params"]
runs, preds = [], []
for seed in EVAL_SEEDS:
    model, enc = tx.fit_xgb(params, fit, val, meta, seed)
    p_val = model.predict_proba(tx.design(val, enc, meta))[:, 1]
    p_test = model.predict_proba(tx.design(test, enc, meta))[:, 1]
    res = {"seed": seed, "best_iteration": int(model.best_iteration),
           "selected": evaluate(test, p_test, val, p_val)}
    runs.append(res)
    preds.append(p_test)
    m = res["selected"]["at_0.5"]
    print(f"seed {seed} it {model.best_iteration:4d} | AUC {m['roc_auc']:.4f} "
          f"BalAcc {m['balanced_accuracy']:.4f}", flush=True)
dump({"model": "xgboost", "params": params, "runs": runs}, os.path.join(EVAL_DIR, "xgboost.json"))
np.save(os.path.join(EVAL_DIR, "xgboost_test_preds.npy"), np.array(preds))
