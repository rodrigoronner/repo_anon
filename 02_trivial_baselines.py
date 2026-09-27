"""Step 2 - Learning-free reference predictors on the test pool."""

import os

import numpy as np

from common import dump, load, metrics, select_threshold
from config import EVAL_DIR

fit, val, test, meta = load()
y = test["target"].values
rng = np.random.default_rng(42)
prior = float(fit["target"].mean())

out = {"test_positive_rate": float(y.mean()), "fit_positive_rate": prior, "n_test": int(len(y))}
refs = {
    "always_positive": (np.ones(len(y)), 0.5),
    "stratified_random": ((rng.random(len(y)) < prior).astype(float), 0.5),
    # Non-learned but informative: the student's own past success rate.
    "student_history_mean": (test["user_mean_correct"].values,
                             select_threshold(val["target"], val["user_mean_correct"])),
    "skill_history_mean": (test["skill_mean_correct"].values,
                           select_threshold(val["target"], val["skill_mean_correct"])),
}
for name, (p, thr) in refs.items():
    out[name] = metrics(y, p, thr)
    print(f"{name:22s} AUC {out[name]['roc_auc']:.4f}  BalAcc {out[name]['balanced_accuracy']:.4f}  "
          f"F1 {out[name]['f1_score']:.4f}")
dump(out, os.path.join(EVAL_DIR, "trivial_baselines.json"))
