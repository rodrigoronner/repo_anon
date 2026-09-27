"""Data loading, metrics, threshold selection and JSON helpers."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from config import DATA_DIR


def load():
    train_all = pd.read_csv(os.path.join(DATA_DIR, "processed_train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "processed_test.csv"))
    with open(os.path.join(DATA_DIR, "processed_meta.json")) as fh:
        meta = json.load(fh)
    fit = train_all[train_all["is_val"] == 0].reset_index(drop=True)
    val = train_all[train_all["is_val"] == 1].reset_index(drop=True)
    return fit, val, test, meta


def auc(y, p):
    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(p, dtype=float)))


def metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    return {
        "threshold": float(thr),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "positive_rate": float(pred.mean()),
    }


def select_threshold(y_val, p_val):
    """Threshold maximising balanced accuracy (Youden's J) on validation data."""
    y = np.asarray(y_val).astype(int)
    p = np.asarray(p_val, dtype=float)
    best_t, best_b = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197))):
        b = balanced_accuracy_score(y, (p >= t).astype(int))
        if b > best_b:
            best_t, best_b = float(t), b
    return best_t


def evaluate(test, p_test, val, p_val):
    """Test metrics at 0.5 and at the validation-selected threshold, overall and by pair type."""
    tau = select_threshold(val["target"], p_val)
    y, p = test["target"].values, np.asarray(p_test)
    seen = test["is_seen_pair"].values.astype(bool)
    out = {"at_0.5": metrics(y, p, 0.5), "calibrated": metrics(y, p, tau),
           "val_auc": auc(val["target"], p_val)}
    for name, mask in (("seen_pairs", seen), ("new_pairs", ~seen)):
        out[name] = {"at_0.5": metrics(y[mask], p[mask], 0.5),
                     "calibrated": metrics(y[mask], p[mask], tau)}
    return out


def dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def read(path):
    with open(path) as fh:
        return json.load(fh)
