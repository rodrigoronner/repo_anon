"""Step 21 - Centralized RecommenderNet with the tuned configuration and with the v1 configuration."""

import importlib
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd

from common import dump, evaluate, load, read
from config import EVAL_DIR, EVAL_SEEDS, N_WORKERS, TUNE_DIR, V1_ARCH, V1_TRAIN

tc = importlib.import_module("11_tune_centralized_dnn")


def configs():
    arch, hp = tc.params_to_config(read(os.path.join(TUNE_DIR, "centralized_dnn_best.json"))["best_params"])
    v1_hp = {k: V1_TRAIN[k] for k in ("optimizer", "lr", "weight_decay", "batch_size")}
    return {"cdnn_tuned": (arch, hp), "cdnn_v1": (V1_ARCH, v1_hp)}


def job(args):
    name, seed = args
    arch, hp = configs()[name]
    fit, val, test, meta = load()
    best, curve = tc.train_centralized(fit, val, meta, arch, hp, seed, test=test)
    res = {"condition": name, "seed": seed, "selected_epoch": best["epoch"],
           "selected": evaluate(test, best["p_test"], val, best["p_val"])}
    m = res["selected"]["at_0.5"]
    print(f"{name:11s} seed {seed} ep {best['epoch']:2d} | AUC {m['roc_auc']:.4f} "
          f"BalAcc {m['balanced_accuracy']:.4f}", flush=True)
    return res, [dict(c, condition=name, seed=seed) for c in curve], best["p_test"]


if __name__ == "__main__":
    cfg = configs()
    jobs = [(n, s) for n in cfg for s in EVAL_SEEDS]
    with get_context("spawn").Pool(N_WORKERS) as pool:
        out = pool.map(job, jobs, chunksize=1)
    for name in cfg:
        sel = [o for o, j in zip(out, jobs) if j[0] == name]
        arch, hp = cfg[name]
        dump({"model": name, "arch": arch, "hp": hp, "max_epochs": tc.MAX_EPOCHS, "patience": tc.PATIENCE,
              "runs": [r for r, _, _ in sel]}, os.path.join(EVAL_DIR, f"{name}.json"))
        np.save(os.path.join(EVAL_DIR, f"{name}_test_preds.npy"), np.array([p for _, _, p in sel]))
    pd.DataFrame([c for _, curve, _ in out for c in curve]).to_csv(
        os.path.join(EVAL_DIR, "cdnn_curves.csv"), index=False)
