"""Step 26 - Evaluation of the conditions added after review, 10 seeds each.

  fl_FedAvg_rowwise          FedAvg with row-wise aggregation of the embedding tables, tuned (step 12)
  fl_FedAvg_rowwise_samecfg  row-wise aggregation with the hyperparameters of tuned (dense) FedAvg,
                             isolating the effect of the aggregation rule
  fl_LogReg                  federated logistic regression on the three features, tuned (step 12)
  clr_tuned                  centralized logistic regression, tuned (step 13)

Federated conditions share the client sample of every round with the main runs of the same
seed. For the recommender conditions, the embedding rows' mean norm and mean distance from
their initial value are recorded on the final model (as in step 25).
"""

import importlib
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch

from common import dump, evaluate, load, read
from config import EVAL_DIR, EVAL_SEEDS, N_ROUNDS, N_WORKERS, TUNE_DIR
from fl_sim import run_federated
from models import build

tc = importlib.import_module("11_tune_centralized_dnn")
tf = importlib.import_module("12_tune_federated")
tl = importlib.import_module("13_tune_logreg_centralized")


def conditions():
    rw, _ = tf.params_to_config(read(os.path.join(TUNE_DIR, "federated_FedAvg_rowwise_best.json"))["best_params"],
                                "FedAvg_rowwise")
    dense, _ = tf.params_to_config(read(os.path.join(TUNE_DIR, "federated_FedAvg_best.json"))["best_params"], "FedAvg")
    lr_f, _ = tf.params_to_config(read(os.path.join(TUNE_DIR, "federated_LogReg_best.json"))["best_params"], "LogReg")
    lr_c = tl.params_to_config(read(os.path.join(TUNE_DIR, "centralized_logreg_best.json"))["best_params"])
    arch = tf.centralized_arch()
    return {"fl_FedAvg_rowwise": ("fl", arch, rw),
            "fl_FedAvg_rowwise_samecfg": ("fl", arch, dict(dense, aggregation="rowwise")),
            "fl_LogReg": ("fl", {"model": "logreg"}, lr_f),
            "clr_tuned": ("central", {"model": "logreg"}, lr_c)}


def embedding_stats(model, init):
    out = {}
    for name in ("user_embedding", "skill_embedding"):
        w, w0 = getattr(model, name).weight.detach(), init[f"{name}.weight"]
        out[name] = {"norm_init": float(w0.norm(dim=1).mean()), "norm_final": float(w.norm(dim=1).mean()),
                     "moved": float((w - w0).norm(dim=1).mean())}
    return out


def job(args):
    name, seed = args
    kind, arch, hp = conditions()[name]
    fit, val, test, meta = load()
    torch.manual_seed(seed)
    init = {k: v.clone() for k, v in build(meta, arch).state_dict().items()}
    rows = []
    if kind == "central":
        best, _ = tc.train_centralized(fit, val, meta, arch, hp, seed, test=test)
        sel, emb = best["epoch"], None
    else:
        rows, best, final = run_federated(fit, val, test, meta, arch, hp, 0.0, seed, N_ROUNDS)
        sel = best["round"]
        emb = embedding_stats(final["model"], init) if "model" not in arch else None
    res = {"condition": name, "seed": seed, "selected": sel,
           "result": evaluate(test, best["p_test"], val, best["p_val"]), "embeddings_final": emb}
    m = res["result"]["at_0.5"]
    print(f"{name:26s} seed {seed} sel {sel:4d} | AUC {m['roc_auc']:.4f} BalAcc {m['balanced_accuracy']:.4f}"
          + (f" | user rows moved {emb['user_embedding']['moved']:.3f}" if emb else ""), flush=True)
    return res, [dict(r, condition=name, seed=seed) for r in rows], best["p_test"]


if __name__ == "__main__":
    conds = conditions()
    jobs = [(n, s) for n in conds for s in EVAL_SEEDS]
    with get_context("spawn").Pool(N_WORKERS) as pool:
        out = pool.map(job, jobs, chunksize=1)
    dump({"conditions": {n: {"kind": k, "arch": a, "hp": h} for n, (k, a, h) in conds.items()},
          "runs": [r for r, _, _ in out]}, os.path.join(EVAL_DIR, "extensions.json"))
    for n in conds:
        np.save(os.path.join(EVAL_DIR, f"{n}_test_preds.npy"),
                np.array([p for (r, _, p), j in zip(out, jobs) if j[0] == n]))
    pd.DataFrame([row for _, rows, _ in out for row in rows]).to_csv(
        os.path.join(EVAL_DIR, "extensions_rounds.csv.gz"), index=False)
