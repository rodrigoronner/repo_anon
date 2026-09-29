"""Step 27 - Robustness analyses requested in review, 10 seeds each, configurations fixed (no re-tuning).

A. Temporal validation. The main protocol draws the validation split at random from the two
   training windows, so validation shares students and windows with the fitting data and can
   reward models that memorize students. Here the model is fitted on the first training window
   (history 0-40% -> labels 40-60%) and validated on the second (labels 60-80%), which lies later
   in every student's history; checkpoints and thresholds are chosen on that later window, and the
   test set is unchanged. The central contrasts are re-estimated: the cost of federation, the value
   of the student embedding, and row-wise aggregation.
B. Proximal term with the same local optimizer. FedProx with mu = 1.0 and with the tuned mu is run
   with the local configuration selected for FedAvg (SGD), so that FedAvg and FedProx differ only
   in the proximal term.
"""

import importlib
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd

from common import dump, evaluate, load, read
from config import EVAL_DIR, EVAL_SEEDS, N_ROUNDS, N_WORKERS, TUNE_DIR
from fl_sim import run_federated

tc = importlib.import_module("11_tune_centralized_dnn")
tf = importlib.import_module("12_tune_federated")
OUT = os.path.join(EVAL_DIR, "robustness")


def temporal_split():
    fit, val, test, meta = load()
    train = pd.concat([fit, val], ignore_index=True)
    return (train[train["window"] == 0].reset_index(drop=True),
            train[train["window"] == 1].reset_index(drop=True), test, meta)


def conditions():
    arch_c, hp_c = tc.params_to_config(read(os.path.join(TUNE_DIR, "centralized_dnn_best.json"))["best_params"])
    hp_f, _ = tf.params_to_config(read(os.path.join(TUNE_DIR, "federated_FedAvg_best.json"))["best_params"], "FedAvg")
    hp_rw, _ = tf.params_to_config(read(os.path.join(TUNE_DIR, "federated_FedAvg_rowwise_best.json"))["best_params"],
                                   "FedAvg_rowwise")
    mu_tuned = read(os.path.join(TUNE_DIR, "federated_FedProx_muTuned_best.json"))["best_params"]["mu"]
    arch_f = tf.centralized_arch()
    c = {}
    for v in ("all", "no_user", "features_only"):
        c[f"T_cdnn_{v}"] = ("temporal", "central", dict(arch_c, inputs=v), hp_c, 0.0)
        c[f"T_FedAvg_{v}"] = ("temporal", "fl", dict(arch_f, inputs=v), hp_f, 0.0)
    c["T_FedAvg_rowwise_samecfg"] = ("temporal", "fl", arch_f, dict(hp_f, aggregation="rowwise"), 0.0)
    c["T_FedAvg_rowwise_tuned"] = ("temporal", "fl", arch_f, hp_rw, 0.0)
    c["S_FedProx_mu1.0_sgd"] = ("standard", "fl", arch_f, hp_f, 1.0)
    c["S_FedProx_muTuned_sgd"] = ("standard", "fl", arch_f, hp_f, float(mu_tuned))
    return c


def job(args):
    name, seed = args
    split, kind, arch, hp, mu = conditions()[name]
    fit, val, test, meta = temporal_split() if split == "temporal" else load()
    rows = []
    if kind == "central":
        best, _ = tc.train_centralized(fit, val, meta, arch, hp, seed, test=test)
        sel = best["epoch"]
    else:
        rows, best, _ = run_federated(fit, val, test, meta, arch, hp, mu, seed, N_ROUNDS)
        sel = best["round"]
    res = {"condition": name, "seed": seed, "selected": sel,
           "result": evaluate(test, best["p_test"], val, best["p_val"])}
    m = res["result"]["at_0.5"]
    print(f"{name:26s} seed {seed} sel {sel:4d} | AUC {m['roc_auc']:.4f} val {res['result']['val_auc']:.4f} "
          f"pos {m['positive_rate']:.2f}", flush=True)
    return res, [dict(r, condition=name, seed=seed) for r in rows], best["p_test"]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    conds = conditions()
    order = sorted(conds, key=lambda n: conds[n][1] == "central")
    jobs = [(n, s) for n in order for s in EVAL_SEEDS]
    with get_context("spawn").Pool(N_WORKERS) as pool:
        out = pool.map(job, jobs, chunksize=1)
    for n in conds:
        np.save(os.path.join(OUT, f"{n}.npy"), np.array([p for (_, _, p), j in zip(out, jobs) if j[0] == n]))
    dump({"conditions": {n: {"split": a, "kind": k, "arch": ar, "hp": h, "mu": m}
                         for n, (a, k, ar, h, m) in conds.items()},
          "runs": [r for r, _, _ in out]}, os.path.join(OUT, "robustness.json"))
    pd.DataFrame([r for _, rows, _ in out for r in rows]).to_csv(os.path.join(OUT, "robustness_rounds.csv.gz"),
                                                                 index=False)
