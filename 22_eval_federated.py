"""Step 22 - Federated evaluation: every tuned strategy plus the v1-protocol ablation, 10 seeds each.

For a given seed all conditions share the client sample of every round and the
local randomness of every (round, client), so comparisons are paired.
"""

import importlib
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd

from common import dump, evaluate, load, read
from config import EVAL_DIR, EVAL_SEEDS, FL_STRATEGIES, N_ROUNDS, N_WORKERS, TUNE_DIR, V1_ARCH, V1_TRAIN
from fl_sim import run_federated

tf = importlib.import_module("12_tune_federated")


def conditions():
    arch = tf.centralized_arch()
    out = {}
    for s in FL_STRATEGIES:
        hp, mu = tf.params_to_config(read(os.path.join(TUNE_DIR, f"federated_{s}_best.json"))["best_params"], s)
        out[s] = (arch, hp, mu)
    out["FedAvg_v1protocol"] = (V1_ARCH, dict(V1_TRAIN), 0.0)
    return out


def job(args):
    name, seed = args
    arch, hp, mu = conditions()[name]
    fit, val, test, meta = load()
    rows, best, final = run_federated(fit, val, test, meta, arch, hp, mu, seed, N_ROUNDS)
    res = {"condition": name, "seed": seed, "mu": mu, "selected_round": best["round"],
           "diverged": bool(rows[-1].get("diverged", False)),
           "selected": evaluate(test, best["p_test"], val, best["p_val"])}
    if final["p_test"] is not None and np.isfinite(final["p_val"]).all():
        res["final"] = evaluate(test, final["p_test"], val, final["p_val"])
    m = res["selected"]["at_0.5"]
    print(f"{name:18s} seed {seed} r{best['round']:4d} | AUC {m['roc_auc']:.4f} "
          f"BalAcc {m['balanced_accuracy']:.4f} pos {m['positive_rate']:.2f}", flush=True)
    return res, [dict(r, condition=name, seed=seed) for r in rows], best["p_test"]


if __name__ == "__main__":
    conds = conditions()
    jobs = [(n, s) for s in EVAL_SEEDS for n in conds]
    with get_context("spawn").Pool(N_WORKERS) as pool:
        out = pool.map(job, jobs, chunksize=1)
    for name, (arch, hp, mu) in conds.items():
        sel = [o for o, j in zip(out, jobs) if j[0] == name]
        dump({"model": name, "arch": arch, "hp": hp, "mu": mu, "n_rounds": N_ROUNDS,
              "runs": [r for r, _, _ in sel]}, os.path.join(EVAL_DIR, f"fl_{name}.json"))
        np.save(os.path.join(EVAL_DIR, f"fl_{name}_test_preds.npy"), np.array([p for _, _, p in sel]))
    pd.DataFrame([r for _, rows, _ in out for r in rows]).to_csv(
        os.path.join(EVAL_DIR, "fl_rounds.csv.gz"), index=False)
