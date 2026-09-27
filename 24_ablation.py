"""Step 24 - Ablation study, 10 seeds per condition, configurations fixed from tuning (no re-tuning).

A. Input blocks. The tuned centralized DNN and tuned FedAvg are retrained without the
   student embedding, without the skill embedding, with the features only and with the
   embeddings only. This locates the cost of federation: in FedAvg a student's embedding
   row is changed only when that student is sampled, and the change is averaged with the
   unchanged copies returned by the other sampled clients.
B. Weight decay on embeddings. Local weight decay acts on every embedding row, including
   the rows of students who are not in the client. The tuned FedAvg, FedProx mu=1.0 and
   FedProx tuned-mu configurations (and the centralized DNN) are retrained with weight
   decay restricted to the MLP.

Paired seeds: federated conditions share the client sample of every round and the local
random state with the main runs of the same seed.
"""

import importlib
import os
import sys
from multiprocessing import get_context

import pandas as pd

from common import dump, evaluate, load, read
from config import EVAL_DIR, EVAL_SEEDS, N_ROUNDS, N_WORKERS, TUNE_DIR, V1_ARCH, V1_TRAIN
from fl_sim import run_federated

tc = importlib.import_module("11_tune_centralized_dnn")
tf = importlib.import_module("12_tune_federated")

INPUT_VARIANTS = ["no_user", "no_skill", "features_only", "embeddings_only"]


def conditions():
    arch_c, hp_c = tc.params_to_config(read(os.path.join(TUNE_DIR, "centralized_dnn_best.json"))["best_params"])
    fl = {}
    for s in ["FedAvg", "FedProx_mu1.0", "FedProx_muTuned"]:
        fl[s] = tf.params_to_config(read(os.path.join(TUNE_DIR, f"federated_{s}_best.json"))["best_params"], s)
    arch_f = tf.centralized_arch()
    out = {}
    for v in INPUT_VARIANTS:
        out[f"cdnn_{v}"] = ("central", dict(arch_c, inputs=v), hp_c, 0.0)
        out[f"FedAvg_{v}"] = ("fl", dict(arch_f, inputs=v), fl["FedAvg"][0], 0.0)
    out["cdnn_noEmbDecay"] = ("central", arch_c, dict(hp_c, decay_embeddings=False), 0.0)
    for s, (hp, mu) in fl.items():
        out[f"{s}_noEmbDecay"] = ("fl", arch_f, dict(hp, decay_embeddings=False), mu)
    # preliminary configuration (no weight decay) with the features only
    out["FedAvg_v1protocol_features_only"] = ("fl", dict(V1_ARCH, inputs="features_only"), dict(V1_TRAIN), 0.0)
    return out


def job(args):
    name, seed = args
    kind, arch, hp, mu = conditions()[name]
    fit, val, test, meta = load()
    rows = None
    if kind == "central":
        best, _ = tc.train_centralized(fit, val, meta, arch, hp, seed, test=test)
        sel = best["epoch"]
    else:
        rows, best, _ = run_federated(fit, val, test, meta, arch, hp, mu, seed, N_ROUNDS)
        sel = best["round"]
    res = {"condition": name, "kind": kind, "seed": seed, "selected": sel,
           "result": evaluate(test, best["p_test"], val, best["p_val"])}
    m = res["result"]["at_0.5"]
    print(f"{name:28s} seed {seed} sel {sel:4d} | AUC {m['roc_auc']:.4f} "
          f"BalAcc {m['balanced_accuracy']:.4f} pos {m['positive_rate']:.2f}", flush=True)
    rows = [dict(r, condition=name, seed=seed) for r in rows] if rows else []
    return res, rows


if __name__ == "__main__":
    # optional condition names: run only those and merge them into an existing ablation.json
    conds = {n: c for n, c in conditions().items() if not sys.argv[1:] or n in sys.argv[1:]}
    # slowest (FedProx mu=1.0, ten local epochs) first so the pool drains evenly
    order = sorted(conds, key=lambda n: (not n.startswith("FedProx_mu1.0"), conds[n][0] == "central"))
    jobs = [(n, s) for n in order for s in EVAL_SEEDS]
    with get_context("spawn").Pool(N_WORKERS) as pool:
        out = pool.map(job, jobs, chunksize=1)
    runs, rounds = [r for r, _ in out], pd.DataFrame([r for _, rows in out for r in rows])
    if sys.argv[1:]:
        old = read(os.path.join(EVAL_DIR, "ablation.json"))
        runs = [r for r in old["runs"] if r["condition"] not in conds] + runs
        old_rounds = pd.read_csv(os.path.join(EVAL_DIR, "ablation_rounds.csv.gz"))
        rounds = pd.concat([old_rounds[~old_rounds["condition"].isin(conds)], rounds], ignore_index=True)
    dump({"conditions": {n: {"kind": k, "arch": a, "hp": h, "mu": m} for n, (k, a, h, m) in conditions().items()},
          "runs": runs}, os.path.join(EVAL_DIR, "ablation.json"))
    rounds.to_csv(os.path.join(EVAL_DIR, "ablation_rounds.csv.gz"), index=False)
