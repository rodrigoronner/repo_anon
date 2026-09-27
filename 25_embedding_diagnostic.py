"""Step 25 - What happens to the embedding tables under centralized and federated training.

For the tuned centralized DNN and tuned FedAvg, with and without weight decay on the
embeddings, reports per seed the mean L2 norm of the student and skill embedding rows at
initialization and after training, and the mean distance each row moved from its initial
value. Federated models are taken at round N_ROUNDS; centralized models are retrained for
the epoch selected on validation in steps 21 and 24.
"""

import importlib
import os
from multiprocessing import get_context

import numpy as np
import torch

from common import dump, load, read
from config import EVAL_DIR, EVAL_SEEDS, N_ROUNDS, N_WORKERS, OUT_DIR, TUNE_DIR
from fl_sim import run_federated
from models import build, make_optimizer, to_tensors, train_epochs

tc = importlib.import_module("11_tune_centralized_dnn")
tf = importlib.import_module("12_tune_federated")


def selected_epoch(condition, seed):
    if condition == "cdnn_tuned":
        runs = read(os.path.join(EVAL_DIR, "cdnn_tuned.json"))["runs"]
        return next(r["selected_epoch"] for r in runs if r["seed"] == seed)
    runs = read(os.path.join(EVAL_DIR, "ablation.json"))["runs"]
    return next(r["selected"] for r in runs if r["condition"] == condition and r["seed"] == seed)


def stats_of(model, init):
    out = {}
    for name in ("user_embedding", "skill_embedding"):
        w, w0 = getattr(model, name).weight.detach(), init[f"{name}.weight"]
        out[name] = {"norm_init": float(w0.norm(dim=1).mean()), "norm_final": float(w.norm(dim=1).mean()),
                     "moved": float((w - w0).norm(dim=1).mean())}
    return out


def job(args):
    cond, seed = args
    fit, val, test, meta = load()
    decay = not cond.endswith("noEmbDecay")
    torch.manual_seed(seed)
    init = {k: v.clone() for k, v in build(meta, tf.centralized_arch()).state_dict().items()}
    if cond.startswith("cdnn"):
        arch, hp = tc.params_to_config(read(os.path.join(TUNE_DIR, "centralized_dnn_best.json"))["best_params"])
        hp = dict(hp, decay_embeddings=decay)
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        model = build(meta, arch)
        opt, t_fit = make_optimizer(model, hp), to_tensors(fit)
        for _ in range(selected_epoch(cond, seed)):
            train_epochs(model, t_fit, 1, hp, opt=opt)
    else:
        hp, mu = tf.params_to_config(read(os.path.join(TUNE_DIR, "federated_FedAvg_best.json"))["best_params"], "FedAvg")
        hp = dict(hp, decay_embeddings=decay)
        _, _, final = run_federated(fit, val, test, meta, tf.centralized_arch(), hp, mu, seed, N_ROUNDS, eval_test=False)
        model = final["model"]
    return {"condition": cond, "seed": seed, **stats_of(model, init)}


if __name__ == "__main__":
    conds = ["cdnn_tuned", "cdnn_noEmbDecay", "FedAvg", "FedAvg_noEmbDecay"]
    jobs = [(c, s) for c in conds for s in EVAL_SEEDS]
    with get_context("spawn").Pool(N_WORKERS) as pool:
        out = pool.map(job, jobs, chunksize=1)
    summary = {}
    for c in conds:
        rs = [r for r in out if r["condition"] == c]
        summary[c] = {e: {k: [float(np.mean([r[e][k] for r in rs])), float(np.std([r[e][k] for r in rs], ddof=1))]
                          for k in ("norm_init", "norm_final", "moved")}
                      for e in ("user_embedding", "skill_embedding")}
        print(c, {e: {k: round(v[0], 3) for k, v in d.items()} for e, d in summary[c].items()}, flush=True)
    dump({"runs": out, "summary": summary}, os.path.join(OUT_DIR, "embedding_diagnostic.json"))
