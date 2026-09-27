"""Step 23 - Equivalence check of the in-process simulator against Flower 1.7 (Ray backend).

The Flower strategy samples, in each round, exactly the clients the simulator
samples (same numpy generator), and every Flower client seeds torch with the
same client_seed(seed, round, client). The tuned FedAvg configuration is run
for CROSSCHECK_ROUNDS rounds; per-round validation and test AUC are compared
with the simulator's rows for the same seed. Remaining differences can only
come from floating-point summation order in the server aggregation.

Usage: python 23_flower_crosscheck.py [seed ...]
"""

import importlib
import os
import sys

import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common import FitIns, ndarrays_to_parameters
from flwr.server.strategy import FedProx

from common import auc, dump, load
from config import EVAL_DIR
from fl_sim import client_partitions, client_seed, n_fit_clients, run_federated
from models import build, predict, to_tensors, train_epochs

CROSSCHECK_ROUNDS = 100
STRATEGY = "FedAvg"

tf = importlib.import_module("22_eval_federated")
ARCH, HP, MU = tf.conditions()[STRATEGY]
FIT, VAL, TEST, META = load()
_, CLIENTS = client_partitions(FIT)
T_VAL, T_TEST = to_tensors(VAL), to_tensors(TEST)


def get_params(m):
    return [v.detach().cpu().numpy() for v in m.state_dict().values()]


def set_params(m, arrs):
    m.load_state_dict({k: torch.tensor(a) for k, a in zip(m.state_dict().keys(), arrs)})


class Client(fl.client.NumPyClient):
    def __init__(self, cid, seed):
        self.cid, self.seed = cid, seed

    def fit(self, parameters, config):
        torch.set_num_threads(1)
        model = build(META, ARCH)
        set_params(model, parameters)
        gparams = [p.detach().clone() for p in model.parameters()] if MU > 0 else None
        torch.manual_seed(client_seed(self.seed, int(config["round"]), self.cid))
        train_epochs(model, CLIENTS[self.cid], HP["local_epochs"], HP, mu=MU, global_params=gparams)
        return get_params(model), len(CLIENTS[self.cid][3]), {}


class MatchedSampling(FedProx):
    """FedProx/FedAvg whose client sample reproduces the simulator's generator."""

    def __init__(self, seed, **kw):
        super().__init__(**kw)
        self.rng = np.random.default_rng(seed)

    def configure_fit(self, server_round, parameters, client_manager):
        proxies = client_manager.all()
        sel = self.rng.choice(len(CLIENTS), n_fit_clients(len(CLIENTS)), replace=False)
        cfg = {"round": server_round, "proximal_mu": self.proximal_mu}
        return [(proxies[str(int(c))], FitIns(parameters, cfg)) for c in sel]


def run_flower(seed):
    rows = []

    def evaluate_fn(rnd, parameters, config):
        if rnd == 0:
            return None
        m = build(META, ARCH)
        set_params(m, parameters)
        rows.append({"round": rnd, "val_auc": auc(VAL["target"], predict(m, T_VAL)),
                     "test_roc_auc": auc(TEST["target"], predict(m, T_TEST))})
        return 0.0, {}

    torch.manual_seed(seed)
    init = build(META, ARCH)
    strategy = MatchedSampling(seed, fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=1,
                               min_evaluate_clients=0, min_available_clients=len(CLIENTS),
                               initial_parameters=ndarrays_to_parameters(get_params(init)),
                               evaluate_fn=evaluate_fn, proximal_mu=MU)
    fl.simulation.start_simulation(
        client_fn=lambda cid: Client(int(cid), seed).to_client(), num_clients=len(CLIENTS),
        config=fl.server.ServerConfig(num_rounds=CROSSCHECK_ROUNDS), strategy=strategy,
        client_resources={"num_cpus": 1},
        ray_init_args={"num_cpus": 8, "include_dashboard": False, "log_to_driver": False})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [42]
    summary = []
    for seed in seeds:
        flw = run_flower(seed)
        sim_rows, _, _ = run_federated(FIT, VAL, TEST, META, ARCH, HP, MU, seed, CROSSCHECK_ROUNDS)
        sim = pd.DataFrame(sim_rows)[["round", "val_auc", "test_roc_auc"]]
        both = flw.merge(sim, on="round", suffixes=("_flower", "_sim"))
        both["seed"] = seed
        both.to_csv(os.path.join(EVAL_DIR, f"flower_crosscheck_seed{seed}.csv"), index=False)
        d = (both["test_roc_auc_flower"] - both["test_roc_auc_sim"]).abs()
        summary.append({"seed": seed, "rounds": int(len(both)), "max_abs_diff_test_auc": float(d.max()),
                        "mean_abs_diff_test_auc": float(d.mean()),
                        "final_test_auc_flower": float(both["test_roc_auc_flower"].iloc[-1]),
                        "final_test_auc_sim": float(both["test_roc_auc_sim"].iloc[-1])})
        print(summary[-1], flush=True)
    dump({"strategy": STRATEGY, "hp": HP, "mu": MU, "results": summary},
         os.path.join(EVAL_DIR, "flower_crosscheck.json"))
