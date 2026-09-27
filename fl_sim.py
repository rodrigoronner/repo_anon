"""In-process FedAvg / FedProx simulation, one client per student.

Each round samples max(MIN_FIT_CLIENTS, FRACTION_FIT * K) clients without
replacement; each runs `local_epochs` of mini-batch training from the current
global weights (plus the proximal term for FedProx) with a freshly created
optimizer; the server sets the global weights to the sample-size-weighted mean
of the returned weights. This is the update rule of Flower's FedAvg/FedProx.

With hp["aggregation"] == "rowwise", the two embedding tables are aggregated
row by row instead: each row is averaged (sample-size weighted) only over the
sampled clients whose data contain that student or skill, and rows that no
sampled client used keep their global value. A student row is then set to the
value returned by that student's own client, which is equivalent to keeping
the student embedding on the client (federated collaborative filtering).

Randomness is fully determined by (seed, round, client): the client sample of
each round comes from numpy's default_rng(seed), and every local update seeds
torch with client_seed(seed, round, client). The Flower cross-check reuses the
same two functions, so both implementations see identical randomness.
"""

import time

import numpy as np
import torch

from common import auc, metrics
from config import FRACTION_FIT, MIN_FIT_CLIENTS
from models import build, predict, to_tensors, train_epochs


def client_seed(seed, rnd, cid):
    return (seed * 1_000_003 + rnd * 10_007 + cid) % (2 ** 31 - 1)


def n_fit_clients(n_clients):
    return max(MIN_FIT_CLIENTS, int(FRACTION_FIT * n_clients))


def client_partitions(fit):
    """List of per-student tensors, ordered by user_id_new (client id = list index)."""
    groups = [(uid, g) for uid, g in fit.groupby("user_id_new")]
    return [uid for uid, _ in groups], [to_tensors(g) for _, g in groups]


def run_federated(fit, val, test, meta, arch, hp, mu, seed, n_rounds,
                  report=None, eval_test=True, log_every=0):
    """Returns (per-round rows, best checkpoint predictions, final predictions and global model).

    report(round, best_val_auc) is called every round; it may raise to prune.
    """
    torch.set_num_threads(1)
    _, clients = client_partitions(fit)
    sizes = np.array([len(c[3]) for c in clients], dtype=float)
    t_val, t_test = to_tensors(val), (to_tensors(test) if eval_test else None)

    torch.manual_seed(seed)
    gmodel, local = build(meta, arch), build(meta, arch)
    rng = np.random.default_rng(seed)
    k = n_fit_clients(len(clients))
    rowwise = hp.get("aggregation", "dense") == "rowwise"
    emb_rows = {"user_embedding.weight": [torch.unique(c[0]) for c in clients],
                "skill_embedding.weight": [torch.unique(c[1]) for c in clients]} if rowwise else {}

    rows, best, t0 = [], None, time.time()
    for rnd in range(1, n_rounds + 1):
        gstate = {n: t.detach().clone() for n, t in gmodel.state_dict().items()}
        gparams = [p.detach().clone() for p in gmodel.parameters()] if mu > 0 else None
        acc = {n: torch.zeros_like(t) for n, t in gstate.items()}
        row_w = {n: torch.zeros(gstate[n].shape[0]) for n in emb_rows}
        sel = rng.choice(len(clients), k, replace=False)
        for c in sel:
            torch.manual_seed(client_seed(seed, rnd, int(c)))
            local.load_state_dict(gstate)
            train_epochs(local, clients[c], hp["local_epochs"], hp, mu=mu, global_params=gparams)
            for n, t in local.state_dict().items():
                if n in emb_rows:
                    r = emb_rows[n][c]
                    acc[n][r] += sizes[c] * t[r]
                    row_w[n][r] += sizes[c]
                else:
                    acc[n] += sizes[c] * t
        tot = sizes[sel].sum()
        new = {n: a / tot for n, a in acc.items() if n not in emb_rows}
        for n, w in row_w.items():
            upd = w > 0
            row = gstate[n].clone()
            row[upd] = acc[n][upd] / w[upd].unsqueeze(1)
            new[n] = row
        gmodel.load_state_dict(new)

        p_val = predict(gmodel, t_val)
        p_test = None
        if not np.isfinite(p_val).all():
            rows.append({"round": rnd, "val_auc": float("nan"), "diverged": True})
            break
        v_auc = auc(val["target"], p_val)
        row = {"round": rnd, "val_auc": v_auc}
        if eval_test:
            p_test = predict(gmodel, t_test)
            row.update({f"test_{m}": v for m, v in metrics(test["target"], p_test).items()})
        rows.append(row)
        if best is None or v_auc > best["val_auc"]:
            best = {"val_auc": v_auc, "round": rnd, "p_val": p_val, "p_test": p_test}
        if log_every and rnd % log_every == 0:
            print(f"    r{rnd:4d} val AUC {v_auc:.4f} (best {best['val_auc']:.4f} @ {best['round']}) "
                  f"{time.time() - t0:.0f}s", flush=True)
        if report is not None:
            report(rnd, best["val_auc"])
    final = {"p_val": p_val, "p_test": p_test, "model": gmodel}
    return rows, best, final
