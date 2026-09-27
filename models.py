"""RecommenderNet (configurable) and the training loop shared by centralized and federated runs."""

import numpy as np
import torch
import torch.nn as nn

from config import FEAT_COLS


# Input blocks fed to the MLP; the ablation study (24_ablation.py) drops some of them.
INPUTS = {"all": ("user", "skill", "feat"), "no_user": ("skill", "feat"), "no_skill": ("user", "feat"),
          "features_only": ("feat",), "embeddings_only": ("user", "skill")}


class RecommenderNet(nn.Module):
    """Student and skill embeddings concatenated with the engineered features, then an MLP."""

    def __init__(self, num_users, num_skills, emb_dim=10, h1=32, h2=16, dropout=0.0, inputs="all"):
        super().__init__()
        self.blocks = INPUTS[inputs]
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.skill_embedding = nn.Embedding(num_skills, emb_dim)
        in_dim = sum({"user": emb_dim, "skill": emb_dim, "feat": len(FEAT_COLS)}[b] for b in self.blocks)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1))

    def forward(self, u, s, f):
        parts = {"user": lambda: self.user_embedding(u), "skill": lambda: self.skill_embedding(s),
                 "feat": lambda: f}
        x = torch.cat([parts[b]() for b in self.blocks], dim=1)
        return self.mlp(x).squeeze(-1)


class LogReg(nn.Module):
    """Logistic regression on the three aggregate features (no embeddings)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(len(FEAT_COLS), 1)

    def forward(self, u, s, f):
        return self.linear(f).squeeze(-1)


def build(meta, arch):
    if arch.get("model") == "logreg":
        return LogReg()
    return RecommenderNet(meta["num_users"], meta["num_skills"], arch["emb_dim"],
                          arch["h1"], arch["h2"], arch["dropout"], arch.get("inputs", "all"))


def to_tensors(df):
    return (torch.tensor(df["user_id_new"].values, dtype=torch.long),
            torch.tensor(df["skill_id_new"].values, dtype=torch.long),
            torch.tensor(df[FEAT_COLS].values, dtype=torch.float32),
            torch.tensor(df["target"].values, dtype=torch.float32))


@torch.no_grad()
def predict(model, tensors):
    model.eval()
    u, s, f, _ = tensors
    return torch.sigmoid(model(u, s, f)).numpy().astype(np.float64)


def make_optimizer(model, hp):
    """Weight decay applies to every parameter unless hp["decay_embeddings"] is False,
    in which case the embedding tables are excluded (ablation study)."""
    wd = hp["weight_decay"]
    if hp.get("decay_embeddings", True):
        params = model.parameters()
    else:
        emb = [model.user_embedding.weight, model.skill_embedding.weight]
        params = [{"params": emb, "weight_decay": 0.0},
                  {"params": list(model.mlp.parameters()), "weight_decay": wd}]
    if hp["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=hp["lr"], weight_decay=wd)
    return torch.optim.SGD(params, lr=hp["lr"], weight_decay=wd)


def train_epochs(model, tensors, epochs, hp, opt=None, mu=0.0, global_params=None):
    """Mini-batch BCE training. Uses the global torch RNG, which the caller seeds.

    With mu > 0 the FedProx proximal term (mu/2)||w - w_global||^2 is added.
    A fresh optimizer is created unless one is passed (centralized training keeps its state).
    """
    u, s, f, y = tensors
    n = len(y)
    opt = opt or make_optimizer(model, hp)
    crit = nn.BCEWithLogitsLoss()
    bs = hp["batch_size"]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for st in range(0, n, bs):
            b = perm[st:st + bs]
            opt.zero_grad()
            loss = crit(model(u[b], s[b], f[b]), y[b])
            if mu > 0:
                loss = loss + (mu / 2.0) * sum(torch.sum((p - g) ** 2)
                                               for p, g in zip(model.parameters(), global_params))
            loss.backward()
            opt.step()
    return opt
