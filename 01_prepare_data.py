"""
Step 1 - Leakage-free data preparation from the raw ASSISTments file.

Every example is "past history -> future outcome", for training and test alike.
Each student's interactions are ordered by order_id (validated against the
ordinal `opportunity` counter) and cut at two points of their own history:

    training examples (expanding window, stacked):
        history [0, 0.4)  -> labels [0.4, 0.6)
        history [0, 0.6)  -> labels [0.6, 0.8)
    test examples:
        history [0, 0.8)  -> labels [0.8, 1.0]

An example is a (student, skill) pair present in a label window; its target is
the pair's success rate in that window, binarized at SUCCESS_THRESHOLD. Its
features are the student's and the skill's success statistics in the matching
history window, so no feature ever contains the example's own label. This
replaces leave-one-out encoding: leave-one-out means are a decreasing function
of the pair's own outcome, which re-introduces the label (validation AUC 0.928
against 0.719 on test; see 03_loo_leakage_diagnostic.py).

VAL_FRACTION of the training examples is held out for tuning, checkpoint and
threshold selection. The scaler is fitted on training examples only.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (DATA_DIR, FEAT_COLS, MIN_SKILL_ATTEMPTS, MIN_STUDENT_INTERACTIONS,
                    RAW_PATH, RAW_SHA256, SPLIT_SEED, SUCCESS_THRESHOLD, TRAIN_FRACTION,
                    VAL_FRACTION)

TRAIN_WINDOWS = [(0.40, 0.60), (0.60, TRAIN_FRACTION)]

with open(RAW_PATH, "rb") as fh:
    sha = hashlib.sha256(fh.read()).hexdigest()
assert sha == RAW_SHA256, f"Raw file checksum mismatch: {sha}"
print(f"Raw file SHA-256 verified: {sha}")

df = pd.read_csv(RAW_PATH, encoding="latin-1", low_memory=False,
                 usecols=["order_id", "user_id", "skill_id", "correct", "opportunity"])
n_raw_rows = len(df)
df = df.dropna(subset=["user_id", "skill_id", "correct", "order_id"])
df["skill_id"] = df["skill_id"].astype(str)
df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
df = df.dropna(subset=["correct"])
df = df[df["correct"].isin([0, 1])].copy()
df["correct"] = df["correct"].astype(int)
n_valid = len(df)

students = df.groupby("user_id").size()
skills = df.groupby("skill_id").size()
df = df[df["user_id"].isin(students[students >= MIN_STUDENT_INTERACTIONS].index)
        & df["skill_id"].isin(skills[skills >= MIN_SKILL_ATTEMPTS].index)].copy()
print(f"Rows {n_raw_rows:,} -> valid {n_valid:,} -> filtered {len(df):,} "
      f"({df['user_id'].nunique()} students, {df['skill_id'].nunique()} skills)")

# order_id as a sequence key: Spearman with `opportunity` within (student, skill).
sub = df.dropna(subset=["opportunity"])
sub = sub[sub.groupby(["user_id", "skill_id"])["order_id"].transform("size") >= 3]
rho = (sub.groupby(["user_id", "skill_id"])[["order_id", "opportunity"]]
          .apply(lambda g: g["order_id"].corr(g["opportunity"], method="spearman")).dropna())
ordering = {"mean_spearman": float(rho.mean()), "pct_pairs_rho_ge_0.99": float((rho >= 0.99).mean()),
            "n_pairs": int(len(rho))}
print(f"order_id vs opportunity: mean rho {ordering['mean_spearman']:.4f}, "
      f"{ordering['pct_pairs_rho_ge_0.99']:.1%} of {ordering['n_pairs']} pairs at rho >= 0.99")

df = df.sort_values(["user_id", "order_id"]).reset_index(drop=True)
pos = df.groupby("user_id").cumcount() / df.groupby("user_id")["correct"].transform("size")
df["pos"] = pos


def examples(history, labels):
    """Pairs in `labels`, with student/skill features computed on `history` only."""
    pairs = labels.groupby(["user_id", "skill_id"])["correct"].agg(
        pair_sum="sum", pair_cnt="count").reset_index()
    pairs["target_correct_rate"] = pairs["pair_sum"] / pairs["pair_cnt"]
    pairs["target"] = (pairs["target_correct_rate"] >= SUCCESS_THRESHOLD).astype(int)
    u = history.groupby("user_id")["correct"].agg(u_sum="sum", u_cnt="count").reset_index()
    s = history.groupby("skill_id")["correct"].agg(s_sum="sum", s_cnt="count").reset_index()
    seen = history[["user_id", "skill_id"]].drop_duplicates().assign(is_seen_pair=1)
    out = pairs.merge(u, on="user_id", how="left").merge(s, on="skill_id", how="left")
    out = out.merge(seen, on=["user_id", "skill_id"], how="left")
    out["is_seen_pair"] = out["is_seen_pair"].fillna(0).astype(int)
    prior = history["correct"].mean()
    out["is_cold_start"] = (out["u_cnt"].isna() | out["s_cnt"].isna()).astype(int)
    out["user_mean_correct"] = (out["u_sum"] / out["u_cnt"]).fillna(prior)
    out["skill_mean_correct"] = (out["s_sum"] / out["s_cnt"]).fillna(prior)
    out["user_interaction_count"] = np.log1p(out["u_cnt"].fillna(0))
    return out, float(prior)


parts, prior_train = [], []
for w, (a, b) in enumerate(TRAIN_WINDOWS):
    part, pr = examples(df[df["pos"] < a], df[(df["pos"] >= a) & (df["pos"] < b)])
    parts.append(part.assign(window=w))
    prior_train.append(pr)
train = pd.concat(parts, ignore_index=True)
test, prior_test = examples(df[df["pos"] < TRAIN_FRACTION], df[df["pos"] >= TRAIN_FRACTION])
test["window"] = len(TRAIN_WINDOWS)

user_map = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
skill_map = {s: i for i, s in enumerate(sorted(df["skill_id"].unique()))}
for frame in (train, test):
    frame["user_id_new"] = frame["user_id"].map(user_map).astype(int)
    frame["skill_id_new"] = frame["skill_id"].map(skill_map).astype(int)

scaler = MinMaxScaler().fit(train[FEAT_COLS])
train[FEAT_COLS] = scaler.transform(train[FEAT_COLS])
test[FEAT_COLS] = scaler.transform(test[FEAT_COLS])

rng = np.random.default_rng(SPLIT_SEED)
train["is_val"] = (rng.random(len(train)) < VAL_FRACTION).astype(int)
test["is_val"] = 0

keep = ["user_id_new", "skill_id_new", *FEAT_COLS, "target_correct_rate", "target",
        "is_seen_pair", "is_cold_start", "is_val", "window"]
train[keep].to_csv(os.path.join(DATA_DIR, "processed_train.csv"), index=False)
test[keep].to_csv(os.path.join(DATA_DIR, "processed_test.csv"), index=False)

fit_part = train[train["is_val"] == 0]
per_user = fit_part.groupby("user_id_new").size()
meta = {
    "raw_sha256": sha, "raw_rows": n_raw_rows, "valid_rows": n_valid,
    "filtered_interactions": int(len(df)),
    "num_users": len(user_map), "num_skills": len(skill_map),
    "train_windows": TRAIN_WINDOWS, "train_fraction": TRAIN_FRACTION,
    "train_examples_per_window": train["window"].value_counts().sort_index().tolist(),
    "val_fraction": VAL_FRACTION, "success_threshold": SUCCESS_THRESHOLD, "split_seed": SPLIT_SEED,
    "n_train_examples": int(len(train)), "n_fit_examples": int(len(fit_part)),
    "n_val_examples": int(train["is_val"].sum()), "n_test_examples": int(len(test)),
    "positive_rate": {"fit": float(fit_part["target"].mean()),
                      "val": float(train.loc[train["is_val"] == 1, "target"].mean()),
                      "test": float(test["target"].mean())},
    "seen_pair_share": {"train": float(train["is_seen_pair"].mean()),
                        "test": float(test["is_seen_pair"].mean())},
    "cold_start": {"train": int(train["is_cold_start"].sum()), "test": int(test["is_cold_start"].sum())},
    "clients_with_fit_examples": int(per_user.size),
    "fit_examples_per_client": {"min": int(per_user.min()), "median": float(per_user.median()),
                                "max": int(per_user.max())},
    "history_prior": {"train": prior_train, "test": prior_test},
    "ordering_validation": ordering,
}
with open(os.path.join(DATA_DIR, "processed_meta.json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print(json.dumps({k: meta[k] for k in ("num_users", "num_skills", "n_fit_examples", "n_val_examples",
                                       "n_test_examples", "positive_rate", "seen_pair_share",
                                       "cold_start", "fit_examples_per_client")}, indent=1))
