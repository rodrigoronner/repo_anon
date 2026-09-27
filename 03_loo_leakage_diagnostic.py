"""Step 3 - Diagnostic: leave-one-out aggregate encoding re-introduces the label.

Rebuilds the construction of the preliminary analysis (chronological 80/20 split,
student and skill means from the training pool, leave-one-out subtraction for
training pairs) and fits the same XGBoost to it. If the encoding were leak-free,
validation AUC (random training pairs) would be close to test AUC.
"""

import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from common import dump
from config import (EVAL_DIR, MIN_SKILL_ATTEMPTS, MIN_STUDENT_INTERACTIONS, RAW_PATH, SPLIT_SEED,
                    SUCCESS_THRESHOLD, TRAIN_FRACTION, VAL_FRACTION)

df = pd.read_csv(RAW_PATH, encoding="latin-1", low_memory=False,
                 usecols=["order_id", "user_id", "skill_id", "correct"]).dropna()
df["skill_id"] = df["skill_id"].astype(str)
df = df[df["correct"].isin([0, 1])]
s, k = df.groupby("user_id").size(), df.groupby("skill_id").size()
df = df[df["user_id"].isin(s[s >= MIN_STUDENT_INTERACTIONS].index)
        & df["skill_id"].isin(k[k >= MIN_SKILL_ATTEMPTS].index)].sort_values(["user_id", "order_id"])
pos = df.groupby("user_id").cumcount() / df.groupby("user_id")["correct"].transform("size")
train_int, test_int = df[pos < TRAIN_FRACTION], df[pos >= TRAIN_FRACTION]

u = train_int.groupby("user_id")["correct"].agg(u_sum="sum", u_cnt="count")
sk = train_int.groupby("skill_id")["correct"].agg(s_sum="sum", s_cnt="count")


def pairs(inter, loo):
    p = inter.groupby(["user_id", "skill_id"])["correct"].agg(p_sum="sum", p_cnt="count").reset_index()
    p["target"] = (p["p_sum"] / p["p_cnt"] >= SUCCESS_THRESHOLD).astype(int)
    p = p.join(u, on="user_id").join(sk, on="skill_id")
    sub = (p["p_sum"], p["p_cnt"]) if loo else (0, 0)
    p["user_mean"] = (p["u_sum"] - sub[0]) / (p["u_cnt"] - sub[1]).replace(0, np.nan)
    p["skill_mean"] = (p["s_sum"] - sub[0]) / (p["s_cnt"] - sub[1]).replace(0, np.nan)
    p["user_count"] = p["u_cnt"] - sub[1]
    return p.fillna(train_int["correct"].mean())


tr, te = pairs(train_int, loo=True), pairs(test_int, loo=False)
val_mask = np.random.default_rng(SPLIT_SEED).random(len(tr)) < VAL_FRACTION
X = ["user_mean", "user_count", "skill_mean"]
model = xgb.XGBClassifier(n_estimators=5000, early_stopping_rounds=100, eval_metric="auc", max_depth=6,
                          learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=0,
                          n_jobs=4, verbosity=0)
model.fit(tr.loc[~val_mask, X], tr.loc[~val_mask, "target"],
          eval_set=[(tr.loc[val_mask, X], tr.loc[val_mask, "target"])], verbose=False)
out = {"validation_auc": float(roc_auc_score(tr.loc[val_mask, "target"],
                                             model.predict_proba(tr.loc[val_mask, X])[:, 1])),
       "test_auc": float(roc_auc_score(te["target"], model.predict_proba(te[X])[:, 1]))}
print(f"Leave-one-out encoding: validation AUC {out['validation_auc']:.3f}, test AUC {out['test_auc']:.3f}")
dump(out, os.path.join(EVAL_DIR, "loo_leakage_diagnostic.json"))
