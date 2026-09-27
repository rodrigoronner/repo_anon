"""Step 34 - Analyses added after review, computed from the saved test predictions.

1. Per-student AUC: AUC computed within each student (students with both classes in the
   test window) and averaged. It measures how well a model orders the skills of one
   student, which is the ordering a recommender uses to choose the next skill.
2. Agreement with the centralized recommender: within-student Spearman correlation of the
   predicted scores and the share of students whose top-ranked skill coincides.
3. Uncertainty including test-set sampling: student-cluster bootstrap of each contrast,
   drawing a random evaluation seed per model in every replicate.
4. Smallest equivalence margin of each pair of aggregation strategies: the margin at which
   TOST (alpha = 0.05) would still conclude equivalence, i.e. the largest absolute bound of
   the 90% CI of the paired difference.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import dump, read
from config import DATA_DIR, EVAL_DIR, EVAL_SEEDS, OUT_DIR

B = 2000
RNG = np.random.default_rng(2026)
test = pd.read_csv(os.path.join(DATA_DIR, "processed_test.csv"))
y = test["target"].to_numpy()
u = test["user_id_new"].to_numpy()
students = np.unique(u)
index = {s: np.flatnonzero(u == s) for s in students}
both = test.groupby("user_id_new")["target"].agg(["min", "max"])
eligible = both[(both["min"] == 0) & (both["max"] == 1)].index.to_numpy()


def preds(name):
    for path in (os.path.join(EVAL_DIR, f"{name}_test_preds.npy"),
                 os.path.join(EVAL_DIR, "ablation_preds", f"{name}.npy")):
        if os.path.exists(path):
            return np.load(path)
    raise FileNotFoundError(name)


MODELS = {
    "xgboost": "XGBoost, centralized", "cdnn_tuned": "Recommender, centralized",
    "fl_FedAvg": "Recommender, FedAvg", "fl_FedProx_mu0.1": "Recommender, FedProx mu=0.1",
    "fl_FedProx_mu0.5": "Recommender, FedProx mu=0.5", "fl_FedProx_mu1.0": "Recommender, FedProx mu=1.0",
    "fl_FedProx_muTuned": "Recommender, FedProx mu tuned",
    "fl_FedAvg_rowwise": "Recommender, FedAvg, row-wise aggregation (tuned)",
    "fl_FedAvg_rowwise_samecfg": "Recommender, FedAvg, row-wise aggregation (FedAvg configuration)",
    "clr_tuned": "Logistic regression, centralized", "fl_LogReg": "Logistic regression, FedAvg",
    "cdnn_features_only": "Recommender without embeddings, centralized",
    "FedAvg_features_only": "Recommender without embeddings, FedAvg",
    "cdnn_v1": "Recommender, centralized, preliminary configuration",
    "fl_FedAvg_v1protocol": "Recommender, FedAvg, preliminary configuration",
}
P = {k: preds(k) for k in MODELS}


def per_student_auc(p):
    """Mean within-student AUC over eligible students (Mann-Whitney form, ties averaged)."""
    df = pd.DataFrame({"u": u, "y": y, "p": p})
    df = df[df["u"].isin(eligible)]
    df["r"] = df.groupby("u")["p"].rank(method="average")
    df["rpos"] = df["r"] * df["y"]
    g = df.groupby("u").agg(n1=("y", "sum"), n=("y", "size"), rpos=("rpos", "sum"))
    g["n0"] = g["n"] - g["n1"]
    return ((g["rpos"] - g["n1"] * (g["n1"] + 1) / 2) / (g["n1"] * g["n0"])).reindex(eligible).to_numpy()


def ms(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1))]


out = {"n_students_eligible": int(len(eligible)), "n_pairs_eligible": int(np.isin(u, eligible).sum()),
       "models": {}, "agreement": {}, "bootstrap": {}, "equivalence_margins": {}}

# 1. global and per-student AUC ---------------------------------------------------------------------
PS = {}
for k, name in MODELS.items():
    PS[k] = np.array([per_student_auc(p) for p in P[k]])  # seeds x eligible students
    out["models"][k] = {"label": name, "global_auc": ms([roc_auc_score(y, p) for p in P[k]]),
                        "per_student_auc": ms(PS[k].mean(axis=1))}
skill_ref = test["skill_mean_correct"].to_numpy()
out["models"]["skill_history_mean"] = {"label": "Skill history mean (no model)",
                                       "global_auc": [float(roc_auc_score(y, skill_ref)), 0.0],
                                       "per_student_auc": [float(per_student_auc(skill_ref).mean()), 0.0]}
out["models"]["student_history_mean"] = {"label": "Student history mean (no model)",
                                         "global_auc": [float(roc_auc_score(y, test["user_mean_correct"])), 0.0],
                                         "per_student_auc": [0.5, 0.0]}

# 2. agreement with the centralized recommender (seed-averaged scores) --------------------------------
ref = P["cdnn_tuned"].mean(axis=0)
for k in ("fl_FedAvg", "fl_FedProx_mu0.1", "fl_FedProx_mu0.5", "fl_FedProx_mu1.0", "fl_FedProx_muTuned",
          "fl_FedAvg_rowwise_samecfg", "fl_FedAvg_rowwise", "fl_LogReg", "FedAvg_features_only", "fl_FedAvg_v1protocol"):
    q = P[k].mean(axis=0)
    rho, top = [], []
    for s in students:
        i = index[s]
        if len(i) >= 3:
            r = stats.spearmanr(ref[i], q[i]).correlation
            if np.isfinite(r):
                rho.append(r)
        if len(i) >= 2:
            top.append(np.argmax(ref[i]) == np.argmax(q[i]))
    out["agreement"][k] = {"spearman_median": float(np.median(rho)), "spearman_mean": float(np.mean(rho)),
                           "n_students": len(rho), "top1_agreement": float(np.mean(top)), "n_top1": len(top)}

# 3. student-cluster bootstrap of contrasts -----------------------------------------------------------
CONTRASTS = {
    "cost_FedAvg": ("cdnn_tuned", "fl_FedAvg"),
    "cost_FedProx_mu0.1": ("cdnn_tuned", "fl_FedProx_mu0.1"),
    "cost_FedProx_mu0.5": ("cdnn_tuned", "fl_FedProx_mu0.5"),
    "cost_FedProx_mu1.0": ("cdnn_tuned", "fl_FedProx_mu1.0"),
    "cost_FedProx_muTuned": ("cdnn_tuned", "fl_FedProx_muTuned"),
    "cost_rowwise": ("cdnn_tuned", "fl_FedAvg_rowwise"),
    "rowwise_minus_dense": ("fl_FedAvg_rowwise", "fl_FedAvg"),
    "cost_features_only": ("cdnn_features_only", "FedAvg_features_only"),
    "centralized_gain_from_embeddings": ("cdnn_tuned", "cdnn_features_only"),
    "fedavg_loss_from_embeddings": ("FedAvg_features_only", "fl_FedAvg"),
    "cost_logreg": ("clr_tuned", "fl_LogReg"),
    "recommender_minus_logreg_centralized": ("cdnn_tuned", "clr_tuned"),
    "recommender_minus_logreg_federated": ("fl_FedAvg", "fl_LogReg"),
    "cost_preliminary": ("cdnn_v1", "fl_FedAvg_v1protocol"),
}
elig_pos = {s: j for j, s in enumerate(eligible)}
for name, (a, b) in CONTRASTS.items():
    g_diff, s_diff = [], []
    for _ in range(B):
        ss = RNG.choice(students, len(students), replace=True)
        ii = np.concatenate([index[s] for s in ss])
        ka, kb = RNG.integers(len(EVAL_SEEDS)), RNG.integers(len(EVAL_SEEDS))
        g_diff.append(roc_auc_score(y[ii], P[a][ka][ii]) - roc_auc_score(y[ii], P[b][kb][ii]))
        jj = [elig_pos[s] for s in ss if s in elig_pos]
        s_diff.append(PS[a][ka][jj].mean() - PS[b][kb][jj].mean())
    point_g = float(np.mean([roc_auc_score(y, p) for p in P[a]]) - np.mean([roc_auc_score(y, p) for p in P[b]]))
    point_s = float(PS[a].mean() - PS[b].mean())
    out["bootstrap"][name] = {
        "a": a, "b": b,
        "global_auc": {"diff": point_g, "ci95": [float(x) for x in np.percentile(g_diff, [2.5, 97.5])]},
        "per_student_auc": {"diff": point_s, "ci95": [float(x) for x in np.percentile(s_diff, [2.5, 97.5])]}}

# 4. smallest equivalence margin of each strategy pair (paired by seed, test AUC at selected round) ---
strategies = ["FedAvg", "FedProx_mu0.1", "FedProx_mu0.5", "FedProx_mu1.0", "FedProx_muTuned"]
sel = {s: np.array([r["selected"]["at_0.5"]["roc_auc"]
                    for r in sorted(read(os.path.join(EVAL_DIR, f"fl_{s}.json"))["runs"], key=lambda r: r["seed"])])
       for s in strategies}
t90 = stats.t.ppf(0.95, len(EVAL_SEEDS) - 1)
for i, a in enumerate(strategies):
    for b in strategies[i + 1:]:
        d = sel[a] - sel[b]
        half = t90 * d.std(ddof=1) / np.sqrt(len(d))
        out["equivalence_margins"][f"{a} vs {b}"] = float(max(abs(d.mean() - half), abs(d.mean() + half)))

dump(out, os.path.join(OUT_DIR, "review_analyses.json"))

# report ----------------------------------------------------------------------------------------------
L = ["# Analyses added after review", "",
     f"Per-student AUC over {out['n_students_eligible']} students with both classes in the test window "
     f"({out['n_pairs_eligible']} of {len(test)} test pairs).", "",
     "| Model | Global AUC | Per-student AUC |", "|---|---|---|"]
for k, m in out["models"].items():
    L.append(f"| {m['label']} | {m['global_auc'][0]:.3f} ± {m['global_auc'][1]:.3f} | "
             f"{m['per_student_auc'][0]:.3f} ± {m['per_student_auc'][1]:.3f} |")
L += ["", "## Agreement with the centralized recommender (seed-averaged scores)", "",
      "| Model | within-student Spearman (median / mean) | same top-ranked skill |", "|---|---|---|"]
for k, a in out["agreement"].items():
    L.append(f"| {MODELS[k]} | {a['spearman_median']:.3f} / {a['spearman_mean']:.3f} (n={a['n_students']}) | "
             f"{a['top1_agreement']:.3f} (n={a['n_top1']}) |")
L += ["", f"## Student-cluster bootstrap (B={B}, random seed per model per replicate)", "",
      "| Contrast | Global AUC diff [95% CI] | Per-student AUC diff [95% CI] |", "|---|---|---|"]
for n, c in out["bootstrap"].items():
    g, s = c["global_auc"], c["per_student_auc"]
    L.append(f"| {n} ({c['a']} − {c['b']}) | {g['diff']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] | "
             f"{s['diff']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] |")
L += ["", "## Smallest equivalence margin (TOST, alpha = 0.05)", ""]
L += [f"- {k}: {v:.4f}" for k, v in out["equivalence_margins"].items()]
with open(os.path.join(OUT_DIR, "REPORT_review.md"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print("\n".join(L))

# LaTeX table: global vs per-student AUC and agreement with the centralized recommender -------------
ROWS = [("skill_history_mean", "Skill history mean (no model)"),
        ("xgboost", "XGBoost, centralized"), ("cdnn_tuned", "Recommender, centralized"),
        ("clr_tuned", "Logistic regression, centralized"),
        ("fl_FedAvg", "Recommender, FedAvg"), ("fl_FedProx_mu0.1", r"Recommender, FedProx $\mu=0.1$"),
        ("fl_FedProx_mu0.5", r"Recommender, FedProx $\mu=0.5$"), ("fl_FedProx_mu1.0", r"Recommender, FedProx $\mu=1.0$"),
        ("fl_FedProx_muTuned", r"Recommender, FedProx, $\mu$ tuned"),
        ("fl_FedAvg_rowwise_samecfg", "Recommender, FedAvg, row-wise (FedAvg config.)"),
        ("fl_FedAvg_rowwise", "Recommender, FedAvg, row-wise (tuned)"),
        ("fl_LogReg", "Logistic regression, FedAvg"),
        ("fl_FedAvg_v1protocol", "Recommender, FedAvg, preliminary config.")]


def pm(x, sd=True):
    return f"{x[0]:.3f} $\\pm$ {x[1]:.3f}" if sd and x[1] > 0 else f"{x[0]:.3f}"


lines = []
for k, lab in ROWS:
    m = out["models"][k]
    ag = out["agreement"].get(k)
    agree = (f"{ag['spearman_median']:.3f} & {100 * ag['top1_agreement']:.1f}\\%" if ag else "-- & --")
    lines.append(f"{lab} & {pm(m['global_auc'])} & {pm(m['per_student_auc'])} & {agree} \\\\")
tex = (r"""\begin{table*}[ht]
\centering
\caption{Global and per-student ROC AUC on the test set (mean $\pm$ SD over 10 seeds), and agreement of federated models with the centralized recommender. Per-student AUC is computed within each of the """ + str(out["n_students_eligible"]) + r""" students with both outcomes in the test window and averaged; it measures how well a model orders the skills of one student. Agreement uses seed-averaged scores: median within-student Spearman correlation, and share of students for whom both models rank the same skill first. Data: own experiments.}
\label{tab:perstudent}
\begin{tblr}{
    colspec = {X[2.6,l] X[1.2,c] X[1.2,c] X[1.1,c] X[1,c]},
    cells   = {font=\small},
    row{1}  = {font=\small\bfseries},
    }
\toprule
Model & Global AUC & Per-student AUC & Spearman with centralized & Same top skill \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tblr}
\end{table*}
""")
with open(os.path.join(OUT_DIR, "latex", "tab_perstudent.tex"), "w") as fh:
    fh.write(tex)
