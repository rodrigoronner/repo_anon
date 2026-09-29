"""Step 35 - Analysis of the robustness runs (step 27): temporal validation and FedProx with local SGD.

Writes outputs/robustness_analysis.json, outputs/REPORT_robustness.md and the LaTeX table
outputs/latex/tab_robustness.tex.
"""

import importlib
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import dump, read
from config import DATA_DIR, EVAL_DIR, EVAL_SEEDS, OUT_DIR

# test data and per-student AUC, with the same definitions as step 34
_test = pd.read_csv(os.path.join(DATA_DIR, "processed_test.csv"))
_y = _test["target"].to_numpy()
_u = _test["user_id_new"].to_numpy()
_students = np.unique(_u)
_index = {s: np.flatnonzero(_u == s) for s in _students}
_b = _test.groupby("user_id_new")["target"].agg(["min", "max"])
_eligible = _b[(_b["min"] == 0) & (_b["max"] == 1)].index.to_numpy()


def _per_student_auc(p):
    df = pd.DataFrame({"u": _u, "y": _y, "p": p})
    df = df[df["u"].isin(_eligible)]
    df["r"] = df.groupby("u")["p"].rank(method="average")
    df["rpos"] = df["r"] * df["y"]
    g = df.groupby("u").agg(n1=("y", "sum"), n=("y", "size"), rpos=("rpos", "sum"))
    g["n0"] = g["n"] - g["n1"]
    return ((g["rpos"] - g["n1"] * (g["n1"] + 1) / 2) / (g["n1"] * g["n0"])).reindex(_eligible).to_numpy()


rv = SimpleNamespace(y=_y, u=_u, students=_students, index=_index, eligible=_eligible, per_student_auc=_per_student_auc)
an = importlib.import_module("30_analysis")
RB = os.path.join(EVAL_DIR, "robustness")
runs = pd.DataFrame(read(os.path.join(RB, "robustness.json"))["runs"])
rounds = pd.read_csv(os.path.join(RB, "robustness_rounds.csv.gz"))
main_rounds = pd.read_csv(os.path.join(EVAL_DIR, "fl_rounds.csv.gz"))
y, u = rv.y, rv.u
B, RNG = 2000, np.random.default_rng(2027)


def preds(name):
    return np.load(os.path.join(RB, f"{name}.npy"))


def summary(name):
    P = preds(name)
    sub = runs[runs["condition"] == name].sort_values("seed")
    return {"global_auc": an.ms([roc_auc_score(y, p) for p in P]),
            "per_student_auc": an.ms([rv.per_student_auc(p).mean() for p in P]),
            "val_auc": an.ms([r["val_auc"] for r in sub["result"]])}


def boot(a, b):
    Pa, Pb = preds(a), preds(b)
    Sa = np.array([rv.per_student_auc(p) for p in Pa]); Sb = np.array([rv.per_student_auc(p) for p in Pb])
    pos = {s: j for j, s in enumerate(rv.eligible)}
    g, st = [], []
    for _ in range(B):
        ss = RNG.choice(rv.students, len(rv.students), replace=True)
        ii = np.concatenate([rv.index[s] for s in ss])
        ka, kb = RNG.integers(len(EVAL_SEEDS)), RNG.integers(len(EVAL_SEEDS))
        g.append(roc_auc_score(y[ii], Pa[ka][ii]) - roc_auc_score(y[ii], Pb[kb][ii]))
        jj = [pos[s] for s in ss if s in pos]
        st.append(Sa[ka][jj].mean() - Sb[kb][jj].mean())
    pg = float(np.mean([roc_auc_score(y, p) for p in Pa]) - np.mean([roc_auc_score(y, p) for p in Pb]))
    return {"global_auc": {"diff": pg, "ci95": [float(x) for x in np.percentile(g, [2.5, 97.5])]},
            "per_student_auc": {"diff": float(Sa.mean() - Sb.mean()),
                                "ci95": [float(x) for x in np.percentile(st, [2.5, 97.5])]}}


def stability(r, cond):
    g = r[(r["condition"] == cond) & r["round"].between(501, 1000)]
    allr = r[r["condition"] == cond]
    return {"late_sd_auc": an.ms(g.groupby("seed")["test_roc_auc"].std(ddof=1)),
            "lag1": an.ms([an.lag1(x.sort_values("round")["test_roc_auc"]) for _, x in g.groupby("seed")]),
            "share_rounds_auc_below_0.5": an.ms(allr.groupby("seed")["test_roc_auc"].apply(lambda x: (x < 0.5).mean())),
            "late_positive_rate": an.ms(g.groupby("seed")["test_positive_rate"].mean())}


out = {"temporal": {}, "temporal_contrasts": {}, "same_optimizer": {}}
T = ["T_cdnn_all", "T_cdnn_no_user", "T_cdnn_features_only", "T_FedAvg_all", "T_FedAvg_no_user",
     "T_FedAvg_features_only", "T_FedAvg_rowwise_samecfg", "T_FedAvg_rowwise_tuned"]
for c in T:
    out["temporal"][c] = summary(c)
CON = {"cost_of_federation": ("T_cdnn_all", "T_FedAvg_all"),
       "cost_features_only": ("T_cdnn_features_only", "T_FedAvg_features_only"),
       "student_embedding_centralized": ("T_cdnn_all", "T_cdnn_no_user"),
       "student_embedding_fedavg": ("T_FedAvg_all", "T_FedAvg_no_user"),
       "rowwise_samecfg_minus_dense": ("T_FedAvg_rowwise_samecfg", "T_FedAvg_all"),
       "rowwise_tuned_minus_dense": ("T_FedAvg_rowwise_tuned", "T_FedAvg_all"),
       "rowwise_samecfg_minus_no_user": ("T_FedAvg_rowwise_samecfg", "T_FedAvg_no_user")}
for k, (a, b) in CON.items():
    out["temporal_contrasts"][k] = {"a": a, "b": b, **boot(a, b)}

# B. same local optimizer: FedProx with FedAvg's SGD configuration vs FedAvg (main runs)
sgd = {"FedAvg (SGD)": ("main", "FedAvg"), "FedProx mu=1.0, SGD": ("rob", "S_FedProx_mu1.0_sgd"),
       "FedProx mu tuned, SGD": ("rob", "S_FedProx_muTuned_sgd"),
       "FedProx mu=1.0, Adam (tuned)": ("main", "FedProx_mu1.0"), "FedProx mu tuned, Adam (tuned)": ("main", "FedProx_muTuned")}
for lab, (src, cond) in sgd.items():
    if src == "main":
        rs = read(os.path.join(EVAL_DIR, f"fl_{cond}.json"))["runs"]
        auc = [r["selected"]["at_0.5"]["roc_auc"] for r in rs]
        stab = stability(main_rounds, cond)
    else:
        auc = [r["at_0.5"]["roc_auc"] for r in runs[runs["condition"] == cond]["result"]]
        stab = stability(rounds, cond)
    out["same_optimizer"][lab] = {"test_auc": an.ms(auc), **stab}
fedavg = sorted(read(os.path.join(EVAL_DIR, "fl_FedAvg.json"))["runs"], key=lambda r: r["seed"])
for cond, lab in (("S_FedProx_mu1.0_sgd", "FedProx mu=1.0, SGD"), ("S_FedProx_muTuned_sgd", "FedProx mu tuned, SGD")):
    fp = runs[runs["condition"] == cond].sort_values("seed")
    out["same_optimizer"][lab]["vs_fedavg_paired"] = an.paired(
        [r["at_0.5"]["roc_auc"] for r in fp["result"]], [r["selected"]["at_0.5"]["roc_auc"] for r in fedavg], margin=0.01)
dump(out, os.path.join(OUT_DIR, "robustness_analysis.json"))

f = lambda x, d=3: f"{x[0]:.{d}f} ± {x[1]:.{d}f}"
cit = lambda c: f"{c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
L = ["# Robustness analyses", "", "## A. Temporal validation (fit on window 1, validate on window 2)", "",
     "| Condition | val AUC | global test AUC | per-student AUC |", "|---|---|---|---|"]
L += [f"| {c} | {f(m['val_auc'])} | {f(m['global_auc'])} | {f(m['per_student_auc'])} |" for c, m in out["temporal"].items()]
L += ["", "| Contrast | global AUC [bootstrap 95% CI] | per-student AUC [95% CI] |", "|---|---|---|"]
L += [f"| {k} ({c['a']} − {c['b']}) | {cit(c['global_auc'])} | {cit(c['per_student_auc'])} |"
      for k, c in out["temporal_contrasts"].items()]
L += ["", "## B. Same local optimizer (SGD configuration of FedAvg)", "",
      "| Condition | test AUC | late SD | lag-1 | rounds AUC<0.5 | late positive rate |", "|---|---|---|---|---|---|"]
for lab, m in out["same_optimizer"].items():
    L.append(f"| {lab} | {f(m['test_auc'])} | {f(m['late_sd_auc'], 4)} | {f(m['lag1'], 2)} | "
             f"{f(m['share_rounds_auc_below_0.5'], 2)} | {f(m['late_positive_rate'], 2)} |")
for lab in ("FedProx mu=1.0, SGD", "FedProx mu tuned, SGD"):
    p = out["same_optimizer"][lab]["vs_fedavg_paired"]
    L.append(f"- {lab} − FedAvg: {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}], TOST p={p['tost_p']:.2g}")
with open(os.path.join(OUT_DIR, "REPORT_robustness.md"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print("\n".join(L))

# LaTeX table ------------------------------------------------------------------------------------------
pcx = lambda v: "0\\%" if v == 0 else ("$<$0.1\\%" if v < 0.001 else f"{100 * v:.0f}\\%")
pmx = lambda x, d=3: f"{x[0]:.{d}f} $\\pm$ {x[1]:.{d}f}"
TLAB = {"T_cdnn_all": "DNN, centralized", "T_cdnn_no_user": "DNN, centralized, without student embedding",
        "T_cdnn_features_only": "DNN, centralized, features only", "T_FedAvg_all": "FedAvg",
        "T_FedAvg_no_user": "FedAvg, without student embedding", "T_FedAvg_features_only": "FedAvg, features only",
        "T_FedAvg_rowwise_samecfg": "FedAvg, row-wise (FedAvg config.)", "T_FedAvg_rowwise_tuned": "FedAvg, row-wise (tuned config.)"}
rows = [r"\SetCell[c=5]{l}\textit{(a) Temporal validation: fitted on the first training window, validated on the second} & & & & \\",
        r"Condition & Validation AUC & Global test AUC & Per-student AUC & \\"]
rows += [f"{TLAB[c]} & {pmx(m['val_auc'])} & {pmx(m['global_auc'])} & {pmx(m['per_student_auc'])} & \\\\"
         for c, m in out["temporal"].items()]
rows += [r"\midrule", r"\SetCell[c=5]{l}\textit{(b) Same local optimizer: FedProx with the SGD configuration of FedAvg} & & & & \\",
         r"Condition & Test AUC & SD rounds 501--1{,}000 & Lag-1 autocorrelation & Rounds with AUC $<$ 0.5 \\"]
BLAB = {"FedAvg (SGD)": "FedAvg (SGD)", "FedProx mu=1.0, SGD": r"FedProx $\mu=1.0$, SGD",
        "FedProx mu tuned, SGD": r"FedProx, $\mu$ tuned, SGD", "FedProx mu=1.0, Adam (tuned)": r"FedProx $\mu=1.0$, Adam (tuned config.)",
        "FedProx mu tuned, Adam (tuned)": r"FedProx, $\mu$ tuned, Adam (tuned config.)"}
for lab, m in out["same_optimizer"].items():
    rows.append(f"{BLAB[lab]} & {pmx(m['test_auc'])} & {m['late_sd_auc'][0]:.3f} & "
                f"{m['lag1'][0]:.2f}".replace("-", "$-$") + f" & {pcx(m['share_rounds_auc_below_0.5'][0])} \\\\")
tex = (r"""\begin{table*}[!htbp]
\centering
\caption[Robustness analyses]{Robustness analyses (mean $\pm$ SD over 10 seeds; configurations fixed, not re-tuned). (a) Temporal validation: the models are fitted on the first training window and the checkpoint is selected on the second, which lies later in every student's history; the test set is unchanged. Because only half of the training examples are used for fitting, absolute values are slightly lower than in the main protocol. (b) FedProx run with the local configuration selected for FedAvg (SGD), compared with FedAvg and with the Adam configurations selected by the FedProx searches; values below 0.1\% of rounds correspond to the first three rounds, before training takes effect. Data: own experiments.}
\label{tab:robustness}
\begin{tblr}{
    colspec = {X[3,l] X[1.2,c] X[1.2,c] X[1.2,c] X[1.1,c]},
    cells   = {font=\small},
    rowsep  = 1pt,
    row{2,12}  = {font=\small\bfseries},
    }
\toprule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tblr}
\end{table*}
""")
with open(os.path.join(OUT_DIR, "latex", "tab_robustness.tex"), "w") as fh:
    fh.write(tex)
print("robustness table written")
