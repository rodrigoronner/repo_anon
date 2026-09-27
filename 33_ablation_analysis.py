"""Step 33 - Analysis of the ablation study (24_ablation.py): summary JSON, report section and LaTeX table.

The unablated reference of every condition is the corresponding main run (steps 21-22),
which shares seeds and, for federated conditions, the client sample of every round.
"""

import importlib
import os

import numpy as np
import pandas as pd

from common import dump, read
from config import EVAL_DIR, OUT_DIR

an = importlib.import_module("30_analysis")
TEX_DIR = os.path.join(OUT_DIR, "latex")
LATE = (501, 1000)

AB = read(os.path.join(EVAL_DIR, "ablation.json"))
ab_runs = pd.DataFrame(AB["runs"])
ab_rounds = pd.read_csv(os.path.join(EVAL_DIR, "ablation_rounds.csv.gz"))
main_rounds = pd.read_csv(os.path.join(EVAL_DIR, "fl_rounds.csv.gz"))

MAIN = {"cdnn": "cdnn_tuned", "FedAvg": "fl_FedAvg", "FedProx_mu1.0": "fl_FedProx_mu1.0",
        "FedProx_muTuned": "fl_FedProx_muTuned"}
PRELIM = "fl_FedAvg_v1protocol"


def runs_of(name):
    """Sorted-by-seed list of per-seed results (main run names or ablation condition names)."""
    if name in MAIN.values() or name == PRELIM:
        rs = read(os.path.join(EVAL_DIR, f"{name}.json"))["runs"]
        return sorted([(r["seed"], r["selected"]) for r in rs])
    sub = ab_runs[ab_runs["condition"] == name]
    return sorted(zip(sub["seed"], sub["result"]))


def metric(name, view="at_0.5", m="roc_auc"):
    return [r[view][m] for _, r in runs_of(name)]


def late(rounds, cond):
    g = rounds[(rounds["condition"] == cond) & rounds["round"].between(*LATE)]
    per = g.groupby("seed")["test_roc_auc"]
    below = rounds[rounds["condition"] == cond].groupby("seed")["test_roc_auc"].apply(lambda x: (x < 0.5).mean())
    return {"late_mean_auc": an.ms(per.mean()), "late_sd_auc": an.ms(per.std(ddof=1)),
            "lag1": an.ms([an.lag1(x.sort_values("round")["test_roc_auc"]) for _, x in g.groupby("seed")]),
            "share_rounds_auc_below_0.5": an.ms(below)}


def summary(name):
    return {k: an.ms(metric(name, v, m)) for k, (v, m) in {
        "auc": ("at_0.5", "roc_auc"), "balacc_0.5": ("at_0.5", "balanced_accuracy"),
        "balacc_cal": ("calibrated", "balanced_accuracy"), "pos_rate": ("at_0.5", "positive_rate")}.items()}


out = {"inputs": {}, "emb_decay": {}}

# A. input blocks: federation cost per variant, and effect of removing each block ---------------------
variants = ["all", "no_user", "no_skill", "features_only", "embeddings_only"]
for v in variants:
    c = MAIN["cdnn"] if v == "all" else f"cdnn_{v}"
    f = MAIN["FedAvg"] if v == "all" else f"FedAvg_{v}"
    out["inputs"][v] = {"centralized": summary(c), "federated": summary(f),
                        "federation_cost_auc": an.welch(metric(c), metric(f)),
                        "federation_cost_balacc_cal": an.welch(metric(c, "calibrated", "balanced_accuracy"),
                                                               metric(f, "calibrated", "balanced_accuracy"))}
    if v != "all":
        out["inputs"][v]["effect_vs_full_centralized"] = an.welch(metric(c), metric(MAIN["cdnn"]))
        out["inputs"][v]["effect_vs_full_federated_paired"] = an.paired(metric(f), metric(MAIN["FedAvg"]))

# B. weight decay restricted to the MLP --------------------------------------------------------------
for base, ref in MAIN.items():
    name = f"{base}_noEmbDecay"
    row = {"with_decay": summary(ref), "without_emb_decay": summary(name)}
    pair = an.paired if base != "cdnn" else an.welch
    row["delta_auc"] = pair(metric(name), metric(ref))
    row["delta_balacc_0.5"] = pair(metric(name, "at_0.5", "balanced_accuracy"), metric(ref, "at_0.5", "balanced_accuracy"))
    if base != "cdnn":
        row["late_with_decay"] = late(main_rounds, base)
        row["late_without_emb_decay"] = late(ab_rounds, name)
        sel = ab_runs[ab_runs["condition"] == name].sort_values("seed")["selected"].tolist()
        row["selected_round_without_emb_decay"] = sel
    out["emb_decay"][base] = row

# C. preliminary configuration (no weight decay) with the features only --------------------------------
pv = "FedAvg_v1protocol_features_only"
out["preliminary"] = {"full": summary(PRELIM), "features_only": summary(pv),
                      "delta_auc_paired": an.paired(metric(pv), metric(PRELIM)),
                      "selected_round_features_only": ab_runs[ab_runs["condition"] == pv].sort_values("seed")["selected"].tolist()}

dump(out, os.path.join(OUT_DIR, "ablation_analysis.json"))


# report ------------------------------------------------------------------------------------------------
def f(x, d=3):
    return f"{x[0]:.{d}f} ± {x[1]:.{d}f}"


def ci(w, d=3):
    return f"{w['diff']:+.{d}f} [{w['ci95'][0]:+.{d}f}, {w['ci95'][1]:+.{d}f}]"


lines = ["# Ablation study", "", "## A. Input blocks (tuned configurations, not re-tuned)", "",
         "| Inputs | cDNN AUC | FedAvg AUC | federation cost AUC (Welch) | cost BalAcc cal. | FedAvg BalAcc cal. |",
         "|---|---|---|---|---|---|"]
for v in variants:
    r = out["inputs"][v]
    lines.append(f"| {v} | {f(r['centralized']['auc'])} | {f(r['federated']['auc'])} | {ci(r['federation_cost_auc'])} "
                 f"| {ci(r['federation_cost_balacc_cal'])} | {f(r['federated']['balacc_cal'])} |")
lines += ["", "Effect of removing a block (ablated − full):", ""]
for v in variants[1:]:
    r = out["inputs"][v]
    lines.append(f"- {v}: centralized {ci(r['effect_vs_full_centralized'])}; "
                 f"FedAvg (paired) {ci(r['effect_vs_full_federated_paired'])}")
lines += ["", "## B. Weight decay restricted to the MLP", "",
          "| Condition | AUC with | AUC without | Δ AUC | BalAcc@0.5 with | without | PosRate with | without "
          "| late SD with | late SD without | lag-1 with | without | AUC<0.5 share with | without |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for base, r in out["emb_decay"].items():
    lw, lo = r.get("late_with_decay"), r.get("late_without_emb_decay")
    tail = (f"| {f(lw['late_sd_auc'], 4)} | {f(lo['late_sd_auc'], 4)} | {f(lw['lag1'], 2)} | {f(lo['lag1'], 2)} "
            f"| {f(lw['share_rounds_auc_below_0.5'], 2)} | {f(lo['share_rounds_auc_below_0.5'], 2)} |") if lw else "| – | – | – | – | – | – |"
    lines.append(f"| {base} | {f(r['with_decay']['auc'])} | {f(r['without_emb_decay']['auc'])} | {ci(r['delta_auc'])} "
                 f"| {f(r['with_decay']['balacc_0.5'])} | {f(r['without_emb_decay']['balacc_0.5'])} "
                 f"| {f(r['with_decay']['pos_rate'])} | {f(r['without_emb_decay']['pos_rate'])} {tail}")
    if "selected_round_without_emb_decay" in r:
        lines.append(f"|  selected rounds without decay: {r['selected_round_without_emb_decay']} |||||||||||||| ")
pr = out["preliminary"]
lines += ["", "## C. Preliminary configuration (no weight decay), features only", "",
          f"- full: AUC {f(pr['full']['auc'])}, BalAcc@0.5 {f(pr['full']['balacc_0.5'])}, "
          f"BalAcc cal {f(pr['full']['balacc_cal'])}, pos {f(pr['full']['pos_rate'])}",
          f"- features only: AUC {f(pr['features_only']['auc'])}, BalAcc@0.5 {f(pr['features_only']['balacc_0.5'])}, "
          f"BalAcc cal {f(pr['features_only']['balacc_cal'])}, pos {f(pr['features_only']['pos_rate'])}",
          f"- paired delta AUC {ci(pr['delta_auc_paired'])}; selected rounds {pr['selected_round_features_only']}"]
with open(os.path.join(OUT_DIR, "REPORT_ablation.md"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))


# LaTeX table -------------------------------------------------------------------------------------------
def pmt(x, d=3):
    return f"{x[0]:.{d}f} $\\pm$ {x[1]:.{d}f}"


def cit(w, d=3):
    n = lambda v: (f"{v:.{d}f}".replace("-", "$-$") if round(v, d) != 0 else f"{0:.{d}f}")
    return f"{n(w['diff'])} [{n(w['ci95'][0])}, {n(w['ci95'][1])}]"


VLAB = {"all": "Student emb.\\ + skill emb.\\ + features (full)", "no_user": "Skill emb.\\ + features",
        "no_skill": "Student emb.\\ + features", "features_only": "Features only", "embeddings_only": "Embeddings only"}
BLAB = {"cdnn": "DNN, centralized", "FedAvg": "FedAvg", "FedProx_mu1.0": r"FedProx $\mu=1.0$",
        "FedProx_muTuned": r"FedProx, $\mu$ tuned"}
rows = [r"\SetCell[c=5]{l}\textit{(a) Input blocks} & & & & \\",
        r"Inputs & Centralized DNN & FedAvg & Federation cost [95\% CI] & \\"]
for v in variants:
    r = out["inputs"][v]
    rows.append(f"{VLAB[v]} & {pmt(r['centralized']['auc'])} & {pmt(r['federated']['auc'])} & "
                f"{cit(r['federation_cost_auc'])} & \\\\")
rows.append(f"FedAvg, preliminary configuration: full / features only & -- & "
            f"{out['preliminary']['full']['auc'][0]:.3f} / {out['preliminary']['features_only']['auc'][0]:.3f} & -- & \\\\")
rows += [r"\midrule", r"\SetCell[c=5]{l}\textit{(b) Weight decay on the embedding tables} & & & & \\",
         r"Condition & AUC, with & AUC, without & SD rounds 501--1{,}000, with / without & Rounds with AUC $<$ 0.5, with / without \\"]
for base, r in out["emb_decay"].items():
    lw, lo = r.get("late_with_decay"), r.get("late_without_emb_decay")
    sd = f"{lw['late_sd_auc'][0]:.3f} / {lo['late_sd_auc'][0]:.3f}" if lw else "--"
    below = (f"{100 * lw['share_rounds_auc_below_0.5'][0]:.0f}\\% / {100 * lo['share_rounds_auc_below_0.5'][0]:.0f}\\%"
             if lw else "--")
    rows.append(f"{BLAB[base]} & {pmt(r['with_decay']['auc'])} & {pmt(r['without_emb_decay']['auc'])} & {sd} & {below} \\\\")
tex = (r"""\begin{table*}[ht]
\centering
\caption{Ablation study: test ROC AUC (mean $\pm$ SD over 10 seeds) of the tuned configurations, not re-tuned. (a) Networks trained with subsets of the inputs; the federation cost is the Welch difference between the centralized DNN and FedAvg. (b) Tuned configurations trained with weight decay on all parameters (``with'', as in the main runs) or on the perceptron only (``without''); the last two columns describe the federated trajectories. Data: own experiments.}
\label{tab:ablation}
\begin{tblr}{
    colspec = {X[2.3,l] X[1.3,c] X[1.3,c] X[1.6,c] X[1.5,c]},
    cells   = {font=\small},
    row{2,10}  = {font=\small\bfseries},
    }
\toprule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tblr}
\end{table*}
""")
with open(os.path.join(TEX_DIR, "tab_ablation.tex"), "w") as fh:
    fh.write(tex)
print("ablation table written")
