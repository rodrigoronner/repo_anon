"""Step 31 - Manuscript figures (PDF + PNG) from the evaluation and tuning outputs."""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import read  # noqa: E402
from config import EVAL_DIR, FIG_DIR, FL_STRATEGIES, OUT_DIR  # noqa: E402
from tuning_utils import storage  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)

INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
TUNED = list(FL_STRATEGIES)
LABEL = {"FedAvg": "FedAvg", "FedProx_mu0.1": r"FedProx $\mu$=0.1", "FedProx_mu0.5": r"FedProx $\mu$=0.5",
         "FedProx_mu1.0": r"FedProx $\mu$=1.0", "FedProx_muTuned": r"FedProx $\mu$ tuned",
         "FedAvg_v1protocol": "FedAvg, preliminary protocol"}
COLOR = {**dict(zip(TUNED, PALETTE)), "FedAvg_v1protocol": MUTED}
STYLE = {**dict(zip(TUNED, STYLES)), "FedAvg_v1protocol": (0, (2, 2))}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 1.4})

A = read(os.path.join(OUT_DIR, "analysis.json"))
S, T = A["summary"], A["trivial"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def ci(values):
    v = np.asarray(values, float)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


# Figure: convergence of the federated conditions ------------------------------------------------
rounds = pd.read_csv(os.path.join(EVAL_DIR, "fl_rounds.csv.gz"))
STABLE = ["FedAvg", "FedProx_mu0.1", "FedProx_mu0.5", "FedAvg_v1protocol"]
UNSTABLE = ["FedProx_mu1.0", "FedProx_muTuned"]


def curve(ax, cond, colname):
    piv = rounds[rounds["condition"] == cond].pivot(index="round", columns="seed", values=colname)
    m = piv.mean(axis=1)
    half = stats.t.ppf(0.975, piv.shape[1] - 1) * piv.std(axis=1, ddof=1) / np.sqrt(piv.shape[1])
    ax.plot(m.index, m, color=COLOR[cond], ls=STYLE[cond], lw=1.1, label=LABEL[cond])
    ax.fill_between(m.index, m - half, m + half, color=COLOR[cond], alpha=0.15, lw=0)


TRACE_SEED = 42


def trace(ax, cond, colname):
    """One seed's trajectory; the mean over seeds would hide the round-to-round oscillation."""
    r = rounds[(rounds["condition"] == cond) & (rounds["seed"] == TRACE_SEED)]
    ax.plot(r["round"], r[colname], color=COLOR[cond], ls="-", lw=0.5, alpha=0.9)
    run = next(x for x in read(os.path.join(EVAL_DIR, f"fl_{cond}.json"))["runs"] if x["seed"] == TRACE_SEED)
    sel = r[r["round"] == run["selected_round"]]
    ax.scatter(sel["round"], sel[colname], s=28, marker="D", color=COLOR[cond], edgecolors="white",
               linewidths=0.8, zorder=5)


fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
panels = [(axes[0, 0], STABLE, "test_roc_auc", "(a) Test ROC AUC, local SGD and preliminary protocol"),
          (axes[0, 1], ["FedAvg"] + UNSTABLE, "test_roc_auc", f"(b) Test ROC AUC, single run (seed {TRACE_SEED})"),
          (axes[1, 0], STABLE + UNSTABLE, "test_balanced_accuracy", "(c) Test balanced accuracy, threshold 0.5"),
          (axes[1, 1], STABLE + UNSTABLE, "test_positive_rate", "(d) Share of test pairs predicted positive")]
for ax, conds, colname, title in panels:
    for cond in conds:
        (trace if ax is axes[0, 1] else curve)(ax, cond, colname)
    if colname == "test_roc_auc":
        ax.axhline(S["cdnn_tuned"]["at_0.5.roc_auc"][0], color=INK, ls="--", lw=0.8, label="Centralized DNN / XGBoost")
        ax.axhline(T["student_history_mean"]["roc_auc"], color=MUTED, ls="-.", lw=0.8,
                   label="Student history mean (no model)")
        ax.set_ylim(0.45, 0.82)
        if ax is axes[0, 1]:
            ax.set_ylim(0.2, 0.82)
            ax.axhline(0.5, color=MUTED, ls=":", lw=0.8)
    if colname == "test_balanced_accuracy":
        ax.axhline(S["cdnn_tuned"]["at_0.5.balanced_accuracy"][0], color=INK, ls="--", lw=0.8)
    if colname == "test_positive_rate":
        ax.axhline(T["test_positive_rate"], color=MUTED, ls="-", lw=0.8, label="Observed positive rate")
    ax.set_title(title, loc="left", fontsize=7.5, color=INK)
for ax in axes[1]:
    ax.set_xlabel("Communication round")
handles, labels = [], []
for ax in axes.ravel():
    for h, lab in zip(*ax.get_legend_handles_labels()):
        if lab not in labels:
            handles.append(h)
            labels.append(lab)
handles.append(plt.Line2D([], [], marker="D", color=MUTED, ls="none", ms=5))
labels.append("Checkpoint selected on validation (panel b)")
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
save(fig, "fig_convergence")

# Figure: forest plot of test performance ------------------------------------------------------------
order = [("xgboost", "XGBoost (centralized)"), ("cdnn_tuned", "DNN (centralized)"),
         *[(f"fl_{s}", LABEL[s]) for s in TUNED],
         ("cdnn_v1", "DNN (centralized), preliminary config."), ("fl_FedAvg_v1protocol", LABEL["FedAvg_v1protocol"])]
runs = {k: read(os.path.join(EVAL_DIR, f"{k}.json"))["runs"] for k, _ in order}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for ax, (view, metric, title, ref) in zip(axes, [
        ("at_0.5", "roc_auc", "(a) Test ROC AUC", T["student_history_mean"]["roc_auc"]),
        ("calibrated", "balanced_accuracy", "(b) Test balanced accuracy, calibrated threshold",
         T["student_history_mean"]["balanced_accuracy"])]):
    for i, (k, _) in enumerate(order):
        m, h = ci([r["selected"][view][metric] for r in runs[k]])
        fed = k.startswith("fl_")
        ax.errorbar(m, i, xerr=h, fmt="o" if fed else "s", ms=4, color=INK if not fed else "#2a78d6",
                    mfc="white" if "v1" in k else None, capsize=2, lw=1)
    ax.axvline(ref, color=MUTED, ls="-.", lw=0.8)
    ax.text(ref, -0.75, " student history mean", fontsize=6.5, color=MUTED, va="center", ha="left")
    ax.set_title(title, loc="left", fontsize=8, color=INK)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([lab for _, lab in order])
axes[0].set_ylim(len(order) - 0.5, -1.1)
fig.tight_layout()
save(fig, "fig_forest")

# Figure: local optimizer in the federated searches ---------------------------------------------------
# (a) best validation AUC reached by each local optimizer in each search; (b) fANOVA importance.
TUNING = {s: A["tuning"][f"federated_{s}"] for s in TUNED}
ROWLAB = {s: LABEL[s] for s in TUNED}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.9), gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.95})
ys = np.arange(len(TUNED))
for y, s in zip(ys, TUNED):
    best = TUNING[s]["best_by_optimizer"]
    chosen = TUNING[s]["best_params"]["optimizer"]
    ax1.plot([best["sgd"], best["adam"]], [y, y], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
    for opt, col, mk in (("sgd", PALETTE[0], "o"), ("adam", PALETTE[1], "^")):
        ax1.scatter(best[opt], y, s=50, marker=mk, color=col, zorder=3,
                    edgecolors=INK if opt == chosen else "white", linewidths=1.0 if opt == chosen else 0.8)
ax1.set_yticks(ys)
ax1.set_yticklabels([ROWLAB[s] for s in TUNED], fontsize=8)
ax1.set_ylim(len(TUNED) - 0.5, -0.5)
ax1.set_xlim(0.760, 0.790)
ax1.set_xlabel("Best validation AUC in the search", fontsize=8)
ax1.grid(axis="y", visible=False)
ax1.set_title("(a) Best trial per local optimizer", loc="left", fontsize=8, color=INK)
ax1.scatter([], [], marker="o", color=PALETTE[0], s=30, label="Local SGD")
ax1.scatter([], [], marker="^", color=PALETTE[1], s=30, label="Local Adam")
ax1.scatter([], [], marker="o", facecolors="white", edgecolors=INK, s=30, label="Selected")
ax1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.32, -0.38), ncol=3, handletextpad=0.2, columnspacing=0.8)

params = [("local_epochs", "Local\nepochs"), ("batch_size", "Batch\nsize"), ("weight_decay", "Weight\ndecay"),
          ("optimizer", "Optimizer"), ("mu", "$\\mu$")]
M = np.full((len(TUNED), len(params)), np.nan)
for i, s in enumerate(TUNED):
    imp = TUNING[s]["param_importances"]
    for j, (k, _) in enumerate(params):
        if k in imp:
            M[i, j] = imp[k]
cmap = matplotlib.colormaps["Blues"].copy()
cmap.set_bad("#f2f1ee")
# pcolormesh keeps the heatmap vector in the PDF (imshow would embed a bitmap)
ax2.pcolormesh(np.arange(M.shape[1] + 1) - 0.5, np.arange(M.shape[0] + 1) - 0.5, np.ma.masked_invalid(M),
               cmap=cmap, vmin=0, vmax=0.8, edgecolors="white", linewidth=1.0)
ax2.set_ylim(M.shape[0] - 0.5, -0.5)
for i in range(M.shape[0]):
    top = np.nanargmax(M[i])
    for j in range(M.shape[1]):
        if np.isnan(M[i, j]):
            ax2.text(j, i, "not\nsearched", ha="center", va="center", fontsize=6.5, color=MUTED)
        else:
            ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                     color="white" if M[i, j] > 0.45 else INK, fontweight="bold" if j == top else "normal")
ax2.set_xticks(range(len(params)))
ax2.set_xticklabels([lab for _, lab in params], fontsize=7.5)
ax2.set_yticks(ys)
ax2.set_yticklabels([ROWLAB[s] for s in TUNED], fontsize=8)
ax2.tick_params(length=0)
ax2.grid(False)
for sp in ax2.spines.values():
    sp.set_visible(False)
ax2.set_title("(b) fANOVA importance (bold: highest in the search)", loc="left", fontsize=8, color=INK)
save(fig, "fig_optuna_optimizer")
print("figures written to", FIG_DIR)
