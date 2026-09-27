"""Step 30 - Statistical analysis of the evaluation runs; writes analysis.json and REPORT.md."""

import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import dump, read
from config import EVAL_DIR, FL_STRATEGIES, N_ROUNDS, OUT_DIR, TUNE_DIR

AUC_MARGIN = 0.01
LATE = (N_ROUNDS // 2 + 1, N_ROUNDS)
BUDGETS = (100, 250, 500, 1000)
TUNED = list(FL_STRATEGIES)
MODELS = {"xgboost": "xgboost.json", "cdnn_tuned": "cdnn_tuned.json", "cdnn_v1": "cdnn_v1.json",
          **{f"fl_{s}": f"fl_{s}.json" for s in TUNED}, "fl_FedAvg_v1protocol": "fl_FedAvg_v1protocol.json"}


def ms(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1))]


def col(runs, view, metric, key="selected"):
    return np.array([r[key][view][metric] for r in runs], float)


def summarize(runs):
    out = {}
    for view in ("at_0.5", "calibrated"):
        for m in ("roc_auc", "balanced_accuracy", "f1_score", "precision", "recall", "accuracy",
                  "positive_rate", "threshold"):
            out[f"{view}.{m}"] = ms(col(runs, view, m))
    for part in ("seen_pairs", "new_pairs"):
        out[f"{part}.roc_auc"] = ms([r["selected"][part]["at_0.5"]["roc_auc"] for r in runs])
    out["val_auc"] = ms([r["selected"]["val_auc"] for r in runs])
    return out


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    t = stats.t.ppf(0.975, df)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"diff": float(d), "ci95": [float(d - t * se), float(d + t * se)],
            "p": float(stats.ttest_ind(a, b, equal_var=False).pvalue), "cohens_d": float(d / pooled)}


def paired(a, b, margin=None):
    d = np.asarray(a, float) - np.asarray(b, float)
    n, sd = len(d), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    out = {"diff": float(d.mean()), "ci95": [float(d.mean() - t * se), float(d.mean() + t * se)],
           "p": float(stats.ttest_rel(a, b).pvalue) if sd > 0 else 1.0,
           "dz": float(d.mean() / sd) if sd > 0 else 0.0}
    if margin is not None and sd > 0:
        out["tost_p"] = float(max(stats.t.sf((d.mean() + margin) / se, n - 1),
                                  stats.t.cdf((d.mean() - margin) / se, n - 1)))
    return out


def holm(ps):
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(ps) - i) * ps[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


def lag1(x):
    x = np.asarray(x, float) - np.mean(x)
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


def tuning_summary():
    out = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(TUNE_DIR), "*_best.json"))):
        best = read(path)
        name = best["study"]
        trials = pd.read_csv(os.path.join(TUNE_DIR, f"{name}_trials.csv"))
        done = trials[trials["state"] == "COMPLETE"]
        entry = {k: best[k] for k in ("best_value", "best_params", "best_user_attrs", "n_trials",
                                      "n_complete", "n_pruned", "n_failed", "param_importances")}
        if "params_optimizer" in trials:
            entry["best_by_optimizer"] = {
                o: float(g["value"].max()) for o, g in trials.groupby("params_optimizer")
                if g["value"].notna().any()}
            entry["complete_by_optimizer"] = done["params_optimizer"].value_counts().to_dict()
        if "params_id_encoding" in trials:
            entry["best_by_id_encoding"] = {
                o: float(g["value"].max()) for o, g in trials.groupby("params_id_encoding")}
        out[name] = entry
    return out


def main():
    runs = {k: read(os.path.join(EVAL_DIR, f))["runs"] for k, f in MODELS.items()}
    A = {"trivial": read(os.path.join(EVAL_DIR, "trivial_baselines.json")),
         "meta": read(os.path.join(os.path.dirname(OUT_DIR), "data", "processed_meta.json")),
         "configs": {k: {kk: vv for kk, vv in read(os.path.join(EVAL_DIR, f)).items() if kk != "runs"}
                     for k, f in MODELS.items()},
         "summary": {k: summarize(v) for k, v in runs.items()},
         "selected_checkpoint": {k: [r.get("selected_round", r.get("selected_epoch", r.get("best_iteration")))
                                     for r in v] for k, v in runs.items()},
         "tuning": tuning_summary()}

    dec = {}
    for view, m in (("at_0.5", "roc_auc"), ("calibrated", "balanced_accuracy"), ("at_0.5", "balanced_accuracy")):
        key = f"{view}.{m}"
        dec[key] = {"architecture_effect_xgb_minus_cdnn": welch(col(runs["xgboost"], view, m),
                                                                col(runs["cdnn_tuned"], view, m)),
                    "federation_cost_v1_config": welch(col(runs["cdnn_v1"], view, m),
                                                       col(runs["fl_FedAvg_v1protocol"], view, m))}
        for s in TUNED:
            dec[key][f"federation_cost_{s}"] = welch(col(runs["cdnn_tuned"], view, m),
                                                     col(runs[f"fl_{s}"], view, m))
        dec[key]["fl_FedAvg_minus_xgboost"] = welch(col(runs["fl_FedAvg"], view, m),
                                                    col(runs["xgboost"], view, m))
        dec[key]["tuned_minus_v1_FedAvg_paired"] = paired(col(runs["fl_FedAvg"], view, m),
                                                          col(runs["fl_FedAvg_v1protocol"], view, m))
    A["decomposition"] = dec

    rounds = pd.read_csv(os.path.join(EVAL_DIR, "fl_rounds.csv.gz"))
    late = rounds[(rounds["round"] >= LATE[0]) & (rounds["round"] <= LATE[1])]
    per_seed = late.groupby(["condition", "seed"]).agg(
        late_auc_mean=("test_roc_auc", "mean"), late_auc_sd=("test_roc_auc", "std"),
        late_val_auc_sd=("val_auc", "std"), late_balacc_mean=("test_balanced_accuracy", "mean"),
        late_f1_sd=("test_f1_score", "std"), late_pos_rate=("test_positive_rate", "mean")).reset_index()
    collapse = rounds.groupby(["condition", "seed"])["test_positive_rate"].apply(
        lambda x: float((x >= 0.99).mean())).rename("collapsed_round_share").reset_index()
    per_seed = per_seed.merge(collapse, on=["condition", "seed"])
    sel = pd.DataFrame([{"condition": r["condition"], "seed": r["seed"],
                         "sel_auc": r["selected"]["at_0.5"]["roc_auc"],
                         "sel_balacc_cal": r["selected"]["calibrated"]["balanced_accuracy"]}
                        for k in runs if k.startswith("fl_") for r in runs[k]])
    per_seed = per_seed.merge(sel, on=["condition", "seed"]).sort_values(["condition", "seed"])
    per_seed.to_csv(os.path.join(EVAL_DIR, "fl_per_seed.csv"), index=False)
    A["late_window"] = LATE
    A["fl_late"] = {c: {m: ms(g[m]) for m in per_seed.columns if m not in ("condition", "seed")}
                    for c, g in per_seed.groupby("condition")}
    A["lag1_autocorrelation_test_auc"] = {
        c: ms([lag1(g.sort_values("round")["test_roc_auc"]) for _, g in gc.groupby("seed")])
        for c, gc in late.groupby("condition")}

    budget = []
    for (c, sd), g in rounds.groupby(["condition", "seed"]):
        g = g.sort_values("round")
        for R in BUDGETS:
            w = g[g["round"] <= R]
            b = w.loc[w["val_auc"].idxmax()]
            budget.append({"condition": c, "seed": sd, "budget": R, "round": int(b["round"]),
                           "test_auc": float(b["test_roc_auc"]),
                           "test_balacc": float(b["test_balanced_accuracy"]),
                           "positive_rate": float(b["test_positive_rate"])})
    budget = pd.DataFrame(budget)
    budget.to_csv(os.path.join(EVAL_DIR, "fl_budget_sensitivity.csv"), index=False)
    A["budget_sensitivity"] = {c: {str(R): {m: ms(g2[m]) for m in ("test_auc", "test_balacc", "positive_rate")}
                                       for R, g2 in g.groupby("budget")}
                               for c, g in budget.groupby("condition")}

    comps = {}
    for metric, margin in (("sel_auc", AUC_MARGIN), ("sel_balacc_cal", AUC_MARGIN),
                           ("late_auc_mean", AUC_MARGIN), ("late_auc_sd", None), ("late_val_auc_sd", None)):
        wide = per_seed.pivot(index="seed", columns="condition", values=metric)[TUNED]
        pairs = {f"{a} vs {b}": paired(wide[a], wide[b], margin) for a, b in itertools.combinations(TUNED, 2)}
        for v, pa in zip(pairs.values(), holm([v["p"] for v in pairs.values()])):
            v["p_holm"] = pa
        fr = stats.friedmanchisquare(*[wide[s] for s in TUNED])
        comps[metric] = {"friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue),
                         "means": {s: ms(wide[s]) for s in TUNED}, "pairs": pairs}
    A["strategy_comparisons"] = comps

    dump(A, os.path.join(OUT_DIR, "analysis.json"))
    report(A)


def f(x, d=3):
    return f"{x[0]:.{d}f} ± {x[1]:.{d}f}"


def report(A):
    S, T = A["summary"], A["trivial"]
    L = ["# Final pipeline — results", "",
         f"Test examples {T['n_test']}, positive rate {T['test_positive_rate']:.3f}", "",
         "| Model | AUC | BalAcc@0.5 | BalAcc cal. | F1@0.5 | PosRate@0.5 | AUC seen | AUC new | val AUC |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n in ("always_positive", "stratified_random", "student_history_mean", "skill_history_mean"):
        m = T[n]
        L.append(f"| {n} | {m['roc_auc']:.3f} | {m['balanced_accuracy']:.3f} | – | {m['f1_score']:.3f} | "
                 f"{m['positive_rate']:.3f} | – | – | – |")
    for n, m in S.items():
        L.append(f"| {n} | {f(m['at_0.5.roc_auc'])} | {f(m['at_0.5.balanced_accuracy'])} | "
                 f"{f(m['calibrated.balanced_accuracy'])} | {f(m['at_0.5.f1_score'])} | "
                 f"{f(m['at_0.5.positive_rate'])} | {f(m['seen_pairs.roc_auc'])} | {f(m['new_pairs.roc_auc'])} | "
                 f"{f(m['val_auc'])} |")
    L += ["", "## Decomposition (Welch, 95% CI)", ""]
    for k, v in A["decomposition"].items():
        for n, r in v.items():
            L.append(f"- {k} | {n}: {r['diff']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p={r['p']:.3g} "
                     f"{'d' if 'cohens_d' in r else 'dz'}={r.get('cohens_d', r.get('dz')):.2f}")
    L += ["", f"## Federated, rounds {A['late_window'][0]}–{A['late_window'][1]}", "",
          "| Condition | mean AUC | SD AUC | SD val AUC | BalAcc | PosRate | collapsed share (all rounds) | lag-1 |",
          "|---|---|---|---|---|---|---|---|"]
    for c, m in A["fl_late"].items():
        L.append(f"| {c} | {f(m['late_auc_mean'])} | {f(m['late_auc_sd'], 4)} | {f(m['late_val_auc_sd'], 4)} | "
                 f"{f(m['late_balacc_mean'])} | {f(m['late_pos_rate'])} | {f(m['collapsed_round_share'], 2)} | "
                 f"{f(A['lag1_autocorrelation_test_auc'][c], 2)} |")
    L += ["", "## Communication-budget sensitivity (checkpoint chosen on validation within the budget; "
          "configurations tuned for 1000 rounds)", "",
          "| Condition | " + " | ".join(f"AUC @{b}" for b in BUDGETS) + " | " +
          " | ".join(f"BalAcc @{b}" for b in BUDGETS) + " |", "|---" * (1 + 2 * len(BUDGETS)) + "|"]
    for c, d in A["budget_sensitivity"].items():
        L.append(f"| {c} | " + " | ".join(f(d[str(b)]["test_auc"]) for b in BUDGETS) + " | " +
                 " | ".join(f(d[str(b)]["test_balacc"]) for b in BUDGETS) + " |")
    L += ["", "## Strategy comparisons (paired by seed)", ""]
    for metric, c in A["strategy_comparisons"].items():
        L += [f"### {metric}: Friedman χ²={c['friedman_chi2']:.2f}, p={c['friedman_p']:.3g}", "",
              "| Pair | Δ | 95% CI | p | p Holm | dz | TOST p |", "|---|---|---|---|---|---|---|"]
        for p, r in c["pairs"].items():
            L.append(f"| {p} | {r['diff']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['p']:.3g} | "
                     f"{r['p_holm']:.3g} | {r['dz']:+.2f} | {r.get('tost_p', float('nan')):.3g} |")
        L.append("")
    L += ["## Tuning", ""]
    for n, t in A["tuning"].items():
        L.append(f"- **{n}**: best val {t['best_value']:.4f}; trials {t['n_trials']} "
                 f"(complete {t['n_complete']}, pruned {t['n_pruned']}, failed {t['n_failed']}); "
                 f"params {t['best_params']}")
        for k in ("best_by_optimizer", "best_by_id_encoding"):
            if k in t:
                L.append(f"  - {k}: {t[k]}")
        L.append(f"  - importances: {t['param_importances']}")
    L += ["", "Selected checkpoints: " + "; ".join(f"{k} {v}" for k, v in A["selected_checkpoint"].items())]
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
