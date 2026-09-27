"""Step 32 - LaTeX (tabularray) tables for the manuscript, generated from analysis.json."""

import os

from common import read
from config import FL_STRATEGIES, OUT_DIR

TEX_DIR = os.path.join(OUT_DIR, "latex")
os.makedirs(TEX_DIR, exist_ok=True)
A = read(os.path.join(OUT_DIR, "analysis.json"))
S, T, M, TU = A["summary"], A["trivial"], A["meta"], A["tuning"]
TUNED = list(FL_STRATEGIES)
SHORT = {"FedAvg": "FedAvg", "FedProx_mu0.1": r"$\mu=0.1$", "FedProx_mu0.5": r"$\mu=0.5$",
         "FedProx_mu1.0": r"$\mu=1.0$", "FedProx_muTuned": r"$\mu$ tuned"}
NAME = {"FedAvg": "FedAvg", "FedProx_mu0.1": r"FedProx $\mu=0.1$", "FedProx_mu0.5": r"FedProx $\mu=0.5$",
        "FedProx_mu1.0": r"FedProx $\mu=1.0$", "FedProx_muTuned": r"FedProx, $\mu$ tuned",
        "FedAvg_v1protocol": "FedAvg, preliminary config."}


def pm(x, d=3):
    return f"{x[0]:.{d}f} $\\pm$ {x[1]:.{d}f}"


def num(x, d=3):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if x < 0 else s


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ptxt(p):
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


def table(label, caption, colspec, header, rows, star=False, note=None, short=None):
    env = "table*" if star else "table"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    cap = f"[{short}]" if short else ""
    tex = (f"\\begin{{{env}}}[ht]\n\\centering\n\\caption{cap}{{{caption}}}\n\\label{{{label}}}\n"
           f"\\begin{{tblr}}{{\n    colspec = {{{colspec}}},\n    cells   = {{font=\\small}},\n"
           f"    row{{1}}  = {{font=\\small\\bfseries}},\n    }}\n\\toprule\n{' & '.join(header)} \\\\\n"
           f"\\midrule\n{body}\n\\bottomrule\n\\end{{tblr}}\n")
    if note:
        tex += f"\\par\\smallskip{{\\footnotesize {note}}}\n"
    tex += f"\\end{{{env}}}\n"
    with open(os.path.join(TEX_DIR, f"{label.replace(':', '_')}.tex"), "w") as fh:
        fh.write(tex)


# Cohort -------------------------------------------------------------------------------------------
pr, fe = M["positive_rate"], M["fit_examples_per_client"]
table("tab:cohort", "Cohort and example construction after filtering. Data: own elaboration from ASSISTments 2009--2010.",
      "p{0.58\\linewidth} X[r]", ["Quantity", "Value"], [
          ["Interactions (raw rows / valid / after filtering)",
           f"{M['raw_rows']:,} / {M['valid_rows']:,} / {M['filtered_interactions']:,}"],
          ["Students (federated clients) / skills", f"{M['num_users']:,} / {M['num_skills']}"],
          ["Training examples: fit / validation", f"{M['n_fit_examples']:,} / {M['n_val_examples']:,}"],
          ["Test examples", f"{M['n_test_examples']:,}"],
          ["Positive rate: fit / validation / test",
           f"{pr['fit']:.3f} / {pr['val']:.3f} / {pr['test']:.3f}"],
          ["Fit examples per client: min / median / max", f"{fe['min']} / {fe['median']:.0f} / {fe['max']}"],
          ["Test pairs whose skill appears in the student's history", f"{M['seen_pair_share']['test']:.3f}"],
          [r"Spearman $\rho$ between \texttt{order\_id} and \texttt{opportunity} (mean)",
           f"{M['ordering_validation']['mean_spearman']:.3f} ({M['ordering_validation']['n_pairs']:,} pairs)"],
      ])

# Search spaces and selections ----------------------------------------------------------------------
x, c = TU["xgboost"], TU["centralized_dnn"]
xp, cp = x["best_params"], c["best_params"]
rows = [
    [r"\SetCell[c=3]{l}\textit{XGBoost} (" + f"{x['n_trials']} trials)", "", ""],
    ["ID encoding", "numeric, categorical, none", xp["id_encoding"]],
    ["Max. depth; learning rate", r"2--10; $10^{-2}$--0.3 (log)", f"{xp['max_depth']}; {xp['learning_rate']:.3f}"],
    ["Min. child weight; subsample; col. sample", "1--50 (log); 0.5--1; 0.5--1",
     f"{xp['min_child_weight']:.2f}; {xp['subsample']:.2f}; {xp['colsample_bytree']:.2f}"],
    [r"$\lambda$; $\alpha$; $\gamma$", r"$10^{-3}$--10 (log); $10^{-3}$--10 (log); 0--5",
     f"{xp['reg_lambda']:.3f}; {xp['reg_alpha']:.3f}; {xp['gamma']:.2f}"],
    [r"\SetCell[c=3]{l}\textit{RecommenderNet, centralized} (" + f"{c['n_trials']} trials)", "", ""],
    ["Embedding; hidden 1; hidden 2", "\\{2,4,8,16,32\\}; \\{16,\\ldots,256\\}; \\{8,\\ldots,128\\}",
     f"{cp['emb_dim']}; {cp['h1']}; {cp['h2']}"],
    ["Dropout; weight decay", r"0--0.7; $10^{-7}$--$10^{-1}$ (log)", f"{cp['dropout']:.1f}; {sci(cp['weight_decay'])}"],
    ["Optimizer; learning rate", r"Adam $10^{-4}$--$3\cdot10^{-2}$, SGD $10^{-2}$--2 (log)",
     f"{cp['optimizer'].upper() if cp['optimizer'] == 'sgd' else 'Adam'}; "
     f"{cp.get('lr_adam', cp.get('lr_sgd')):.4f}"],
    ["Batch size", "\\{32,64,128,256\\}", f"{cp['batch_size']}"],
    [r"\SetCell[c=3]{l}\textit{Federated training} (" + f"{TU['federated_FedAvg']['n_trials']} trials per strategy; "
     r"architecture frozen from the centralized search)", "", ""],
]
for s in TUNED:
    p = TU[f"federated_{s}"]["best_params"]
    lr = p.get("lr_adam", p.get("lr_sgd"))
    opt = "Adam" if p["optimizer"] == "adam" else "SGD"
    extra = f"; $\\mu$={p['mu']:.3f}" if "mu" in p else ""
    rows.append([NAME[s], r"\SetCell[c=2]{l}" + f"{opt}, lr {lr:.4f}, $E$={p['local_epochs']}, "
                 f"$B$={p['batch_size']}, weight decay {sci(p['weight_decay'])}{extra}", ""])
EXTRA_STUDIES = [("federated_FedAvg_rowwise", "FedAvg, row-wise aggregation"),
                 ("federated_LogReg", "Logistic regression, FedAvg"),
                 ("centralized_logreg", "Logistic regression, centralized")]
if all(os.path.exists(os.path.join(OUT_DIR, "tuning", f"{n}_best.json")) for n, _ in EXTRA_STUDIES):
    rows.append([r"\SetCell[c=3]{l}\textit{Added conditions} (search spaces as above)", "", ""])
    for n, lab in EXTRA_STUDIES:
        best = read(os.path.join(OUT_DIR, "tuning", f"{n}_best.json"))
        p = best["best_params"]
        lab = f"{lab} ({best['n_trials']} trials)"
        opt = "Adam" if p["optimizer"] == "adam" else "SGD"
        parts = [f"{opt}, lr {p.get('lr_adam', p.get('lr_sgd')):.4f}"]
        if "local_epochs" in p:
            parts.append(f"$E$={p['local_epochs']}")
        parts += [f"$B$={p['batch_size']}", f"weight decay {sci(p['weight_decay'])}"]
        rows.append([lab, r"\SetCell[c=2]{l}" + ", ".join(parts), ""])
table("tab:search", "Optuna search spaces (TPE sampler, validation AUC objective) and selected configurations. "
      "Federated space: local optimizer \\{SGD, Adam\\} with the learning-rate ranges of the centralized search "
      "(Adam up to $10^{-1}$), local epochs $E\\in\\{1,2,5,10\\}$, batch size $B\\in\\{16,32,64\\}$, weight decay "
      "$10^{-7}$--$10^{-1}$, and $\\mu\\in[10^{-3},1]$ (log) for the tuned-$\\mu$ condition. Data: own experiments.",
      "p{0.40\\textwidth} p{0.31\\textwidth} X", ["Hyperparameter", "Search space", "Selected"], rows, star=True)

# Main results --------------------------------------------------------------------------------------
rows = []
for n, lab in (("always_positive", "Always positive"), ("student_history_mean", "Student history mean"),
               ("skill_history_mean", "Skill history mean")):
    m = T[n]
    rows.append([lab, f"{m['roc_auc']:.3f}", f"{m['balanced_accuracy']:.3f}", "--", f"{m['f1_score']:.3f}",
                 f"{m['positive_rate']:.3f}"])
for k, lab in [("xgboost", "XGBoost, centralized"), ("cdnn_tuned", "DNN, centralized"),
               *[(f"fl_{s}", NAME[s]) for s in TUNED],
               ("cdnn_v1", "DNN, centralized, preliminary config."), ("fl_FedAvg_v1protocol", NAME["FedAvg_v1protocol"])]:
    m = S[k]
    rows.append([lab, pm(m["at_0.5.roc_auc"]), pm(m["at_0.5.balanced_accuracy"]),
                 pm(m["calibrated.balanced_accuracy"]), pm(m["at_0.5.f1_score"]), pm(m["at_0.5.positive_rate"])])
table("tab:main", "Test-set performance (mean $\\pm$ SD over 10 seeds). Checkpoint and calibrated threshold are "
      "selected on validation data. The first three rows are learning-free references. Data: own experiments.",
      "l X[c] X[c] X[c] X[c] X[c]",
      ["Model", "ROC AUC", "Bal. acc. (0.5)", "Bal. acc. (calibr.)", "F1 (0.5)", "Pred. positive"], rows, star=True)

# Decomposition ---------------------------------------------------------------------------------------
D = A["decomposition"]
rows = []
spec = [("architecture_effect_xgb_minus_cdnn", "Architecture: XGBoost $-$ centralized DNN")]
spec += [(f"federation_cost_{s}", f"Federation cost: centralized DNN $-$ {NAME[s]}") for s in TUNED]
spec += [("federation_cost_v1_config", "Federation cost, preliminary configuration"),
         ("fl_FedAvg_minus_xgboost", "FedAvg $-$ centralized XGBoost")]
for key, lab in spec:
    a, b = D["at_0.5.roc_auc"][key], D["calibrated.balanced_accuracy"][key]
    rows.append([lab, f"{num(a['diff'])} [{num(a['ci95'][0])}, {num(a['ci95'][1])}]",
                 f"{num(b['diff'])} [{num(b['ci95'][0])}, {num(b['ci95'][1])}]"])
table("tab:decomposition", "Architecture effect and cost of federation: difference of means with Welch 95\\% "
      "confidence interval over 10 seeds per model. Data: own experiments.",
      "X[2.2,l] X[c] X[c]", ["Contrast", "ROC AUC", "Balanced accuracy (calibrated)"], rows, star=True,
      short="Architecture effect and cost of federation")

# Strategy comparisons ---------------------------------------------------------------------------------
C = A["strategy_comparisons"]
rows = []
for pair in C["sel_auc"]["pairs"]:
    a, b = pair.split(" vs ")
    r, st = C["sel_auc"]["pairs"][pair], C["late_auc_sd"]["pairs"][pair]
    rows.append([f"{SHORT[a]} vs {SHORT[b]}", f"{num(r['diff'], 4)}", f"[{num(r['ci95'][0], 4)}, {num(r['ci95'][1], 4)}]",
                 pval(r["p_holm"]), num(r["dz"], 2), pval(r["tost_p"]), f"{num(st['diff'], 4)}", pval(st["p_holm"])])
fr, fs = C["sel_auc"], C["late_auc_sd"]
table("tab:strategies", "Pairwise comparison of the tuned aggregation strategies, paired by seed ($n=10$); $\\mu$ values denote FedProx. "
      "$\\Delta$AUC: test AUC at the validation-selected round; TOST: equivalence test with margin $\\pm$0.01 AUC; "
      "$\\Delta$SD: difference in round-to-round SD of test AUC over rounds "
      f"{A['late_window'][0]}--{A['late_window'][1]} (stability). $p$ values Holm-adjusted within each metric. "
      f"Friedman tests: AUC $\\chi^2={fr['friedman_chi2']:.2f}$, {ptxt(fr['friedman_p'])}; "
      f"SD $\\chi^2={fs['friedman_chi2']:.2f}$, {ptxt(fs['friedman_p'])}. Data: own experiments.",
      "X[1.9,l] X[c] X[2,c] X[c] X[0.8,c] X[c] X[c] X[c]",
      ["Pair", "$\\Delta$AUC", "95\\% CI", "$p_{\\text{Holm}}$", "$d_z$", "TOST $p$", "$\\Delta$SD", "$p_{\\text{Holm}}$"],
      rows, star=True)
print("tables written to", TEX_DIR)

# Communication budget ----------------------------------------------------------------------------------
B = A["budget_sensitivity"]
budgets = ["100", "250", "500", "1000"]
rows = [[NAME[c]] + [pm(B[c][b]["test_auc"]) for b in budgets]
        for c in TUNED + ["FedAvg_v1protocol"]]
table("tab:budget", "Test ROC AUC (mean $\\pm$ SD over 10 seeds) when the checkpoint is chosen on validation data "
      "within a budget of $R$ communication rounds. Configurations were tuned for $R=1{,}000$. Data: own experiments.",
      "l X[c] X[c] X[c] X[c]", ["Condition", "$R=100$", "$R=250$", "$R=500$", "$R=1{,}000$"], rows, star=True)
print("budget table written")
