# Robustness analyses

## A. Temporal validation (fit on window 1, validate on window 2)

| Condition | val AUC | global test AUC | per-student AUC |
|---|---|---|---|
| T_cdnn_all | 0.790 ± 0.001 | 0.783 ± 0.002 | 0.729 ± 0.007 |
| T_cdnn_no_user | 0.790 ± 0.001 | 0.785 ± 0.001 | 0.733 ± 0.006 |
| T_cdnn_features_only | 0.776 ± 0.000 | 0.778 ± 0.001 | 0.720 ± 0.000 |
| T_FedAvg_all | 0.773 ± 0.002 | 0.767 ± 0.003 | 0.725 ± 0.004 |
| T_FedAvg_no_user | 0.784 ± 0.002 | 0.777 ± 0.002 | 0.730 ± 0.006 |
| T_FedAvg_features_only | 0.775 ± 0.000 | 0.778 ± 0.001 | 0.720 ± 0.000 |
| T_FedAvg_rowwise_samecfg | 0.761 ± 0.003 | 0.747 ± 0.003 | 0.729 ± 0.005 |
| T_FedAvg_rowwise_tuned | 0.750 ± 0.002 | 0.739 ± 0.005 | 0.701 ± 0.009 |

| Contrast | global AUC [bootstrap 95% CI] | per-student AUC [95% CI] |
|---|---|---|
| cost_of_federation (T_cdnn_all − T_FedAvg_all) | +0.0158 [+0.0077, +0.0252] | +0.0041 [-0.0180, +0.0220] |
| cost_features_only (T_cdnn_features_only − T_FedAvg_features_only) | -0.0001 [-0.0020, +0.0020] | +0.0000 [+0.0000, +0.0000] |
| student_embedding_centralized (T_cdnn_all − T_cdnn_no_user) | -0.0018 [-0.0074, +0.0039] | -0.0042 [-0.0261, +0.0183] |
| student_embedding_fedavg (T_FedAvg_all − T_FedAvg_no_user) | -0.0096 [-0.0186, -0.0026] | -0.0054 [-0.0207, +0.0136] |
| rowwise_samecfg_minus_dense (T_FedAvg_rowwise_samecfg − T_FedAvg_all) | -0.0206 [-0.0333, -0.0078] | +0.0046 [-0.0127, +0.0219] |
| rowwise_tuned_minus_dense (T_FedAvg_rowwise_tuned − T_FedAvg_all) | -0.0280 [-0.0420, -0.0126] | -0.0234 [-0.0444, +0.0044] |
| rowwise_samecfg_minus_no_user (T_FedAvg_rowwise_samecfg − T_FedAvg_no_user) | -0.0302 [-0.0425, -0.0181] | -0.0008 [-0.0183, +0.0189] |

## B. Same local optimizer (SGD configuration of FedAvg)

| Condition | test AUC | late SD | lag-1 | rounds AUC<0.5 | late positive rate |
|---|---|---|---|---|---|
| FedAvg (SGD) | 0.777 ± 0.001 | 0.0069 ± 0.0007 | 0.99 ± 0.00 | 0.00 ± 0.00 | 0.78 ± 0.01 |
| FedProx mu=1.0, SGD | 0.769 ± 0.002 | 0.0085 ± 0.0007 | 0.99 ± 0.00 | 0.00 ± 0.00 | 0.80 ± 0.01 |
| FedProx mu tuned, SGD | 0.777 ± 0.001 | 0.0070 ± 0.0007 | 0.99 ± 0.00 | 0.00 ± 0.00 | 0.78 ± 0.01 |
| FedProx mu=1.0, Adam (tuned) | 0.775 ± 0.003 | 0.1771 ± 0.0098 | -0.35 ± 0.17 | 0.40 ± 0.05 | 1.00 ± 0.00 |
| FedProx mu tuned, Adam (tuned) | 0.778 ± 0.004 | 0.1416 ± 0.0098 | -0.06 ± 0.03 | 0.12 ± 0.02 | 1.00 ± 0.00 |
- FedProx mu=1.0, SGD − FedAvg: -0.0075 [-0.0083, -0.0068], TOST p=2.1e-05
- FedProx mu tuned, SGD − FedAvg: -0.0001 [-0.0002, -0.0000], TOST p=1.1e-20
