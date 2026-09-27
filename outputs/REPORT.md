# Final pipeline — results

Test examples 9575, positive rate 0.643

| Model | AUC | BalAcc@0.5 | BalAcc cal. | F1@0.5 | PosRate@0.5 | AUC seen | AUC new | val AUC |
|---|---|---|---|---|---|---|---|---|
| always_positive | 0.500 | 0.500 | – | 0.783 | 1.000 | – | – | – |
| stratified_random | 0.503 | 0.503 | – | 0.658 | 0.669 | – | – | – |
| student_history_mean | 0.689 | 0.639 | – | 0.739 | 0.635 | – | – | – |
| skill_history_mean | 0.699 | 0.634 | – | 0.714 | 0.587 | – | – | – |
| xgboost | 0.789 ± 0.001 | 0.703 ± 0.003 | 0.706 ± 0.005 | 0.808 ± 0.002 | 0.698 ± 0.007 | 0.791 ± 0.002 | 0.789 ± 0.001 | 0.813 ± 0.001 |
| cdnn_tuned | 0.788 ± 0.002 | 0.690 ± 0.023 | 0.714 ± 0.002 | 0.806 ± 0.013 | 0.716 ± 0.076 | 0.788 ± 0.002 | 0.790 ± 0.002 | 0.806 ± 0.001 |
| cdnn_v1 | 0.775 ± 0.003 | 0.684 ± 0.009 | 0.702 ± 0.003 | 0.804 ± 0.004 | 0.722 ± 0.024 | 0.777 ± 0.004 | 0.774 ± 0.004 | 0.789 ± 0.006 |
| fl_FedAvg | 0.777 ± 0.001 | 0.670 ± 0.008 | 0.701 ± 0.003 | 0.813 ± 0.003 | 0.774 ± 0.024 | 0.780 ± 0.001 | 0.774 ± 0.002 | 0.782 ± 0.002 |
| fl_FedProx_mu0.1 | 0.778 ± 0.001 | 0.615 ± 0.014 | 0.702 ± 0.002 | 0.813 ± 0.002 | 0.881 ± 0.021 | 0.780 ± 0.001 | 0.777 ± 0.001 | 0.785 ± 0.002 |
| fl_FedProx_mu0.5 | 0.777 ± 0.001 | 0.672 ± 0.009 | 0.702 ± 0.002 | 0.811 ± 0.003 | 0.764 ± 0.026 | 0.780 ± 0.001 | 0.775 ± 0.002 | 0.784 ± 0.002 |
| fl_FedProx_mu1.0 | 0.775 ± 0.003 | 0.500 ± 0.000 | 0.699 ± 0.002 | 0.783 ± 0.000 | 1.000 ± 0.000 | 0.776 ± 0.003 | 0.777 ± 0.003 | 0.776 ± 0.002 |
| fl_FedProx_muTuned | 0.778 ± 0.004 | 0.500 ± 0.000 | 0.700 ± 0.004 | 0.783 ± 0.000 | 1.000 ± 0.000 | 0.779 ± 0.004 | 0.779 ± 0.004 | 0.779 ± 0.001 |
| fl_FedAvg_v1protocol | 0.630 ± 0.008 | 0.573 ± 0.015 | 0.600 ± 0.007 | 0.779 ± 0.005 | 0.855 ± 0.039 | 0.615 ± 0.012 | 0.644 ± 0.009 | 0.629 ± 0.007 |

## Decomposition (Welch, 95% CI)

- at_0.5.roc_auc | architecture_effect_xgb_minus_cdnn: +0.0002 [-0.0012, +0.0016] p=0.738 d=0.15
- at_0.5.roc_auc | federation_cost_v1_config: +0.1448 [+0.1386, +0.1511] p=3.95e-15 d=22.71
- at_0.5.roc_auc | federation_cost_FedAvg: +0.0115 [+0.0101, +0.0129] p=8.71e-12 d=7.86
- at_0.5.roc_auc | federation_cost_FedProx_mu0.1: +0.0104 [+0.0091, +0.0116] p=1.92e-09 d=8.15
- at_0.5.roc_auc | federation_cost_FedProx_mu0.5: +0.0116 [+0.0102, +0.0129] p=1.25e-11 d=8.12
- at_0.5.roc_auc | federation_cost_FedProx_mu1.0: +0.0133 [+0.0110, +0.0155] p=2.74e-09 d=5.64
- at_0.5.roc_auc | federation_cost_FedProx_muTuned: +0.0105 [+0.0077, +0.0133] p=2.24e-06 d=3.63
- at_0.5.roc_auc | fl_FedAvg_minus_xgboost: -0.0117 [-0.0128, -0.0106] p=1.08e-14 d=-10.15
- at_0.5.roc_auc | tuned_minus_v1_FedAvg_paired: +0.1468 [+0.1406, +0.1531] p=1.51e-12 dz=16.78
- calibrated.balanced_accuracy | architecture_effect_xgb_minus_cdnn: -0.0080 [-0.0117, -0.0043] p=0.000483 d=-2.11
- calibrated.balanced_accuracy | federation_cost_v1_config: +0.1013 [+0.0959, +0.1067] p=5.64e-14 d=18.24
- calibrated.balanced_accuracy | federation_cost_FedAvg: +0.0128 [+0.0103, +0.0153] p=8.46e-09 d=4.86
- calibrated.balanced_accuracy | federation_cost_FedProx_mu0.1: +0.0116 [+0.0097, +0.0134] p=1.82e-10 d=5.80
- calibrated.balanced_accuracy | federation_cost_FedProx_mu0.5: +0.0121 [+0.0101, +0.0140] p=1.46e-10 d=5.80
- calibrated.balanced_accuracy | federation_cost_FedProx_mu1.0: +0.0150 [+0.0131, +0.0169] p=2.67e-12 d=7.45
- calibrated.balanced_accuracy | federation_cost_FedProx_muTuned: +0.0134 [+0.0101, +0.0167] p=6.43e-07 d=3.97
- calibrated.balanced_accuracy | fl_FedAvg_minus_xgboost: -0.0048 [-0.0087, -0.0008] p=0.0206 d=-1.16
- calibrated.balanced_accuracy | tuned_minus_v1_FedAvg_paired: +0.1005 [+0.0948, +0.1062] p=1.92e-11 dz=12.63
- at_0.5.balanced_accuracy | architecture_effect_xgb_minus_cdnn: +0.0126 [-0.0040, +0.0293] p=0.121 d=0.76
- at_0.5.balanced_accuracy | federation_cost_v1_config: +0.1112 [+0.0996, +0.1227] p=3.76e-12 d=9.21
- at_0.5.balanced_accuracy | federation_cost_FedAvg: +0.0200 [+0.0029, +0.0370] p=0.0258 d=1.15
- at_0.5.balanced_accuracy | federation_cost_FedProx_mu0.1: +0.0752 [+0.0569, +0.0935] p=2.95e-07 d=3.92
- at_0.5.balanced_accuracy | federation_cost_FedProx_mu0.5: +0.0177 [+0.0006, +0.0349] p=0.0441 d=1.01
- at_0.5.balanced_accuracy | federation_cost_FedProx_mu1.0: +0.1902 [+0.1736, +0.2068] p=9.06e-10 d=11.60
- at_0.5.balanced_accuracy | federation_cost_FedProx_muTuned: +0.1902 [+0.1736, +0.2068] p=9.06e-10 d=11.60
- at_0.5.balanced_accuracy | fl_FedAvg_minus_xgboost: -0.0326 [-0.0387, -0.0265] p=7.45e-08 d=-5.24
- at_0.5.balanced_accuracy | tuned_minus_v1_FedAvg_paired: +0.0976 [+0.0849, +0.1103] p=3.06e-08 dz=5.51

## Federated, rounds 501–1000

| Condition | mean AUC | SD AUC | SD val AUC | BalAcc | PosRate | collapsed share (all rounds) | lag-1 |
|---|---|---|---|---|---|---|---|
| FedAvg | 0.767 ± 0.001 | 0.0069 ± 0.0007 | 0.0071 ± 0.0007 | 0.659 ± 0.003 | 0.783 ± 0.007 | 0.05 ± 0.01 | 0.99 ± 0.00 |
| FedAvg_v1protocol | 0.629 ± 0.007 | 0.0021 ± 0.0010 | 0.0023 ± 0.0008 | 0.581 ± 0.007 | 0.820 ± 0.017 | 0.18 ± 0.05 | 0.95 ± 0.03 |
| FedProx_mu0.1 | 0.770 ± 0.002 | 0.0077 ± 0.0012 | 0.0083 ± 0.0015 | 0.619 ± 0.006 | 0.869 ± 0.008 | 0.27 ± 0.05 | 0.99 ± 0.00 |
| FedProx_mu0.5 | 0.768 ± 0.001 | 0.0075 ± 0.0009 | 0.0073 ± 0.0011 | 0.655 ± 0.002 | 0.794 ± 0.006 | 0.11 ± 0.02 | 0.99 ± 0.00 |
| FedProx_mu1.0 | 0.507 ± 0.028 | 0.1771 ± 0.0098 | 0.1792 ± 0.0100 | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.00 ± 0.00 | -0.35 ± 0.17 |
| FedProx_muTuned | 0.683 ± 0.008 | 0.1416 ± 0.0098 | 0.1424 ± 0.0101 | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.00 ± 0.00 | -0.06 ± 0.03 |

## Communication-budget sensitivity (checkpoint chosen on validation within the budget; configurations tuned for 1000 rounds)

| Condition | AUC @100 | AUC @250 | AUC @500 | AUC @1000 | BalAcc @100 | BalAcc @250 | BalAcc @500 | BalAcc @1000 |
|---|---|---|---|---|---|---|---|---|
| FedAvg | 0.668 ± 0.015 | 0.718 ± 0.003 | 0.752 ± 0.003 | 0.777 ± 0.001 | 0.549 ± 0.018 | 0.603 ± 0.009 | 0.639 ± 0.008 | 0.670 ± 0.008 |
| FedAvg_v1protocol | 0.604 ± 0.011 | 0.619 ± 0.011 | 0.625 ± 0.011 | 0.630 ± 0.008 | 0.500 ± 0.000 | 0.523 ± 0.016 | 0.559 ± 0.023 | 0.573 ± 0.015 |
| FedProx_mu0.1 | 0.639 ± 0.019 | 0.689 ± 0.014 | 0.750 ± 0.005 | 0.778 ± 0.001 | 0.500 ± 0.000 | 0.507 ± 0.009 | 0.570 ± 0.017 | 0.615 ± 0.014 |
| FedProx_mu0.5 | 0.655 ± 0.016 | 0.706 ± 0.008 | 0.751 ± 0.003 | 0.777 ± 0.001 | 0.507 ± 0.010 | 0.575 ± 0.012 | 0.626 ± 0.012 | 0.672 ± 0.009 |
| FedProx_mu1.0 | 0.773 ± 0.007 | 0.774 ± 0.006 | 0.775 ± 0.003 | 0.775 ± 0.003 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| FedProx_muTuned | 0.777 ± 0.002 | 0.778 ± 0.001 | 0.778 ± 0.003 | 0.778 ± 0.004 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |

## Strategy comparisons (paired by seed)

### sel_auc: Friedman χ²=13.28, p=0.00999

| Pair | Δ | 95% CI | p | p Holm | dz | TOST p |
|---|---|---|---|---|---|---|
| FedAvg vs FedProx_mu0.1 | -0.0011 | [-0.0019, -0.0004] | 0.00609 | 0.0548 | -1.13 | 2.38e-10 |
| FedAvg vs FedProx_mu0.5 | +0.0001 | [-0.0007, +0.0009] | 0.829 | 1 | +0.07 | 2.03e-10 |
| FedAvg vs FedProx_mu1.0 | +0.0018 | [-0.0005, +0.0041] | 0.112 | 0.675 | +0.56 | 1.02e-05 |
| FedAvg vs FedProx_muTuned | -0.0010 | [-0.0041, +0.0021] | 0.488 | 1 | -0.23 | 4.88e-05 |
| FedProx_mu0.1 vs FedProx_mu0.5 | +0.0012 | [+0.0007, +0.0017] | 0.000466 | 0.00466 | +1.69 | 1.25e-11 |
| FedProx_mu0.1 vs FedProx_mu1.0 | +0.0029 | [+0.0007, +0.0051] | 0.0152 | 0.121 | +0.95 | 2.4e-05 |
| FedProx_mu0.1 vs FedProx_muTuned | +0.0001 | [-0.0026, +0.0029] | 0.907 | 1 | +0.04 | 1.08e-05 |
| FedProx_mu0.5 vs FedProx_mu1.0 | +0.0017 | [-0.0005, +0.0039] | 0.114 | 0.675 | +0.55 | 6.91e-06 |
| FedProx_mu0.5 vs FedProx_muTuned | -0.0011 | [-0.0038, +0.0017] | 0.41 | 1 | -0.27 | 2.34e-05 |
| FedProx_mu1.0 vs FedProx_muTuned | -0.0028 | [-0.0051, -0.0004] | 0.0265 | 0.186 | -0.84 | 3.5e-05 |

### sel_balacc_cal: Friedman χ²=5.44, p=0.245

| Pair | Δ | 95% CI | p | p Holm | dz | TOST p |
|---|---|---|---|---|---|---|
| FedAvg vs FedProx_mu0.1 | -0.0012 | [-0.0036, +0.0012] | 0.275 | 1 | -0.37 | 7.91e-06 |
| FedAvg vs FedProx_mu0.5 | -0.0007 | [-0.0023, +0.0009] | 0.342 | 1 | -0.32 | 1.88e-07 |
| FedAvg vs FedProx_mu1.0 | +0.0022 | [-0.0005, +0.0050] | 0.093 | 0.744 | +0.59 | 5.7e-05 |
| FedAvg vs FedProx_muTuned | +0.0006 | [-0.0033, +0.0045] | 0.724 | 1 | +0.12 | 0.000209 |
| FedProx_mu0.1 vs FedProx_mu0.5 | +0.0005 | [-0.0009, +0.0019] | 0.429 | 1 | +0.26 | 4.41e-08 |
| FedProx_mu0.1 vs FedProx_mu1.0 | +0.0035 | [+0.0014, +0.0056] | 0.0047 | 0.047 | +1.18 | 3.1e-05 |
| FedProx_mu0.1 vs FedProx_muTuned | +0.0019 | [-0.0019, +0.0056] | 0.298 | 1 | +0.35 | 0.000446 |
| FedProx_mu0.5 vs FedProx_mu1.0 | +0.0030 | [+0.0009, +0.0050] | 0.01 | 0.0901 | +1.03 | 1.47e-05 |
| FedProx_mu0.5 vs FedProx_muTuned | +0.0013 | [-0.0018, +0.0045] | 0.36 | 1 | +0.30 | 7.83e-05 |
| FedProx_mu1.0 vs FedProx_muTuned | -0.0016 | [-0.0047, +0.0015] | 0.27 | 1 | -0.37 | 9.1e-05 |

### late_auc_mean: Friedman χ²=36.88, p=1.91e-07

| Pair | Δ | 95% CI | p | p Holm | dz | TOST p |
|---|---|---|---|---|---|---|
| FedAvg vs FedProx_mu0.1 | -0.0024 | [-0.0036, -0.0013] | 0.000854 | 0.00171 | -1.55 | 5.04e-08 |
| FedAvg vs FedProx_mu0.5 | -0.0003 | [-0.0012, +0.0005] | 0.401 | 0.401 | -0.28 | 4.67e-10 |
| FedAvg vs FedProx_mu1.0 | +0.2600 | [+0.2404, +0.2796] | 2.48e-10 | 1.74e-09 | +9.49 | 1 |
| FedAvg vs FedProx_muTuned | +0.0847 | [+0.0791, +0.0904] | 8.28e-11 | 7.45e-10 | +10.73 | 1 |
| FedProx_mu0.1 vs FedProx_mu0.5 | +0.0021 | [+0.0015, +0.0027] | 3.19e-05 | 9.57e-05 | +2.42 | 1.88e-10 |
| FedProx_mu0.1 vs FedProx_mu1.0 | +0.2624 | [+0.2421, +0.2828] | 3.2e-10 | 1.77e-09 | +9.22 | 1 |
| FedProx_mu0.1 vs FedProx_muTuned | +0.0872 | [+0.0815, +0.0929] | 6.91e-11 | 6.91e-10 | +10.95 | 1 |
| FedProx_mu0.5 vs FedProx_mu1.0 | +0.2603 | [+0.2403, +0.2804] | 2.95e-10 | 1.77e-09 | +9.31 | 1 |
| FedProx_mu0.5 vs FedProx_muTuned | +0.0851 | [+0.0793, +0.0909] | 1.04e-10 | 8.35e-10 | +10.45 | 1 |
| FedProx_mu1.0 vs FedProx_muTuned | -0.1753 | [-0.1984, -0.1522] | 3.48e-08 | 1.39e-07 | -5.43 | 1 |

### late_auc_sd: Friedman χ²=34.16, p=6.91e-07

| Pair | Δ | 95% CI | p | p Holm | dz | TOST p |
|---|---|---|---|---|---|---|
| FedAvg vs FedProx_mu0.1 | -0.0007 | [-0.0016, +0.0001] | 0.0856 | 0.171 | -0.61 | nan |
| FedAvg vs FedProx_mu0.5 | -0.0005 | [-0.0009, -0.0001] | 0.0149 | 0.0448 | -0.95 | nan |
| FedAvg vs FedProx_mu1.0 | -0.1702 | [-0.1771, -0.1632] | 1.01e-12 | 1.01e-11 | -17.54 | nan |
| FedAvg vs FedProx_muTuned | -0.1347 | [-0.1417, -0.1277] | 8.89e-12 | 5.1e-11 | -13.77 | nan |
| FedProx_mu0.1 vs FedProx_mu0.5 | +0.0002 | [-0.0004, +0.0009] | 0.462 | 0.462 | +0.24 | nan |
| FedProx_mu0.1 vs FedProx_mu1.0 | -0.1694 | [-0.1767, -0.1622] | 1.51e-12 | 1.22e-11 | -16.78 | nan |
| FedProx_mu0.1 vs FedProx_muTuned | -0.1339 | [-0.1407, -0.1272] | 6.86e-12 | 4.8e-11 | -14.17 | nan |
| FedProx_mu0.5 vs FedProx_mu1.0 | -0.1697 | [-0.1768, -0.1625] | 1.36e-12 | 1.22e-11 | -16.97 | nan |
| FedProx_mu0.5 vs FedProx_muTuned | -0.1342 | [-0.1411, -0.1272] | 8.49e-12 | 5.1e-11 | -13.84 | nan |
| FedProx_mu1.0 vs FedProx_muTuned | +0.0355 | [+0.0264, +0.0446] | 9.83e-06 | 3.93e-05 | +2.80 | nan |

### late_val_auc_sd: Friedman χ²=33.68, p=8.67e-07

| Pair | Δ | 95% CI | p | p Holm | dz | TOST p |
|---|---|---|---|---|---|---|
| FedAvg vs FedProx_mu0.1 | -0.0013 | [-0.0024, -0.0001] | 0.0362 | 0.108 | -0.78 | nan |
| FedAvg vs FedProx_mu0.5 | -0.0002 | [-0.0009, +0.0004] | 0.465 | 0.465 | -0.24 | nan |
| FedAvg vs FedProx_mu1.0 | -0.1722 | [-0.1792, -0.1651] | 1.04e-12 | 1.04e-11 | -17.49 | nan |
| FedAvg vs FedProx_muTuned | -0.1354 | [-0.1427, -0.1280] | 1.35e-11 | 9.43e-11 | -13.14 | nan |
| FedProx_mu0.1 vs FedProx_mu0.5 | +0.0011 | [-0.0001, +0.0022] | 0.0715 | 0.143 | +0.65 | nan |
| FedProx_mu0.1 vs FedProx_mu1.0 | -0.1709 | [-0.1786, -0.1633] | 2.35e-12 | 1.88e-11 | -15.97 | nan |
| FedProx_mu0.1 vs FedProx_muTuned | -0.1341 | [-0.1417, -0.1265] | 1.87e-11 | 1.12e-10 | -12.67 | nan |
| FedProx_mu0.5 vs FedProx_mu1.0 | -0.1720 | [-0.1795, -0.1644] | 1.91e-12 | 1.72e-11 | -16.35 | nan |
| FedProx_mu0.5 vs FedProx_muTuned | -0.1351 | [-0.1428, -0.1274] | 2.06e-11 | 1.12e-10 | -12.53 | nan |
| FedProx_mu1.0 vs FedProx_muTuned | +0.0368 | [+0.0278, +0.0458] | 6.69e-06 | 2.68e-05 | +2.93 | nan |

## Tuning

- **centralized_dnn**: best val 0.8070; trials 108 (complete 50, pruned 58, failed 0); params {'optimizer': 'adam', 'lr_adam': 0.005472075340482052, 'emb_dim': 32, 'h1': 128, 'h2': 128, 'dropout': 0.4, 'weight_decay': 0.0031137629439129755, 'batch_size': 64}
  - best_by_optimizer: {'adam': 0.8069813999199213, 'sgd': 0.7991870806488796}
  - importances: {'weight_decay': 0.539050555018941, 'dropout': 0.3555465072007208, 'h2': 0.059705382400837434, 'batch_size': 0.016746530896637236, 'h1': 0.014010607261891591, 'emb_dim': 0.009913206955634845, 'optimizer': 0.005027210265337066}
- **federated_FedAvg**: best val 0.7822; trials 38 (complete 23, pruned 15, failed 0); params {'optimizer': 'sgd', 'lr_sgd': 0.21708861222739265, 'local_epochs': 2, 'batch_size': 32, 'weight_decay': 0.0030564236337135248}
  - best_by_optimizer: {'adam': 0.777974738834492, 'sgd': 0.7822480253339643}
  - importances: {'local_epochs': 0.43474905587049234, 'batch_size': 0.3011647501386891, 'weight_decay': 0.22305247823195276, 'optimizer': 0.04103371575886591}
- **federated_FedProx_mu0.1**: best val 0.7853; trials 38 (complete 16, pruned 22, failed 0); params {'optimizer': 'sgd', 'lr_sgd': 0.14586756276351537, 'local_epochs': 1, 'batch_size': 16, 'weight_decay': 0.008686809107072178}
  - best_by_optimizer: {'adam': 0.7733890243754473, 'sgd': 0.7853395454931509}
  - importances: {'weight_decay': 0.4526130409171992, 'local_epochs': 0.4155555004571582, 'batch_size': 0.08463370682521844, 'optimizer': 0.04719775180042415}
- **federated_FedProx_mu0.5**: best val 0.7855; trials 38 (complete 18, pruned 20, failed 0); params {'optimizer': 'sgd', 'lr_sgd': 0.3608639932902437, 'local_epochs': 1, 'batch_size': 16, 'weight_decay': 0.0030945924008762664}
  - best_by_optimizer: {'adam': 0.7765739695943896, 'sgd': 0.7854754364891591}
  - importances: {'batch_size': 0.4238883909440155, 'local_epochs': 0.29484872388694805, 'optimizer': 0.18432378345531744, 'weight_decay': 0.09693910171371908}
- **federated_FedProx_mu1.0**: best val 0.7739; trials 38 (complete 16, pruned 22, failed 0); params {'optimizer': 'adam', 'lr_adam': 0.013712723599669366, 'local_epochs': 10, 'batch_size': 16, 'weight_decay': 0.07892657903676713}
  - best_by_optimizer: {'adam': 0.7738610029240831, 'sgd': 0.7643941324209224}
  - importances: {'local_epochs': 0.3685664953206856, 'batch_size': 0.2629970770855866, 'optimizer': 0.19198298581831286, 'weight_decay': 0.17645344177541497}
- **federated_FedProx_muTuned**: best val 0.7791; trials 38 (complete 22, pruned 16, failed 0); params {'optimizer': 'adam', 'lr_adam': 0.07129439007411074, 'local_epochs': 1, 'batch_size': 16, 'weight_decay': 0.08998522793041448, 'mu': 0.017313728464657275}
  - best_by_optimizer: {'adam': 0.7790630801150219, 'sgd': 0.7726391972724735}
  - importances: {'mu': 0.36221705370801216, 'weight_decay': 0.3348017852460561, 'local_epochs': 0.1765056173966979, 'optimizer': 0.06670681214495124, 'batch_size': 0.05976873150428262}
- **xgboost**: best val 0.8159; trials 108 (complete 108, pruned 0, failed 0); params {'id_encoding': 'numeric', 'max_depth': 6, 'learning_rate': 0.060321171713341955, 'min_child_weight': 3.4861760418460723, 'subsample': 0.6757699336988072, 'colsample_bytree': 0.5833804882071538, 'reg_lambda': 0.026481938706688603, 'reg_alpha': 0.06747072086403312, 'gamma': 1.066047945310664}
  - best_by_id_encoding: {'categorical': 0.8111145488297602, 'none': 0.8051978305973138, 'numeric': 0.8158749802836724}
  - importances: {'id_encoding': 0.8102502355273578, 'gamma': 0.07944698118684207, 'subsample': 0.03874073807137375, 'colsample_bytree': 0.026334599086277048, 'learning_rate': 0.014779500548339427, 'reg_alpha': 0.012822526140578959, 'min_child_weight': 0.008170847236847151, 'max_depth': 0.007150030893751259, 'reg_lambda': 0.002304541308632807}

Selected checkpoints: xgboost [187, 208, 334, 209, 187, 213, 172, 302, 262, 245]; cdnn_tuned [27, 17, 6, 11, 19, 30, 11, 19, 13, 15]; cdnn_v1 [9, 12, 10, 8, 8, 6, 11, 8, 11, 8]; fl_FedAvg [948, 936, 1000, 993, 987, 995, 976, 996, 962, 989]; fl_FedProx_mu0.1 [950, 938, 994, 971, 898, 973, 874, 983, 963, 996]; fl_FedProx_mu0.5 [948, 933, 957, 974, 890, 973, 875, 980, 962, 989]; fl_FedProx_mu1.0 [361, 951, 585, 204, 43, 260, 28, 61, 28, 30]; fl_FedProx_muTuned [1000, 827, 922, 951, 361, 901, 695, 467, 542, 595]; fl_FedAvg_v1protocol [964, 422, 988, 997, 473, 362, 976, 384, 738, 371]
