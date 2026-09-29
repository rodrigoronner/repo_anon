# Analyses added after review

Per-student AUC over 825 students with both classes in the test window (8534 of 9575 test pairs).

| Model | Global AUC | Per-student AUC |
|---|---|---|
| XGBoost, centralized | 0.789 ± 0.001 | 0.731 ± 0.004 |
| Recommender, centralized | 0.788 ± 0.002 | 0.736 ± 0.004 |
| Recommender, FedAvg | 0.777 ± 0.001 | 0.734 ± 0.003 |
| Recommender, FedProx mu=0.1 | 0.778 ± 0.001 | 0.732 ± 0.002 |
| Recommender, FedProx mu=0.5 | 0.777 ± 0.001 | 0.733 ± 0.002 |
| Recommender, FedProx mu=1.0 | 0.775 ± 0.003 | 0.712 ± 0.004 |
| Recommender, FedProx mu tuned | 0.778 ± 0.004 | 0.712 ± 0.007 |
| Recommender, FedAvg, row-wise aggregation (tuned) | 0.767 ± 0.004 | 0.712 ± 0.009 |
| Recommender, FedAvg, row-wise aggregation (FedAvg configuration) | 0.777 ± 0.001 | 0.738 ± 0.003 |
| Logistic regression, centralized | 0.780 ± 0.001 | 0.720 ± 0.000 |
| Logistic regression, FedAvg | 0.779 ± 0.001 | 0.720 ± 0.000 |
| Recommender without embeddings, centralized | 0.779 ± 0.001 | 0.720 ± 0.000 |
| Recommender without embeddings, FedAvg | 0.779 ± 0.000 | 0.720 ± 0.000 |
| Recommender, FedAvg, no weight decay on the embedding tables | 0.731 ± 0.003 | 0.723 ± 0.006 |
| Recommender, centralized, preliminary configuration | 0.775 ± 0.003 | 0.718 ± 0.008 |
| Recommender, FedAvg, preliminary configuration | 0.630 ± 0.008 | 0.685 ± 0.011 |
| Skill history mean (no model) | 0.699 ± 0.000 | 0.720 ± 0.000 |
| Student history mean (no model) | 0.689 ± 0.000 | 0.500 ± 0.000 |

## Agreement with the centralized recommender (seed-averaged scores)

| Model | within-student Spearman (median / mean) | same top-ranked skill |
|---|---|---|
| Recommender, FedAvg | 0.976 / 0.920 (n=949) | 0.887 (n=1063) |
| Recommender, FedProx mu=0.1 | 0.984 / 0.940 (n=949) | 0.927 (n=1063) |
| Recommender, FedProx mu=0.5 | 0.986 / 0.939 (n=949) | 0.930 (n=1063) |
| Recommender, FedProx mu=1.0 | 0.852 / 0.794 (n=949) | 0.637 (n=1063) |
| Recommender, FedProx mu tuned | 0.858 / 0.803 (n=949) | 0.697 (n=1063) |
| Recommender, FedAvg, row-wise aggregation (FedAvg configuration) | 0.973 / 0.915 (n=949) | 0.844 (n=1063) |
| Recommender, FedAvg, row-wise aggregation (tuned) | 0.952 / 0.885 (n=949) | 0.775 (n=1063) |
| Logistic regression, FedAvg | 0.842 / 0.781 (n=949) | 0.680 (n=1063) |
| Recommender without embeddings, FedAvg | 0.842 / 0.781 (n=949) | 0.680 (n=1063) |
| Recommender, FedAvg, preliminary configuration | 0.829 / 0.740 (n=949) | 0.560 (n=1063) |

## Student-cluster bootstrap (B=2000, random seed per model per replicate)

| Contrast | Global AUC diff [95% CI] | Per-student AUC diff [95% CI] |
|---|---|---|
| cost_FedAvg (cdnn_tuned − fl_FedAvg) | +0.0115 [+0.0044, +0.0188] | +0.0016 [-0.0114, +0.0147] |
| cost_FedProx_mu0.1 (cdnn_tuned − fl_FedProx_mu0.1) | +0.0104 [+0.0038, +0.0163] | +0.0042 [-0.0082, +0.0160] |
| cost_FedProx_mu0.5 (cdnn_tuned − fl_FedProx_mu0.5) | +0.0116 [+0.0048, +0.0181] | +0.0033 [-0.0098, +0.0150] |
| cost_FedProx_mu1.0 (cdnn_tuned − fl_FedProx_mu1.0) | +0.0133 [+0.0057, +0.0217] | +0.0244 [+0.0072, +0.0421] |
| cost_FedProx_muTuned (cdnn_tuned − fl_FedProx_muTuned) | +0.0105 [+0.0023, +0.0223] | +0.0239 [+0.0047, +0.0477] |
| cost_rowwise (cdnn_tuned − fl_FedAvg_rowwise) | +0.0217 [+0.0126, +0.0326] | +0.0242 [+0.0042, +0.0478] |
| rowwise_minus_dense (fl_FedAvg_rowwise − fl_FedAvg) | -0.0102 [-0.0219, -0.0003] | -0.0225 [-0.0455, -0.0030] |
| cost_features_only (cdnn_features_only − FedAvg_features_only) | +0.0004 [-0.0026, +0.0026] | +0.0000 [+0.0000, +0.0000] |
| centralized_gain_from_embeddings (cdnn_tuned − cdnn_features_only) | +0.0092 [+0.0023, +0.0154] | +0.0160 [+0.0003, +0.0314] |
| fedavg_loss_from_embeddings (FedAvg_features_only − fl_FedAvg) | +0.0020 [-0.0049, +0.0091] | -0.0144 [-0.0301, +0.0029] |
| cost_logreg (clr_tuned − fl_LogReg) | +0.0003 [-0.0028, +0.0042] | +0.0000 [+0.0000, +0.0000] |
| recommender_minus_logreg_centralized (cdnn_tuned − clr_tuned) | +0.0088 [+0.0020, +0.0156] | +0.0160 [+0.0000, +0.0323] |
| recommender_minus_logreg_federated (fl_FedAvg − fl_LogReg) | -0.0024 [-0.0103, +0.0057] | +0.0144 [-0.0021, +0.0299] |
| cost_preliminary (cdnn_v1 − fl_FedAvg_v1protocol) | +0.1448 [+0.1227, +0.1670] | +0.0329 [+0.0040, +0.0686] |

## Smallest equivalence margin (TOST, alpha = 0.05)

- FedAvg vs FedProx_mu0.1: 0.0017
- FedAvg vs FedProx_mu0.5: 0.0007
- FedAvg vs FedProx_mu1.0: 0.0036
- FedAvg vs FedProx_muTuned: 0.0035
- FedProx_mu0.1 vs FedProx_mu0.5: 0.0016
- FedProx_mu0.1 vs FedProx_mu1.0: 0.0047
- FedProx_mu0.1 vs FedProx_muTuned: 0.0024
- FedProx_mu0.5 vs FedProx_mu1.0: 0.0035
- FedProx_mu0.5 vs FedProx_muTuned: 0.0033
- FedProx_mu1.0 vs FedProx_muTuned: 0.0047
