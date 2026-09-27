# Ablation study

## A. Input blocks (tuned configurations, not re-tuned)

| Inputs | cDNN AUC | FedAvg AUC | federation cost AUC (Welch) | cost BalAcc cal. | FedAvg BalAcc cal. |
|---|---|---|---|---|---|
| all | 0.788 ± 0.002 | 0.777 ± 0.001 | +0.011 [+0.010, +0.013] | +0.013 [+0.010, +0.015] | 0.701 ± 0.003 |
| no_user | 0.785 ± 0.001 | 0.777 ± 0.002 | +0.008 [+0.007, +0.010] | +0.008 [+0.007, +0.010] | 0.703 ± 0.001 |
| no_skill | 0.778 ± 0.002 | 0.778 ± 0.001 | -0.000 [-0.002, +0.002] | +0.003 [+0.002, +0.005] | 0.699 ± 0.001 |
| features_only | 0.779 ± 0.001 | 0.779 ± 0.000 | +0.000 [-0.000, +0.001] | -0.000 [-0.001, +0.000] | 0.702 ± 0.001 |
| embeddings_only | 0.779 ± 0.002 | 0.705 ± 0.002 | +0.074 [+0.072, +0.076] | +0.062 [+0.058, +0.066] | 0.642 ± 0.005 |

Effect of removing a block (ablated − full):

- no_user: centralized -0.003 [-0.004, -0.002]; FedAvg (paired) +0.000 [-0.002, +0.002]
- no_skill: centralized -0.011 [-0.013, -0.009]; FedAvg (paired) +0.001 [-0.000, +0.002]
- features_only: centralized -0.009 [-0.010, -0.008]; FedAvg (paired) +0.002 [+0.001, +0.003]
- embeddings_only: centralized -0.010 [-0.011, -0.008]; FedAvg (paired) -0.072 [-0.074, -0.071]

## B. Weight decay restricted to the MLP

| Condition | AUC with | AUC without | Δ AUC | BalAcc@0.5 with | without | PosRate with | without | late SD with | late SD without | lag-1 with | without | AUC<0.5 share with | without |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cdnn | 0.788 ± 0.002 | 0.780 ± 0.001 | -0.008 [-0.010, -0.007] | 0.690 ± 0.023 | 0.698 ± 0.008 | 0.716 ± 0.076 | 0.688 ± 0.031 | – | – | – | – | – | – |
| FedAvg | 0.777 ± 0.001 | 0.731 ± 0.003 | -0.046 [-0.048, -0.044] | 0.670 ± 0.008 | 0.638 ± 0.006 | 0.774 ± 0.024 | 0.759 ± 0.018 | 0.0069 ± 0.0007 | 0.0031 ± 0.0005 | 0.99 ± 0.00 | 0.97 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 |
|  selected rounds without decay: [734, 978, 978, 999, 898, 931, 976, 788, 962, 741] |||||||||||||| 
| FedProx_mu1.0 | 0.775 ± 0.003 | 0.658 ± 0.011 | -0.117 [-0.125, -0.109] | 0.500 ± 0.000 | 0.579 ± 0.012 | 1.000 ± 0.000 | 0.856 ± 0.025 | 0.1771 ± 0.0098 | 0.0032 ± 0.0012 | -0.35 ± 0.17 | 0.88 ± 0.05 | 0.40 ± 0.05 | 0.00 ± 0.00 |
|  selected rounds without decay: [762, 953, 942, 530, 679, 518, 990, 974, 844, 972] |||||||||||||| 
| FedProx_muTuned | 0.778 ± 0.004 | 0.557 ± 0.013 | -0.221 [-0.231, -0.210] | 0.500 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.1416 ± 0.0098 | 0.0261 ± 0.0040 | -0.06 ± 0.03 | 0.06 ± 0.06 | 0.12 ± 0.02 | 0.43 ± 0.02 |
|  selected rounds without decay: [285, 306, 109, 648, 925, 537, 898, 713, 939, 801] |||||||||||||| 

## C. Preliminary configuration (no weight decay), features only

- full: AUC 0.630 ± 0.008, BalAcc@0.5 0.573 ± 0.015, BalAcc cal 0.600 ± 0.007, pos 0.855 ± 0.039
- features only: AUC 0.765 ± 0.021, BalAcc@0.5 0.500 ± 0.000, BalAcc cal 0.688 ± 0.017, pos 1.000 ± 0.000
- paired delta AUC +0.135 [+0.117, +0.152]; selected rounds [17, 130, 40, 6, 27, 81, 53, 87, 7, 156]
