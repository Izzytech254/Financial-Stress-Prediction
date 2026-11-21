# v2 Experiment Log

Baseline: AUC 0.90659, LogLoss 0.24899. Frozen 5-fold seeds 42/2026/7. Feature selection is fold-internal.

| Hypothesis | Feature set | OOF AUC | OOF LL | dAUC vs LGB base | SE | dLL | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | baseline | 0.90536 | 0.25097 | +0.00000 | 0.00000 | +0.00000 | keep |
| cust_tr | cust_tr | 0.90501 | 0.25115 | -0.00043 | 0.00042 | +0.00019 | reject |
| cust_tt | cust_tt | 0.90508 | 0.25101 | -0.00043 | 0.00021 | +0.00004 | reject |
| history | history | 0.91023 | 0.24488 | +0.00473 | 0.00057 | -0.00609 | keep |
| combined | combined | 0.91030 | 0.24447 | +0.00483 | 0.00052 | -0.00650 | keep |
| confirm | combined | 0.91202 | 0.24254 | +0.00540 | 0.00042 | -0.00645 | kept |
| final blend (LGB+Cat only) | combined | 0.91256 | 0.24095 | +0.00599 | 0.00066 | -0.00804 | ACCEPT |
