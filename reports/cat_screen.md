# CatBoost v3 screen

- Frozen seed `42`, `N_SELECT=500`, `iterations=700`

| config | AUC | LogLoss | last AUC | last LL | Platt CV LL | 60/40 proxy |
|---|---:|---:|---:|---:|---:|---:|
| CAT_TUNED_700 | 0.912943 | 0.241122 | 0.914696 | 0.240631 | 0.240849 | 0.179333 |
| CAT_PARAMS_700 | 0.913165 | 0.240601 | 0.915346 | 0.239503 | 0.240188 | 0.178847 |

Runtime: `2218.4s`.
