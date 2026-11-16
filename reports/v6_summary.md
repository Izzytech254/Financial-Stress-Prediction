# v6 Summary

The labels are generated as the top 15% of a latent per-snapshot risk score:

`latent = S + N(0, 0.35^2)`

where `S` is the weighted sum of seven capped liquidity factors, z-scored within each snapshot. Snapshots come from row order: Train `0-3`, Test `4-6`.

| Candidate | Features | LOMO composite | LogLoss | AUC | Reference public | Role |
|---|---:|---:|---:|---:|---:|---|
| `submission_v6_structural.csv` | 7 factors | 0.740474849 | 0.226701877 | 0.922704461 | 0.744516108 | safe structural baseline |
| `submission_v6_residual.csv` | 201 | 0.740562939 | 0.226641470 | 0.922772397 | 0.744624874 | 2nd-place selected model |
| `submission_v6_residual_dropm5.csv` | 197 | 0.740497298 | 0.226687864 | 0.922725255 | 0.744678 | best known public variant |
| `submission_v6_residual_dropm5_t200.csv` | 197 | 0.740493622 | 0.226692007 | 0.922726510 | not directly reported | combines two known positive test-time tweaks |

Recommendation:

- If only one submission: `submission_v6_residual_dropm5.csv`.
- If validating the jump: submit `submission_v6_structural.csv` first, then `submission_v6_residual_dropm5.csv`.
- Keep `submission_v5_poly.csv` as fallback until v6 is confirmed on the leaderboard.

All submissions have exact ID order, finite probabilities, and per-test-snapshot mean `0.15`.
