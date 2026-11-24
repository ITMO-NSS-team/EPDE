# Structural early stopping on full records

EPDE compares the set of structures on the reported first Pareto front after each epoch. For patience 5, the first possible trigger is epoch 6. The search ran to its full budget. Earlier runs were replayed and their full discovery paths checked against the original archives.

The count stopped early excludes a trigger at the budget itself. Epochs saved is the median among runs that would stop early. Late recoveries had a complete registered system on the final front but lacked it at the stopping epoch. Early-only recoveries show the reverse.

| Group | Noise | Patience | Assessed | Stopped early | Epochs saved | Hits at stop | Hits at full budget | Late recoveries | Early-only recoveries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Core | 0% | 3 | 45 | 44 | 39.0 | 34 | 34 | 0 | 0 |
| Core | 0% | 5 | 45 | 41 | 35 | 34 | 34 | 0 | 0 |
| Core | 0% | 10 | 45 | 28 | 27.0 | 34 | 34 | 0 | 0 |
| Core | 5% | 3 | 45 | 43 | 37 | 29 | 29 | 0 | 0 |
| Core | 5% | 5 | 45 | 38 | 30.5 | 29 | 29 | 0 | 0 |
| Core | 5% | 10 | 45 | 33 | 24 | 29 | 29 | 0 | 0 |
| Oscillator control | 0% | 3 | 3 | 3 | 41 | 3 | 3 | 0 | 0 |
| Oscillator control | 0% | 5 | 3 | 3 | 34 | 3 | 3 | 0 | 0 |
| Oscillator control | 0% | 10 | 3 | 3 | 19 | 3 | 3 | 0 | 0 |
| Oscillator control | 5% | 3 | 3 | 3 | 18 | 3 | 3 | 0 | 0 |
| Oscillator control | 5% | 5 | 3 | 3 | 8 | 3 | 3 | 0 | 0 |
| Oscillator control | 5% | 10 | 3 | 0 | – | 3 | 3 | 0 | 0 |

[Per-run stop epochs and structural recovery](patience.csv)
