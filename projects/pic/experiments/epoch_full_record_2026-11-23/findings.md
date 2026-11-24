# Full-record structural discovery at fixed epochs

96 of 96 runs have completed discovery and front assessment. A hit requires one candidate on the first Pareto front containing the complete law or a verified equivalent system. Accepted identities count once per run. Lorenz identities must together span all three laws. Each condition has three paired seeds per dataset. The oscillator control is reported separately.

| Dataset | Clean native | 5% native | Clean multiscale CV | 5% multiscale CV |
|---|---:|---:|---:|---:|
| Core suite | 37/45 | 29/45 | 34/45 | 29/45 |
| ac | 3/3 | 0/3 | 3/3 | 0/3 |
| burgers | 3/3 | 3/3 | 3/3 | 3/3 |
| burgers_inviscid | 3/3 | 3/3 | 3/3 | 2/3 |
| duffing | 0/3 | 0/3 | 0/3 | 0/3 |
| kdv | 3/3 | 3/3 | 3/3 | 3/3 |
| kdv_cossin | 3/3 | 0/3 | 3/3 | 0/3 |
| ks | 3/3 | 3/3 | 3/3 | 3/3 |
| lorenz | 3/3 | 3/3 | 3/3 | 3/3 |
| lv | 3/3 | 2/3 | 3/3 | 3/3 |
| ns | 1/3 | 2/3 | 1/3 | 2/3 |
| ode | 3/3 | 3/3 | 0/3 | 3/3 |
| pde_compound | 3/3 | 1/3 | 3/3 | 1/3 |
| pde_divide | 0/3 | 3/3 | 0/3 | 3/3 |
| vdp | 3/3 | 2/3 | 3/3 | 2/3 |
| wave | 3/3 | 1/3 | 3/3 | 1/3 |
| ode_simple | 3/3 | 3/3 | 3/3 | 3/3 |

Completed runs used 36,868–299,753 fitness calls and started with 32–42 individuals. All completed scalar and coupled runs passed the check for their configured 50 and 150 epochs, respectively.

The clean inviscid Burgers fronts contain a verified identity of the measured similarity solution on all three seeds. This identity was added after the original assessment. The earlier 3,000-fit counts below use the registry from that campaign. [Clean-run audit](clean_misses.json)

## Comparison with the earlier 3,000-fit campaign

Both campaigns have archive fronts for ordinary discrepancy and multiscale CV. Each entry compares the same datasets and seeds on those fronts. Lorenz and Navier–Stokes also gained samples in the epoch campaign. Thirteen core datasets used the same observations at each matched noise level.

| Inputs | Noise | Archive objective | Paired runs | Earlier hits | Epoch hits | Gains | Losses |
|---|---:|---|---:|---:|---:|---:|---:|
| Same samples | 0% | ordinary | 39 | 30 | 30 | 0 | 0 |
| Same samples | 0% | multiscale | 39 | 29 | 30 | 1 | 0 |
| Same samples | 5% | ordinary | 39 | 24 | 24 | 1 | 1 |
| Same samples | 5% | multiscale | 39 | 24 | 24 | 1 | 1 |
| Expanded samples | 0% | ordinary | 6 | 2 | 4 | 2 | 0 |
| Expanded samples | 0% | multiscale | 6 | 2 | 4 | 2 | 0 |
| Expanded samples | 5% | ordinary | 6 | 0 | 5 | 5 | 0 |
| Expanded samples | 5% | multiscale | 6 | 0 | 5 | 5 | 0 |

## Historical Lorenz identity credit

These identities are included in the recovery totals above. Literal supports remain available in the per-run records.

| Noise | Front | Assessed | Catalog hits | Verified identity-rule hits | Candidates with unresolved identity |
|---:|---|---:|---:|---:|---:|
| 0% | native | 3 | 0 | 3 | 0 |
| 0% | multiscale | 3 | 0 | 3 | 0 |
| 5% | native | 3 | 0 | 3 | 0 |
| 5% | multiscale | 3 | 0 | 3 | 0 |

[Experiment protocol](protocol.md)

[Per-run results](runs.csv) · [All four front definitions by dataset](by_dataset.csv)
