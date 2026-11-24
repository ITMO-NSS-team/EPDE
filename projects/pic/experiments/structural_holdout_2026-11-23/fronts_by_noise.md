# Complete-system recovery on clean and noisy data

The endpoint is membership of a correct complete system on the unrestricted first nondominated front. Each condition covers the same 15 datasets and 12 search seeds, with 3,000 search fit equivalents per run. The 0% inputs are the stored observations. At 5%, Gaussian noise with standard deviation 0.05 times each field’s standard deviation is added before discovery. The physical grid, trajectory and registered equations are shared between conditions. Search, window selection, candidate frequencies, fitting and validation use the fields from their own condition. Slab CV adds offline coefficient refits.

| Discrepancy with coefficient stability | Clean front hits / 180 | 5% noise front hits / 180 | Median front size, clean | Median front size, 5% noise |
|---|---:|---:|---:|---:|
| Contiguous holdout | 144 | 79 | 6 | 7 |
| Full record, ordinary discrepancy | 120 | 100 | 12 | 13 |
| Full record, two-slab CV | 119 | 100 | 7 | 7.5 |
| Full record, all-scale slab CV | 119 | 100 | 6 | 7 |

## Full-record discovery and all-scale slab CV by dataset

Counts are out of 12 paired search seeds for each dataset. A generated coupled system counts only if its component laws occurred together in an actual candidate.

| Dataset | Generated, clean | On front, clean | Generated, 5% noise | On front, 5% noise |
|---|---:|---:|---:|---:|
| ac | 12 | 12 | 9 | 0 |
| burgers | 12 | 12 | 12 | 12 |
| burgers_inviscid | 12 | 10 | 11 | 10 |
| duffing | 0 | 0 | 0 | 0 |
| kdv | 12 | 12 | 12 | 12 |
| kdv_cossin | 12 | 12 | 0 | 0 |
| ks | 12 | 12 | 12 | 12 |
| lorenz | 1 | 1 | 0 | 0 |
| lv | 12 | 12 | 12 | 11 |
| ns | 0 | 0 | 0 | 0 |
| ode | 12 | 0 | 12 | 10 |
| pde_compound | 12 | 12 | 8 | 8 |
| pde_divide | 8 | 0 | 12 | 12 |
| vdp | 12 | 12 | 12 | 10 |
| wave | 12 | 12 | 12 | 3 |
| ode_simple (separate control) | 12 | 12 | 12 | 12 |

The oscillator identity control is excluded from the 180-run totals. Clean seeds vary search randomness on the same stored fields. Noisy seeds vary both the added noise draw and search randomness. The original sampling and numerical discretization remain in both conditions. For full-record CV, discovery saw the full record and adjacent weak windows can overlap raw observations. [Per-dataset paired counts](fronts_by_noise.csv) · [Run-level results with noise labels](fronts_by_noise_runs.csv)

## Comparison with the November 7–8 universal-search benchmark

That campaign ran 13 systems at 0% and 5% added noise, with three seeds per condition. With its full historical credit rule, every system succeeded on at least one clean seed. Wave was the only system that missed on all three noisy seeds. The original first-front counts separate the effect of the scoring rules on the same archived fronts.

| Historical first-front scoring rule | Clean / 39 | 5% noise / 39 |
|---|---:|---:|
| Listed structure under the historical strict test | 28 | 25 |
| Also credit algebraic identities | 31 | 28 |
| Also credit registered truncation companions and the NS pressure–Poisson forms | 38 | 30 |

Eleven named dynamics occur in both primary suites. The old simple harmonic oscillator is the current separate control. Solar heat remains cataloged outside the core suite. The current core includes the forced oscillator, Duffing, inviscid Burgers and forced KdV. The same 15 names were already present in the October core-15 campaign. [November universal-search report](../../../robot/epde_u/FINDINGS_RU.md) · [October core-15 report](../baseline_clean_2026-10-09/README.md)

On the 11 shared dynamics, current all-scale CV returned the registered complete system in 97/132 clean and 80/132 noisy runs. On the four additional core dynamics, the corresponding counts were 22/48 and 20/48.

The November campaign used 50 epochs for scalar laws and 150 for coupled systems, with the final full-record search front. The current campaign caps search at 3,000 fit equivalents and reconstructs the eligible front from actual archived systems. Its truth registry accepts specific listed supports and equivalents. The old Lorenz run used 100,000 time samples; the current loader supplies a 1,041-sample attractor segment. The old Navier–Stokes grid was 50 × 50 × 100 and the current grid is 50 × 20 × 36. These simultaneous changes preclude a code-regression estimate from the suite totals. [Earlier loaders](../../../robot/epde_u/run_auto.py) · [Current loaders](../../epde_bench/datasets.py) · [Current scoring](../live_operator_suite_2026-11-23/assess.py)

### Lorenz with the original identity checker on the same current fronts

| Scoring of the current all-scale CV front | Clean / 12 | 5% noise / 12 |
|---|---:|---:|
| Current registered supports | 1 | 0 |
| November 7 identity and full-system rank check | 7 | 2 |

Only the evaluation of the already-discovered front members changes in this check. The historical checker accepts algebraic rewritings when the three equations span all three Lorenz laws. [Paired Lorenz checks](historical_lorenz.csv) · [Replay code](historical_lorenz.py)
