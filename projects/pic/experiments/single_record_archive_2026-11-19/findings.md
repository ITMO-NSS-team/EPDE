# Single-trajectory discovery and validation in EPDE

For wave, holding out space and keeping the full time series gave better exact-support recovery than holding out time. For nonlinear diffusion, the largest loss came before validation. Nine of the twelve time-split searches never scored the exact active structure. Both findings came from plain solver-free searches with native WAPE, directional dyad and weak least squares. No term-count objective or compromise selection was used.

The 48 ten-epoch searches comprised six fresh paired seeds per system, both temporal directions and a whole-record baseline, plus a follow-up with both spatial directions for wave. Each split began from raw 5% noisy measurements. Each discovery and validation half used disjoint time points or disjoint spatial positions, with the middle row or column omitted. Native weak-window selection, token frequencies, fitting and source dyad used only the discovery half. The validation matrices used its frozen window geometry and evaluated tokens at the actual held-out coordinates.

## Entire held-out fronts

Each cell counts runs with the exact active support in any target orientation on the held-out WAPE–source-dyad front. The study kept each eligible equation's native coefficients fixed. The blocked-WLS cutoff of one gave the same recovery counts as WAPE eligibility alone in every table cell. Front sizes count distinct support–target combinations after the blocked check.

| System and split, discovery → validation | Exact support scored | Returned front after validation | Terminal population after validation | All scored candidates after validation | Median archive front size |
|---|---:|---:|---:|---:|---:|
| Wave, early → late | 6/6 | 0/6 | 0/6 | 0/6 | 9 |
| Wave, late → early | 6/6 | 0/6 | 0/6 | 0/6 | 10 |
| Wave, left → right | 4/6 | 3/6 | 3/6 | 4/6 | 4 |
| Wave, right → left | 6/6 | 2/6 | 2/6 | 3/6 | 7 |
| Nonlinear diffusion, early → late | 2/6 | 1/6 | 1/6 | 1/6 | 8 |
| Nonlinear diffusion, late → early | 1/6 | 0/6 | 0/6 | 1/6 | 8 |

The whole-record in-sample baselines returned the wave law in 0/6 runs and the diffusion law in 3/6. Using all scored candidates changed those counts to 1/6 and 3/6, respectively. These baselines use the full data for fitting and scoring. The held-out rows in the table were used for ranking, so a subsequent assessment of the selected front would require fresh data.

## Wave residuals and window size

On early → late splits the median held-out WAPE of the wave equation with its known coefficient 0.04 was 0.874. Refitting this support on the discovery half gave 0.878. With the same native window, the known coefficient gave a median WAPE of 0.0021 on the clean held-out field. On late → early splits the corresponding values were 0.957, 0.959 and 0.00086. No diagnostic reference was inserted in any search or front.

An exploratory check widened the temporal radius only during WAPE validation. Starting from the native radius, it doubled the radius while at least two centers fit, preserving the stride, polynomial order, spatial geometry, trained coefficients, source dyad and block check. This reduced noisy known-coefficient WAPE to 0.333 forward and 0.229 in reverse. Exact-support recovery stayed at zero in both directions. Cleaner validation residuals alone could not rescue these time-split searches.

The spatial follow-up preserved all 81 time coordinates. Nine of the twelve searches selected temporal radius 16. Seven of those nine scored the wave law and all seven retained it on the archive front. The remaining three chose radius 8 or 2, all scored the law, and none retained it on the archive front. This is an association across six paired fields. The split direction, signal and selected geometry vary together.

## Nonlinear diffusion proposals

Among the nine half-record runs that never scored the exact diffusion support, every discovery archive contained all three required terms across its candidates. Each also contained a candidate with the physical time target that was one predictor addition or substitution from the true structure. Such a candidate was present in five terminal populations and three returned fronts. Only one of the nine archives contained the full law plus exactly one extra term. The fit of the known equation on the discovery half gave median held-out WAPE of 0.337 forward and 0.088 in reverse.

The next proposal test is a fixed-budget one-term addition or replacement using terms already evaluated on the discovery data. Compare it with the existing mutation operator under an equal number of candidate evaluations. Use every proposed candidate's WAPE and dyad for competition, keep the held-out observations for validation, and measure exact-support generation separately from final-front survival. The neighbor counts establish reachability. An unbiased proposal operator must still find the relevant edits among many alternatives.

## Reproduction and checks

Run `run_split.py` and `score_split.py` for the temporal experiments and full-record baselines. Use `--space` with each for the wave spatial follow-up. Run `window_check.py` for the temporal sensitivity study and `summarize.py` for the tables. The 48 archives captured 348,072 scoring calls and 200,254 distinct source-eligible fitted points summed over runs. Independent reconstruction reproduced every eligible source WAPE within 2.6e-15 absolute error. Reconstructed first fronts from the terminal populations matched the returned native fronts, and the saved hashes matched across reversed folds. The same six seeds underlie the direction comparisons. These experiments cover two benchmarks at one noise level and search budget.
