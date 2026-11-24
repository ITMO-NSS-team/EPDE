# Longer wave windows and Allen–Cahn transfer

The wave experiment improved the measurement of the physical law on every original archive and recovered it on zero additional fronts. For Allen–Cahn, a majority of the previously winning rivals lost their error advantage on each of three new trajectories. Matched zero-diffusion controls show that much of this reversal can also occur when diffusion is absent. The replay used archived candidates at 3,000 total fits, with coefficients and training stability frozen. [Protocol](protocol.md)

## Wave

The training-only rule selected a temporal half-width of 14 or 16 samples, up from 4 or 8. The estimated weak time-derivative noise fraction met the 10% training threshold on 11/12 archives. On the other archive its lowest feasible value was 17.5%. The physical-coefficient result uses the registered wave law as a measurement control. The discovery results below use only actual archived handovers. [Training decisions](wave_selection.csv)

| Original held-out measurement | Original windows | Selected windows |
|---|---:|---:|
| Median WAPE for the published wave coefficient | 92.8% | 34.2% |
| Median WAPE for the best actual correct equation, among 11 archives containing one | 95.8% | 71.7% |
| Median WAPE of its lowest-error dominating front member, across those archives | 14.6% | 11.0% |
| Correct law on six-candidate list | 0/12 | 0/12 |
| Correct law on full first front | 0/12 | 0/12 |
| Median first-front size | 6 | 7 |

On 16 fresh held-out noise draws shared across the 12 archives, the registered law's median WAPE fell from 93.6% to 34.8%. Both six-candidate lists and both unrestricted first fronts had zero hits in 192 archive–draw combinations per method. All draws used the same clean trajectory. Wider windows also reduced rivals' errors. The frozen training coefficients and stability scores still influence this ranking. [Summary](wave_summary.csv) · [Per-archive scores and returns](wave_results.csv)

## Allen–Cahn

For each archive, a win means the original correct equation had a lower held-out WAPE than its frozen rival, using the median paired difference over 16 noise draws. The first comparison uses the lowest-error front member that dominated the correct equation on the original field. Different rivals include different extra terms. The second compares directly to the archived reaction-only equation when available. Each column reuses the same archives across three periodic initial profiles. The two diffusivities share the initial profile and measurement noise. [Archive-level margins](ac_by_archive.csv)

| Leading spatial mode of initial field | Full law beats original winning rival, diffusion present | Same comparison, diffusion absent | Full law beats reaction-only, diffusion present | Same comparison, diffusion absent |
|---|---:|---:|---:|---:|
| 1 | 9/11 | 9/11 | 2/5 | 2/5 |
| 3 | 10/11 | 7/11 | 3/5 | 1/5 |
| 6 | 8/11 | 7/11 | 3/5 | 1/5 |

All 11 rivals originally dominated on both error and stability. A lower transferred WAPE suffices to break that original domination because stability was frozen. Ten of these eleven pairs predicted the same target. The one cross-target pair lost on every profile and control, so restricting the comparison to common targets gives the same win numerators out of ten. Against the five reaction-only rivals, switching diffusion on improved the paired margin in four of five archives for each higher mode. The median increase was 0.20 percentage points for mode 3 and 0.67 for mode 6. The small number of archived pairs and prescribed profiles limits inference for unfamiliar dynamics. This pairwise test measures transfer of the original ranking and does not rank the full Allen–Cahn archive on a new field. [All paired draws](ac_transfer.csv)

## Reproduction

Run `python wave.py`, `python allen_cahn.py`, then `python summarize.py` from this directory in the EPDE research environment. The wave baseline reproduces the original archive WAPEs and first-front sizes. All Allen–Cahn pairs reproduce the original recorded WAPEs before transfer. A coarse simulation on 512 spatial nodes failed the convergence tolerance on two profiles. Production results use 2,048 nodes, checked against 1,024 with twice the time step. The maximum absolute field difference across the three diffusion-bearing trajectories was below 2.6 × 10⁻⁷ on the observation grid. The zero controls were checked against exact pointwise reaction evolution. [Solver checks](ac_convergence.csv) · [Wave replay](wave.py) · [Allen–Cahn replay](allen_cahn.py)
