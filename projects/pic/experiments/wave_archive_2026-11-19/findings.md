# Wave recovery before and after Pareto pruning

Six fresh ten-epoch solver-free WAPE–dyad searches each evaluated both orientations of the wave law. Three searches retained it in their final population, and all three returned it on their first front. The experiment saved 42,494 scoring calls and 25,727 distinct fitted points, including source-ineligible fits. Among the 24,834 points with valid source objectives there were 23,354 distinct support–target combinations, counted separately per seed. All archived coefficients and discovery dyads were frozen for validation. Reconstructing the training WAPE directly from the saved coefficients reproduced every eligible score within 1.2e-15 absolute error.

Recovery counts below compare the candidates returned by EPDE, the entire terminal population and the archive of all eligible evaluated fits. Each cell counts noisy observations on which the exact two-derivative support appeared anywhere on the WAPE–dyad front after the blocked-WLS eligibility check. Validation used no inserted equation. Each validation profile had six independently fitted archives and eight paired noise draws per archive.

| Validation field | Returned front | Terminal population | Scored archive |
|---|---:|---:|---:|
| Original discovery field with fresh noise | 24/48 | 24/48 | 32/48 |
| Original plus third mode at benchmark speed | 24/48 | 24/48 | 48/48 |
| Original plus fourth mode at benchmark speed | 24/48 | 24/48 | 48/48 |
| Independent three-mode wave at speed 0.20 | 24/48 | 24/48 | 48/48 |
| Same initial conditions at speed 0.22 | 16/48 | 16/48 | 40/48 |
| Same initial conditions at speed 0.40 | 0/48 | 0/48 | 0/48 |

Even on the original discovery measurements, the offline source archive contained the exact law on its own front in 4/6 searches, compared with 3/6 returned by EPDE. Changing the initial conditions provided additional separation. On the independent wave the ungated archive front also retained the law in 48/48 observations. The blocked check reduced the median number of surviving support–target pairs from four to three, with two to eight surviving after the check. Two common competitors were the wave support plus a fourth spatial derivative (20/48 across three seeds) and an affine spatial relation between the field and its second spatial derivative (16/48 across two seeds).

On seed 919103, noise draw 7, two nondominated candidates had the same spatial-second-derivative target. The wave law had validation WAPE 0.10184 and discovery dyad 0.05646. Adding a fourth spatial derivative lowered WAPE to 0.09150 and raised dyad to 1.08059. Both survived the blocked check. WAPE evaluates the weak residual with fixed coefficients. Dyad refits the same support across dyadic slabs and measures the relative spread of each coefficient. The two scores reveal the tradeoff directly.

On the speed controls the structural wave law is still true, but its physical coefficient changes from 0.04 to 0.0484 or 0.16. WAPE and dyad alone retained the originally fitted structure in 48/48 and 44/48 checks respectively across the full archive. Adding the blocked check gave 40/48 at the smaller shift and 0/48 at double speed. The eight rejections at speed 0.22 all came from one discovery seed. The cut at one detects the large shift in this experiment and frequently accepts the smaller mismatch.

The independent field was generated analytically, using three standing modes and nonzero initial velocity. It shares no measurements with the discovery field. The repository copies of the standard 81-by-81 benchmark examined here contain identical measurements, so transfer to a second measured trajectory remains open. The 48 checks per profile share six fitted archives and a prescribed initial condition. The results measure candidate retention under those conditions.

For reproduction, run `capture.py`, `validate.py` and `report.py` in order. The archive and validation files in `results` preserve candidate identifiers, weights, objectives and all front memberships. The aggregations are in `summary.csv`, `paired_results.csv` and `primary_supports.csv`.
