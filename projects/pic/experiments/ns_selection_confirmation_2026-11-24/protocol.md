# Joint equation selection on clean Navier–Stokes

Use search seeds 24112401, 24112402 and 24112403 on all 200 × 50 × 100 measurements of the same clean flow. All methods share one native search trajectory per seed. Keep the full-record experiment's plain-mode weak-form NSGA-II settings, native population sizing, frequency profiling, WAPE, dyadic coefficient instability and 150 epochs with patience disabled.

Use the native front after epoch 100 for one round of candidate assembly. Joint reassignment uses only the equations already together in each parent. It tries admissible assignments to fields and derivative targets. The exchange arm additionally replaces one equation with a copy of another equation from the same parent under field and axis exchanges allowed by the observed primitive vocabulary. Both arms check that relabelling the fields and axes produces exactly the corresponding assignments. Benchmark laws enter only when scoring finished candidates.

The follow-up exchange-only comparison uses the same saved parents and field and axis maps. Replace just the outgoing slot. Carry the donor’s derivative target through the same map and require it to be admissible in the destination. In a second variant, try every admissible target of the copy in that slot. In both cases retain the exact fitted equations in every other slot. Verify equivariance under field and axis relabelling for each variant. These comparisons were added after the original three-arm results.

For each seed the native arm sets the fit budget with its 150-epoch run. Charge one fit for each new support–target pair attempted on the original weak rows, including rejected fits. Reuse an identical fit already handed over by the checkpoint. At the exact checkpoint clock, only observed front members are reused to avoid ambiguity about handover order. Exclude later terms and points from the proposal inputs. If proposals exceed the remaining budget, admit entire groups in increasing order of support size, factor count, total derivative order and pivot order. Skip groups that do not fit. The assembly arms spend the remaining fits on the same native continuation. Proposals enter the comparison together as complete systems and do not feed back into evolutionary reproduction.

Record whether a single candidate contains both momentum equations and continuity, allowing the registered continuity-equivalent rewrites. For each arm compare every matching candidate against every admitted handover and assembly on the six per-slot objectives. Use EPDE's exemption for matching equation structures in a slot. Also record the number of matches on the optimizer's own last completed native epoch front. Its incremental archive keeps one representative per structure and can differ from a comparison against all historical candidates.

Replaying seeds 923701–923703 at their final epoch checks what caused the earlier gain. In that retrospective comparison the assembly fits are additional. Report those results separately from the fresh equal-budget comparison.

Run the following commands in PowerShell from this experiment directory for each fresh seed. Use Python from the EPDE environment.

```powershell
$env:PYTHONHASHSEED = '0'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
python discover.py 24112401
python score.py 24112401
```

After scoring the six archives, run `python placement.py 24112403` and `python placement.py 923701` to trace the ordering of the complete systems on the two seeds where joint reassignment gained a Pareto-front hit over choosing a target for the copy. Then run `python report.py` to rebuild the tables.

Run `python momentum_beam.py` with the same environment to trace the first 5,000 equation evaluations on seed 24112402. Compare the replayed equation records with the archive and check the full 150-epoch history for both momentum laws. When all required predictors are admissible and both forms miss the five-predictor shortlist, replay that exact beam at widths 1,000, 2,000 and 4,000. Track the best-ranked subset of each accepted form at each size. As a separate check, rank the forms using the most frequent observed partner-target combination for each velocity time derivative. [Beam and retention audit](results/24112402/momentum_beam.json)

## Reusing equations across parents and epochs

Run `python donor_history.py SEED` for each of the three archived confirmation seeds. Compare donors from the epoch-100 front with donors from the union of epoch-end fronts 1 through 100. Both rules allow the three distinct equations to come from different parents. They admit systems with three observed equation structures, or two observed structures and one copy of either under the existing field and axis exchanges. Try all admissible derivative targets in each placement. Use the same observations, weak rows, refit charges and native continuation budget as in the previous comparison.

Donor selection and fitting use observations and search history alone. During assessment, compare each accepted complete system against every admitted native handover and the full set of candidate combinations. Enumerate competitors in batches and stop testing a hit once a dominator is found. These are additional comparisons on the seeds that motivated the retention rule.

## Independent search-seed confirmation of donor retention

Use the next three unused seeds 24112404, 24112405 and 24112406 on the same clean full record. Apply the two cross-parent donor rules above at epoch 100, under the same per-seed fit budget from 150 native epochs. Keep beam width 1,000 and the seven-term cap. The primary endpoint is at least one complete system on the global first front. Tabulate paired gains and losses, systems assembled, charged refits and elapsed time for each arm. Time each arm with an empty weak-column cache. Record native elapsed time at its allotted fit cutoff separately from assembly and assessment time. Run `discover.py` and then `donor_history.py` with each seed.

As a sensitivity check, challenge primary survivors with all later native handovers through epoch 150. Build the paired tables with `python confirm_retention.py`. For the earlier missed seed, `python donor_dominance.py` reconstructs the twelve complete candidates, searches all budget-admitted native rivals and reports a least-changed witness per candidate. [Dominance audit](dominance_24112401.md)
