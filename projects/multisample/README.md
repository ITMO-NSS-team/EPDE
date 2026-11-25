# Multisample fitting

Set `multisample_mode='joint'` on `EpdeSearch` or `EpdeMultisample` to fit
shared coefficients across trajectories. The default `'mean'` averages
coefficients fitted separately. Each record has its own grid for preprocessing
and instability scoring. Use the same variables in the same order in every
record.

Pass a list of `Trajectory` objects to `EpdeSearch.fit(data=...)`. Create
each trajectory with `createDomain` and `createTrajectory`. `EpdeMultisample`
also accepts `(grids, fields)` pairs with a dictionary of named arrays or a
single array for one variable. Custom domain weights must be positive
throughout the interior and zero on the excluded boundary.

`multisample_weighting='samples'` gives every record the same total regression
weight. `'points'` keeps the original row weights. The discrepancy and
instability objectives use their usual weights.

Joint fitting works in the `plain`, `assignment`, and `repair` right-part modes.
Joint `repair` uses VWSR with a basis-free instability metric such as `dyad`
or `chi2`. Its sparse fit uses normal equations, so extreme magnitudes may
require rescaling the data.

## Comparing complete systems across recordings

Use `multisample_discrepancy='per_record'` with solver-free multiobjective
NSGA-II, WAPE and instability to expose every recording's error to the
optimizer. Pair it with `multisample_mode='joint'` to fit one coefficient
vector per equation. With R recordings and E equations, each system has
R × E WAPE values ordered by trajectory ID and then equation, followed by E instabilities.
The default discrepancy setting, `'mean'`, averages each equation's WAPE
over its recordings.

Read the complete system front from `DiscoveryResult.from_search(search).front`.
Each system carries `objective_labels` alongside its `objectives`.
Distinct fitted systems with equal objectives and incomparable fits of the
same structure remain on the front. For a runnable example, the noise
design and numerical output, see the [joint discovery protocol](../pic/experiments/joint_multisample_2026-11-24/protocol.md).
