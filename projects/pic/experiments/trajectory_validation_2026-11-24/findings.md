# Validating discovered equations on additional trajectories

Twelve new full-record searches used 5% correlated measurement noise. The additional recordings were simulated from the benchmark dynamics. Each method scores the same candidates from a given run. The original-record comparison reconstructs the ordinary archive front. The native optimizer can return a different front.

## Multisample Pareto front

Each recording contributes a separate weak-form equation-fit error (WAPE) for each equation. Each equation also contributes its training coefficient instability. All objectives are minimized. The mean-WAPE eligibility rule is shared across comparisons. The full front retains every nondominated archived system, including ties. Archive order is for lookup. Held-out errors are available per recording and equation for every front member.

| Validation and front construction | Correct law on front | Median front size |
|---|---:|---:|
| Original recording | 4/12 | 17.5 |
| New initial states, averaged WAPE per equation | 7/12 | 12.5 |
| Same initial state, separate recording objectives | 6/12 | 57.5 |
| New initial states, separate recording objectives | 7/12 | 56 |

| Dataset | Generated | On multisample front with new initial states |
|---|---:|---:|
| Forced oscillator | 3/3 | 3/3 |
| Van der Pol | 3/3 | 2/3 |
| Duffing | 0/3 | 0/3 |
| Lotka–Volterra | 3/3 | 2/3 |

## Per-equation WAPE and instability fronts

Each cell counts runs whose front contains a registered law. The denominator counts evaluable runs.

| Dataset | Original recording | New draws, same trajectory | New initial states | Disjoint halves | Buffered halves |
|---|---:|---:|---:|---:|---:|
| Forced oscillator | 1/3 | 1/3 | 3/3 | 0/3 | 0/3 |
| Van der Pol | 1/3 | 1/3 | 2/3 | 1/3 | 1/3 |
| Duffing | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Lotka–Volterra | 2/3 | 3/3 | 2/3 | 2/3 | 2/3 |
| All four | 4/12 | 5/12 | 7/12 | 3/12 | 3/12 |

## Retrospective controls

Each cell counts runs whose front contains a registered law.

| Measurements | Original recording | New draws, same trajectory | New initial states, mean WAPE | Buffered halves | New initial states, multisample front |
|---|---:|---:|---:|---:|---:|
| clean | 6/12 | 6/12 | 6/12 | 6/12 | 6/12 |
| gaussian | 8/12 | 9/12 | 9/12 | 7/12 | 9/12 |
| correlated | 1/12 | 1/12 | 9/12 | 2/12 | 9/12 |

On clean Lotka–Volterra data the original archive front recovered the system in 3/3 runs and new-state validation recovered it in 0/3. This limits the case for a global scoring change.

## Single-record confirmation

Six separate searches saw only the first 153 samples. The adjacent and buffered validation segments each have 153 samples. The 14-sample buffer also moves the validation segment later in time.

| Dataset | Generated on prefix | Training front | Adjacent segment | Buffered segment |
|---|---:|---:|---:|---:|
| Forced oscillator | 1/3 | 1/3 | 1/3 | 1/3 |
| Van der Pol | 3/3 | 0/3 | 0/3 | 1/3 |

Three seeds per dataset share the stored discovery trajectory and the same two validation initial states. Different noisy recordings are independent draws of measurement error. Real-data use requires additional measurements on the specified coordinates with compatible forcing. The earlier full-record disjoint and buffered halves reuse the discovered library and refit coefficients. In the six prefix searches above, both validation segments score coefficients fitted only on the prefix.

Reintegrating the catalog models from the stored start gave relative deviations of ode 1.2%, vdp 0.2%, duffing 0.0%, lv 4.1%/3.8% on the rest of each original trajectory. For the second-order systems the first two position samples determine the initial velocity. The two Lotka–Volterra figures correspond to its two fields. [Simulator check](simulator_check.json)

## Forced-KdV dispersion measurement

Twelve fresh paired cases per noise family. Each estimator uses the catalog support. The average and cross-record projection see the same two independently measured fields. The true coefficient is −1. Errors and coefficients are medians. Held-out WAPE uses a third recording and the same raw weak columns for every arm.

| Noise | Method | Relative dispersion-column error | Coefficient | Absolute coefficient error | Clean WAPE | Held-out WAPE |
|---|---|---:|---:|---:|---:|---:|
| gaussian | single | 2.9563 | -0.0015 | 0.9985 | 0.5389 | 0.5454 |
| gaussian | average | 2.1238 | -0.0036 | 0.9964 | 0.5381 | 0.5515 |
| gaussian | cross_record | 0.6569 | -0.0213 | 0.9787 | 0.5395 | 0.8183 |
| correlated | single | 9.8575 | 0.0001 | 1.0001 | 0.5403 | 0.5845 |
| correlated | average | 7.4841 | 0.0000 | 1.0000 | 0.5395 | 0.5861 |
| correlated | cross_record | 2.2139 | -0.0073 | 0.9927 | 0.5980 | 2.9785 |

### Calibration on an unforced record

Each method gets two 81 × 81 recordings. The average uses two forced measurements. The joint and staged fits use the same forced and unforced measurements. Staging estimates the two intrinsic coefficients from the unforced record before fitting the forcing amplitude on the forced record. All medians below use a separate cohort of 12 cases per family.

| Noise | Method | Dispersion coefficient | Absolute error | Clean forced WAPE | Noisy forced WAPE | Noisy WAPE at exact coefficients |
|---|---|---:|---:|---:|---:|---:|
| gaussian | average_forced | -0.0037 | 0.9963 | 0.5376 | 0.5488 | 13.2100 |
| gaussian | joint | -0.0019 | 0.9981 | 0.6199 | 0.6280 | 13.2100 |
| gaussian | staged | -0.9916 | 0.0097 | 0.5326 | 13.1981 | 13.2100 |
| correlated | average_forced | 0.0000 | 1.0000 | 0.5403 | 0.5573 | 50.0367 |
| correlated | joint | 0.0001 | 1.0001 | 0.6237 | 0.6398 | 50.0367 |
| correlated | staged | -0.9549 | 0.0489 | 2.1038 | 48.9817 | 50.0367 |

The staged calibration improved the dispersion coefficient under both noise families. On the noisy forced recording, even the exact physical coefficients gave a large weak residual. Use the calibration result when comparable unforced measurements and a shared intrinsic law are available. Identifying the full forced law through search remains an open question.

## Navier–Stokes generation

Each entry counts archives whose actual system handovers contained the named complete equation. The last column asks whether the three registered equations could be combined from different handovers even allowing arbitrary slot reassignment.

| Noise | x-momentum | y-momentum | Continuity | Assembled full system | All required donors available |
|---|---:|---:|---:|---:|---:|
| gaussian | 2/3 | 3/3 | 3/3 | 2/3 | 2/3 |
| correlated | 0/3 | 1/3 | 3/3 | 0/3 | 0/3 |

For the three correlated-noise seeds, candidate generation for x-momentum is the first bottleneck. A ranking change can only select equations already present in the archive.

## CPU use

The full-record launch logs recorded host CPU use of 22.3% to 54.0% at admission. The shared logical CPU pool was [0, 1, 2, 3, 4, 5]. The experiment ran up to two single-threaded search workers at below-normal priority.

[Protocol and reproduction](protocol.md) · [Per-run fronts](pareto_runs.csv) · [Dispersion measurements](kdv_measurement.csv) · [KdV calibration](kdv_calibration.csv) · [Navier–Stokes donor audit](ns_donors.csv)
