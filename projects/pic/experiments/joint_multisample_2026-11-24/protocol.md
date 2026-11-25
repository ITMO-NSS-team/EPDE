# Joint discovery across noisy recordings

The experiment asks whether repeating a measurement or changing the initial condition helps EPDE recover the governing system. Every recording supplied to a search participates in preprocessing, shared parameter fitting and system selection. Recovery means a registered system occurs anywhere on the complete nondominated front. All measurements are used for discovery, so recovery and coefficient errors describe this benchmark and do not measure prediction on unseen recordings.

## Using multisample discovery

Pass one grid and a dictionary of fields per recording. Use the same variable names, time units, forcing phase and coordinate convention in every recording. Each recording keeps its own grid and derivative calculation. The joint fit estimates one set of physical coefficients per equation. The `samples` weighting gives each recording equal weight even when the grids have different lengths.

```python
import epde
from epde.interface.results import DiscoveryResult

samples = [(t_first, {"u": u_first}), (t_second, {"u": u_second})]
with epde.EpdeMultisample(
    use_solver=False, device="cpu", multiobjective_mode=True, engine="nsga2",
    system_rps_mode="plain", multisample_mode="joint", multisample_weighting="samples",
    evaluation_form="weak", discrepancy_metric="wape", instability_metric="dyad",
    multisample_discrepancy="per_record", population_size=8, training_epochs=50,
) as search:
    search.fit(samples=samples, max_deriv_order=(2,))
    result = DiscoveryResult.from_search(search)
    for fitted_system in result.front:
        print(list(zip(fitted_system["objective_labels"], fitted_system["objectives"])))
```

For each recording and equation, WAPE is the sum of absolute residuals divided by the sum of absolute target values on that recording's evaluation rows. A zero denominator makes the candidate ineligible. Weak evaluation uses the weak-form design rows. Dyadic coefficient instability measures sensitivity to the fitting windows. Three recordings and two equations give this eight-entry vector.

```text
WAPE(record 1, u), WAPE(record 1, v),
WAPE(record 2, u), WAPE(record 2, v),
WAPE(record 3, u), WAPE(record 3, v),
instability(u), instability(v)
```

All objectives are minimized. A system dominates another only if it is at least as good on every entry and strictly better on at least one. The search retains distinct fitted systems with equal vectors and systems that trade off objectives, including fits sharing the same symbolic terms. An identical fitted system encountered twice counts once. The plain fitter uses its existing equation-level eligibility checks, including a mean WAPE ceiling. NSGA-II uses Pareto levels and crowding for its bounded working population. The returned front includes eligible systems encountered throughout the search, including siblings evaluated before that population is truncated.

## Paired benchmark

Each of five noise draws covers the forced oscillator (`ode`), Van der Pol (`vdp`), Duffing (`duffing`) and Lotka–Volterra (`lv`). Each system has five arms.

| Arm | Recordings | Initial conditions |
|---|---:|---|
| `one` | 1 | Base state |
| `repeat3`, `repeat5` | 3, 5 | Base state in every recording |
| `diverse3`, `diverse5` | 3, 5 | Base state followed by two or four prescribed alternative states |

All clean trajectories are integrated with DOP853 using the earlier trajectory experiment's states. Each observed field receives stationary Gaussian AR(1) noise with lag-one correlation 0.8 and nominal standard deviation 5% of that clean field's standard deviation. The noise seed is `948000 + 10 * draw + recording_index`. The first recording is identical across all arms within a draw. Three-record arms are nested in their respective five-record arms. Repeated and diverse measurements at corresponding positions use the same standardized innovations. The search seed is fixed per system at 947301–947304 in the order above.

The scalar systems receive 50 epochs and Lotka–Volterra receives 150. Each search uses a working population of eight, fourth-order finite-difference preprocessing and weak-form evaluation. The archive receives eligible initialization siblings before NSGA-II applies the population cap. The epoch logs record the generation count.

## Frequency fitting throughout the search

The pilot estimates up to two spectral peaks from the supplied recordings. The revised policy allows sine and cosine to appear alone or in a product. At initialization and in every epoch, an eligible fitted equation receives a continuous frequency fit in each observed band on its axis. Eligibility requires a target free of trigonometric factors and predictors whose sine and cosine factors share one frequency on one axis. A run-local cache reuses fits when the equation slot, target, predictor terms and intercept choice recur, including when a child inherits a different starting frequency. There is no encounter-count cutoff.

Within each band, the default `systematic` policy samples 17 evenly spaced frequencies, including the edges, plus the spectral seed. It refines each sampled local minimum within its neighboring grid points, including one-sided minima at the band edges. Very narrow minima can fall between grid points. The earlier `per_support` policy used one bounded minimization across the entire band. At each trial frequency, one linear coefficient vector is fitted to all weak-form rows, weighting each recording equally. The original fits and eligible alternatives from every resolved minimum enter the system search. Pareto dominance uses every recording’s WAPE for every equation and each equation’s instability.

## Structural proposals

The default pilot disables the residual-ranked fill mutation. Alongside ordinary evolutionary proposals, it tests each way to remove one predictor from an encountered equation that has at least two predictors. For a single-factor target it also traverses combinations of single-factor library terms, advancing one combination of each permitted size per visit. This schedule uses the configured term library and excludes terms related to the target and the targets of partner equations. Evolution continues to propose products. When several equations need fitting in a coupled system, visits rotate structural expansion among their slots. Available epochs and visits determine how much of the library is explored.

For each matched draw, the report also pools the two policies’ complete fronts and recomputes dominance on every objective. A system dominated in its own run is already dominated in the pooled evaluation. The combined front therefore includes all ties and trade-offs from both evaluated pools. Its rows identify the source archive and system index for each surviving fit.

The controller admits at most two searches concurrently. Each worker uses single-threaded numerical libraries, below-normal Windows priority and the existing six-of-eight logical CPU affinity mask. The shared admission check requires host load at or below 62.5% before starting a worker.

## Reproduction and artifacts

Run these commands from this directory. Completed archives are checked against their SHA-256 receipts and reused.

| Command | Output |
|---|---|
| `python pilot.py --recheck-misses` | Repeats the prior per-support pilot’s oscillator and Duffing front misses on the same recordings and seeds. |
| `python pilot.py` | Runs or resumes the full 100-search design under `systematic/`. |
| `python report.py --partial` | Regenerates the [current findings](systematic/findings.md) for committed runs, including paired comparisons with the preceding policy. Omit `--partial` after all 100 runs. |
| `python pilot.py --frequency-policy per_support --recheck-misses` | Repeats the historical 34-run comparison with one minimization per band. Its [findings](per_support/findings.md) remain available. |
| `python pilot.py --worker lv diverse5 0 --smoke` | Runs one epoch of the coupled benchmark. |
| `python pilot.py --frequency-policy first12` | Reproduces the historical first-12 policy under `results/`. Use the same option with `report.py` to regenerate its report. |

The [spectral-seed baseline](spectral_seeds_only/findings.md) contains the earliest 100 searches. A generator truth test inadvertently skipped continuous frequency fitting. The subsequent [first-12 pilot](findings.md) profiled the first twelve eligible structures in encounter order. Its 100 archives remain under `results/`. The [per-support comparison](per_support/paired_policies.csv) and [systematic-proposal comparison](systematic/paired_policies.csv) use separate archives and verify the sample hashes, seeds, population and epoch settings for each pair.

Every `results/<system>/<arm>/<draw>/archive.json.gz` stores sample hashes, seeds, options, exact term parameters, fitted equation points, evaluated system combinations, indices of nondominated systems, and the frequency-profile log. Each equation point contains its variable, target label index, signed term coefficients, mean WAPE, dyadic instability and WAPE for each recording. System indices identify actual combinations produced by the search. An independent nondominated sort of those combinations must reproduce the exact front before the worker commits its archive.

Each policy report summarizes catalog-law generation, membership on the front, front sizes and structure recurrence across noise draws. The current [coefficient ranges](systematic/coefficient_ranges.csv.gz) give within-draw and across-draw minima and maxima for aligned structures appearing in at least two draws in an arm. The [catalog errors](systematic/benchmark_parameter_errors.csv) cover every matching system on the front, normalized to a common pivot. Structural matching allows catalog equation permutations and the benchmark's 1% relative frequency tolerance. The [per-search results](systematic/runs.csv) preserve the draw pairing. These ranges are descriptive and the optimizer seed is fixed.
