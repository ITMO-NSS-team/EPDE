# GOLEM as EPDE's evolutionary backend

`epde_golem` runs EPDE's equation discovery on [GOLEM](https://github.com/aimclub/GOLEM)'s
evolutionary optimizer instead of the native MOEA/D-DD engine, keeping every
domain component identical, and benchmarks the two against each other on EPDE's
own functional-test scenarios.

Using it is a one-line change:

```python
from epde_golem import EpdeGolemSearch as EpdeSearch   # instead of epde.interface.interface.EpdeSearch
```

## Requirements

**This needs a patched GOLEM.** The backend passes `collect_garbage`,
`mating_selection_types` and `SelectionTypesEnum.nsga2`, none of which exist in
released GOLEM. They come from two pull requests to that project; until those
land, apply `benchmarks/golem/golem_required_patches.diff` to a GOLEM checkout:

```bash
git -C /path/to/GOLEM apply /path/to/EPDE/benchmarks/golem/golem_required_patches.diff
pip install -e /path/to/GOLEM
```

GOLEM requires Python >= 3.10. It caps `scipy < 1.13` while EPDE pins
`scipy == 1.15.2`; the environment used here resolved to scipy 1.12, which both
work with.

## Layout

```
epde_golem/                     the backend package
epde_golem/tests/               its tests (16)
benchmarks/golem/               benchmark harness, scenarios, scoring, probes
benchmarks/golem/results/       raw records and figures (written by the harness)
```

Raw benchmark records are not committed; `benchmarks/golem/run_all.sh`
regenerates them.

## What the integration does

`epde_golem` swaps only the *population engine*. Everything that defines the
search problem stays EPDE's:

| component | source |
|---|---|
| candidate representation (`SoEq`: a system of `Equation`s over token products) | EPDE |
| initial population (`SystemsPopulationConstructor`, plus EPDE's hard reject of fit-degenerate candidates) | EPDE |
| mutation (`SystemMutation` → `EquationMutation` → `TermMutation`) | EPDE |
| crossover (`ChromosomeCrossover` → `EquationCrossover` → `TermCrossover`) | EPDE |
| evaluation (right-part selection → sparse regression → LinReg coefficients → objective readers) | EPDE |
| **selection, survival, population schedule** | **GOLEM** |

The operators are not re-implemented: `extract_epde_operators` reaches into the
strategy director EPDE already assembled and reuses those exact instances, so
both arms run byte-identical domain operators with byte-identical parameters.

The initial population is vetted the same way in both arms. EPDE's
`InitialParetoLevelSorting` treats a fit-degenerate chromosome (one whose sparse
regression collapsed it) as a *hard* reject and re-rolls it; that is a property
of the problem, not of MOEA/D, so the GOLEM arm applies it too — otherwise the
comparison would partly be measuring initialisation quality. Duplicates are left
alone: MOEA/D needs one distinct solution per weight vector, GOLEM does not.
Every vetting evaluation is counted, so the arms stay budget-comparable.

Concretely, what differs between the arms:

| | EPDE native | GOLEM |
|---|---|---|
| selection | PBI decomposition over Das–Dennis weight vectors, Gale–Shapley solution↔weight marriage | SPEA-2 |
| survival | per-sector crowding, replace the PBI-worst neighbour | steady-state: SPEA-2 over parents ∪ offspring |
| archive | Pareto levels of the population | unbounded Pareto front (`ParetoFront`) |
| diversity | weight sectors + exact-duplicate history | structural-uniqueness check every N generations |
| schedule | fixed | adaptive operator probabilities, optional bandit mutation agent |

### Files

* `epde_golem/graph.py` — `SoEqGraph`: a GOLEM `OptGraph` mirroring the system
  structure (term nodes → equation nodes → system node) and carrying the `SoEq`
  as payload. The mirror is what gives GOLEM a meaningful `descriptive_id`, so
  dedup, structural diversity and history all work on the real structure.
* `epde_golem/operators.py` — EPDE's mutation/crossover wrapped in GOLEM's
  operator protocol, plus a verification rule and an extra "grow" mutation
  exposed as a separate action so GOLEM's bandit agent can learn when to use it.
* `epde_golem/objective.py` — EPDE's evaluation chain behind a GOLEM
  `Objective`; the first metric evaluates, the rest read the cached vector.
* `epde_golem/optimizer.py` — `GolemEpdeOptimizer` (interface-compatible with
  both `MOEADDOptimizer` and `SimpleOptimizer`) and `EpdeGolemSearch` (an
  `EpdeSearch` subclass that builds it). Using it is a one-line change:

```python
from epde_golem import EpdeGolemSearch as EpdeSearch   # instead of epde.interface.interface.EpdeSearch
```

Both of EPDE's modes are covered. `multiobjective_mode=True` replaces
MOEA/D-DD; `multiobjective_mode=False` replaces `SimpleOptimizer`, with GOLEM
running tournament selection over EPDE's scalar objective. The two modes are
assembled by different EPDE directors with different block labels and different
operator nesting, and `extract_epde_operators` handles both — including the
single-objective mutation's "only elites are immutable" contract, which the
native pipeline stamps in a separate elitism stage that GOLEM replaces.

## Running the benchmark

```bash
PYTHONHASHSEED=0 python benchmarks/golem/run_benchmark.py \
    --scenarios wave burgers kdv allen_cahn van_der_pol lotka_volterra \
    --seeds 0 1 2 3 4 --pop-size 16 --epochs 2 --tag main
python benchmarks/golem/analyze.py benchmarks/golem/results/benchmark_main.jsonl
```

`PYTHONHASHSEED=0` is not optional for reproducibility — see the findings below.

Three arms per (scenario, seed):

* `native` — EPDE as shipped.
* `golem_eq_gen` — GOLEM with the same population size and generation count.
* `golem_eq_budget` — GOLEM run until it has spent the same number of
  equation-level fitness evaluations the native arm consumed. One MOEA/D
  generation costs many GOLEM generations (MOEA/D visits every weight sector
  separately and retries for unique offspring), so this is the like-for-like
  comparison.

The scenarios are transcribed from `tests/functional/scenarios/`, i.e.
EPDE's own examples, together with their analytic ground-truth equations.

Every run records four recovery criteria, because on this domain they come
apart:

* **front** — the true equation is on the final non-dominated front (what
  `EpdeSearch.equations()` prints);
* **kept** — it is anywhere in the returned set (front plus dominated
  survivors);
* **up-to-factor** — the true law was found multiplied by a redundant common
  token, e.g. `u³·(u_tt − 0.04 u_xx) = 0`;
* **ever** — it was on the front at the end of *some* generation, even if it
  was displaced later.

`benchmarks/golem/run_all.sh` runs the full matrix; `benchmarks/golem/analyze.py` prints
the table; `benchmarks/golem/plots.py` draws the figures.

## Results

Six scenarios, five seeds, population 16. "Recovery" is the fraction of runs
whose final non-dominated front contains the analytic ground-truth equation.

### With EPDE's shipped objectives (discrepancy + coefficient instability)

| arm | recovery | recovery (ever on the front) | time | evaluations |
|---|---:|---:|---:|---:|
| EPDE (MOEA/D-DD) | 73.3% | 80.0% | 118.6 s | 13946 |
| GOLEM, same generations | 53.3% | 53.3% | 4.4 s | 560 |
| GOLEM, same evaluation budget | 83.3% | 96.7% | 117.8 s | 14011 |

### With discrepancy + structural complexity

| arm | recovery | time | evaluations |
|---|---:|---:|---:|
| EPDE (MOEA/D-DD) | 83.3% | 111.3 s | 14134 |
| GOLEM, same generations | 50.0% | 4.1 s | 544 |
| GOLEM, same evaluation budget | **100%** | 109.6 s | 14204 |

Three things fall out of this.

**The engines are equivalent on single equations and differ on systems.** On
wave, burgers, kdv, allen_cahn and van_der_pol both reach the same median
discrepancy — to the printed digits — so at these budgets the problem is not
optimizer-limited. The entire gap is `lotka_volterra`, the one scenario that
searches for a *system* of two equations: 0% for MOEA/D-DD in all five seeds,
100% for GOLEM in all five, at the same budget and the same wall clock.

**"Same generations" is not the same amount of work.** One MOEA/D-DD epoch
visits every weight sector, so it costs O(pop_size) sector passes where a
generational EA costs one. Keeping `population_size=16, training_epochs=2` and
swapping the engine therefore buys a 25× cheaper, 27× faster — and
correspondingly shallower — search. Both numbers are real; they just answer
different questions.

**The objective set matters more than the engine.** With `use_pic=True` the
second Pareto axis is coefficient instability, and *nothing* in the objective
penalises extra terms. A stronger optimizer then legitimately finds
lower-discrepancy forms that are the true law times a redundant factor, or a
sum of two such copies — mathematically the same PDE, structurally not its
parsimonious form. Recovery on `wave` goes 20% → 100% for both engines by
changing that axis to complexity, a bigger swing than anything the choice of
engine produced.

### Can GOLEM be improved *for this domain*?

Mostly no, and the data says why. Nine configurations, 30 runs each, all at 10%
of the native evaluation budget:

| configuration | recovery | time | median discrepancy |
|---|---:|---:|---:|
| baseline (steady-state + SPEA-2) | 73% | 11.3 s | 1.896e-03 |
| + NSGA-II survival selection | 77% | 11.2 s | 1.896e-03 |
| + structure-keyed fitness cache | 77% | 12.2 s | 1.896e-03 |
| population 32 instead of 16 | 67% | 11.6 s | 1.896e-03 |
| population 48 instead of 16 | 70% | 11.4 s | 1.896e-03 |
| 3 restarts, budget split | 77% | 11.7 s | 1.498e-03 |
| 5 restarts, budget split | 70% | 11.3 s | 1.896e-03 |
| 8 restarts, budget split | 67% | 10.4 s | 1.896e-03 |
| 5 restarts + NSGA-II | 73% | 11.4 s | 1.498e-03 |

The spread is 67–77% against a ±8pp standard error on 30 binary trials: nothing
here is distinguishable from the baseline. The median discrepancy column is the
explanation — every configuration converges to the same value. Changing
selection, operator set, population size or restart count changes how the
search gets there, not where it arrives.

Two of the changes were expected to help and did not, in an informative way.
Mating-pool selection with replacement genuinely fixes a missing mechanism (see
finding 2 below), but *adding* selection pressure to a landscape where the
recovery failure is over-convergence makes things slightly worse (67% vs 73%).
The structure-keyed fitness cache hits 2–5% of the time: EPDE's mutation
randomises whole terms, so identical structures rarely recur and there is
nothing to memoize.

`lotka_volterra` is the honest counter-example to all of it: 0% for every one of
the nine configurations at 10% budget, 100% at full budget. There the binding
constraint is the number of evaluations, and no amount of selection cleverness
substitutes for it.

Two follow-ups checked that 10% was not simply past the point where the search
algorithm still matters. A harsher budget (3% of native, 30 runs per
configuration) separates nothing either — baseline 57%, NSGA-II 57%, cache 57%,
3 restarts 53%. And a budget ladder on `lotka_volterra` runs the *other* way:

| budget | baseline | + NSGA-II | 3 restarts |
|---:|---:|---:|---:|
| 20% | 20% | 0% | 0% |
| 40% | 20% | 20% | 20% |
| 60% | 80% | 20% | 20% |
| 100% | 100% | — | — |

On the one scenario that needs sustained convergence, splitting the budget
across restarts or spreading the front with crowding distance *costs*
recovery. Five seeds put a ~18pp standard error on those numbers, so this is a
direction rather than a proof — but it is the opposite of the expected one, and
it is consistent with the rest: this domain rewards spending the budget, not
redistributing it.

**What would actually move the needle**, on this evidence, is not the
evolutionary engine at all:

1. Use complexity, not coefficient instability, as the second Pareto axis —
   worth 20 percentage points of recovery, more than any engine change.
2. Spend more evaluations on systems of equations; single equations saturate
   early and extra search only buys overfitted forms.
3. Reduce the cost *per* evaluation. The right-part-selection sweep evaluates
   every term as a candidate target, so a candidate system costs roughly
   (terms + 1) fitness evaluations. That is a domain-side cost inside EPDE, and
   it dominates everything the optimizer does.

## Findings about GOLEM

### 1. `gc.collect()` once per individual (fixed)

`BaseGraphEvaluationDispatcher._evaluate_graph` ran a full `gc.collect()` after
*every* graph evaluation. A gen-2 collection walks the whole live heap, so on a
domain whose objective keeps a large working set alive (EPDE's token pool,
cached derivative tensors, torch) it cost ~85 ms per graph — about 70 % of the
optimiser's wall clock, more than the evaluations themselves.

Fix: move it to once per evaluated population, and make it switchable via
`OptimizationParameters.collect_garbage`. Measured on the `wave` scenario
(pop 8, 2 generations): 2.7 s → 0.8 s, identical results. GOLEM's own unit
suite (478 tests) passes unchanged.

### 2. Multi-objective + generational scheme has no selection pressure (warned)

In multi-objective mode:

* elitism is deliberately disabled (`Elitism._is_elitism_applicable` returns
  `False` when `multi_objective`, and there is a test pinning that);
* the generational scheme replaces the whole population with the offspring
  (`Inheritance.direct_inheritance`), selecting nothing;
* the one remaining selection call, in `ReproductionController`, asks for a
  mating pool of `min(len(population), …)` — never smaller than the population
  it selects from — and `default_selection_behaviour` answers "return them all"
  whenever `len(individuals) <= pop_size`.

Net effect: every parent breeds regardless of fitness and every offspring
survives regardless of fitness. The run *looks* healthy — generations tick over,
the Pareto archive fills — it just searches like a random walk.
`benchmarks/golem/probe_selection.py` measures it directly:

```
scheme=generational   no-op selections: 5/6
scheme=steady_state   no-op selections: 5/10
```

`steady_state` selects the next population out of parents ∪ offspring and does
apply pressure; it is what GOLEM's own multi-objective example uses. Since the
elitism behaviour is pinned by a test, the change here is a warning in
`GPAlgorithmParameters.__post_init__` rather than a silent semantics change, and
the integration defaults to `steady_state`.

### 3. No mating-pool selection, no NSGA-II, no restarts (added)

Three mechanisms GOLEM lacks were added while investigating the above. They are
in `benchmarks/golem/golem_required_patches.diff` and are off by default, so existing behaviour is
unchanged:

* `SelectionTypesEnum.tournament_with_replacement` plus
  `GPAlgorithmParameters.mating_selection_types` — a mating pool built by
  binary tournament *with replacement*, which is the only way to apply mating
  pressure when the requested pool is not smaller than the population it is
  drawn from (see finding 2).
* `SelectionTypesEnum.nsga2` — Pareto rank plus crowding distance. GOLEM's only
  multi-objective survival selection was SPEA-2, whose k-th-nearest-neighbour
  density estimate and one-at-a-time truncation are O(N³) in the worst case;
  NSGA-II is O(M N log N) and spreads the kept solutions more evenly.
* Restarts (in `epde_golem`, not GOLEM itself): split the budget across
  independent searches and union their archives. `optimise` runs one population
  to the stop condition and returns; there is no restart mechanism.

On this domain none of them measurably improved recovery (see Results), but
they close real gaps in the library's operator inventory.

### 4. `optimise()` clobbers the final population (worked around)

`PopulationalOptimizer.optimise` ends with
`self._update_population(self.best_individuals, 'final_choices')`, which assigns
the Pareto archive to `self.population`. The last evolved population — the
diverse, partly-dominated candidates a user of a multi-objective run still wants
— is unreachable after the call. The integration snapshots it through the
iteration callback instead.

## Findings about EPDE

### Runs were not reproducible without `PYTHONHASHSEED` (on the version studied)

EPDE seeds `random`, `numpy` and `torch`, but on commit `3fe9efc` — the one
this whole study ran on — its search path also depended on Python's per-process
string-hash salt, so two runs with identical explicit seeds diverged:

```
without PYTHONHASHSEED:  2968 evals / 2513 evals   (same seed, same config)
with PYTHONHASHSEED=0:   2968 evals / 2968 evals
```

Not a marginal difference: one salt recovered the wave equation, the other did
not.

The cause is a single line in `EqRightPartSelector.simplify_equation`:

```python
common_factors = list(frozenset.intersection(*equation_terms))
```

The elements are label tuples, so the `frozenset` iteration order follows the
hash salt — and the loop below walks that list *while mutating the terms*,
reducing the order of each common factor and dropping factors whose power
reaches zero. With two or more common factors the simplified equation differs,
and the search diverges. Sorting the list made the run identical across salts
1, 2, 3, 4, 7 and 11.

**Scope.** Instrumenting the call shows the triggering case (≥ 2 common
factors) fires 4 times in 954 calls on `wave` at `3fe9efc` — enough to decide
whether the equation was recovered — but **0 times** across `wave`, `kdv`,
`van_der_pol` and `lotka_volterra` on current `main` (`7d8b9d3`, ~5300 calls),
where `wave` is reproducible across salts without the patch. The construct is
still order-dependent, so the fix is worth having, but it closes a latent
dependence rather than a failure reproducible on `main` today.

Either way: fix `PYTHONHASHSEED` before benchmarking anything in EPDE. The
symptom is silent irreproducibility, not an error.
