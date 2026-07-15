# PySINDy vs EPDE NEW pipeline — comparative table

**n = 30 reps per (system, pipeline).** EPDE coefficient relative error
is averaged across the structurally-successful reps (per-system success
counts listed below the aggregate table). PySINDy is a single
deterministic shot per system. EPDE source:
`projects/thesis/results/<system>/new_rep*.json`. PySINDy source:
`Pysindy clean statistics.docx`.

## EPDE configuration (new pipeline)

- **Optimizer**: MOEA/D-DD with `training_epochs = 5`,
  `population_size = 16` for single-equation systems and **32 for coupled
  systems** (lv, lorenz, ns). Multi-objective mode on,
  `use_solver = false`.
- **Preprocessor**: finite-difference derivative estimation (`FD`).
- **Token pool per system**:
  - Variable tokens for each state variable, powers 1..3
    (`data_fun_pow = 3`).
  - Derivative tokens for each (variable, axis, order) with powers
    1..2 (`deriv_fun_pow = 2`). Max derivative order is `(2,)` for ODEs
    and `(2, 4)` (time and space respectively) for 1+1D PDEs.
  - Single consolidated grid coordinate family `x` with `dim` as the
    axis-discriminating parameter (powers 1..2).
  - Trigonometric tokens `sin`, `cos` with `freq ∈ [1.99999999,
    2.00000001]` (narrow window pinned to 2.0).
  - Optional adapter-specific extras (e.g. forced-oscillator
    custom tokens).
- **Equation shape**: up to 10 terms per equation
  (`equation_terms_max_number = 10`); each term has 1 or 2 factors
  drawn with probabilities `[0.65, 0.35]`.

## Metric definitions

- **EPDE success**: fraction of the 30 reps for which any Pareto-0
  candidate's canonical equation system matches the system's truth
  (or one of its declared `truth_alternatives`) under the canonical
  metric. Match is structural-only — coefficient magnitudes ignored.
- **EPDE mean term-set distance**: per-rep best-Hamming averaged over
  all 30 reps. *Hamming* here is the symmetric-difference size of the
  unordered factor signatures, bipartite-matched across equations in
  multi-equation systems and minimised across truth alternatives.
  0 = perfect structural match; bigger = more wrong/missing terms.
- **EPDE mean time**: mean wall-clock per rep at 5 epochs.
- **EPDE coef Σ rel-err**: across only the structurally-successful
  reps, the equations are aligned with the truth (or matched
  alternative), normalised so the truth's target term has coefficient
  1, and the per-term relative differences summed. Smaller is better.
  Mirrors PySINDy's "Σ rel-err".
- **EPDE term univ (k≤2)**: per-system upper bound on the number of
  distinct 1-to-2-factor term signatures the EPDE token pool can
  produce. Computed as `C(F, 1) + C(F, 2)` where `F` is the per-system
  factor-signature universe (variables × powers + derivatives ×
  powers + grid × dim × powers + trig). Closest analogue to PySINDy's
  fixed `library size`.

## Aggregate table

| System | PySINDy time (s) | PySINDy lib | PySINDy Σ rel-err | EPDE success | EPDE mean term-set distance | EPDE mean time (s) | EPDE coef Σ rel-err | EPDE term univ (k≤2) |
|---|---|---|---|---|---|---|---|---|
| Forced Damped Oscillator (ODE) | 0.002 | 10 | 1.59e-2 | 93% | 0.1 | 61.5 | 1.78e-2 | 66 |
| Van der Pol oscillator (ODE) | 0.001 | 9 | 6.84e-2 | 100% | 0.0 | 87.5 | 8.78e-3 | 66 |
| Lorenz system (coupled 3D ODE) | 0.007 | 60 | 7.66e-4 | 0% | 10.5 | 54.7 | — | 325 |
| Lotka-Volterra (coupled 2D ODE) | 0.003 | 20 | 1.41e-2 | 50% | 2.7 | 204.6 | 1.34e-2 | 171 |
| Allen-Cahn (1+1D PDE) | 0.019 | 26 | 6.10e-3 | 80% | 0.2 | 197.0 | 1.12e-1 | 231 |
| Burgers inviscid (1+1D PDE) | 0.099 | 14 | 2.47e-2 | 100% | 0.0 | 228.3 | 0.000 | 231 |
| Burgers viscous (1+1D PDE, ν=0.1) | 0.042 | 26 | 1.85e-2 | 67% | 0.3 | 273.8 | 1.38e-2 | 231 |
| Korteweg-de Vries (1+1D PDE) | 0.374 | 26 | 1.61e-2 | 70% | 1.8 | 1198.6 | 3.35e-2 | 231 |
| KdV with cos(t)·sin(x) source (1+1D PDE) | 0.010 | 30 | 6.78e-4 | 80% | 1.2 | 167.2 | 5.30e-2 | 253 |
| Kuramoto-Sivashinsky (1+1D PDE) | 1.671 | 26 | 3.60e-2 | 60% | 3.4 | 1907.0 | 3.75e-2 | 231 |
| Wave equation (1+1D PDE, c²=0.04) | 0.013 | 26 | 1.70e-2 | 100% | 0.0 | 122.2 | 6.77e-3 | 231 |
| Compound rational PDE (1+1D) | 0.106 | 27 | 1.42e-3 | 43% | 2.1 | 438.6 | 9.83e-3 | 231 |
| Rational PDE (1+1D) | 0.233 | 33 | 3.23e-3 | 67% | 1.9 | 377.3 | 2.48e-4 | 231 |

**n_success per system** (denominator for EPDE coef Σ rel-err): Forced
Damped Oscillator 28, Van der Pol 30, Lorenz 0, Lotka-Volterra 15,
Allen-Cahn 24, Burgers inviscid 28, Burgers viscous 21, Korteweg-de
Vries 21, KdV with cos·sin source 24, Kuramoto-Sivashinsky 18, Wave 30,
Compound rational PDE 13, Rational PDE 20.

## Head-to-head on coefficient quality

| System | PySINDy | EPDE | Winner |
|---|---|---|---|
| Forced Damped Oscillator | 1.59e-2 | 1.78e-2 | ≈ tie |
| Van der Pol oscillator | 6.84e-2 | 8.78e-3 | **EPDE** (~8×) |
| Lorenz system | 7.66e-4 | — | PySINDy (EPDE 0% structural) |
| Lotka-Volterra | 1.41e-2 | 1.34e-2 | ≈ tie |
| Allen-Cahn | 6.10e-3 | 1.12e-1 | PySINDy (~18×) |
| Burgers inviscid | 2.47e-2 | 0.000 | **EPDE** (exact) |
| Burgers viscous | 1.85e-2 | 1.38e-2 | **EPDE** (~1.3×) |
| Korteweg-de Vries | 1.61e-2 | 3.35e-2 | PySINDy (~2×) |
| KdV with cos·sin source | 6.78e-4 | 5.30e-2 | PySINDy (~80×) |
| Kuramoto-Sivashinsky | 3.60e-2 | 3.75e-2 | ≈ tie |
| Wave equation | 1.70e-2 | 6.77e-3 | **EPDE** (~2.5×) |
| Compound rational PDE | 1.42e-3 | 9.83e-3 | PySINDy (~7×) |
| Rational PDE | 3.23e-3 | 2.48e-4 | **EPDE** (~13×) |

## Compute cost contrast

| Axis | PySINDy | EPDE |
|---|---|---|
| Strategy | Closed-form sparse regression | Evolutionary search over equation structures |
| Repeatability | Deterministic single shot | Stochastic, 30 reps per system |
| Per-attempt time | 1 ms – 1.7 s | 55 – 1907 s |
| Library / term universe | 9 – 60 fixed terms | 66 – 325 unordered 1-or-2-factor combinations |
