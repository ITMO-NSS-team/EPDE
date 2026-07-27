# NEW vs LEGACY pipeline — aggregate comparison

**n = 30 reps per (system, pipeline)** before crash exclusions
(NS NEW is n=12 after removing 18 crashed reps; ns legacy is not
yet run, see Notes).
Source: `projects/thesis/results/<system>/<pipeline>_rep*.json`.
Success rates are reported as point estimate with the underlying
numerator/denominator (the denominator is the number of completed
reps, which can be less than 30 when a pipeline crashed on a rep).

- **success**: fraction of reps whose canonical equation system
  matches one of the truth alternatives under the canonical metric
  (structure-only; coefficients ignored).
- **H**: per-rep best Hamming distance (0 = perfect structural
  match; bigger = more wrong/missing terms), mean±std across reps.
- **coef err**: mean per-term relative coefficient error
  `|c_disc - c_truth| / |c_truth|` against the matching truth
  alternative, anchored at the truth's target term so target-flipped
  discoveries are scored correctly. Success-only (a structurally
  wrong rep has no canonical partner to compare against); `-` when
  no successful rep in the cell.
- **runtime**: per-rep wall-clock at 5 epochs, mean±std.

| System | n | Legacy success | Legacy H (mean±std) | Legacy coef err (mean±std) | NEW success | NEW H (mean±std) | NEW coef err (mean±std) | runtime L (mean±std) | runtime N (mean±std) |
|---|---|---|---|---|---|---|---|---|---|
| ac | 30 | 0% (0/30) | 5.4±1.8 | - | 100% (30/30) | 0.0±0.0 | 0.056±0.000 | 422.2±736.0s | 372.6±87.1s |
| burgers_inviscid | 30 | 20% (6/30) | 2.4±1.4 | 0.000±0.000 | 90% (27/30) | 0.5±1.5 | 0.000±0.000 | 3434.1±6619.8s | 421.0±70.8s |
| burgers_viscous | 30 | 0% (0/30) | 6.5±2.8 | - | 90% (27/30) | 0.1±0.3 | 0.007±0.000 | 4587.4±6977.2s | 1026.9±241.5s |
| kdv | 30 | 3% (1/30) | 3.7±1.6 | 0.017±0.000 | 97% (29/30) | 0.0±0.2 | 0.009±0.004 | 2956.8±7381.9s | 3332.9±285.2s |
| kdv_cossin | 30 | 0% (0/30) | 8.5±1.2 | - | 80% (24/30) | 1.2±2.5 | 0.018±0.000 | 1742.2±4401.2s | 167.2±15.2s |
| ks | 30 | 0% (0/30) | 5.7±2.2 | - | 73% (22/30) | 0.3±0.4 | 0.013±0.000 | 5455.1±5108.8s | 4254.5±1201.7s |
| lorenz | 30 | 0% (0/30) | 29.9±2.4 | - | 0% (0/30) | 5.7±1.0 | - | 106.8±152.4s | 709.4±125.6s |
| lv | 30 | 0% (0/30) | 17.1±1.9 | - | 63% (19/30) | 1.4±1.9 | 0.004±0.000 | 44.0±15.7s | 627.7±111.9s |
| ns | 12 | - | - | - | 0% (0/12) | 13.0±1.2 | - | - | 7018.7±2148.5s |
| ode | 30 | 0% (0/30) | 6.2±0.7 | - | 93% (28/30) | 0.1±0.3 | 0.006±0.000 | 26.8±5.9s | 76.8±6.6s |
| pde_compound | 30 | 0% (0/30) | 5.7±0.7 | - | 90% (27/30) | 0.5±1.5 | 0.005±0.000 | 4151.6±5267.8s | 769.1±83.8s |
| pde_divide | 30 | 0% (0/30) | 9.7±1.2 | - | 80% (24/30) | 0.7±1.4 | 0.000±0.000 | 4324.2±5930.4s | 589.9±52.8s |
| vdp | 30 | 7% (2/30) | 4.7±2.4 | 0.003±0.000 | 100% (30/30) | 0.0±0.0 | 0.003±0.000 | 121.5±28.5s | 138.4±19.4s |
| wave | 30 | 3% (1/30) | 4.1±1.9 | 0.010±0.000 | 100% (30/30) | 0.0±0.0 | 0.007±0.000 | 288.8±794.6s | 245.2±43.4s |

## Discovery dynamics (mean±std)

- **unique cands**: number of distinct objective-vector candidates
  ever explored during the search (full sidecar dedup, matches the
  cloud point count on the figures). Bigger = wider search.
- **epoch identified**: 0-indexed epoch at which the truth-matching
  solution first appeared in the Pareto-0 set (success-only; failed
  reps excluded). `-` means no rep in that cell succeeded so there
  is no epoch to average.

| System | Legacy unique cands (mean±std) | Legacy epoch identified | NEW unique cands (mean±std) | NEW epoch identified |
|---|---|---|---|---|
| ac | 51.5±5.0 | - | 38.0±4.6 | 0.0±0.0 |
| burgers_inviscid | 54.2±6.5 | 1.0±1.3 | 19.7±3.8 | 0.4±1.0 |
| burgers_viscous | 48.2±6.8 | - | 32.9±5.3 | 0.1±0.3 |
| kdv | 55.2±5.3 | 0.0±0.0 | 35.9±4.3 | 1.3±1.5 |
| kdv_cossin | 46.6±7.0 | - | 34.0±4.0 | 1.6±1.3 |
| ks | 53.7±7.2 | - | 40.8±5.0 | 0.5±0.7 |
| lorenz | 134.0±0.0 | - | 142.2±8.1 | - |
| lv | 99.0±0.0 | - | 89.2±7.3 | 1.7±1.3 |
| ns | - | - | 93.4±4.6 | - |
| ode | 61.0±0.0 | - | 29.7±3.5 | 0.0±0.0 |
| pde_compound | 42.7±4.1 | - | 38.1±3.5 | 1.7±1.4 |
| pde_divide | 46.2±5.6 | - | 33.4±4.0 | 1.6±1.3 |
| vdp | 43.1±5.1 | 3.0±1.4 | 31.6±4.1 | 0.0±0.0 |
| wave | 42.7±5.0 | 3.0±0.0 | 42.1±4.5 | 0.0±0.0 |

## Runtime breakdown — single-seed per-phase decomposition

Single-seed (seed=0), 3-epoch NEW-pipeline profile via
`projects/thesis/profile_loop_stats.py`. The probes here are wall-
clock timers added to `epde/_loop_stats.py` (gated on
`EPDE_LOOP_STATS=1`); see `profile_results/timer_compare.txt` for
the full side-by-side. Four 1+1D PDE systems with otherwise
identical NEW-pipeline config (population_size=16, factors_num=[1,2],
eq sparsity interval [1e-12, 1e-4]):

| System | grid (n_samples) | wall (s) | EqRPS.apply | EqRPS.gram_super | EqRPS.term_sweep | mut + crossover |
|---|---|---|---|---|---|---|
| wave | 4 225 | 92.4 | 95.7 % | 42.8 % | 52.4 % | 3.1 % |
| burgers_viscous | 16 686 | 409.3 | 98.8 % | 64.7 % | 34.0 % | 0.8 % |
| kdv | 66 010 | 1 301.7 | 99.6 % | 76.2 % | 23.4 % | 0.2 % |
| ks | 164 820 | 1 492.5 | 99.7 % | **80.5 %** | 19.1 % | 0.2 % |

**Bottleneck**: `EqRightPartSelector._precompute_super_gram` —
the per-equation `X^T diag(w) X` precompute that the RPS term-sweep
slices for each candidate target. Per-call cost ranges from
8 µs/sample (wave) to 16 µs/sample (ks), i.e. the precompute scales
~`n_samples^1.18` — mildly superlinear (cache pressure as the
mesh exceeds ~10 k samples). Per-call gram: 35 ms (wave), 189 ms
(burgers), 815 ms (kdv), 2 567 ms (ks). The gram share of wall
grows monotonically with grid size (43 → 65 → 76 → 80.5 %).

The inner term-sweep (sparsity → `PhysicsInformedLasso.fit` →
fitness) is *not* the dominant cost on KdV: 304 s combined for
all three sub-phases vs. 991 s for the gram precompute alone.
Mutation, crossover, and `Term.evaluate` / `Equation.evaluate`
together account for < 0.5 % on kdv (< 4 % even on wave) — the
evolutionary loop overhead is not where the time goes.

Implication: any KdV speed-up has to come from the gram precompute.
Three obvious levers (not implemented here, only flagged):

1. **Subsample the grid before precompute.** The structural sparsity
   selection does not need 66 010 collocation points; ~5–10× downsampling
   would push KdV per-call gram into the burgers regime without
   changing the structural-success rate (the WAPE/Reg objectives
   already include a stability term that does not need full
   resolution).
2. **Cache the gram across outer RPS iters.** `_precompute_super_gram`
   is called from inside the
   `while not (objective.simplified and objective.is_correct_right_part)`
   outer loop in `EqRightPartSelector.apply` — if the structure
   hash is unchanged across iters, the gram could be reused. KdV
   shows 1 216 gram calls vs 1 049 `EqRPS.apply` calls (~1.16 calls
   per apply), so the bound is ~14 % of current gram cost.
3. **Move the matmul to BLAS / sketched gram.** `GramSetup.precompute_super`
   builds a windowed `X^T diag(w) X` — a randomized sketch
   (Gaussian, count-sketch) would trade exact gram for an O(sqrt(n_samples))
   speedup at the same recall on KdV-sized grids.

Lever 1 is the largest single hammer (~10× per-call reduction on
KdV); lever 2 is cheap to verify but bounded; lever 3 is the
biggest engineering lift. None of these is implemented in the
current pipeline.

## Notes

- The Docker sweep has been steadily populating legacy cells on the
  2-D PDE systems. Most still return 0% structural success but
  provide a baseline mean Hamming, runtime distribution, and
  exploration-breadth count.
- The slow legacy cohorts (burgers_inviscid, burgers_viscous, ks,
  pde_compound, pde_divide) have now all reached the full 30 reps.
  Remaining partial / in-progress cohorts:
  - **ns legacy**: not yet started; ns NEW also incomplete (12/30,
    18 crashed via the serializer issue).

  The success-rate denominator is the completed count; the
  `n` column reflects the surviving cohort size after exclusion.
- NEW beats legacy on every system that legacy attempted, by
  margins ranging from +68 pp (burgers_inviscid: 22% → 90%,
  the smallest NEW gain) to +100 pp (ac, vdp, wave all go to
  100%). Lorenz is the only tie (both 0%).
- **Lorenz** stays at 0% structural success under both pipelines
  even with the bumped population_size=48. NEW's mean Hamming
  dropped from 6.7 → 5.7 with the larger weight pool. Both
  pipelines explore very wide candidate sets (134 / 142 unique
  candidates per rep) — the search isn't lacking breadth, the
  truth structure just doesn't dominate the Pareto trade-off.
- **Discovery is mostly immediate when it lands**: every NEW
  success-cell has mean `epoch identified` ≤ 1.7, and on the easy
  systems (ac, ode, vdp, wave) it lands at epoch 0 every single
  rep. Failed reps simply never converge.
- **Exploration breadth (unique cands)** is systematically smaller
  on NEW than legacy. Legacy's wider search doesn't translate to
  more hits because it lacks the WAPE+Reg combination that lets
  good candidates dominate the Pareto front.
- **Coefficient error is tiny when structure matches**: NEW
  cells with successful reps land at ≤ 1.8% mean relative coef
  error on every system except ac (5.6%, where the truth carries
  a 1e-4 diffusion coefficient that amplifies any FD noise). The
  ~0% entries on burgers_inviscid and pde_divide are real — those
  systems' truth alternatives carry analytic ±1 coefficients that
  EPDE recovers exactly. Legacy succeeds rarely but with similar
  precision when it does (vdp 0.3%, wave 1.0%, kdv 1.7%) — when
  the structure is right, both pipelines fit clean coefficients.
- NS surviving cohort (n=12) shows 0% structural success — the
  surviving reps scatter across different incorrect structures.

## Combined: legacy + new + 2×2×2 ablation cells

Adds the 6 off-diagonal cells of the 2×2×2 factorial over the three
NEW components: **W**APE fitness, **I**nstability objective,
**R**egularizer. `legacy` = (000), `new` = (111). The 6 off-diagonal
cells are produced by `projects/thesis/run_ablation.py`; the
`X/.` triple shows which axes are ON for that cell.

ODE, wave, and lv now all have all 8 cells at full 30-rep cohorts.
The other systems have legacy + new only — their off-diagonal cells
have not been run yet and are omitted rather than shown empty.

Source: `projects/thesis/thesis_ablation_aggregate.py`
(`thesis_ablation_summary.json` for the raw numbers).

| System | Cell | W | I | R | n | Success | H (mean±std) | coef err (mean±std) | runtime (mean±std) | unique cands (mean±std) | epoch identified (mean±std) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ac | legacy | . | . | . | 30 | 0% (0/30) | 5.4±1.8 | - | 422.2±736.0s | 51.5±5.0 | - |
| ac | new | X | X | X | 30 | 100% (30/30) | 0.0±0.0 | 0.056±0.000 | 372.6±87.1s | 38.0±4.6 | 0.0±0.0 |
| burgers_inviscid | legacy | . | . | . | 30 | 20% (6/30) | 2.4±1.4 | 0.000±0.000 | 3434.1±6619.8s | 54.2±6.5 | 1.0±1.3 |
| burgers_inviscid | new | X | X | X | 30 | 90% (27/30) | 0.5±1.5 | 0.000±0.000 | 421.0±70.8s | 19.7±3.8 | 0.4±1.0 |
| burgers_viscous | legacy | . | . | . | 30 | 0% (0/30) | 6.5±2.8 | - | 4587.4±6977.2s | 48.2±6.8 | - |
| burgers_viscous | new | X | X | X | 30 | 90% (27/30) | 0.1±0.3 | 0.007±0.000 | 1026.9±241.5s | 32.9±5.3 | 0.1±0.3 |
| kdv | legacy | . | . | . | 30 | 3% (1/30) | 3.7±1.6 | 0.017±0.000 | 2956.8±7381.9s | 55.2±5.3 | 0.0±0.0 |
| kdv | new | X | X | X | 30 | 97% (29/30) | 0.0±0.2 | 0.009±0.004 | 3332.9±285.2s | 35.9±4.3 | 1.3±1.5 |
| kdv_cossin | legacy | . | . | . | 30 | 0% (0/30) | 8.5±1.2 | - | 1742.2±4401.2s | 46.6±7.0 | - |
| kdv_cossin | new | X | X | X | 30 | 80% (24/30) | 1.2±2.5 | 0.018±0.000 | 167.2±15.2s | 34.0±4.0 | 1.6±1.3 |
| ks | legacy | . | . | . | 30 | 0% (0/30) | 5.7±2.2 | - | 5455.1±5108.8s | 53.7±7.2 | - |
| ks | new | X | X | X | 30 | 73% (22/30) | 0.3±0.4 | 0.013±0.000 | 4254.5±1201.7s | 40.8±5.0 | 0.5±0.7 |
| lorenz | legacy | . | . | . | 30 | 0% (0/30) | 29.9±2.4 | - | 106.8±152.4s | 134.0±0.0 | - |
| lorenz | new | X | X | X | 30 | 0% (0/30) | 5.7±1.0 | - | 709.4±125.6s | 142.2±8.1 | - |
| lv | legacy | . | . | . | 30 | 0% (0/30) | 17.1±1.9 | - | 44.0±15.7s | 99.0±0.0 | - |
| lv | wape | X | . | . | 30 | 0% (0/30) | 11.8±2.2 | - | 80.4±7.1s | 98.5±10.7 | - |
| lv | instab | . | X | . | 30 | 0% (0/30) | 13.7±1.5 | - | 70.3±2.7s | 93.4±7.6 | - |
| lv | reg | . | . | X | 30 | 17% (5/30) | 2.7±1.3 | 0.004±0.000 | 368.0±24.0s | 71.4±5.3 | 2.6±1.5 |
| lv | wape_instab | X | X | . | 30 | 0% (0/30) | 13.0±1.7 | - | 60.9±3.5s | 90.0±6.7 | - |
| lv | wape_reg | X | . | X | 30 | 33% (10/30) | 2.0±1.4 | 0.004±0.000 | 428.3±12.1s | 79.9±6.6 | 2.2±1.0 |
| lv | instab_reg | . | X | X | 30 | 23% (7/30) | 2.5±1.5 | 0.004±0.000 | 345.5±8.1s | 82.6±8.3 | 2.3±1.3 |
| lv | new | X | X | X | 30 | 63% (19/30) | 1.4±1.9 | 0.004±0.000 | 627.7±111.9s | 89.2±7.3 | 1.7±1.3 |
| ns | legacy | . | . | . | 0 | - | - | - | - | - | - |
| ns | new | X | X | X | 12 | 0% (0/12) | 13.0±1.2 | - | 7018.7±2148.5s | 93.4±4.6 | - |
| ode | legacy | . | . | . | 30 | 0% (0/30) | 6.2±0.7 | - | 26.8±5.9s | 61.0±0.0 | - |
| ode | wape | X | . | . | 30 | 0% (0/30) | 6.2±2.0 | - | 25.2±4.4s | 49.7±6.7 | - |
| ode | instab | . | X | . | 30 | 0% (0/30) | 7.0±1.4 | - | 28.8±9.4s | 44.6±4.9 | - |
| ode | reg | . | . | X | 30 | 3% (1/30) | 2.5±1.0 | 0.006±0.000 | 55.6±5.4s | 20.1±3.2 | 0.0±0.0 |
| ode | wape_instab | X | X | . | 30 | 0% (0/30) | 6.7±2.2 | - | 32.2±4.8s | 47.1±7.0 | - |
| ode | wape_reg | X | . | X | 30 | 20% (6/30) | 1.7±1.7 | 0.006±0.000 | 66.5±5.6s | 38.1±3.4 | 0.0±0.0 |
| ode | instab_reg | . | X | X | 30 | 30% (9/30) | 0.8±0.6 | 0.006±0.000 | 65.1±9.5s | 39.7±4.0 | 0.0±0.0 |
| ode | new | X | X | X | 30 | 93% (28/30) | 0.1±0.3 | 0.006±0.000 | 76.8±6.6s | 29.7±3.5 | 0.0±0.0 |
| pde_compound | legacy | . | . | . | 30 | 0% (0/30) | 5.7±0.7 | - | 4151.6±5267.8s | 42.7±4.1 | - |
| pde_compound | new | X | X | X | 30 | 90% (27/30) | 0.5±1.5 | 0.005±0.000 | 769.1±83.8s | 38.1±3.5 | 1.7±1.4 |
| pde_divide | legacy | . | . | . | 30 | 0% (0/30) | 9.7±1.2 | - | 4324.2±5930.4s | 46.2±5.6 | - |
| pde_divide | new | X | X | X | 30 | 80% (24/30) | 0.7±1.4 | 0.000±0.000 | 589.9±52.8s | 33.4±4.0 | 1.6±1.3 |
| vdp | legacy | . | . | . | 30 | 7% (2/30) | 4.7±2.4 | 0.003±0.000 | 121.5±28.5s | 43.1±5.1 | 3.0±1.4 |
| vdp | new | X | X | X | 30 | 100% (30/30) | 0.0±0.0 | 0.003±0.000 | 138.4±19.4s | 31.6±4.1 | 0.0±0.0 |
| wave | legacy | . | . | . | 30 | 3% (1/30) | 4.1±1.9 | 0.010±0.000 | 288.8±794.6s | 42.7±5.0 | 3.0±0.0 |
| wave | wape | X | . | . | 30 | 3% (1/30) | 2.1±1.3 | 0.010±0.000 | 1592.1±4576.4s | 43.8±3.6 | 2.0±0.0 |
| wave | instab | . | X | . | 30 | 0% (0/30) | 3.9±1.6 | - | 894.4±177.5s | 44.2±4.1 | - |
| wave | reg | . | . | X | 30 | 10% (3/30) | 2.2±1.0 | 0.007±0.000 | 216.8±18.3s | 15.3±2.8 | 0.0±0.0 |
| wave | wape_instab | X | X | . | 30 | 0% (0/30) | 3.5±1.7 | - | 522.8±332.9s | 44.9±5.0 | - |
| wave | wape_reg | X | . | X | 30 | 73% (22/30) | 0.8±1.4 | 0.007±0.000 | 216.6±11.8s | 34.0±3.4 | 0.0±0.0 |
| wave | instab_reg | . | X | X | 30 | 10% (3/30) | 2.1±0.8 | 0.007±0.000 | 187.4±11.0s | 40.9±5.2 | 0.0±0.0 |
| wave | new | X | X | X | 30 | 100% (30/30) | 0.0±0.0 | 0.007±0.000 | 245.2±43.4s | 42.1±4.5 | 0.0±0.0 |

### Marginal contribution per axis (ODE, wave, and lv — full 8-cell factorial)

Mean delta across the 4 mutually-exclusive (off, on) cell pairs along
each axis. Requires all 8 cells to have ≥1 rep — ODE, wave, and lv
are currently the systems with that.

| System | Axis | mean delta success | mean delta H | n pairs |
|---|---|---|---|---|
| lv | WAPE | +14.2 pp | -1.94 | 4 |
| lv | Instab | +9.2 pp | -0.76 | 4 |
| lv | Reg | +34.2 pp | -11.74 | 4 |
| ode | WAPE | +20.0 pp | -0.43 | 4 |
| ode | Instab | +25.0 pp | -0.50 | 4 |
| ode | Reg | +36.7 pp | -5.27 | 4 |
| wave | WAPE | +38.3 pp | -1.48 | 4 |
| wave | Instab | +5.0 pp | +0.07 | 4 |
| wave | Reg | +46.7 pp | -2.10 | 4 |

### ODE single-axis breakdown (n=30 cohorts)

Each pair below holds two of the three axes fixed and toggles the
third, giving the direct effect of that axis at that backdrop.
Format: `(off success → on success)`.

**WAPE axis** (varying W with I, R held fixed):

- I=., R=.: legacy 0% → wape 0% (no effect at the (0,0,0) backdrop)
- I=X, R=.: instab 0% → wape_instab 0% (no effect when only Instab is on)
- I=., R=X: reg 3% → wape_reg 20% (+17 pp — Reg primes the pump for WAPE)
- I=X, R=X: instab_reg 30% → new 93% (+63 pp — biggest WAPE jump,
  needs both Instab and Reg already on)

**Instability axis** (varying I with W, R held fixed):

- W=., R=.: legacy 0% → instab 0% (no effect on its own)
- W=X, R=.: wape 0% → wape_instab 0%
- W=., R=X: reg 3% → instab_reg 30% (+27 pp — Instab adds a lot once Reg is on)
- W=X, R=X: wape_reg 20% → new 93% (+73 pp — Instab is the gating
  axis for the headline result)

**Reg axis** (varying R with W, I held fixed):

- W=., I=.: legacy 0% → reg 3% (+3 pp on its own)
- W=X, I=.: wape 0% → wape_reg 20% (+20 pp)
- W=., I=X: instab 0% → instab_reg 30% (+30 pp)
- W=X, I=X: wape_instab 0% → new 93% (+93 pp — Reg is the dominant
  single-axis contributor)

The pattern: **R is the single biggest lever** (every R-on pair
gains 3-93 pp; no cell without R reaches double digits). **W and I
on their own do nothing** on ODE, but they're strongly synergistic
with R — the (W,R) and (I,R) pairs each clear 20%, and only the
full (W,I,R) corner reaches the 93% headline.

### wave single-axis breakdown (n=30 cohorts)

Same construction as the ODE breakdown above, on wave's full 8-cell
factorial.

**WAPE axis** (varying W with I, R held fixed):

- I=., R=.: legacy 3% → wape 3% (no effect at the (0,0,0) backdrop)
- I=X, R=.: instab 0% → wape_instab 0% (no effect when only Instab is on)
- I=., R=X: reg 10% → wape_reg 73% (+63 pp — WAPE's big jump once Reg is on)
- I=X, R=X: instab_reg 10% → new 100% (+90 pp — biggest WAPE jump,
  needs both Instab and Reg already on)

**Instability axis** (varying I with W, R held fixed):

- W=., R=.: legacy 3% → instab 0% (-3 pp — slightly hurts on its own)
- W=X, R=.: wape 3% → wape_instab 0% (-3 pp)
- W=., R=X: reg 10% → instab_reg 10% (no effect once Reg is on, alone)
- W=X, R=X: wape_reg 73% → new 100% (+27 pp — Instab closes the last
  gap to 100%)

**Reg axis** (varying R with W, I held fixed):

- W=., I=.: legacy 3% → reg 10% (+7 pp on its own)
- W=X, I=.: wape 3% → wape_reg 73% (+70 pp — (WAPE, Reg) is the key combo)
- W=., I=X: instab 0% → instab_reg 10% (+10 pp)
- W=X, I=X: wape_instab 0% → new 100% (+100 pp — Reg completes the
  full corner)

The pattern: as on ODE, **Reg is the biggest single lever** (+46.7 pp
mean; no cell without Reg clears single digits). But wave leans far
more on **WAPE** — the (WAPE, Reg) cell alone reaches 73%, whereas on
ODE no two-axis cell exceeds 30% and the full (W,I,R) corner is needed
for the headline. **Instability is the weak axis here**: flat or
slightly negative at the no-Reg backdrops, contributing only the final
+27 pp that closes wape_reg's 73% to the 100% headline. So the gating
axis flips between the two systems — Instability gates ODE, WAPE gates
wave — but both still need all three axes to reach the top corner.

### lv single-axis breakdown (n=30 cohorts)

Same construction as the ODE and wave breakdowns above, on lv's full
8-cell factorial. lv is the only one of the three where the headline
(W,I,R) corner does not clear 90% — it tops out at 63% — so the
single-axis effects are smaller in absolute terms but the pattern is
otherwise comparable.

**WAPE axis** (varying W with I, R held fixed):

- I=., R=.: legacy 0% → wape 0% (no effect at the (0,0,0) backdrop)
- I=X, R=.: instab 0% → wape_instab 0% (no effect when only Instab is on)
- I=., R=X: reg 17% → wape_reg 33% (+17 pp — WAPE roughly doubles
  success once Reg is on)
- I=X, R=X: instab_reg 23% → new 63% (+40 pp — biggest WAPE jump,
  needs both Instab and Reg already on)

**Instability axis** (varying I with W, R held fixed):

- W=., R=.: legacy 0% → instab 0% (no effect on its own)
- W=X, R=.: wape 0% → wape_instab 0% (no effect when only WAPE is on)
- W=., R=X: reg 17% → instab_reg 23% (+7 pp — small bump once Reg is on)
- W=X, R=X: wape_reg 33% → new 63% (+30 pp — Instab closes part of
  the remaining gap to the headline 63%)

**Reg axis** (varying R with W, I held fixed):

- W=., I=.: legacy 0% → reg 17% (+17 pp — the only axis that lifts
  the (0,0,0) corner off zero on its own)
- W=X, I=.: wape 0% → wape_reg 33% (+33 pp)
- W=., I=X: instab 0% → instab_reg 23% (+23 pp)
- W=X, I=X: wape_instab 0% → new 63% (+63 pp — biggest Reg jump,
  completes the corner)

The pattern: as on ODE and wave, **Reg is the dominant single lever**
(+34.2 pp mean; every R-off cell sits at exactly 0%, and only R-on
cells score above zero). What is distinctive about lv is that **the
two-axis cells are intermediate but no single pairing dominates**:
(WAPE, Reg) lands at 33%, (Instab, Reg) at 23%, and the full corner
only reaches 63% — versus ODE 93% and wave 100%. WAPE contributes
more than Instab at every backdrop (+14.2 pp mean vs +9.2 pp), but
neither is gating in the way Instab gates ODE or WAPE gates wave.
Hamming-wise the story is even cleaner: Reg alone drops mean H from
17.1 (legacy) to 2.7 (-11.74 ΔH per pair), so the bulk of lv's
"correctness" comes from Reg's pruning, with WAPE and Instab providing
modest further refinements (ΔH -1.94 and -0.76).
