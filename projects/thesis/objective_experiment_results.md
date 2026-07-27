# A complementary objective that is lowest for the true equation

**Question.** Find a MOEA/D objective, complementary to discrepancy, that the
**true** equation minimises — so it is ranked below **truly-wrong** forms
(overfits, wrong structures, under-specified), **without** penalising valid
analytical identities (which are credited as correct discoveries). Tested
solver-free; we want to know how far this gets noiseless and under noise.

Code: `projects/thesis/hadamard/objective_experiment.py`
(`--summarize` reprints the scoreboard from `objective_experiment.json`).

## Setup

For each of the 14 systems we assemble candidate equations and tag them:

| category | source |
|---|---|
| `true` | seeded truth (`configs/<sys>.yaml`) |
| `identity` | valid analytical alternatives (`truth_alternatives`) |
| `wrong:missing` | a true term dropped (constructed) |
| `wrong:overfit` | truth + a spurious extra term (constructed) |
| `wrong:discovered` | structurally-wrong equations the pipeline actually produced (`results/<sys>/*rep*.json`, structural failures) |

Each candidate is scored solver-free (lower = better) by three objectives,
across a noise sweep (0/1/4/8 % of std). Multi-equation systems aggregate per
equation (mean for `disc`/`gengap`, sum for `instab`, as MOEA/D does):

- `disc` — normalised weighted residual of the OLS fit (the discrepancy term).
- `instab` — the **current complement**: sum of `VaryingCoefSetup` per-term
  scores (vcoef instability).
- `gengap` — **candidate complement**: held-out normalised residual under
  K-fold CV over grid points (an exact law/identity holds on every fold; an
  overfit/accidental fit does not).

A system "passes" an objective at a noise level if `true` is **strictly below
every truly-wrong** candidate (`separation > 0`) **and** the valid identities
stay below the wrong forms (not penalised).

## Result

Scoreboard — number of systems (out of 14) where the true equation is ranked
strictly below all truly-wrong forms; `id` counts systems where identities are
kept in the good cluster:

| noise % | discrepancy | instability | gengap | instab **or** gengap |
|--------:|:-----------:|:-----------:|:------:|:--------------------:|
| 0       | 0/14  · 14 id | **14/14 · 14 id** | 2/14 · 14 id | **14/14** |
| 1       | 0/14  · 11 id | 4/14 · 11 id  | 5/14 · 11 id | 7/14 |
| 4       | 0/14  · 11 id | 4/14 · 11 id  | 3/14 · 11 id | 6/14 |
| 8       | 0/14  · 12 id | 4/14 · 11 id  | 2/14 · 11 id | 5/14 |

(Instability is **14/14 noiseless** after removing the `1e-30` denominator floor
in `VaryingCoefSetup.score` — see finding 2.)

## Findings

1. **Discrepancy alone is insufficient — even noiseless (0/14).** The
   `wrong:overfit` form (truth + a spurious term) ties or beats the truth on
   discrepancy to machine precision in every system, because extra degrees of
   freedom never raise the residual. A complement is mandatory, not optional.

2. **Instability is a perfect complement noiseless: 14/14.** The true equation
   is the strict minimiser of vcoef-instability below all truly-wrong forms in
   every system, and it is **identity-friendly**: valid identities stay in the
   good cluster in 14/14, so it does not fight them. (Before removing the
   denominator floor it was 13/14: `burgers_inviscid` had two degenerate
   `u_xx ≈ 0` discoveries whose coefficients collapse to ~0; the old
   `(γ₀²+1e-30)` floor let their score sink to ~0 and tie the truth. Dropping
   the floor — `score = (Var(γ₀)+NC)/γ₀²`, `→∞` when `γ₀→0` — pushes those
   trivial forms off the front. The fix touches the live MOEA/D objective and
   the regularizer; the seeded-truth fit path is unaffected: kdv still recovers
   `[-6.04, -1.02]` with `coef_stability = 3.7e-5`.) This validates the
   solver-free + instability design noiseless and answers the core question
   affirmatively for that regime.

3. **The frontier is noise.** Every solver-free complement collapses under
   noise (instability and gengap to ~4/14, combined to 5–7/14), and identity
   safety slips to ~11/14 (under noise the line between a valid identity and a
   wrong form blurs). This empirically confirms the working hypothesis that the
   noisy regime needs solvers — the solver-free residual/derivative signal is
   too corrupted for the complement to separate true from wrong.

4. **Instability and generalisation are complementary, not redundant.** They
   pass on *different* systems (instability robust on lorenz/lv/wave/burgers;
   gengap robust on ks/lorenz), so the **union reaches 14/14 noiseless** and is
   strictly better than either alone under noise. A combined complement (Pareto
   over both, or a min/normalised sum) is the most promising direction.

## Honest limits

- On a **single trajectory**, a truly-wrong form that fits the sampled solution
  *exactly* is an accidental identity and is **indistinguishable** from a valid
  one by any solver-free objective — only a different initial condition (i.e. a
  solver) can separate them. The complement can only catch wrong forms that are
  not exact on the data (overfits, approximate fits), which is why noise (where
  the overfit captures noise) is exactly where it is needed and where the
  solver-free signal also degrades.
- `gengap` here is the held-out residual; for overfit detection the held-out
  *gap* (test − train) is the sharper signal and is worth testing next.
- Noise uses one realisation per level (fixed seed); an ensemble would tighten
  the noisy-regime numbers.

## Next directions

- **Combined complement**: `instab` ⊕ `gengap` (Pareto or normalised min) —
  14/14 noiseless, best-available under noise.
- **gengap as a gap** (test − train) and **cross-IC generalisation** once a
  solver is in the loop (the principled fix for the single-trajectory limit).
- The experiment harness scores any candidate objective in one pass — drop in
  new complements (e.g. a noise-debiased instability) and re-run `--summarize`.
