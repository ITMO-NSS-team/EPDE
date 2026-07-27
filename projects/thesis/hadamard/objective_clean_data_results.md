# The best Hadamard-stability complementary objective (clean data)

**Question.** Find a mathematically-solid, experiment-proven objective — used as
*both* the MOEA/D complement to discrepancy *and* the sparsity regularizer —
grounded in **Hadamard stability**: the true equation (or a valid analytical
alternative) may have worse discrepancy than an overfit, yet be *stable*.
Validate on 14 systems. **Clean data only** (solver-free noise hits systems
unequally → out of scope this round; this relaxes the correlated-noise C3
constraint). Not restricted to existing ideas/baselines.

**Method.** A 9-agent research panel formalized 6 distinct Hadamard-stability
paradigms (inverse conditioning; generalized coefficient-field; resampling /
stability-selection; generalization / cross-block forecast; forward symbol /
dispersion consistency; information-theoretic evidence/MDL) → adversarial
synthesis → critique. Survivors were implemented as objectives
`J(f,t,w,gshape,var)→float` in `objectives_hadamard.py` and scored solver-free on
seeded candidates (one pool build per system, **no evolutionary reps**):

- `bic` — parameter-count description length `Neff·ln(nRSS)+k·ln(Neff)` (Occam).
- `biw` — Block-Invariance Wald ratio (CRLB-normalized between-block F-test).
- `biw_cv` — robust block region-variation (across-block coef std/|mean|).
- `gengap_block` / `gengap_gap` — contiguous-time extrapolation residual (+drift).

Compared against the incumbents `disc` (discrepancy) and `instab_both` /
`instab_var` (vcoef `(Var(γ₀)+NC_deb)/γ₀²`). Two experiments:
`_obj_gate.py` (the burgers/ac/kdv C1 gates) and `_obj_sweep14.py` (all 14 systems
with **hard** wrong forms: the coordinate-modulated `_REPLACE` failures + riders).

## Result — full 14-system sweep, clean data, hard wrong forms

| objective | systems passing (true & all identities strictly below every wrong) |
|---|---|
| `disc` | 0/14 |
| **`instab_both`** | **14/14** |
| `instab_var` | 13/14 (fails wave) |
| `bic` | 1/14 |
| `biw_cv` | 11/14 (fails lorenz, kdv, wave) |
| `biw`, `gengap_*` | C1-dead (see gate) |

**vcoef `instab_both` is the unique 14/14**, even against coordinate-modulated
hard forms — stronger than the prior easy-set 14/14.

## Why every "more principled" alternative fails

The root cause is one fact: **on clean *solver-free* data the regression residual
is deterministic FD/discretization error, not stochastic noise** (the June audit's
`R = Var_emp/Var_analytic ≪ 1`). This breaks each paradigm at its foundation:

- **BIW (CRLB Wald) — explodes on exact fits.** The denominator is the per-block
  sampling variance σ²(AᵀWA)⁻¹ → 0 for a near-exact fit, so FD-derivative drift
  blows the Wald ratio up. On burgers the *true* PDE scores `biw=2.1e4` while its
  algebraic shadows score ~0.3 — it ranks the true law *less stable than its own
  shadows*. Backwards. (gate: fails burgers, kdv.)
- **BIC (parsimony) — fooled by FD-error absorption.** A collinear spurious term
  that is algebraically equal to an active one (`x·u_x = u` on burgers) reduces the
  FD-error residual more than the `+ln(Neff)` parsimony penalty costs, so BIC ranks
  the overfit *below* the truth. 1/14. (Only safe as a *nested* regularizer, not a
  cross-candidate objective.)
- **biw_cv (robust block region-variation) — false-flags drifting valid
  identities.** Normalizing by coefficient magnitude (not variance) fixes the
  explosion and reaches 11/14, but explicit per-block OLS picks up *legitimate*
  coefficient drift of valid identities across subdomains — the kdv soliton-
  collision band, wave's time-modulated identity shadows, chaotic lorenz — so it
  violates C1 there. It is a coarser version of what vcoef does robustly.
- **gengap (extrapolation) — does not separate.** True law and overfit both
  extrapolate to a held-out time block equally well on clean data; the drift term
  inherits BIW's explosion.

**Why vcoef survives:** it measures coefficient-*field structure* normalized by the
coefficient magnitude (`/γ₀²`, scale-free like a CV, never `/0`), **debiased**
against the FD/noise floor (`Σ max(γ_k²−λVar(γ_k),0)`), and fit **globally and
jointly** over an orthonormal cosine basis (Frisch-Waugh-decoupled) rather than as
independent block OLS — so it is robust to both the discretization floor (which
sinks BIW/BIC) and the localized identity drift (which sinks biw_cv).

## 'var' vs 'nc' vs 'both' — resolved on the hard set: **both**

`instab_var` (significance channel alone) reaches 13/14 but **fails wave**: the
space-modulated `u_xx·sin(2x)` shadow is "significant", so only the **NC
region-variation channel** flags it (its coefficient field must vary as 1/sin).
On the hard coordinate-modulated set NC is load-bearing → keep the production
**`both`** channel; do not switch to var-only. (On plain confusers var alone
suffices, which is why earlier plain-confuser sweeps favored var.)

## Recommendation

**Keep the production vcoef instability `(Var(γ₀)+NC_deb)/γ₀²` (`both` channel) as
the Hadamard-stability objective and regularizer.** It is already wired as both.
This study validates it as the *best* available on clean data — the unique 14/14
survivor of a 6-paradigm adversarial panel + a hard coordinate-modulated screen —
and explains mechanistically why the residual-variance (BIW) and fit-reward (BIC)
alternatives cannot compete when the residual is discretization-dominated. No
combination improves on `both` (no system needs a complement; `biw_cv` passes only
a subset of what `both` already passes).

## Honest limits

- **Exact-shadow / single-trajectory limit (unchanged).** A truly-wrong form that
  fits the sampled solution *exactly* is an accidental identity, information-
  theoretically inseparable from a valid one by any solver-free objective; only a
  different IC (a solver) separates them. All objectives here target *approximate*/
  overfit wrong forms.
- **FDC (forward symbol/dispersion) was dropped**, not refuted: it needs per-term
  derivative-order parsing of the candidate (the previously-rejected model parse),
  degenerates to NA on the 4 ODE systems, and *ties* the forward-consistent wave
  `u_xx²` shadow — so it cannot exceed `both`'s 14/14. Revisit only if a
  noisy-regime study re-opens the question.

Artifacts: `objectives_hadamard.py` (objective module), `_obj_gate.py` (C1 gates),
`_obj_sweep14.py` (14-system sweep). Panel transcript: the Jun-2026 workflow
`hadamard-objective-research`.
