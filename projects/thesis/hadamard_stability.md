# Data-driven Hadamard stability of the EPDE pipeline

This note defines how we measure **Hadamard stability** for the thesis, why the
measures are **purely data-driven**, and how they connect to the existing
varying-coefficient stability machinery. Code lives in
`projects/thesis/hadamard/`; results in `hadamard.json` /
`hadamard_results.md` and `hadamard/plots/`.

## 1. Hadamard well-posedness and the Petrowsky criterion

A problem is *Hadamard well-posed* if a solution (i) **exists**, (ii) is
**unique**, and (iii) **depends continuously on the data**. Condition (iii) —
"stability" in Hadamard's sense — is the one we operationalise.

For a linear(ised) evolution equation `∂ₜu = L u` the Cauchy problem is
well-posed in `L²`/Sobolev iff the temporal growth rate of the Fourier symbol
is **bounded above uniformly in wavenumber** (the Petrowsky condition):

```
sup_k  Re s(k)  <  ∞,        u ~ exp(i k·x + s(k) t).
```

The canonical ill-posed example is the backward heat equation `uₜ = −ν u_xx`,
whose symbol `s(k) = +ν k²` grows without bound: an arbitrarily small
high-wavenumber perturbation of the data is amplified by `exp(ν k² t)`, so the
solution map is unbounded. Parabolic (`s ~ −k²`), hyperbolic (`s ~ ±i c k`) and
dispersive (`s ~ i k³`) operators all satisfy `sup_k Re s(k) < ∞` and are
well-posed.

## 2. Two maps in the discovery setting

Continuous dependence applies to **two distinct maps**, and we measure both.

* **Forward solution operator** `S_t : data(IC) → solution`. A discovered
  equation may fit the data yet correspond to a Hadamard-*ill-posed* operator
  (a backward-heat form), which is useless for prediction. Whether the dynamics
  behind the measured field are well-posed is a property *of the data*.

* **Inverse discovery operator** `T : data → equation`. PDE identification is a
  classically ill-posed inverse problem; we ask how much the recovered equation
  (its coefficients on a fixed support) moves under small data perturbations.

We estimate **both from data**: the forward verdict from the field's own
growth spectrum (never parsing or simulating a model), the inverse modulus by
perturbing the data and re-fitting (referenced to self-consistency across
perturbed datasets, never to a ground-truth equation).

## 3. Forward well-posedness, estimated from the field alone

Implemented in `hadamard/forward_spectral.py`.

### 3.1 Global spectral abscissa via DMD
Arrange the field as snapshots `X = [x₀ … x_{T−1}]` (state = flattened space, or
the stacked variables for ODEs / multi-field PDEs). Exact DMD fits the best
linear propagator `x_{t+1} ≈ A x_t`; its eigenvalues map to continuous-time
rates `μ_i = log(λ_i)/Δt`. The **spectral abscissa** `α = max_i Re μ_i` is the
least-stable growth rate.

* The reported `α` is taken at the **energy-capturing rank** (smallest rank
  holding 99.99 % of the singular-value energy), so low-energy noise directions
  — whose DMD eigenvalues drift off the unit circle and inflate the apparent
  growth of conservative systems — are excluded.
* A **rank-robustness sweep** still climbs to full rank. If `α` keeps rising
  there because *energetic* high modes genuinely grow, that is the data-side
  ill-posedness signal; if only noise modes drive the rise, the energy-rank `α`
  is unmoved.
* `α_fb` is the **forward–backward DMD** abscissa (geometric mean of forward and
  inverse-backward operators), which removes the leading noise bias.

DMD works for ODEs as the `k = 0` case: the eigenvalues approximate the
data-estimated Jacobian spectrum, so `α` is a Lyapunov-type IC-sensitivity
indicator (chaotic Lorenz gives `α > 0`).

### 3.2 Empirical dispersion relation
For PDEs, FFT the field over space to get modal time series `û(k,t)`, and for
each energetic wavenumber fit an **order-2 Prony** model
`û(k,t) ≈ Σ_m c_m exp(s_m t)`, taking `Re s(k) = max_m Re s_m` (the per-`k`
spectral abscissa). Order ≥ 2 is essential: a real cosine `cos(ωt)` is a
conjugate pair `exp(±iωt)`, which a one-mode fit aliases into spurious growth;
order 2 recovers `Re s ≈ 0`. Roots are **amplitude-pruned** (a surplus root
carrying negligible signal energy is dropped) so a single decaying mode is not
mistaken for growth.

From the curve `Re s(k)` we report `sup_k Re s(k)`, the high-`k` trend exponent
`p` (`Re s ~ |k|^p`), and the amplification `G = max_k exp(Re s(k)·T)` over the
observation horizon `T`.

### 3.3 Classification (`classify_wellposedness`)
| trend at high `k`                              | type        | well-posed |
|------------------------------------------------|-------------|-----------|
| `Re s` grows (`p>0`, increasing)               | ill-posed   | no        |
| `Re s → −∞` like `|k|²`                         | parabolic   | yes       |
| `Re s ≈ 0`, `Im s ~ |k|` (linear)              | hyperbolic  | yes       |
| `Re s ≈ 0`, `Im s ~ |k|^q`, `q>1`              | dispersive  | yes       |
| mixed signs                                    | mixed       | borderline|

The "≈ 0" test is scaled to the **oscillation rate** `max|Im s|` (not the noise
floor), so a fast conservative oscillation with a tiny noisy `Re s` reads as
bounded rather than diffusive.

### 3.4 Honest limits (these are reported, not hidden)
* **Resolved band.** The data only resolves `|k| ≤ k_max = π/Δx`. Well-posedness
  beyond the grid scale is an *extrapolation* of the high-`k` trend, never a
  certificate.
* **Temporal sampling.** A growth rate cannot be separated from a transient if
  the field is observed for a fraction of an oscillation period. We compute the
  energy-weighted number of observed periods `obs_cycles`; if the dynamics are
  oscillatory and `obs_cycles < 0.5`, a non-well-posed verdict is downgraded to
  **`undetermined`**. (This is exactly what happens to the wave dataset, which
  spans ≈ 0.4 periods of its dominant mode.)
* **Linearisation.** DMD/Prony are linear/Koopman approximations; for strongly
  nonlinear or bounded, non-translation-invariant domains (NS cylinder wake)
  the global DMD abscissa is the more reliable read than the FFT dispersion.

## 4. Inverse continuous-dependence, from perturb-and-observe

Implemented in `hadamard/inverse_stability.py`, reusing the seeded-support
feature evaluation that `noise_stability_sweep.py` already drives
(`build_pool_only` + `eq.evaluate`); the support is fixed, so coefficient
vectors are column-aligned across perturbed datasets.

1. **Condition number** of the column-equilibrated weighted library matrix
   `A = √W · Θ`. For a fixed support this is the local Lipschitz constant of the
   coefficient map, `‖Δθ‖/‖Δdata‖ ≤ cond(A)`; equilibration isolates genuine
   collinearity from unit scaling.
2. **Analytic covariance** `Cov = σ̂² (ΘᵀWΘ)⁻¹` for *all* active coefficients,
   hence relative standard errors `SE_i/|θ_i|`. This generalises the
   `Var(γ₀)` diagonal that `stability.VaryingCoefSetup` already computes for the
   constant block.
3. **Monte-Carlo noise ensemble**: empirical coefficient CV and the ratio
   `R = Var_empirical / Var_analytic`. Across the five systems `R ≈ 0.003–0.5`,
   reproducing the June-2026 audit's finding (`R ≈ 0.01–0.23`, see
   `stability_audit.md`) that the analytic covariance over-states the
   seed-to-seed spread because the dominant perturbation on real data is
   **deterministic discretisation**, not the injected noise. This is *why*
   family (4) includes a differentiation perturbation.
4. **Perturbation-response → Lipschitz slope**: relative coefficient change
   `‖Δθ‖/‖θ‖` versus perturbation magnitude, over three data-level families —
   additive noise (`σ = nl·0.01·std`), grid decimation (stride), and Gaussian
   pre-smoothing (a differentiation-scheme proxy). The log-log slope is the
   local Lipschitz exponent: `≈1` Lipschitz/well-posed, `>1` super-linear. Empirically
   the **noise** slope is ≈ 1 everywhere (well-posed w.r.t. stochastic noise),
   while the **resolution/diffscheme** slopes are ≈ 2 (super-linear sensitivity
   to discretisation) — the same discretisation-dominance story as `R ≪ 1`.

### Framing the existing varying-coefficient work
The `Var(γ₀) + NC_deb` per-term score (`stability.py`, default `gram_mode =
vcoef`) is the *analytic local-covariance* piece of this inverse map: it is the
linearised sensitivity of a term's coefficient field to the data, which is why
the audit correctly re-cast `Var(γ₀)` as a structural-misfit detector rather
than a significance test. The MC ratio `R` is the empirical counterpart that
exposes when that analytic sensitivity is dominated by deterministic error.

## 5. Coupling of the two halves

The two maps are not independent: forward-sensitive data is intrinsically
harder to invert stably. Lorenz (forward abscissa `α = +0.30`, chaotic
IC-separation) shows the highest inverse condition number among the ODEs (its
coupled `uv` equation, `cond ≈ 14`); LV (`α = −3.2`, contracting) is
well-conditioned (`cond ≈ 2.3`). The forward growth rate of the dynamics
lower-bounds how well a single trajectory constrains the inverse problem. See
`hadamard/plots/coupling.png`.

This also yields an optional **well-posedness filter**: a discovered/candidate
dataset whose data-estimated forward spectrum is ill-posed (or whose inverse
condition number is huge) can be flagged or rejected before it is trusted for
prediction.

## 6. Results summary (5 systems)

| system | forward type | `α` (DMD) | inverse `cond` (worst) | `R` | noise Lip. |
|--------|-------------|-----------|------------------------|-----|-----------|
| lv     | well-posed (contracting) | −3.2 | 2.3 | 0.16–0.29 | ≈1 |
| lorenz | well-posed; chaotic IC sens. | **+0.30** | **13.8** | 0.003–0.009 | ≈1 |
| wave   | **undetermined** (0.4 periods) | 0.49 | 1–? | 0.13 | 0.6 |
| kdv    | dispersive (well-posed) | ≈0 | 5.2 | 0.52 | ≈0 |
| ns     | mixed (marginal limit cycle) | ≈0 | 4.2 | 0.06–0.11 | ≈1.8 |

(Numbers regenerate via the commands below; see `hadamard_results.md` for the
full tables.)

## 7. Reproduce

```bash
# synthetic sanity (heat/backward-heat/advection/KdV/linear-ODE)
python projects/thesis/hadamard/test_synthetic.py
# full battery over the 5 systems -> hadamard.json
python projects/thesis/hadamard/run_hadamard.py
# tables + figures -> hadamard_results.md, hadamard/plots/
python projects/thesis/hadamard/plot_hadamard.py
```

The forward analysis is pure numpy/scipy (no EPDE imports); the inverse
analysis reuses the thesis runner's pool-only path. Extend to all 14 systems
with `--systems …` (see `noise_stability_sweep.SYSTEMS`).
