# Audit: stability statistics + regularizer — findings and improvement proposals

Scope: `epde/operators/common/stability.py` (`VaryingCoefSetup`) and
`epde/operators/common/sparsity.py` (`PhysicsInformedLasso`). Evidence:
`audit_t1_nc_null.py` … `audit_t6_nc_pde.py` (this audit) plus the session's
`formula_*.py` / `verify_channels_14.py` experiments. All tests seeded and
reproducible; the RFE+CD replication used for mitigation A/Bs matches the
production class exactly (`repl=OK` on 27/27 cases, T4).

## 1. What the statistics actually are (audit + T1 + T2)

Per term j the score is `s_j = (Var(γ_{j,0}) + NC_deb,j) / γ_{j,0}²` with
`γ̂ = (Θ_B^T W Θ_B)^{-1} Θ_B^T W y`, `Var = σ̂²·[(Θ_B^T W Θ_B)^{-1}]_ii`,
`σ̂² = RSS_const/(N_eff − nf)` (constants-only fit, stability.py:737-742), and
`NC_deb = Σ_{k≥1} max(γ_k² − λ·Var_k, 0)` (score, lines 808-860).

**Finding 1a — `Var(γ₀)` is not a sampling variance on real data; it is a
deterministic misfit ratio.** Monte-Carlo over 20 noise seeds (T2):
`R = Var_emp/Var_est = 0.01–0.23` on ode/lv/wave/burgers — the estimator
over-states the true seed-to-seed variance 5–100×, because σ̂² is dominated by
deterministic residual (discretization/smoothing distortion, identical across
seeds), not stochastic noise. Consequences:
- the "1/t² significance" reading is wrong on real data; the channel is in fact
  a **structural-misfit detector** — which is precisely why it catches
  coordinate/trig-modulated spurious terms (their constants-only fit has huge
  misfit) and why it tracks the target's noise floor σ̂²/⟨y²⟩ under noise;
- "fixing" the calibration naively would destroy the degeneracy defense
  (the u·cos spurious cv 13.2 would collapse to ~0.1 and be kept).

**Finding 1b — `NC_deb` carries a positive-part bias, and the debias breaks
under correlated noise.** Synthetic null (T1, constant-coefficient truth):
- iid noise: `E[NC_deb(λ=1)] ≈ 0.48·Σ_k Var_k`, matching the Gaussian
  positive-part prediction exactly; grows linearly with mode count B−1.
- spatially correlated noise (corr length 3, the realistic derivative-noise
  case): `Var_k` is under-estimated ~7.4× → the subtraction removes only ~10%
  of the noise energy; even λ=3 barely helps. This is the likely source of the
  PDE truth NC floors (kdv 0.38, ks 0.52, ns 2.2 at 2% noise).

**Finding 1c — score scale-invariance (verified to 1e-12):** both channels are
invariant under per-feature rescaling, so **no feature normalization is needed
for the statistics**; normalization matters only for the comparison measure in
the keep rule (`formula_normalize_invariance.py`).

## 2. Regularizer mechanics (audit + T3 + T4)

Keep rule: term j survives the CD iff `|ρ_j| ≥ cv_j · max_corr`
(sparsity.py:337-348, 390-393), i.e. cv is the per-term penalty as a fraction
of the LASSO λ_max; cv ≥ 1 ⇒ auto-prune. Selection only — magnitudes come from
the relaxed weighted-OLS refit (456-471).

**Finding 2a — latent weighted/unweighted inconsistency, currently harmless.**
The inner CD metric is unweighted (`ρ = X_j·resid` line 390, `norm_sq = ΣX²`
line 285, `X_T_y` line 286) while init/refit/statistics are weighted. T3:
`g_func` is **uniform** in the current pipeline (w_ratio = 1.0 on all systems),
so decisions are identical today (9/9 same). It bites only if a non-uniform
weak-form window is ever configured.

**Finding 2b — the full RFE+CD cascade over-prunes true terms under noise, and
no within-template mitigation helps.** T4, production-validated, truth+spurious
candidates over noise {0.5, 2, 8}%:

| variant                      | trueKeep | spurPrune |
|------------------------------|----------|-----------|
| production (cv·max_corr)     | 12/27    | 22/27     |
| cap `min(cv,1)`              | 12/27 (no-op) | 22/27 |
| NC-only cv                   | 16/27    | **9/27** (loses the defense) |
| frozen anchor (no cascade)   | 11/27    | 23/27     |

Over-pruning is not caused by cv>1, the re-anchoring cascade, or the channel
mix; it is the comparison itself — noise-degraded `|ρ_j|` of weak true terms vs
misfit-inflated `cv_j·max_corr`. lv and wave survive perfectly at all noise
levels (the noise-robust group).

**Finding 2c — the burgers_inviscid blind spot is collinearity, invisible to
ANY per-term stability statistic.** T5: the modulated spurious `u·u_x·t` is
**0.946 weighted-correlated** with the true `u·u_x`; the joint fit splits the
true coefficient (−0.74 + −0.164 instead of −1.0) and both coefficient fields
are genuinely near-constant (β std ≈ 0.01) ⇒ both look stable (spur cv 0.147).
Finer basis does nothing (K=12: identical cv) — there is no spatial variation
to detect. The x-modulated variant is correctly handled (γ₀≈−5e-5, cv 3.4).
This is the same family as the open cos-modulation degeneracy.

## 3. Objective level (session formula_* + T6)

- ODE systems: truth NC = 0 exactly under noise (debias removes everything) ⇒
  **NC-only instability restores the truth to the Pareto front** where the
  current sum is dominated 4/5 (`formula_var_vs_ncdeb.py`).
- PDE systems (T6, 2% noise, wrong-structure competitors ADD/REPL/MISS):
  **identical domination outcomes under sum and NC-only on all 5 PDEs**
  (wave/pde_compound/burgers_viscous PASS; ac, pde_divide FAIL under BOTH —
  pre-existing fragility, not an NC-only regression).
- Both WAPE and the instability are monotone in the target's noise floor
  σ̂²/⟨y²⟩, so no per-term reweighting or fit-normalization can recover the
  ranking once a lower-order spurious target halves the noise floor
  (`formula_wape_objectives.py`).

## 4. Ranked improvement proposals

**P1 (recommended, cheap, verified): NC-only instability OBJECTIVE; keep
Var+NC in the regularizer.** Evidence: ODE Pareto rescue + PDE neutrality (T6)
+ regularizer needs Var for the degeneracy defense (T4: NC-only spurPrune
9/27). Change: equation-level objective sums `NC_deb/γ₀²` instead of
`(Var+NC_deb)/γ₀²` (one branch in the score aggregation; regularizer path
unchanged). Since the search returns the non-dominated front, the domination
tests are selection-level evidence: non-dominated ⇒ in the returned set. The
remaining caveat is exploration only — the evolution must *generate* the true
structure (the noisy ode runs returned only `u_t`-target equations, suggesting
`u_tt`-target candidates are rarely visited); a small noisy-ODE ablation
(~5 reps) would confirm the end-to-end gain.

**P2 (recommended, cheap, validated): post-RFE collinearity SWAP test (+ pair
sweep).** After the RFE converges, for each survivor test |corr_w| > 0.9
PRUNED terms as replacements (swap on RSS win or 1%-tie with lower
complexity), then resolve surviving pairs by drop-sweep. Validated in T8:
trueKeep 12→14/27, spurPrune 22→24/27, zero regressions — fully fixes the
burgers_inviscid blind spot (T5) and the coefficient bias (−0.74 → −1.0).
A drop-only sweep without the swap is a no-op (the RFE kills the wrong twin
first). Per-term stability cannot do this by construction.

**P3 (validated prototype, T7 `audit_t7_corr_debias.py`): correlation-aware
debias for NC_deb via spectral inflation factors.**
`Var_k^corr = Var_k · F_d(πk/n_d) · Π_{d'≠d} F_{d'}(0)`, with `F_d` a
Newey-West spectral ratio from the lag-1..8 autocorrelation of the
constants-only residual along axis d (iid ⇒ F≡1, correction vanishes; cost
O(N·L) per solve). Validation:
- synthetic null: corr-noise debias effectiveness restored 12% → **51%**
  removed (= the iid level; the remaining 49% is the universal positive-part
  bias, composable with a λ/B correction); iid case unchanged (55%→55%);
- real @2%: kdv_cossin's noise-artifact NC floors **vanish** (u_xxx cv_nc
  5.45 → 0.000, u_x 0.021 → 0.000) while its GENUINE modulation survives
  (cos(t)sin(x): 0.24 → 0.20); kdv truth floor 0.37 → 0.29; spurious stay
  caught (wave 1.42 → 1.27, kdv 4.6e4 → 2.5e4).
Synergy with P1: with corrected NC the modulated-truth system kdv_cossin —
the NC-rescue failure case — drops its truth objective from ~5.7 to ~0.2,
extending the noise rescue beyond constant-coefficient truths.

All-14-system validation (T7b `audit_t7_all14.py`, 2% noise, truth+spurious):
- **no regression on any system**: ODE truth NC stays 0 (lv: 5e-5); spurious
  detection retained everywhere (wave 1.42→1.27, kdv 4.6e4→2.5e4, ks 572→523,
  pde_compound 4.09→3.94, burgers_viscous 13.2→9.9, ns ~unchanged);
- effectiveness tiers: **kdv_cossin 5.72→0.20 (28×)**; moderate kdv 0.38→0.29,
  ks 0.54→0.45, pde_compound 0.14→0.08; **neutral ns 2.15→2.13 and pde_divide
  8.9→8.5** — their floors are NOT short-range correlated noise (deterministic
  misfit / longer-range correlation), i.e. the upstream target-noise problem;
- ac a wash (true 0.46→0.56, spur 0.44→0.66 — detection ratio actually
  improves); burgers_inviscid blind-spot spurious gains a nonzero NC
  (0→2.7e-3) but remains far below threshold — P2 still required.
Implementation: inside `_solve_gammas` (residual is already formed there),
flag-gated.

**P3 scope limit (T8 `audit_t8_reg_corr.py`): the corrected debias does NOT
belong in the regularizer — tested in BOTH forms.**
(1) cv = Var + NC_corrected: trueKeep 12→11/27, spurPrune unchanged — no gain,
ac regresses. (2) cv = NC_corrected only: 16/27 + spurPrune 10/27 vs
uncorrected NC-only's 16/27 + 9/27 — one case better, still catastrophic vs
base (22/27): the ODE-family degenerate spurious have NC = 0 under ANY
debias (insignificant constants, visible only to Var), so no NC calibration
can replace the Var channel. **P3 is objective-only.**

**P2 placement (T8): a post-RFE pair-sweep alone is ineffective, the SWAP
form works.** At 2–8% noise the RFE keeps the WRONG member of the collinear
pair (kills true `u·u_x`, keeps spurious `u·u_x·t`), so no surviving pair
exists for a drop-only sweep. The validated form (`collin_swap` in
`audit_t8_reg_corr.py`): for each survivor, test highly-correlated (|corr_w| >
0.9) PRUNED terms as replacements — swap on clear RSS win, or within 1% with
lower complexity; then the drop-sweep for surviving pairs. Result:
**trueKeep 12→14/27, spurPrune 22→24/27, zero regressions** — both gains are
the burgers_inviscid blind-spot cases (`..` → `TS` at 2% and 8%), fixing T5's
failure in both directions.

**Anchor formula (T9 `audit_t9_anchor.py`): max_corr is not the bottleneck;
no alternative dominates.** Racing threshold scales in the production harness
(8 systems × 3 noise, trueKeep/spurPrune): base `cv·max_corr` 12/22, median
anchor 14/21, self-norm (`r_j ≥ cv_j`) 11/23, own-correlation 14/20, hard
cv<1-only 19/13. Every formula lands on the same ~34-total trade-off frontier
— the choice only redistributes errors between true-term protection and
spurious rejection. The `hard` pole shows both that correlation pressure
causes 7 of the over-prunes AND that it does irreplaceable spurious-killing
(13/27 without it, incl. sub-1-cv spurious like lv's 0.93). Verdict: keep
`max_corr` (median is a marginal +1-net alternative, within single-seed noise
of this 27-case benchmark); the real levers remain the swap test and upstream
noise handling.

**Keep-rule factorial (T10/T11 `audit_t10_factorial.py`/`audit_t11_minmax.py`):
cv form × anchor × normalization, 68 cells.** (trueKeep/spurPrune out of 27;
production = `sum|max|raw` 12/22, 12 perfect-TS cases.)
- **Normalization never improves the leaders**: top-5 cells are all `raw`;
  L2-unit is a near no-op; min-max only shifts along the frontier toward
  stricter pruning — `mm01`+`sqrt` reaches **27/27 spurPrune** at trueKeep
  10-11 (the "maximum strictness" corner, if spurious rejection is the
  priority).
- **Anchor: median > max ≈ own > self**, consistently across cv forms.
- **cv form: var ≥ sqrt ≈ sum ≫ nc** (nc collapses spurPrune ≤ 9 in every
  cell — Var is irreplaceable in the regularizer, mirroring the objective
  where NC-only wins; the two uses want opposite channels).
- Best cell **`var|median|raw`: 16 TS, 17/20** (+4 perfect cases, +5 trueKeep,
  −2 spurPrune vs production); `sqrt|median|raw` (14 TS, 14/24) strictly
  dominates production on all three aggregates.
- Caveat: single seed, 27 constructed cases — ±2 differences are within noise.
  If changing the rule, the evidence supports the median anchor and a
  var-weighted cv, gated by a multi-seed re-run (per-case detail in the
  JSONs).

**Channel-formula grid (T12 `audit_t12_channel_formulas.py`): Var/NC internal
formulas, 34 cells.** cv = (VarF + NcF)/γ₀², VarF ∈ {none, current,
spectral-corrected} × NcF ∈ {none, λ=1, λ=2, sum-level debias,
bias-subtracted, spectral-corrected} × anchor {max, median}:
- **NC formula choice is irrelevant to the regularizer's binary decisions**:
  v1n1 = v1n2 = v1nS = v1nP (identical 14/14/21 at median) — the λ/positive-
  part refinements matter for the continuous objective score, not for
  keep/prune.
- **Dropping NC from the cv entirely is best** (n0 rows top the table), and
  the **spectral-corrected Var helps**: `vCn0|median` = **16 TS, trueKeep 18,
  spurPrune 21** — the best cell across all grids (T9–T12) vs production
  `v1n1|max` 12/12/22. The vC factor is per-equation (Π F_d(0)), acting as an
  equation-adaptive strictness multiplier — noisier equations get stricter
  thresholds.
- v0 rows (no Var) collapse spurPrune ≤ 10 regardless of NC sophistication —
  final confirmation that Var is the regularizer's irreplaceable channel.
- **Cumulative conclusion of the keep-rule search**: the channels separate
  perfectly by use — regularizer cv = spectrally-corrected **Var only**
  (median anchor), objective = corrected **NC only**. Same gate as before:
  multi-seed confirmation, then search-level ablation.

**Bounded cv transforms (T13 `audit_t13_bounded.py`): smooth compression adds
the final increment.** transform(cv) ∈ {id, cap1=min(cv,1), sat=cv/(1+cv),
log=log(1+cv)} on bases {v1n0, vCn0, v1n1} × anchors:
- **New best: `vCn0*log|median` — 17 TS, trueKeep 19, spurPrune 21** (and
  `*sat` 17/19/20); vs unbounded vCn0|median 16/18/21 and production 12/12/22.
- Mechanism: compression is ~identity below cv≈1 but softens the 1–3 zone
  (log(1+2)=1.1) where noise-inflated TRUE terms lived, while huge-cv spurious
  thresholds stay far above their |ρ| — selective rescue of the borderline
  zone, ordering preserved.
- The hard cap1 is inferior to smooth log/sat (loses spurPrune); transforms
  only help with the **median** anchor (with max they're a no-op/noise).
- **Final cumulative keep rule** (T9–T13, 126 cells): cv = log(1 +
  F₀·Var(γ₀)/γ₀²) (spectral-corrected Var only), thr = cv·median|Xᵀy| —
  +7 trueKeep / −1 spurPrune / +5 perfect cases over production. Gate:
  multi-seed confirmation + search-level ablation.

**Clean-data gate (T14 `audit_t14_clean.py`, noise 0, 9 equations):** the
finalist `vCn0*log|median` is **9/9 perfect — no clean regression** (production
also 9/9). The clean risk was real but specific: plain Var-only `v1n0` loses
the ac spurious on clean data (8/9) because clean σ̂² is tiny and the Var
channel weakens — and **the spectral factor F₀ is what repairs it** (the
clean deterministic residual is highly autocorrelated → F₀ inflates Var back
to catching strength). So vC is not just noise-adaptive strictness; it is
also the clean-case fix that makes dropping NC from the cv safe.

**Full-benchmark validation (T15 `audit_t15_all14.py`): ALL 14 systems ×
noise {0, 0.5, 2, 8}%, 76 cases, finalists head-to-head:**

| rule | perfect TS | trueKeep | spurPrune |
|---|---|---|---|
| **vclog+swap (proposed stack)** | **58/76** | **61** | 70 |
| vclog (`log(1+F₀·Var/γ₀²)`·median) | 57 | 60 | 67 |
| vc (unbounded) | 56 | 59 | 67 |
| vcn1log (+NC hedge) | 54 | 54 | 67 |
| varmed (UNcorrected Var-only) | 52 | 61 | **61** |
| production (`(Var+NC)/γ₀²`·max) | 45 | 45 | 71 |

- The 8-system grid ranking generalizes unchanged to the full benchmark.
- **Proposed stack vs production: +13 perfect cases (+29%), +16 trueKeep,
  −1 spurPrune** — near-parity on spurious rejection with massively better
  true-term protection.
- The swap test remains additive (+1 TS, +3 spurPrune on top of vclog).
- varmed confirms T14 at scale: WITHOUT the spectral factor, Var-only loses
  ~9 spurious cases (61 vs 70) — F₀ is what makes dropping NC safe.
Remaining gate: search-level ablation (the rule changes are selection-level
validated; end-to-end discovery confirmation pending).

**Time-only basis modes (T16 `audit_t16_timeonly.py`, all 14 × 4 noise):**
restricting the cosine basis to the time axis (`modes = (K_t, 1, …)`):
- **Exactly neutral for the finalist** (vclogT = vclog 57/60/67, vcT = vc):
  by Frisch–Waugh decoupling, γ₀ and Var(γ₀) come from the constants-only
  block and F₀ from the residual — Var-only rules never touch the basis, so
  the mode domain is irrelevant to them by construction.
- **A real improvement for NC-carrying rules**: production with time-only
  modes goes 45/45/71 → **49/51/69** (+6 trueKeep, −2 spurPrune) — the
  spatial NC floors (ks, burgers_viscous, ns, kdv) were over-pruning true
  terms. If NC stays in a keep rule, restrict it to time. Still loses to the
  finalist (49 < 57).
- Caveat for the OBJECTIVE side (NC-based): time-only NC would also drop the
  PDE truth floors there, but would blind the score to purely SPATIAL
  modulation — the documented cos(x)·u_xx degeneracy family — so it is not
  recommended for the objective without a spatial-degeneracy guard.

**P4 (low, consistency): weight the CD metrics** (`ρ`, `norm_sq`, `X_T_y`)
so the optimizer matches its init/refit. No behavioral change today (T3,
g_func uniform); prevents silent divergence if a non-uniform window arrives.

**P5 (documentation/honesty): rename the "significance" channel.** It is a
deterministic misfit statistic on real data (T2); docstrings should say so, and
no future "calibration fix" should be applied to it without re-testing the
degeneracy defense (it works *because* it is miscalibrated as a t-test).

**Explicitly rejected by evidence:** cv cap (T4 no-op), frozen anchor (T4
worse), NC-only in the regularizer (T4 spurPrune collapse), basis refinement
for the collinearity blind spot (T5 no effect), score-formula reweighting for
noise robustness (A1/A2/A3, `formula_ab_noise.py`), fit-metric replacement
WAPE→relL2 variants (`formula_wape_objectives.py`).

## 5. Open weaknesses (no in-template fix found)

- Regularizer over-pruning of weak true terms under noise (T4) — the honest
  lever is upstream noise-robust derivative estimation, or a survival test that
  consults the residual increase of removal (a MISS-test) rather than |ρ| alone.
- ac / pde_divide truths are Pareto-dominated by wrong structures already at 2%
  noise under ANY tested objective variant (T6) — target-noise-floor problem,
  upstream of the statistics.
