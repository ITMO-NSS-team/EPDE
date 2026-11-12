# EPDE solver-based fitness (PINN) study

*Tracked on branch `pinn` since 2026-11-12. **Part I** is the current assignment. **Part II** is the record of the first study (Sep–Oct 2026, DeepXDE backend), kept as background. Numbers are quoted from the result files named next to them; when a number and a file disagree, the file wins.*

---

# Part I. The assignment: a general PINN for equation discovery (Nov 2026)

Read A (the problem) and B (what is already known), then do the tasks C to F in order. H says how to work. Part II is reference material; B points into it.

Words used throughout:
- **Candidate:** an equation, or a system of equations, proposed by EPDE's search, with its coefficients already fitted.
- **Record:** the observed data of one system on its grid.
- **True form, wrong form:** the equation that generated a record, and a deliberately wrong one (a term dropped). Both are refitted to the data before a solve.
- **Gate:** `ln(RMSE_wrong / RMSE_true)` for one seed. Above 0 means the true form scores better.
- **Arm:** one named solver configuration. Arms differ from their control by one mechanism, so a difference in results has one cause.
- **PINN:** a physics-informed neural network: a network trained so that it satisfies the equation's residual.
- **PI:** the supervisor of this study.

## A. Why the PINN exists

EPDE's search scores every candidate by a *discrepancy*. The solver-free discrepancy is WAPE. It evaluates the candidate's terms on finite-difference derivatives of the data and measures how far the target term is from the fitted combination of the others. It works in derivative space and never asks what field the equation would produce.

The PINN answers that question. For a candidate with fixed coefficients it returns the field U that the equation produces, and the discrepancy becomes the error of U against the original data:

```
discrepancy(equation) = RMSE (or MSE) of U against the observed field
```

**Rules for the objective**
- One value per equation, scored on that equation's own variable, never averaged over variables (§2, directive 9a).
- Scored only on points the solve was not given. A network that fits the data it was shown scores well whatever the equation is.
- The same solve is reused for forecasting: continued past the end of the record, it is the candidate's forecast.

**The data regime** is what the solve is given. There are two, one arm each:

| Regime | The solve is given | Scored on |
|---|---|---|
| `tail` | the observations on the first 80 % of time levels | the last 20 % of time levels. This is the first study's definition, and it is a forecasting test inside the record. |
| `ic` | the initial state only; on an open domain also the observed values on the side boundaries | the whole record after the initial state |

**The reference.** A candidate that can be solved for its highest time derivative can also be integrated by an ordinary numerical integrator, with no network. That field, U_ref, has two uses:
- It gives the objective a perfect solver would report. This tells us whether the objective itself separates the true form from the wrong one.
- It gives the PINN's own error, ‖U − U_ref‖. This tells us whether the optimiser reached the solution.

The first study could not separate these two questions. Lorenz failed the gate under every arm of its Stage 2, while an integrator started from the true state at the train/tail split gave a gate of about +7 (§5.0). The failure was the optimiser's.

The PINN stays the solver because the search proposes forms an integrator cannot take: targets that are not a time derivative, constraint equations such as continuity, open domains. The reference exists only for the benchmark forms. Its job is to check the PINN there.

## B. What is already known

Every row was measured. "Local" means the result files exist only on the machine that ran them.

| Finding | What it means for this work | Where |
|---|---|---|
| A learning rate moves the gate: Allen-Cahn (AC) +0.195 … +0.650 and Duffing −2.03 … +4.28 across rates. | No tuned learning rate. Use a line search, curvature or a trust region. | §2 directive 1, §4 |
| L-BFGS alone, with the inputs scaled to [−1, 1], ties the Adam recipe on AC, Duffing and Burgers. Without the input scaling it fails Duffing (−1.81). | This is the starting recipe, arm `G0`. | §5 |
| DoG, Prodigy and a learning rate computed from the data fail a system each. A second data-computed rate clears AC by only 0.01. | Not repeated. | §5.0, "Recalculation" |
| Levenberg–Marquardt on a [32]×4 network beats L-BFGS at completion on AC and Duffing, loses on Burgers, and is slow. | The first optimiser arm. | §5.0, "Recalculation" |
| In float32, L-BFGS can stop on a zero line-search step while the loss is still falling (Duffing, Lotka-Volterra). Float64 removes the stall but runs to the iteration cap. | Float64 is an arm. Every row records why the optimiser stopped. | §6.2, §5.0 |
| Choosing the iterate on a validation block inverts the gate. | Always score the last iterate. | §4, §8 rule 6 |
| Loss weights taken from the gradient norm at the start are fragile. On Lorenz they are probed at an equilibrium and range from 0.008 to 11 by seed. | Loss balancing is a component to study. | §5.0, "Stage 3" |
| Dividing every loss term by the variance of its own signal gave a loss that trains at unit weights (double pendulum, Duffing, wave; Aug 2026). | The `var` loss terms with `unit` weights are the first balancing arm. | `projects/pic/data/dp/pinn_test_autoscale.py`, flag `PHYS_NORM`; results local |
| A coefficient-constancy statistic used as a loss term collapses the field (3 of 3 systems). A sum of log-terms collapses onto an equilibrium (6 of 6 seeds). | Not repeated. `DY-mon` (section F) is a diagnostic, never a loss term. | same script, flags `STAT`, `LOSS_FORM`; results local |
| The training loss cannot see a missing weak term: AC with and without diffusion tie on it, while integrating the two forms separates them 16–24× (3 seeds, Sep 14). | The objective must compare the solved field with the data. A loss value is not an objective. | `projects/pic/data/ac/pinn_test_autoscale.py`, flag `DIFFUSION`; results local |
| A scale-invariant observation loss collapses on fields that change sign. The scale-invariant physics loss gave near-zero gates in Stage 1 but propagated Lotka-Volterra (LV) on 3 of 3 seeds in Stage 2. | One physics-loss arm (`sinv_live`); no observation variant. | §5, §5.0, §6.3b |
| Outputs scaled by the train-window mean and standard deviation win on Navier-Stokes (NS) and propagate LV on 12 of 12 solves, but alone they fail Duffing at the time bar (−0.845). | The output-transform arm. | §5, §5.0 |
| The equilibrium trap: a freshly initialised network outputs about 0, which is a zero-residual equilibrium of Lorenz and LV, and fitted solutions slide onto an equilibrium where there is no data. | Report a collapse flag with every score. Short time windows remove the trap. | §5.0, "Stage 3" |
| Marching the tail in short windows from the last observed state (arm TM-obs) passes the five systems whose tail is an initial-value problem. Marching the whole window, growing domains and causal weights all fail, because none fits the train window without a learning rate. | `G-tm` is the second baseline arm. `DY-cont` (section F) targets the train-window fit. | §5.0, "Stage 3" |
| On an open domain (NS) the tail is not an initial-value problem: the pressure drifts by a free function and TM-obs scores −2.20. | An open domain needs boundary data or the global solve. | §5.0, "Stage 3" |
| Under DeepXDE a solve's result depended on what its process had solved before (−20 % on AC, −53 % on Duffing). | Keep one fresh process per solve until a probe shows the general PINN is free of this. | §4, §8 rule 1 |
| Wall-clock time moved by up to 49 % between identical runs. | Compare arms at equal iteration budgets and report seconds beside them. | §5.0, "Stage 2" |

## C. Task 1: build the general PINN

Build the package `projects/pinn/gpinn/`. Nothing in it exists yet.

**Why not DeepXDE.** The first study changed losses, optimisers and weighting through hooks around DeepXDE; `epde/integrate/deepxde_integration.py` grew to over 1,900 lines. A plain PyTorch solver that owns its training loop makes each of those a named option.

**All code lives inside `projects/pinn/`.**
- The library imports only numpy, torch and scipy: no deepxde, and nothing from `epde/` or `projects/pic/`.
- Helpers that already exist elsewhere are copied in, not imported. Each copy names its source file and commit and leaves the DeepXDE-only parts behind.
- One file, `bridge_epde.py`, is the only one that imports `epde`. It turns a fitted EPDE system into the library's own `Problem`.
- Nothing under `epde/` changes.
- The datasets stay under `projects/pic/data/` and are read by path. They are data, not code.

| File (to build) | Content | Copy the logic from |
|---|---|---|
| `spec.py` | The equation spec and its residual, `Σ c_k · (product of factors) + free term − target`, on numpy or torch; the list of derivatives a spec needs. | `epde/integrate/residual_terms.py` (`EquationSpec`, `residual`, `required_fields`) |
| `data.py` | `Problem`: the specs, the axes, the inner mask, the observed fields. The time split (what the solve is given, what is scored). The data profile (bounds, periodic axes). | `epde/integrate/heldout.py` (`time_split`), `epde/integrate/data_profile.py` (`build_profile`) |
| `bridge_epde.py` | A fitted EPDE system → `Problem`. It reads the grids, the inner mask and the observed fields from the samples manager, as `_held_out_replica` in `projects/pinn/gate.py` does. | new |
| `net.py` | The network (tanh layers, Glorot start) and its transforms: affine inputs, periodic embedding, outputs scaled by train moments, a hard initial state. | `hard_ic_transform` in `epde/integrate/time_marching.py` |
| `fields.py` | The autograd table `{(variable, orders): tensor}` for the derivatives a spec needs, sharing the lower orders. `orders` counts derivatives per axis; axis 0 is time. | new |
| `terms.py` | Loss terms as residual vectors. Physics: `mse`, `var` (divided by the variance of the target term), `sinv_live` (divided pointwise by the summed size of the terms). Observation: `mse`, `var`. | new; the scale-invariant ratio is `cancellation_ratio` in `residual_terms.py` |
| `balance.py` | Loss weights: `unit`, `variance`, `grad_init` (1/gradient norm at the start, then frozen: the first study's rule), `grad_ema` (the same, tracked during training). | new |
| `optim.py` | `lbfgs` (an own loop around torch's strong-Wolfe L-BFGS that records why it stopped and calls back after every iterate), `adam`, `lm` (Levenberg–Marquardt on the residual vectors), `dog`, `prodigy`. | `epde/integrate/lr_free.py` (`DoG`, `Prodigy`; `levenberg_marquardt` for the damping rule) |
| `marching.py` | The regimes `tail` and `ic`. Window plans: windows of length τ_min, dyadic slabs, and the observed state at a window's left edge. | `tau_min`, `window_edges`, `window_caps`, `observed_edge_state` in `time_marching.py`; `_dyadic_slab_edges` in `epde/operators/common/survival.py` |
| `solve.py` | `solve(problem, config, seed)` returns the prediction, the stats and a per-iterate trace. `forecast` continues a finished solve past the record, window by window, from its last state. | new |
| `score.py` | RMSE and MSE per equation on the scoring points. The same for two trivial fields, "keep the last given state" and "the mean of the given data"; a score no better than those raises the collapse flag. | new |
| `reference.py` | The exact integrator: `scipy.integrate.solve_ivp` for systems of ordinary differential equations, a Fourier method of lines for periodic 1-D equations. It reads the right-hand side from the same spec and refuses a form it cannot integrate (NS). | new |
| `stats.py` | Paired-seed contrasts, the smallest difference a design can detect, the number of seeds needed. | `contrast`, `mde`, `seeds_needed`, `arm_table`, `contrast_table` in `projects/pic/data/pinn_common.py` |

**Conventions**
- The config is one flat dict. Every row stores it whole. An unknown key raises: the DeepXDE adapter ignored unknown keys silently, and that hid mistakes.
- A solve is a pure function of `(problem, config, seed)`. On the CPU in float64 the same three give the same bits.
- Nothing computed from the data may read the scoring points. Replace them by NaN and the solve must not change.

**The two baseline recipes**
- **`G0`, the starting recipe.** It repeats the first study's arm S1 (§10).
  - Network: [64]×4 tanh, float32.
  - Optimiser: L-BFGS only, up to 4000 iterations (strong-Wolfe line search, history 100, gradient tolerance 1e-8).
  - Inputs scaled to [−1, 1]; 4000 collocation points.
  - Loss weights `grad_init`; regime `tail`.
- **`G-tm`, tail marching.** It repeats the first study's arm TM-obs.
  - No solve on the train window.
  - The tail is cut into windows of length τ_min = min over variables of std(x) / rms(dx/dt), measured on the train levels and snapped to whole grid levels.
  - The first window starts from the observed state at the last train level; each later window starts from the end of the one before and from its weights.
  - A window takes its start state exactly, through the hard initial-state form, so there is no continuity penalty.
  - The iteration budget is split over the windows by their length.

**Tests**, in `projects/pinn/tests/`:
- the derivative table against an analytic field;
- the bridge: the copied residual equals EPDE's own on the true forms of the six systems (the one test that imports `epde`);
- an import scan: only `bridge_epde.py` imports `epde`, and nothing imports `deepxde`;
- a repeatable solve: CPU, float64, same seed, same bits;
- no leak: NaN on the scoring points leaves the solve unchanged;
- the score equals the fitness host's rule on a toy field (predict on the full grid, keep the inner points, take the scored time levels);
- the reference reproduces the record for the true forms;
- an unknown config key is refused;
- one Levenberg–Marquardt step lowers the loss on a linear toy problem.

**Done when** `python -m pytest projects/pinn/tests -q` passes and `tests/unit` gives the counts in section H.

## D. Task 2: reference, harness and baseline

- **Harness.** Add a backend `gpinn` to `projects/pinn/gate.py`, next to `deepxde` and `basis`.
  - It reaches the library through `bridge_epde.py`.
  - `run_basis_solve` shows how a second backend is wired.
  - Reuse the six systems and their forms (`system_forms`, `build_search`, `fitted_system`) and the driver (`drive`).
- **Arms.** `G0`, `G-tm`, and `REF`, the integrator. `REF` has no seed, so it runs once per form, like the basis arms (`SEEDLESS_BACKENDS`).
- **What `REF` integrates.** Under `tail` it starts from the observed state at the last train level and covers the scored tail. Under `ic` it starts from the initial state and covers the record. A second-order equation (Duffing) also needs the first time derivative at the start; take it by backward differences, as `observed_edge_state` does.
- **Rows.** Keep today's fields (`tag`, `system`, `arm`, `form`, `seed`, `mode`, `config`, `rmse`, `seconds`, the optimiser's exit record, `env`, `provenance`), so `analyze.py` reads the new rows. Add `iterations`, `ref_error` (RMSE to the reference, per variable) and `collapse` (per variable). Add the library's files to `PROVENANCE_FILES`.
- **Baseline batch**, tag `gpinn_base`:
  - systems: Duffing, LV, Lorenz;
  - forms: true and wrong; seeds 0–2;
  - arms: `G0`, `G-tm`, `REF`, and the DeepXDE arm `TM-obs-3k` as a control on the same code;
  - cost: about 1.5 GPU hours, one fresh process per solve.

**Done when**
- `G-tm` has the reference's gate sign on every seed of the three systems;
- its error to the reference is reported per system and form;
- `G0` repeats the known failure on LV and Lorenz (§5.0). If it passes there, find out why before going on.

## E. Task 3: the training study

**The question.** Which loss, optimiser and balancing reach the reference solution reliably and cheaply on every kind of record: one ordinary differential equation, a coupled system, a partial differential equation?

**Every solve reports four things:**
1. the error to the reference;
2. the gate, per equation and seed;
3. the cost, in iterations and in seconds;
4. the collapse flag.

**Arms.** Each arm is `G0` with one component changed. The evidence for each is in section B.

| Component | In `G0` | First arms |
|---|---|---|
| Optimiser | L-BFGS only | Levenberg–Marquardt on a [32]×4 network; float64 |
| Loss weights | `grad_init` | `var` terms with `unit` weights; `grad_ema` |
| Physics loss | `mse` | `sinv_live` |
| Outputs | the raw network | outputs scaled by train moments |
| Solve order | one global solve | tail marching (`G-tm`); regime `ic` |

**Rules**
- **Systems, in order of cost:** Duffing, LV, Lorenz; then AC and Burgers; then NS. Do not start NS before an arm passes the five others.
- **Write the arms and the contrasts into the source before the batch runs.** A contrast chosen after seeing the rows is not evidence (§8 rule 7).
- **Compare arms at equal iteration budgets.** Report seconds beside them.
- **Decision rule.** An arm is eligible when its gate is above 0 on every seed of every system and it raises no collapse flag. Among eligible arms the lower error to the reference wins; arms closer than the seed spread tie.
- **Do not repeat** what section B lists as already failed.

## F. Task 4: can the dyad idea help the training?

**What dyad is.** It is EPDE's newest instability objective (`dyad_scores` in `epde/operators/common/survival.py`). It cuts each axis into 2, 4, 8, … equal slabs, refits the candidate's coefficients on every slab, and scores how much they scatter between slabs. A true equation has the same coefficients in every slab at every scale.

**How it relates to the PINN objective.** Dyad lets the coefficients move per slab and measures their scatter. The PINN holds the coefficients fixed and measures how far the solved state drifts from the data. They test the same thing from two sides, and they can share the same slabs.

**The slabs here** are 2, 4, 8, … equal, nested slabs of the time axis, for as long as a slab keeps at least the τ_min window's number of levels.

Five ways to use them, one arm each:

| Arm | Mechanism | Control |
|---|---|---|
| `DY-march` | March over the dyadic slabs, each slab started from the end of the one before. | `G-tm` (τ_min windows) |
| `DY-cont` | Inside the train window, solve the short slabs first, each from its own observed start state, and use their union as the start of the long solve. This targets what killed full-window marching on Lorenz: no fit of the train window without a learning rate. | `G0` |
| `DY-int` | The physics term is the residual integrated over each slab, at every dyadic scale. For a target that is a first time derivative the target integrates to u(end) − u(start), so no time derivative of the network is taken, and distant times are coupled directly. | `G0` |
| `DY-bal` | Each slab's residual is divided by that slab's own signal energy, so quiet and busy stretches weigh the same. | `G0` |
| `DY-mon` | A diagnostic only. Refit the candidate's coefficients per slab on the network's own field and compare them with the fixed ones. Where they disagree the solve has not converged; where the columns vanish it has collapsed. It needs no reference and no truth. | none |

**Drop rule.** An arm is dropped unless it beats its control on the error to the reference or on cost, on at least one system, in a resolved paired contrast, without losing the gate on any system.

**Judge `DY-mon` differently:** against the reference, does its verdict agree with the true error to the reference on the solves of tasks 2 and 3?

## G. Later (not part of this assignment)

- **Port the winning recipe into EPDE** as a solver backend beside `deepxde` and `basis`. This touches `fitness.py`, `objectives.py` and `search_config.py`, which other work is changing, so it needs the PI's go-ahead.
- **End-to-end discovery** with the PINN objective against WAPE: found the true form or not, distance to it, solve time, reproducibility of a rerun.
- **A forecasting demo:** keep the end of a record away from the search, discover, forecast, compare.
- **Near-miss wrong forms.** Today's wrong forms drop a whole term. A real search mostly meets the true form plus a small extra term. Never use an equivalent true form as a wrong form (§12).
- **Noisy and real records** (§12).

## H. How to work

- **Branch.** Everything is on branch `pinn`. Paths in this doc are relative to the repository root.
- **Start a fresh baseline.** EPDE's default finite differences became 4th-order on Oct 22. The rows of Part II were made with 2nd-order differences, so their candidates have other coefficients. Never read old and new tags in one `analyze.py` call.
- **Results.** Rows go to `projects/pinn/results/<tag>.jsonl`. That folder is not tracked. Quote a number in this doc only with the name of the file it came from.
- **Bench rules.** §8 applies in full. The ones that cost the most when broken:
  - one fresh process per solve;
  - nothing CPU-heavy beside a timed GPU batch;
  - pin the thread counts for anything compared bit by bit;
  - per-seed rows stay unaggregated, with contrasts fixed in advance;
  - one mechanism per arm.
- **Commits** only with the PI's OK: explicit paths, a prose body with the numbers, no trailers (§8 rule 12).
- **Environment.** §11.
- **Tests.**
  ```
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest projects/pinn/tests -q
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/unit -q
  ```
  On `pinn` on Nov 11 the unit suite gave 2548 passed, 1 skipped, 5 failed. The 5 are bitwise pins of the DeepXDE solve in `tests/unit/test_lr_free_solve.py`. They were captured under 2nd-order differences and fail since the default changed. They are not caused by this work; whether to recapture them is the PI's call.

### Progress log
Add one line per finished step: date, what ran (tag), result file, verdict.

| Date | Step | Result file | Verdict |
|---|---|---|---|
| 2026-11-12 | Assignment written; doc tracked | — | — |

---

# Part II. Record of the DeepXDE study (Sep–Oct 2026)

*Written between Sep 29 and Nov 11 as the onboarding doc of the first study. The text is kept as it was, with stale statements corrected on Nov 12. Its "next" lists (§6.5, §13) are superseded by Part I.*

---

## 1. TL;DR

- **What EPDE does.** EPDE discovers differential equations from data with an evolutionary search. Each candidate equation needs a *fitness*. The **solver-based** fitness solves the candidate with a PINN (DeepXDE). The candidate's coefficients stay fixed; the PINN sees an observation term on the first 80 % of time. The fitness is the RMSE on the **held-out last 20 %**.
- **The question.** Does that fitness rank the **true** equation above a **wrong** one? We measure it with the **gate**:
  `gate = ln(RMSE_wrong / RMSE_true)` per seed (> 0 means the true form wins). An arm's headline number is the **minimum over seeds**.
- **The bar.** The gate must be > 0 on **every seed of every system**. Gate and cost are reported against the shipped Adam recipe.
- **Current goal.** Make the PINN solve **learning-rate-free, or with an lr computed from the data**, without losing the gate. It must also work for **arbitrary forms, multi-dimensional domains and coupled systems**.
- **State on Oct 7 (see §5.0):**
  - **The bar is met lr-free on all six systems** (Duffing, AC, Burgers, LV, Lorenz, NS) by the routed recipe **R-route-3k**, read the pre-registered way (every seed > 0 at T_B and 0.9 T_B). Score +0.360; AC binds.
  - **R-route-3k** routes by the data's periodicity profile:
    - **TM-obs** (`lbfgs_maxiter` 3000) where the held-out tail is an initial-value problem (no spatial axis, or only periodic ones). TM-obs marches only the tail, in short windows, from the observed state at the last training level.
    - **`G-m000`** (S1 + moment outputs) on open domains (NS).
  - **Lorenz** is now a resolved win over the shipped recipe on 6 seeds. Its failure was an **equilibrium trap**, not chaos.
  - The shipped recipe **B fails LV and Lorenz**. Stage 2's 24-cell grid found no arm passing everything. All results use the no-bias rule.
  - **The Stage-3 code is on branch `pinn`** (off `domain_refactor`; commits `9b42f48`, `1da0d71`), checked out in the worktree `../EPDE_lorenz`. The exact as-run tree behind the rows is tag `pinn-stage3-asrun`. The fixed-basis backend stays frozen.
- *History:* Stage 1 (Sep 28–29) found S1 ties B on AC, Duffing and Burgers. Stage 1b (lr alternatives) found nothing better than B.

---

## 2. What we want: the PI's directives

Quotes are verbatim.

1. **"Ideally i want no lr or data-driven one."** Adam's lr is dimensional: it carries the units of the weights. It also measurably moves the gate:
   - AC +0.195 … +0.650 across lr 1e-5 … 1e-2;
   - Duffing −2.03 … +4.28.

   Prefer line search, curvature, trust region, or a length scale taken from the data.
2. **Adoption bar:** gate > 0 on every seed of every system. Gate and cost vs Adam are reported for the PI to decide.
3. **"Lets focus on pinns then cos we need to do search on arbitrary forms and multidimensions/coupled systems."**
   - The fixed-basis backend (Chebyshev/Fourier + L-BFGS, ≤ 2-D box domains) is **frozen as an opt-in**: no grid, no discovery.
   - The network route (S1 and its Stage-2 grid) is the one to finish.
   - Gate systems to add: **Lotka-Volterra, Navier-Stokes wake, Lorenz.**
4. **"Make it a grid search"** for the non-lr options (periodic embedding × collocation density × precision), but **only on the winning lr-free route.**
5. **A scale-invariant discrepancy as the physics (DE) loss.** Two variants, `sinv_live` and `sinv_data`. **Gradient loss weighting is kept** ("Do not want to remove gradient weightening though").
6. **Shared data fit.** Fit ONE network to the observations once per dataset and reuse it as the start of every candidate's PINN. This is *not* the old ANN derivative preprocessor.
7. **"Idk if any floor or eps are needed."** The scale-invariant losses ship with no floor; floored variants are separate arms.
8. **"Do not tune every configuration each time, collect all info and adjust after all sweep (fix only critical ones)."**
9. **Compare, don't fuse.** One mechanism per arm; combinations only as labelled follow-ups.
9a. **"Do not average for coupled, each equation has its own objective."** One gate per equation.
9b. **"Make sure that there is no bias in pde. It is only there if equation has it."** See §6.3a.
10. **Housekeeping rules:**
    - one fresh process per solve;
    - no timed CPU work beside a GPU batch;
    - no commits without the PI's OK, and no AI co-author trailers;
    - don't blanket-kill processes (target PIDs);
    - `projects/thesis` is retired; the solver study lives in `projects/pinn`.

---

## 3. How the solver-based fitness works

### 3.1 The solve (DeepXDE backend)
For a candidate (single equation or system):
1. Its coefficients come from EPDE's own fit on finite-difference derivatives (`weights_final`) and **stay fixed**.
2. The PINN minimises the physics residual `Σ c_k·term_k + b − target` on collocation points over the **whole** domain. It adds an observation term `u ≈ u_obs` on the inner grid points of the **first `train_frac` = 0.8 of time levels**. There are no IC/BC terms.
3. The fitness is the RMSE of the PINN's `u` against the observed field on the **held-out last 20 % of time levels**.
   - `heldout.time_split` is the ONE rule for what is trained on and what is scored.
   - For a system, each equation is scored on its own variable.
4. Shipped recipe (`search_config._default_deepxde_config`):
   - net `[64]*4` tanh;
   - Adam `lr 1e-4` × 2000, then L-BFGS ≤ 2000;
   - 4000 collocation points;
   - loss weights `'gradient'` (1/‖∇L_i‖ at init, frozen);
   - `train_frac 0.8`, `val_frac 0`;
   - ≈ 120 s per solve on the RTX 2070 SUPER.

   ⚠ A bare `DeepXDEAdapter()` does **not** use these. Its fallbacks are the old recipe (`[50]*4`, lr 1e-3, 10000 epochs, no L-BFGS, 'variance' weights). Unknown `deepxde_config` keys are silently ignored.

### 3.2 Code map
Line numbers are as of commit `ede3c28`.

| What | Where |
|---|---|
| Fitness host | `epde/operators/common/fitness.py`: `SolverBasedFitness` (~L229), `apply` (~L333; backend dispatch ~L354), `_apply_deepxde` (~L454-549). The latter observes `samples.get((eq.main_var_to_explain,(1.0,)))`, solves per trajectory, masks the held-out block, and turns failures into `LOSS_NAN_VAL = 1e7`. |
| Discrepancy | `epde/operators/common/objectives.py`: `Discrepancy('deepxde')`, whose `_compute_deepxde` (~L512) is the held-out RMSE. `cancellation_parts` (~L52) is the ONE term-mass formula, shared by the scale-invariant objective and the `sinv_*` physics losses. |
| The PINN adapter | `epde/integrate/deepxde_integration.py`, `DeepXDEAdapter` (~L592). Config keys and validation are in `__init__` (~L593-830). Methods: `_equation_system_to_pde_func` (the residual), `loss_weights`/`_gradient_weights`, `_set_lbfgs_budget`, `_install_lbfgs_stall_guard` (exit record: gtol/maxiter/maxeval/converged/nonfinite/stop_training/no_step), `_solve_trajectory` (the sequence), `_apply_transforms`, `_apply_init`, `_data_fit`/`_shared_data_fit`/`prepare_shared_fit`, `_setup_sinv`, `_data_driven_lr`. |
| Helpers | `epde/integrate/heldout.py` (`time_split`, `variance_weight`, `DeepXDEConfigError`); `residual_terms.py` (solver-independent residual, `cancellation_ratio`); `data_profile.py` (train-window-only field profile: bounds, periodicity, spectral size); `lr_free.py` (DoG, Prodigy, Levenberg–Marquardt); `basis_integration.py` (the frozen basis backend). |
| Config | `epde/interface/search_config.py`: `SolverConfig` (`use_solver`, `solver_backend` ∈ autograd/deepxde/basis, `deepxde_config`, `basis_config`), `_default_deepxde_config` (a comment justifies every value), `_default_basis_config`. |
| Routing | `epde/optimizers/moeadd/strategy.py`: `deepxde`/`basis` → `Discrepancy('deepxde')`. |
| Turning it on | `EpdeSearch(use_solver=True, solver_backend='deepxde', multiobjective_mode=True, deepxde_config={...})`. Smallest example: `tests/system/solver_backends.py`, run as `PYTHONPATH=. python tests/system/solver_backends.py deepxde 2 cpu`. Full discovery example: `projects/pic/data/ac/ac.py` (slow: ~2 min per candidate solve). |

---

## 4. History (what has been done), with key numbers

- **Jun 2026: fitness hosts.** Five copy-pasted operators became `SolverFreeFitness` / `SolverBasedFitness` plus pluggable objective "fillers".
- **Aug 2026: standalone PINN testbed** (`projects/pic/data/<sys>/pinn_test_autoscale.py`, `pinn_common.py`). Findings:
  - A truth-free, observation-free loss is provably under-determined: the force-free collapse scores 0.
  - An **observation term** plus coefficients from the net's own field became the default; it beats fixed FD coefficients 3.9–46×.
  - **Evaluation protocol:** *integrate* the recovered equation from the split state beside a true-coefficient control. Scoring the net past the split measured nothing.
  - **Paired-seed statistics** (`pinn_common.contrast/arm_table/contrast_table/mde/seeds_needed`).
- **Aug 28: solver path migration.** Both backends were dead after the multisample refactor and have been repaired.
- **Sep 2: "the solver discrepancy was 100 % PINN loss".**
  - `pinn_loss_mult = 1e4` meant the ranking was the PINN's training loss, not its error; the error term was even anti-correlated with correctness.
  - Now `pinn_loss_mult = 0`, a relative data term, and normalised MOEA/D sectors.
- **Sep 14–18: DeepXDE backend rebuilt** (the §3.1 design).
  - The old path fed the *target* term (du/dt) to the solver as if it were u.
  - A residual bug dropped `u^3`'s power, so AC's 5u − 5u³ cancelled.
  - **Run-order artifact.** A solve's result depends on what its process solved before: −20 % on AC, −53 % on Duffing, and reseeding doesn't help. Hence **one fresh process per solve.**
  - **Loss weighting.** 'variance' and 'gradient' were compared; 'gradient' is shipped. "lr is not removable by gradient weighting: dimensional, not numerical."
  - **lr ladder.** AC per-seed spread across lr 0.494 (a pre-registered 0.10 would have been acceptable).
  - The first lr-free attempt (L-BFGS only, cold start) **failed Duffing**: −0.54 worst seed.
  - **val_frac must be 0.** A 10 % validation block gives AC/Duffing −0.141/−2.242; choosing its best iterate gives −0.212/−2.754.
  - **`lbfgs_gtol_rel`:** keep `None`.
  - **L-BFGS cap:**

    | cap | 1000 | 2000 | 3000 | 4000 |
    |---|---|---|---|---|
    | AC | −0.011 | +0.650 | +0.816 | +0.829 |
    | Duffing | −0.299 | +4.277 | +4.277 | +4.277 |

    The cap stays at 2000 for now.
  - The old shipped default *inverted* the AC gate (−0.131/−0.011/−0.096).
- **Sep 28: lr-free plan and gate harness.**
  - Baseline reproduces the Sep 18 references bitwise.
  - Burgers joins the gate: +4.342.
  - New modules and the basis backend.
- **Sep 28–29: Stage 1 sweep** (§5).
- **Sep 29: Stage 1b/1c launched** (§6).
  - The coupled systems were added to the harness.
  - A `translate_equation` bug was fixed: its dict overload gave every equation the LAST variable, so a solver fitted every observation term to it.
  - The Duffing float32 stall was confirmed (§6.2).

Commits on `domain_refactor` from Sep 29 (`e23de09` … `a72d351`) hold all of this code: the translator fix, the RPS scrub/ban, the PINN testbed, the solver backends, the harness in `projects/pinn`, and the per-equation gate for coupled systems.

---

## 5.0 Results, Oct 2026: Stage 2 and Stage 3 (the current answer)

Everything here is under the no-bias rule.
- Stage 2 analysis: `results/stage2_analysis.txt`.
- Stage 3 code: branch `pinn` (worktree `../EPDE_lorenz`); as-run tree: tag `pinn-stage3-asrun` (`0f59348` + the work). Results stay local in `../EPDE_lorenz/projects/pinn/results/` (`tmobs2_analysis.txt`, tags `lorenz_m1*`, `tmobs*`, `tmobs2*`).

### The routed recipe (worst-seed gate at T_B)

| Recipe | Duffing | AC | Burgers | LV (u) | Lorenz (v) | NS (u) | Verdict |
|---|---|---|---|---|---|---|---|
| **R-route-3k** | +5.80 | **+0.36** | +3.76 | +4.15 | +1.57 (6 seeds) | +0.91 | **eligible** (+0.360) |
| R-route (4000 it.) | +5.80 | +0.65 | over T_B | +4.28 | +2.39 | +0.91 | eligible at own budget (+0.647) only |
| B (shipped) | +3.48 | +0.62 | +4.34 | −0.14 | −0.97 | +0.16 | fails LV, Lorenz |
| S1 (G-0000) | +1.57 | +0.31 | +4.44 | −0.11 | −0.28 | +0.25 | fails LV, Lorenz |
| G-m000 | +4.99 | +0.59 | +4.74 | −0.27 | −0.99 | +0.91 | fails LV, Lorenz |

**Per seed (3k):**
- Duffing +5.80 / +5.81 / +5.92
- AC +0.48 / +1.06 / +0.36
- Burgers +4.36 / +4.37 / +3.76
- LV +4.35 / +4.15 / +4.24
- Lorenz +4.87 / +1.57 / +3.92 / +4.65 / +5.04 / +4.41

**Median solve time:** Duffing 11 s, LV 16 s, Lorenz 24 s, AC 50 s, Burgers 66 s, against B's 79–140 s. NS uses the global fit (T_B 385 s).

**Paired contrasts vs B:**
- Lorenz (v): TM-obs +4.82 (t 7.14, 6/6) and TM-obs-3k +4.18 (t 6.58, 6/6). Both resolved.
- LV: +4.38 (t 85) and +4.31 (t 48). Both resolved.
- Duffing: +1.58, unresolved (~4 seeds).
- AC and Burgers: about level.

**The 3000-iteration budget** brings Burgers to 61–70 s (0.9 T_B = 98 s). It costs AC margin (+0.65 → +0.36) and some Lorenz margin. The 3k − 4k contrast is unresolved on Duffing, AC, LV and Lorenz, and unreadable on Burgers, where the 4k solves ran past T_B.

### Stage 3: why Lorenz failed and what fixed it

Diagnosis and verification outputs are in the session scratchpad `lorenz/`.
- **Equilibrium trap.**
  - A Glorot net outputs about 0, which is an exact zero-residual equilibrium (the origin) of the true Lorenz form. S1's true tail sits there: v 7.65, against 7.648 for a constant 0.
  - With moment outputs, most tails land near C+, on the wrong lobe (v about 13–14).
  - B fits the train window (RMSE about 0.3) but still slides onto an equilibrium in the tail. The exact ODE from B's own split state gives v 4.06 against the PINN's 7.50.
  - Following the dynamics would LOWER the PINN loss (about 9 % for B, 24 % for G-m000), so this is an optimisation basin, as in Rohrhofer et al., TMLR 2023.
- **The oracle is lenient.**
  - Integrating from the exact split state gives gate_v +6.99 on this window and +5.1..+7.3 over 8 windows.
  - It stays positive under split-state errors up to 6 % of the attractor spread.
  - The "+3.8..+5.1" in gate.py is not reproducible.
- **Gradient weights are probed at that equilibrium.**
  - The eq_w weight ranges 0.008..11 across seeds.
  - On seed 0 the true v-equation's weight is smaller than the wrong form's (8.7× under B, 5.4× under S1).
- **Round 1** (Lorenz s0, own budget):

  | Arm | Result | Time | Verdict |
  |---|---|---|---|
  | TM-obs | true v 0.028, wrong v 13.08 (= oracle 13.09); gate_v +6.16 | 44 s | the only arm that propagated |
  | TM (full-window march) | gate −0.505; train RMSE ~11–12 | | killed |
  | DE (growing domain) | stays at the origin | | killed |
  | CW (causal weights) | flat tail, gate −0.600 | | killed |

  None of S1, TM, DE or CW fits the 4-unit train window lr-free (train RMSE u/v about 7–12, w 12–22). TM-obs takes the split state from the last observed level instead.
- **NS needs the global fit.**
  - TM-obs alone gives ns:u −2.20, a resolved loss vs B (t −19).
  - The candidate's p-equation fixes only p_y, so in the tail p is free up to h(t, x). 99.9–100 % of the true form's p error is that y-independent part.
  - Its x-gradient drives u through −p_x: corr(du_err/dt, −h_x) = 0.98 / 0.87 / 0.98. The u error grows linearly (0.02 → 0.65).
  - The wrong form has no −p_x and does not drift (u 0.04).
  - So an open domain's tail is not an initial-value problem.

### Stage 2 (24-cell grid on S1 + loss arms; no arm eligible)

- **Three cells pass 4 systems** (G-mpg0, G-m0g0, G-mp0d): two independent solve sets on LV and Duffing, AC margins +0.58..+0.86 at T_B (G-m0g0 dips to +0.48 at 0.9 T_B). All three fail Lorenz.
- **On the global fit, LV propagates** with moment outputs (12/12 solves at own budget, but seed 2 can arrive after T_B) or with L-ps (3/3). Without either, only seed 2 propagates. Tail marching (TM-obs) needs neither.
- **Resolved factor effects (3 seeds):**
  - moments win on NS (+0.465, t 4.59);
  - gradient collocation loses on Burgers (−0.250);
  - collocation re-placement alone moves AC (−0.167).

  No float64 or periodic-embedding main effect resolves; the gradient-warp × float64 interaction loses on Lorenz (−0.083, t −7.74).
- **Scale-invariant observation losses collapse on sign-changing fields.** AC L-od converges in 5 s at RMSE 0.856.
- **The float32 L-BFGS partial-fit stall also hits LV:** four float32 solves stop at u RMSE 0.10–0.14 where their float64 twins reach 0.015–0.036. Float64 fixes it but always runs to the 4000 cap.
- **Wall-clock reads are fragile.**
  - AC S1 repeated Stage 1 iterate for iterate but ran 49 % slower on one seed, which moved its T_B gate +0.784 → +0.306.
  - Burgers' B-ctl took 144 s one day and 113 s the next.
- **The bar needs a collapse guard.** L-od "passes" lorenz:v (+0.011) collapse-vs-collapse.

### Recalculation (no-bias rule) of Stages 0/1/1b

Only B (+0.615) and S1 (+0.497) are eligible on AC, Duffing and Burgers. Every other arm failed:
- D2 holds the bar by a hair (AC +0.010 / +0.006 at T_B / 0.9 T_B) but is flagged unreadable: it adds GPU work before training, and the S3-null check moved;
- D1, S10 and P2 fail Duffing; P1 fails AC (−0.918);
- S7f and S8f fail;
- S5-32 (LM) runs past T_B everywhere; at completion it beats L-BFGS on AC and Duffing and loses on Burgers.

`results/recalc_analysis.txt`.

---

## 5. Results so far (Stage 1, old forced-constant rule; superseded)

Full report: `results/stage1_analysis.txt`. Worst-seed gates are shown as **own budget (at T_B)**. "Own budget" is the arm's own iteration cap. T_B is the wall-clock of a shipped solve on that system: 124 / 117 / 109 s. The primary rule reads each arm at T_B.

| Arm | AC | Duffing | Burgers | Notes |
|---|---|---|---|---|
| **B** shipped (Adam 1e-4 + L-BFGS) | +0.650 | +4.277 | +4.342 | the reference |
| **S1** L-BFGS only + affine inputs | +0.734 (**+0.611**) | +4.576 (+4.576) | +4.558 (+4.515) | passes every seed at T_B; below B per seed on AC; uncapped AC solves ~175 s |
| Basis, 3–4× data Nyquist (frozen) | +0.84 | +5.74 | +4.29 | CPU, float64, deterministic |
| S0 cold L-BFGS only | +0.649 (+0.281) | **−1.807** (−1.807) | +4.528 (+4.461) | Duffing fails (raw t ∈ [0, 20] into tanh) |
| S2 output moments | +0.666 (+0.583) | +0.913 (**−0.845**) | +4.756 (+4.726) | |
| S3 per-solve data prefit | +0.701 (**−0.015**) | +1.205 (**−1.245**) | +4.388 (+4.364) | the prefit's time kills it at T_B |
| S4 lstsq last layer | nan | nan | +0.113 | diverges exactly on u³ equations |
| S6 float64 | +0.749 (+0.607) | **−1.121** (−0.483) | +4.379 (+4.308) | float64 alone doesn't fix Duffing |
| S7 sinv_live (no floor) | +0.003 | +0.057 | +0.004 | near-zero everywhere |
| S8 sinv_data (no floor) | **−0.448** (−0.949) | **−0.344** (−0.960) | **−0.718** (−0.546) | 94–100 % of its weight on \|u\| < 1e-3 points |
| S9 shared data fit | +0.590 (+0.362) | **−0.532** (−0.532) | +4.266 (+4.246) | |
| B9 shared fit + Adam recipe | +0.271 (+0.271) | **−0.676** (−1.301) | +4.140 (+4.148) | |

- **Pre-registered paired contrasts** (3 seeds, t_crit 4.30):
  - S1−S0 **wins** on Duffing (+6.09, t = 7.5).
  - S7−S0 **loses** on AC and Burgers and **wins** on Duffing.
  - S9−S0 **wins** Duffing only.
  - B9−B **loses** on AC.
  - Most other contrasts are unresolved; `mde` and `seeds_needed` are printed for them.
- **Noise:**
  - Null arms that should equal S0 bitwise don't on AC and Duffing (GPU process state). On Duffing a null change moves per-seed gates by ~1.2 nats, so arms that add GPU work before training are flagged UNREADABLE.
  - Controls (B-ctl at the start and end of each batch) have always matched bitwise.
- **Artifacts:** BL-n2 AC +4.16 and BL-sinv-live AC +4.73 are ringing (an under-resolved wrong form), not real wins. The mechanical ranking rule has no artifact check, so it put BL-n2 first.
- **Timing contamination is real.** B-4k beside a CPU sweep ran 289 s instead of ~200 s. Even ONE single-threaded CPU job slows a GPU solve ~20 % (measured Sep 29).

---

## 6. Stage details (all batches finished), and next

### 6.1 Stage 1b/1c (old rule; recalculated under the no-bias rule, see §5.0)
Script: `run_stage1b_gpu.sh`, tags `stage1b_start`, `stage1b`, `stage1b_end`, `stage1b_basis`.

Arms, per system:

| Arm | Mechanism |
|---|---|
| S1-cap | S1 with L-BFGS ≤ 2000, a real solve at the shipped time |
| S10 | S1 + shared data fit |
| S7f / S8f | scale-invariant losses with a 1e-3 floor |
| P1 / P2 | DoG / Prodigy replace Adam, lr = None |
| D1 / D2 | lr from the data: prefit distance / the grid lr with the lowest data-only loss |
| S0-32 / S5-32 | L-BFGS vs Levenberg–Marquardt on a [32]×4 net |

The CPU basis cells run last. Pre-registered contrasts: S1-cap−B, S10−S1, S7f−S0, S8f−S0, P1−B, P2−B, D1−B, D2−B, S5-32−S0-32.

AC so far (my arithmetic from the rows, not yet an analyzed table):

| Arm | Seed gates | Worst |
|---|---|---|
| S1-cap | +0.530 / +0.765 / +0.542 | **+0.530** (B +0.650) |
| S10 | +0.803 / +0.733 / +0.642 | +0.642 |
| S7f | −0.038 / +0.329 / +0.001 | |

S7f seed 2 collapsed: both forms "converged" in ~100 iterations at RMSE 0.855.

- **Rows with contaminated timings** (their RMSEs are fine) in `stage1b.jsonl`:
  - lines 3–18 overlapped a CPU diagnostic from 09:28 to 10:18;
  - lines 21–22 overlapped the unit suite from 10:27 to 10:29.

  They will be rerun with `--force` and listed in `analyze.py`'s `CONTAMINATED`.

### 6.2 The Duffing float32 stall (confirmed Sep 29)
The shared data fits for Duffing stop early. Each fit is an observation-only L-BFGS fit, the start used by S9/S10. The rerun `duffing_fit_stop.py` read the optimizer's own state at the stop:

| Fit | Stop | Iterations | Data loss | max\|grad\| |
|---|---|---|---|---|
| float32 seeds 0/1/2 | **zero line-search step** | 1238 / 1478 / 2467 | 9.9e-5 / 1.7e-4 / 8.2e-5 | 1e-2 … 1e-1 |
| float64 seeds 0/1 | eval budget (25k) | 13945 / 15845 | **9.0e-7 / 1.8e-5** | 3e-4 / 1.5e-3 |

- In float32 the strong-Wolfe line search stops finding a decrease it can confirm. The gradient is still 10⁶× above tolerance and the loss is still dropping.
- Float64 keeps going, to a 10–100× lower loss, at ~15 min per fit on one CPU thread.
- DeepXDE's own docs name float64 as the fix for early L-BFGS stops.
- **Next:** an opt-in `shared_fit_precision='float64'` (patch written), float64 shared fits, and S9/S10 rerun on them.

### 6.3 Stage 2 grid (on the winning route, S1)
- **24 cells** `G-<o><p><c><x>` on S1 (epochs 0, L-BFGS ≤ 4000, affine inputs); `0` means S1's default:

  | Factor | Code | Level |
  |---|---|---|
  | output transform | `m` | `train_moments`; **S1+S2 = `G-m000`**, added by the PI |
  | periodic embedding | `p` | `auto`: (cos, sin) of the axes the data profile finds periodic |
  | collocation | `g` | data-gradient density: sequential inverse CDF, any number of axes, NS included |
  | collocation | `f` | flat warp, the placement null (role null, never ranked) |
  | precision | `d` | float64 |

- **Inert cells run seed 0 only.**
  - Which cells:
    - periodic on systems with no periodic axis (Duffing, LV, Lorenz);
    - flat in float32, which is the identity after the cast.
  - `analyze.py` checks seed 0 bitwise against the reference cell and reuses the reference's other seeds; a mismatch marks the cell UNREADABLE (rerun with `drive --no-alias`).
  - NS runs no periodic cell, since its open box is not pre-registered as periodic.
- **Pre-registered contrasts:** each factor alone vs `G-0000`, plus moments × float64 and gradient warp × float64.
- **Order:** the full grid runs on Duffing, AC, Burgers, LV and Lorenz; NS then gets a reduced set (B, `G-0000`, `G-m000`).
- **Cost:** about 46 GPU hours in all (~34 h grid, ~9 h NS, ~3 h controls).
- **Script:** `projects/pinn/run_stage2_gpu.sh`. It starts with the periodicity check (`gate.py profile`), the candidate printout (`gate.py forms`) and coupled-system smoke solves checked by `gate.py check`.
- The adapter patches (periodic embedding, n-D collocation warp, float64 shared fit, no-bias rule, host order guard) are written, tested in a separate worktree (176 solver tests pass), and applied after the Stage-1b batch.

### 6.3b Scale-invariant data loss (PI, Sep 29: "lets try sinv for data as well")
- **The data loss.** The observation equation u = u_obs has two terms, so its cancellation ratio is ρ_obs = (u − u_obs) / (|u| + |u_obs|), in [0, 1]:
  - `obs_loss='sinv_live'`;
  - `'sinv_data'` divides by the frozen 2|u_obs| and is unbounded where the data cross zero.

  This is the same pointwise formula as the scale-invariant objective and the physics `sinv_*` losses.
- **Aggregation stays L2** (DeepXDE's mean ρ²); the objective is mean |ρ|.
  - The property that matters holds in both: the same relation scores identically whichever term is the target (checked exactly on AC and LV, Sep 29).
  - Swapping the target changes the discrepancy only through the least-squares REFIT: AC with u_xx as target is ×1.51 in L1 and ×2.61 in L2; every other swap moves it by under 1%.
- **Arms on S1**, each against G-0000: `L-os` (data live), `L-od` (data mass), `L-ps` (physics live), `L-pos` (both live). They run on the five systems right after B in Stage 2 (~6 GPU h).
- **When they go in.** The recalculation's tags pin the code state, so the option is applied between the recalculation and Stage 2. A watcher (`stage2_boundary_v2.sh`) applies the patches and runs the full unit suite; Stage 2 waits on `results/STAGE2_HOLD` until `results/STAGE2_GO` exists (`STAGE2_BLOCKED` records a failure).
- **The objective is unchanged.** An exact 0/0 does not occur in these data.

### 6.3a No bias in the PDE unless the equation has one (PI, Sep 29)
- **The rule.** A candidate's PDE carries a free coefficient only when the equation has one: its support slot `weights_internal[-1]` is non-zero. EPDE's coefficient step already follows this rule (it refits an intercept only then, and leaves `weights_final[-1] = 0` otherwise).
- **Where the bias came from.** The gate harness forced a fitted intercept into every true and wrong form (Stage 0/1/1b rows). That is where LV's wrong form got its +11.6 "recruitment".
- **What changed:**
  - Stage 2 refits a form with a free term only when the form writes a non-zero one (`+ 0.0` = none).
  - The adapter adds the free coefficient to the residual only when the equation has one.
  - Rows record `intercept_rule` (`forced` for the old rows, `from_form` from Stage 2) and `free_term` per equation.
  - `analyze.py` refuses to mix the two rules: read Stage 2 with `--tags stage2_start,stage2,stage2_end`.
  - `--legacy-intercept` reproduces the old rule only for the bitwise patch-neutrality control (tag `stage2_legacy`, read with `--tags baseline,stage2_legacy`).

### 6.4 Coupled systems
They are in `projects/pinn/gate.py` and ran in Stage 2 and Stage 3; results are in §5.0.

| System | Data | Setup | Wrong form (why) |
|---|---|---|---|
| `lv` | `pic/data/lv/{t_20,data_20}.npy`, synthetic, α = β = γ = δ = 20 | all 301 levels, boundary 10 | drop prey growth: the refit gives constant recruitment, a bounded damped spiral, the *lowest* oracle gate (~4.4) of the 4 single drops. Dropping predator mortality blows up in the tail. |
| `ns` | `pic/data/ns/cylinder_nektar_wake.mat`, Raissi, Re = 100 | first 50 levels, every 2nd y and x (50×20×36), boundary (2, 2, 7). Equations: x-momentum (target u_t), continuity (v_y), y-momentum solved for p_y | drop −p_x from x-momentum (static residual 10× the truth). ⚠ Only ∇p enters the physics, so the p-equation's objective (RMSE of p) carries a free gauge drift in the tail. |
| `lorenz` | `pic/data/lorenz/{t,lorenz}.npy` | window t ∈ [20, 25.2], every 5th sample, re-zeroed. Tail = 0.92 Lyapunov times (λ₁ = 0.900) | drop −x·z (the y-equation's only nonlinearity). Oracle gate +3.8 … +5.1 over 8 windows. |

**Gates for coupled systems are per equation** (PI, Sep 29: "do not average for coupled, each equation has its own objective").
- `analyze.py` turns a coupled system into one gate per equation, labelled `lv:u`, `lv:v`, and so on (`--systems ac,lv`).
- The bar applies to the equations whose text differs between the true and the wrong form: `lv:u`, `ns:u` and `lorenz:v`.
- The other equations are the same equation in both candidates. Their gates are reported as **coupling only**: how much the wrong equation corrupts a correct one through the joint solve. They are not part of the bar or the ranking.
- The same holds in EPDE itself: `SolverBasedFitness` gives each equation of a system its own fitness (the RMSE of its variable); the host's averaging path is unreachable.

Warnings:
- The LV data come from a hand-written RK4 with two stage-3 bugs, so the record is LV only to ~0.4 %; the refit absorbs it.
- **NS is expensive:** measured about 6.5 min per shipped solve (B median 385 s; S1 / G-m000 about 480–490 s; TM-obs 273 s), ~2–3 GB of GPU memory. Two known costs:
  - the adapter differentiates a fresh output slice per factor, which misses DeepXDE's Jacobian cache (18 vs ~7 autograd passes);
  - there is one observation BC per variable at identical points.
- A **PINN that fails to propagate** gives a TRUE-form RMSE near the persistence or mean baseline (an equilibrium or persistence level: LV ≈ 2.3 / 2.4, Lorenz ≈ 7.5–7.7 at the origin or ≈ 13–14 near C+ with moment outputs). Read that as "solve failed", not as a verdict. Report RMSE / tail std beside the gate.

### 6.5 Next (Oct 7)
*Superseded by Part I (Nov 12). Kept as the state at the end of the first study.*

- **Done Nov 11:** the Stage-3 work is committed on branch `pinn` off `domain_refactor` (`9b42f48` adapter + `time_marching.py` + 129 tests; `1da0d71` harness arms, routed analysis, run scripts). The stored `tmobs2_analysis.txt` reproduces byte for byte on it. Not pushed, not merged.
- **Port R-route into `SolverBasedFitness`**, routing at solve time from `data_profile`. Merge around the other uncommitted `epde/` work.
- **Budget:** per-system budget, or an iteration-matched bar, to win back AC's margin.
- **Robustness:**
  - a collapse guard for the bar;
  - a smoothed split state for noisy and real data.
- **More seeds:** Duffing and AC; find out why Lorenz seed 1 is the weak one.
- **Phase 5:** end-to-end discovery with the adopted route against the shipped one. Metrics: found_true, Hamming distance, solve wall-clock, rerun reproducibility.

---

## 7. How to run the gate

Code lives in **`projects/pinn/`**.

**The Sep 29 queue (started 13:46, finished): `run_queue.sh`.**
- Results live in `projects/pinn/results/`. The old `projects/pic/data/solver_gate/` is a retired copy.
- Order:
  1. patch neutrality under the old rule (`gate.py neutrality --tag stage2_legacy`; stops the queue on failure);
  2. the recalculation of Stages 0/1/1b under the no-bias rule (`run_recalc_gpu.sh`, tags `r_*`);
  3. the Stage-2 grid (`run_stage2_gpu.sh`);
  4. the basis cells on CPU (`run_recalc_basis.sh`).
- Logs: `results/neutrality.log`, `recalc_gpu.log`, `stage2.log`, `recalc_basis.log`.
- **Reads:**
  ```bash
  python analyze.py --tags r_baseline,r_start,r_stage1,r_stage1b,r_end        # recalculated Stages 0/1/1b
  python analyze.py --tags stage2_start,stage2,stage2_end --systems ac,duffing,burgers,lv,lorenz
  python analyze.py --tags stage2,stage2_end --systems ns                      # NS on its own
  ```
- `drive` refuses a tag whose rows come from another code state or intercept rule. It exits non-zero when a solve writes no row.

```bash
cd projects/pinn
python gate.py list                                   # the arm registry
python gate.py solve  --system ac --arm B --form true --seed 0 [--mode trace]
python gate.py drive  --system ac --arms B,B-ctl --seeds 0-2 [--mode trace] --tag mytag
python gate.py prefit --system duffing --arm S10 --seeds 0-2   # before any S9/B9/S10 drive
python gate.py probe  --system duffing --arm B --forms true,wrong,true   # run-order diagnostic
python analyze.py [--tags baseline,stage1,...] [--systems ac,duffing,burgers]
```

- **Systems:** `ac`, `duffing`, `burgers`, `lv`, `ns`, `lorenz`. **Forms:** `true`, `wrong`.
- **drive:**
  - appends rows to `results/<tag>.jsonl`;
  - is **resumable** (it skips existing `(system, arm, form, seed, mode)`; `--force` reruns);
  - runs control arms last;
  - runs basis and null arms on seed 0 only.
- **Modes:**
  - `solve` has minimal instrumentation.
  - `trace` records the held-out RMSE at every L-BFGS iterate (a read-only wrap of `torch.optim.lbfgs._strong_wolfe`). One trace at cap C gives the gate at every cap ≤ C, validated bit-exact on 60 real solves. `analyze.py` replays traces at caps 1000–4000 and at wall-clock budgets T_B and 0.9 T_B.
- **Row fields:**
  - identity: tag, system, arm, form, seed, mode, the full merged config;
  - `rmse`: a float for one variable. For a coupled system it is `{var: RMSE}`, one value per equation, **never averaged**. Each equation is its own objective in EPDE's search, scored on its own variable (see §6.4);
  - timing: `seconds`, `prelude_seconds`, `phases`;
  - solver records: `lbfgs_exit`, `init_stats`, `sinv_stats`, `lr_stats`, `lm_stats`, `pairing` (θ₀ digest), `loss_weights_lbfgs`, `coeffs`, `equation`;
  - `env` (threads, torch, GPU, deepxde);
  - `provenance` (HEAD, epde diff hash, **sha256 of the decisive files**: the real code identity).
- **Trace files:** `results/traces/<tag>/<system>__<arm>__<form>__<seed>.json`. Row `rows[i]` = `[n, heldout_rmse, gmax_before, loss_after, gmax_after, step_t, evals, wall, instr]`.
- **Run scripts:** `run_stage0.sh`, `run_stage1_gpu.sh`, `run_stage1_basis.sh` and `run_stage1b_gpu.sh` record how each tag was produced. Run them with bash.
  - ⚠ On a fresh Windows checkout with `core.autocrlf=true` they may get CRLF endings; convert them back to LF before running.
- **Winner rule** (`analyze.py`):
  - eligible = finite, and gate > 0 on every seed of every system at T_B *and* 0.9 T_B;
  - score = the min gate at T_B;
  - arms within 0.10 nats tie.

---

## 8. Rules of the bench (hard-won)

1. **One fresh process per solve.** Never compare solves that shared a process.
2. **Controls.** B-ctl at the start and end of each batch must equal B bitwise. Null arms must equal their control bitwise. Pairing digests of θ₀ are checked, not assumed.
3. **Nothing CPU-heavy beside a GPU batch.**
   - Even a 1-thread job costs ~20 %.
   - If it happens, record which rows overlapped (rows have no timestamps: note `wc -l results/*.jsonl` before and after), flag them `CONTAMINATED`, and rerun them.
   - The machine is shared, so **ask before starting GPU work** while a batch is running.
4. **Pin threads** for anything compared bitwise: `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, plus `PYTHONHASHSEED=0` for the EPDE search itself.
5. **Train window only.** Anything computed from data (profiles, moments, shared fits, lr rules) must be bit-identical when the tail is replaced by NaN. There are tests for this.
6. **Always score the last iterate.** Validation-based selection is anti-discriminative.
7. **Statistics.** Report unaggregated per-seed rows plus pre-specified paired contrasts. An unresolved contrast reports `mde` and `seeds_needed`, never "no difference".
8. **One mechanism per arm.** Adjust configs only after a full sweep, except for critical bugs.
9. **Don't read a code change into a running batch.** Children import the repo fresh, so editing `epde/` mid-batch changes later rows. Never `git checkout/stash/reset` mid-batch: with autocrlf it rewrites files.
10. **Stopping a batch.**
    - Stop the driving shell, then the specific `gate.py` PIDs. Killing only the driver lets the shell loop continue.
    - Children whose console was killed die with rc 0xC0000142.
11. **Scripts:** write multi-line scripts to files and run them (no Bash heredocs; they break on quoting here).
12. **Commits.** Explicit paths only (never `git add -A`: results and traces are large). Imperative subject, prose body with numbers, no trailers, and only with the PI's OK.

---

## 9. Where to look

| Path | What |
|---|---|
| `projects/pinn/gate.py`, `analyze.py`, `run_*.sh` | the gate harness (tracked) |
| `projects/pinn/results/` | `baseline.jsonl`, `stage1_start/stage1/stage1_end.jsonl` (network), `stage1_basis.jsonl` (basis), `stage1b*.jsonl`, `shared_fits.jsonl` + `shared_fits/*.pt`, `traces/<tag>/`, `stage1_analysis.txt` (the Stage-1 report), the Stage-2 and Stage-3 rows and analyses (§5.0), `*.log` (the shell output of each batch). Untracked: the files exist only on the machine that ran them. |
| `epde/integrate/*.py` | adapter and helpers (§3.2) |
| `tests/unit/test_solver_path.py` | adapter behaviour: time split, weights, L-BFGS budget, stall guard, exit record, device, validation checkpoint, a real fixed-coefficient solve |
| `tests/unit/test_lr_free_solve.py` | **bitwise pins** of the shipped solve under 2nd-order differences; five fail since the default changed, see Part I, section H (`'adam+lbfgs': 6f9cc574…`, `'lbfgs-only': 361e35ab…`, S1 `c28c0dc3…`), null checks, option validation, shared-fit cache/tail/RNG, DoG/Prodigy/LM mechanics |
| `tests/unit/test_basis_backend.py`, `test_data_profile.py` | the basis backend and the data profile |
| `projects/pic/data/pinn_common.py`, `stat_common.py`, `<sys>/pinn_test_autoscale.py`, `dp/cv_metric.py` | the standalone PINN testbed. Configure by editing module constants, run from the script's own folder. |
| `projects/pic/data/<system>/` | all datasets |
| `projects/thesis/` | **retired**. The solver-free thesis workspace, never solver-related. Its code is on branch `main` under `projects/thesis/`; the useful facts are in §12. |

---

## 10. Glossary of arms

`_S0` = {epochs 0, lbfgs_maxiter 4000, pairing_digest}.

| Arm | Meaning |
|---|---|
| **B** / B-ctl / B-4k | the shipped recipe / its bitwise duplicate / the same with the L-BFGS cap at 4000 |
| **S0** | no Adam: L-BFGS only from the Glorot init; the control of every S arm |
| S0-nolr, S3-null | null checks (lr = None; prefit with 0 iterations); must equal S0 |
| **S1** | S0 + `input_transform='affine'` (grid bounds → [-1, 1]) |
| S1-cap | S1 with L-BFGS ≤ 2000 |
| S2 | S0 + `output_transform='train_moments'` (u = μ + σ·net) |
| S3 | S0 + `init='data_prefit'` (observation-only L-BFGS per solve) |
| S4 | S0 + `init='lstsq_last_layer'` |
| S6 | S0 + `precision='float64'` |
| S7 / S8 | S0 + `pde_loss='sinv_live'` / `'sinv_data'` (residual ÷ term mass, net's own / data's) |
| S7f / S8f | the same with `sinv_floor_rel=1e-3` |
| S9 / B9 | shared data fit as the start, under S0 / under the shipped recipe |
| S10 | S1 + shared data fit |
| P1 / P2 | DoG / Prodigy instead of Adam (`first_order`), lr = None |
| D1 / D2 | Adam with lr from `lr_rule='prefit_distance'` / `'data_selected'` |
| S0-32 / S5-32 | L-BFGS / Levenberg–Marquardt (`second_order='lm'`) on a [32]×4 net |
| BL, BL-* | fixed-basis backend cells (frozen): resolution `n1`–`n4`, `r5`, data start, no preconditioning, `t4`, `c3`, sinv losses, `pred` (prediction-convergence refinement), floored sinv |
| `_smoke` | harness mechanics only |

---

## 11. Environment

- Python 3.14.2, conda env `EPDE_3_14` (`C:/Users/NSSLab/anaconda3/envs/EPDE_3_14`).
- torch 2.10.0+cu128, deepxde 1.15.0, numpy 2.5.1, scipy 1.18.0.
- GPU: RTX 2070 SUPER (8 GB). CPU: i7-9700 (8 cores).
- `import deepxde` moves torch's default device to CUDA process-wide; the adapter contains it. Never import deepxde in CPU-only checks.
- **Unit suite:**
  ```
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/unit -q
  ```
  Sep 29: 983 passed, 1 skipped (the CUDA-fallback test, skipped because CUDA is present), ~107 s. Nov 11, on `pinn`: 2548 passed, 1 skipped, 5 failed (Part I, section H).

---

## 12. System catalog and future gate systems (salvaged from the retired thesis)

The 14 benchmark systems (true equations, `dx0 = t`; data under `projects/pic/data/`):

| System | True equation | Data |
|---|---|---|
| ac | u_t = 1e-4 u_xx + 5u − 5u³ | `ac/ac_data.npy` (51×128) |
| burgers_inviscid | u_t = −u u_x | `burgers/burgers_sln_100.csv` |
| burgers_viscous | u_t = −u u_x + 0.1 u_xx | `burgers/burgers.mat` |
| kdv | u_t = −6 u u_x − u_xxx | `kdv/kdv_sindy.mat` |
| kdv_cossin | kdv + cos(t)sin(x) | `kdv/data.csv` |
| ks | u_t = −u u_x − u_xx − u_xxxx | `ks/kuramoto_sivishinky.mat` |
| lorenz | σ = 10, ρ = 28, β = 8/3 | `lorenz/lorenz.npy`, `t.npy` |
| lv | u' = 20u − 20uv; v' = 20uv − 20v | `lv/data_20.npy`, `t_20.npy` |
| ns | cylinder wake, Re = 100, (u, v, p)(t, y, x) | `ns/cylinder_nektar_wake.mat` |
| ode | u'' = −4u − sin(2t)u' + 1.5t | `ode/ode_data.npy` |
| pde_compound | u_t = u_x² + u u_xx | `pde_compound/PDE_compound.npy` |
| pde_divide | x u_t = −2u_x + 0.5 x u_xx | `pde_divide/PDE_divide.npy` |
| vdp | u'' = 0.2(1 − u²)u' − u | `vdp/vdp_data.npy` |
| wave | u_tt = 0.04 u_xx | `wave/wave_sln_80.csv` |

- **Equivalent true forms.** Some different-looking equations hold *exactly* on the data. A gate WRONG form must **not** be one of them, because it would solve as well as the truth:
  - **Lorenz w-equation:**
    - dw/dt = −8/3 w + 0.1 u·du/dt + u²;
    - also 0.5u² + 0.5v² − 0.005(du/dt)² ≡ uv.
  - **LV:** du/dt + dv/dt = 20(u − v), plus its 2nd-derivative companion.
  - **KdV:** three soliton identities.
  - **Inviscid Burgers:** u = x·u_x, and one more characteristic identity.
- **Why coupled systems matter.** Solver-free baseline (new pipeline, n = 30):

  | System | Successes |
  |---|---|
  | AC | 30/30 |
  | viscous Burgers | 27/30 |
  | LV | 19/30 |
  | **Lorenz** | **0/30** |
  | **NS** | **0/12** (18 runs crashed) |

  The solver-based fitness is meant to close that gap.
- **Window choices differ from the thesis:**
  - the thesis used Lorenz `t[:1000]`, an off-attractor transient with a 0.14 T_λ tail;
  - it used LV `[:150]`, whose tail would be ~0.07 time units;
  - it wrote NS continuity with target u_x.
- **Real-data candidates** (thesis configs/adapters on `main`):
  - `silverbox` (a real electronic Duffing, y'' = a y + b y³ + d y' + g u_in, R² 0.987; the cubic sits below the derivative-noise floor);
  - `pend_single` (θ'' = −64.8θ − 0.65θ', R² 0.996; centre θ on π);
  - `lv_real` (lynx–hare);
  - `jhtdb_plane` (a 2-D slice of 3-D NS, R² 0.991).

---

## 13. Open questions / good starter tasks

*Superseded by Part I. Items 2, 4, 5 and 6 are still open.*

1. **Coupled-system baselines** (done in Stages 2 and 3, §5.0).
   - B and S1 on `lv` and `lorenz` (cheap: ~2–4 min per solve).
   - An NS `_smoke` run plus one timed B solve before committing to NS arms.
   - Report RMSE / tail std beside every gate.
2. **NS cost.** Measure the two adapter-side savings (full-output Jacobian so DeepXDE's cache hits; one merged observation BC). Both change the baseline, so each is a labelled arm, not a silent change.
3. **Duffing stall follow-up** (superseded). The shared-fit arms were not continued after Stage 1b. Float64 is an arm of Part I, section E.
4. **Ranking-rule artifact check.** The pre-registered rule ranked a ringing artifact (BL-n2) first. Propose a check, e.g. the resolution-convergence of the wrong form.
5. **Live-search solver variance.** It is still unmeasured how much a candidate's fitness moves across seeds inside a real EPDE search (per-candidate seeding alone won't fix the process-state part).
6. **Equivalent-forms audit** of every gate's wrong form (§12).
7. **Phase 5** discovery driver (`run_discovery.py` is planned, not written). Now Part I, section G.

---

## 14. Known caveats and stale spots

- `REFERENCE_GATE` in `analyze.py` is declared but never read. `REFERENCE_RMSE` (Sep 18 values) is what the integrity check uses.
- The Stage-1 ranking counts basis artifacts as eligible (§5).
- Several "no floor" statements predate the floored Stage-1c arms. The unfloored arms are the default; the floored ones are follow-ups the PI chose.
- Every row in this record was produced with EPDE's 2nd-order finite differences (`np.gradient`). On AC's 128×51 grid that biases the diffusion coefficient ~36 %, which sets AC's true-form floor at ~1.5e-3. Since Oct 22 the default is 4th-order, so new solves get other candidate coefficients and cannot be compared with these rows (Part I, section H).
- Code comments in `epde/` still cite three retired thesis files; they will be repointed to git revisions.
- Trace paths written before the move use Windows backslashes. `projects/pinn/analyze.py` reads both forms; `projects/pinn/gate.py` now writes forward slashes.
