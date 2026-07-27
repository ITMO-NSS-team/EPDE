# Datasets — Section 4.5

Reference for the 14 systems used in the LEGACY vs NEW EPDE pipeline
comparison. For each system this doc lists the ground-truth equation(s),
state variables, data file, generator, adapter behaviour, and any
per-system config override that the pipeline picks up at runtime.

Most data is reused from the in-tree PIC SR benchmark
(`projects/pic/data/<system>/`); `burgers_viscous`, `kdv`, and `ks` use
the SINDy benchmark `.mat` files; `ns` uses a Nektar++ cylinder-wake
reference. All trajectories are clean by default — the runtime applies
additive Gaussian noise with magnitude `noise_level * 0.01 * std(data)`
when `noise_level > 0`. Default per-rep settings (from
`configs/defaults.yaml`): population_size=16, training_epochs=5, FD
preprocessor, `equation_terms_max_number=10`, `data_fun_pow=3`,
`deriv_fun_pow=2`, `TrigonometricTokens(freq ∈ [1.999, 2.000])`.

## Summary

| System | Class | Dim | # Eqs | Pop size | Data file | Truth alts |
|---|---|---|---|---|---|---|
| ac | PDE | 1+1D | 1 | 16 | `pic/data/ac/ac_data.npy` | — |
| burgers_inviscid | PDE | 1+1D | 1 | 16 | `pic/data/burgers/burgers_sln_100.csv` | 2 |
| burgers_viscous | PDE | 1+1D | 1 | 16 | `pic/data/burgers/burgers.mat` | — |
| kdv | PDE | 1+1D | 1 | 16 | `pic/data/kdv/kdv_sindy.mat` | 3 |
| kdv_cossin | PDE | 1+1D | 1 | 16 | `pic/data/kdv/data.csv` | — |
| ks | PDE | 1+1D | 1 | 16 | `pic/data/ks/kuramoto_sivishinky.mat` | — |
| lorenz | ODE | scalar t | 3 | **48** | `pic/data/lorenz/lorenz.npy` | — |
| lv | ODE | scalar t | 2 | **32** | `pic/data/lv/data_20.npy` | 1 (pair) |
| ns | PDE | 2+1D | 3 | **48** | `pic/data/ns/cylinder_nektar_wake.mat` | — |
| ode | ODE | scalar t | 1 | 16 | `pic/data/ode/ode_data.npy` | — |
| pde_compound | PDE | 1+1D | 1 | 16 | `pic/data/pde_compound/PDE_compound.npy` | — |
| pde_divide | PDE | 1+1D | 1 | 16 | `pic/data/pde_divide/PDE_divide.npy` | — |
| vdp | ODE | scalar t | 1 | 16 | `pic/data/vdp/vdp_data.npy` | — |
| wave | PDE | 1+1D | 1 | 16 | `pic/data/wave/wave_sln_80.csv` | — |

Axis convention: `dx0 = t`, `dx1 = x` (or `y` for 2+1D NS), `dx2 = x`
for NS. `power: N` is the factor exponent in the canonical-token form.

## ac — Allen-Cahn

- **Class.** 1+1D reaction-diffusion PDE.
- **State.** `u(t, x)`.
- **Truth.**
  ```
  0.0001 * d^2u/dx1^2 + -5.0 * u^3 + 5.0 * u = du/dx0
  ```
- **Data.** `pic/data/ac/ac_data.npy`, shape (51, 128), generator
  `pic/data/ac/ac.py` (finite-difference PDE solver).
- **Adapter.** Loads the full grid.

## burgers_inviscid — Inviscid Burgers

- **Class.** 1+1D PDE.
- **State.** `u(t, x)`.
- **Truth.**
  ```
  -1.0 * u * du/dx1 = du/dx0
  ```
- **Truth alternatives.** The dataset records `u(x, t) = x / (t + c)`
  trajectories, so two characteristic identities also hold and EPDE
  routinely converges to them (~50/90 Pareto-0 solutions across the
  30 reps):
  ```
  1.0 * u = x{dim:1} * du/dx1
  0.5 * x{dim:0} * u + -0.5 * x{dim:1} = du/dx1 * x{dim:1}
  ```
- **Data.** `pic/data/burgers/burgers_sln_100.csv`, shape (101, 101);
  generator `pic/data/burgers/burgers.py`. Adapter grids the data on
  `t ∈ [0, 1]`, `x ∈ [-1000, 0]`.

## burgers_viscous — Viscous Burgers (ν = 0.1)

- **Class.** 1+1D PDE.
- **State.** `u(t, x)`.
- **Truth.**
  ```
  -1.0 * u * du/dx1 + 0.1 * d^2u/dx1^2 = du/dx0
  ```
- **Data.** `pic/data/burgers/burgers.mat`, `usol` field shape
  (256, 101), reused from the SINDy benchmark.

## kdv — Korteweg-de Vries

- **Class.** 1+1D PDE.
- **State.** `u(t, x)`.
- **Truth.**
  ```
  -6.0 * du/dx1 * u + -1.0 * d^3u/dx1^3 = du/dx0
  ```
- **Truth alternatives.** The dataset records a soliton family, so
  three analytic identities are also valid:
  ```
  -1.0 * u^3 + 1.0 * (du/dx1)^2 = u * d^2u/dx1^2
  -3.0 * u^2 * du/dx1 + 1.0 * du/dx1 * d^2u/dx1^2 = d^3u/dx1^3 * u
  -0.333 * u * du/dx0 + -0.333 * du/dx1 * d^2u/dx1^2 = u^2 * du/dx1
  ```
  The third is the temporal companion: multiply the KdV equation by
  `u` and substitute the spatial identity to eliminate `u * d^3u/dx1^3`.
- **Data.** `pic/data/kdv/kdv_sindy.mat`, `usol` field shape (512, 201),
  SINDy benchmark.

## kdv_cossin — KdV with cos(t)sin(x) source

- **Class.** 1+1D PDE with explicit spatio-temporal source.
- **State.** `u(t, x)`.
- **Truth.**
  ```
  -6.0 * du/dx1 * u + -1.0 * d^3u/dx1^3 + 1.0 * cos(t)sin(x) = du/dx0
  ```
- **Data.** `pic/data/kdv/data.csv`, shape (81, 81), generator
  `pic/data/kdv/kdv.py` (`kdv_data` with the custom source).
- **Note.** The token `cos(t)sin(x)` is a single product token built by
  the kdv_cossin adapter — not the `TrigonometricTokens` from defaults.

## ks — Kuramoto-Sivashinsky

- **Class.** 1+1D PDE (4th-order spatial).
- **State.** `u(t, x)`.
- **Truth.**
  ```
  -1.0 * u * du/dx1 + -1.0 * d^2u/dx1^2 + -1.0 * d^4u/dx1^4 = du/dx0
  ```
- **Data.** `pic/data/ks/kuramoto_sivishinky.mat`, `uu` field shape
  (1024, 251), generator `pic/data/ks/ks.py`.

## lorenz — Lorenz system

- **Class.** Coupled 3D ODE (chaotic attractor; σ=10, ρ=28, β=8/3).
- **State.** `(u, v, w)` over `t`.
- **Truth.**
  ```
  10.0 * v + -10.0 * u            = du/dx0
  28.0 * u + -1.0 * u*w + -1.0 * v = dv/dx0
  1.0 * u*v + -2.667 * w          = dw/dx0
  ```
- **Data.** `pic/data/lorenz/lorenz.npy` (100000, 3), `t.npy` (100000,).
  The adapter slices the first **1000 samples** for tractability.
- **Override.** `population_size: 48` — the 3-equation truth space
  needs more weight vectors than the runner's auto-bump (32) to cover
  per-equation (discrepancy, complexity) trade-offs.

## lv — Lotka-Volterra

- **Class.** Coupled 2D ODE (predator-prey; α=β=γ=δ≈20).
- **State.** `(u, v)` over `t`.
- **Truth.**
  ```
  20.0 * u + -20.0 * u*v          = du/dx0
  20.0 * u*v + -20.0 * v          = dv/dx0
  ```
- **Truth alternative.** Adding the two ODEs cancels the `u*v` cross
  term, yielding a derivable pair that EPDE picks up in ~5/30 reps:
  ```
  -1.0 * dv/dx0 + 20.0 * u + -20.0 * v        = du/dx0
  20.0 * du/dx0 + -20.0 * dv/dx0 + -1.0 * d^2u/dx0^2 = d^2v/dx0^2
  ```
  Both equations hold on every LV trajectory (residuals 1.3% and 8.3%
  against the data; the 2nd-derivative form picks up FD noise) so
  the pair credits as an analytically-correct discovery.
- **Data.** `pic/data/lv/data_20.npy` (301, 2), `t_20.npy` (301,). The
  adapter slices the first **150 samples**. The `_20` suffix names
  the rate constants; least-squares fit to the trajectory yields
  19.86, 19.94, 19.94, 19.97 — rounded to the integer generator value
  of 20.
- **Override.** `population_size: 32`, pinned explicitly so it doesn't
  rely on the runner's auto-bump for 2-equation systems.

## ns — 2D incompressible Navier-Stokes (cylinder wake, Re=100)

- **Class.** Coupled 2+1D PDE (3 equations).
- **State.** `(u, v, p)(t, y, x)`. Axis convention: `dx0=t`, `dx1=y`,
  `dx2=x`. `1/Re = 0.01`.
- **Truth.**
  ```
  -1.0 * u*du/dx2 + -1.0 * v*du/dx1 + -1.0 * dp/dx2
    + 0.01 * d^2u/dx2^2 + 0.01 * d^2u/dx1^2 = du/dx0      [momentum-u]
  -1.0 * u*dv/dx2 + -1.0 * v*dv/dx1 + -1.0 * dp/dx1
    + 0.01 * d^2v/dx2^2 + 0.01 * d^2v/dx1^2 = dv/dx0      [momentum-v]
  -1.0 * dv/dx1 = du/dx2                                  [continuity]
  ```
  Continuity is rewritten in `du/dx2` form so EPDE has a per-equation
  target derivative for it.
- **Data.** `pic/data/ns/cylinder_nektar_wake.mat` (Nektar++ cylinder
  wake reference; `U_star` is (N=5000, 2 components, T=200)). The
  adapter loads **the first 50 snapshots** and reshapes to
  `(T, ny, nx)`.
- **Override.** `population_size: 48` — same rationale as lorenz.

## ode — Forced damped oscillator

- **Class.** Scalar ODE (2nd order, periodic forcing + linear drive).
- **State.** `u(t)`.
- **Truth.**
  ```
  -4.0 * u + -1.0 * du/dx0 * sin{freq:2, dim:0} + 1.5 * x_0{dim:0}
    = d^2u/dx0^2
  ```
  i.e. `u'' + sin(2t)·u' + 4u = 1.5·t`.
- **Data.** `pic/data/ode/ode_data.npy`, shape (320,). Adapter
  reconstructs the time grid `t = arange(0, 16, 0.05)`.
- **Note.** The `sin{freq:2}` factor relies on `TrigonometricTokens`
  with the narrow `freq ∈ [1.999..., 2.000...]` band declared in
  `defaults.yaml`. The narrow window prevents the search from
  drifting onto spurious nearby frequencies.

## pde_compound — Synthetic compound PDE

- **Class.** 1+1D synthetic PDE with quadratic derivative term.
- **State.** `u(t, x)`.
- **Truth.**
  ```
  1.0 * (du/dx1)^2 + 1.0 * d^2u/dx1^2 * u = du/dx0
  ```
- **Data.** `pic/data/pde_compound/PDE_compound.npy`, shape (251, 100),
  generator `pic/data/pde_compound/pde_compound.py`.

## pde_divide — Synthetic rational PDE

- **Class.** 1+1D synthetic PDE with explicit spatial coefficient.
- **State.** `u(t, x)`.
- **Truth.** Both sides multiplied by `x` to keep all terms polynomial:
  ```
  -2.0 * du/dx1 + 0.5 * d^2u/dx1^2 * x_1{dim:1} = du/dx0 * x_1{dim:1}
  ```
- **Data.** `pic/data/pde_divide/PDE_divide.npy`, shape (251, 100).
  Adapter grids on `t ∈ [0, 0.5]`, `x ∈ [1, 2]`. LS fit confirms
  generator coefficients `(-2.000, 0.500)` with residual / RMS < 0.1%
  — the YAML uses the rounded integer/rational values.

## vdp — Van der Pol oscillator

- **Class.** Scalar ODE (limit-cycle, weak nonlinearity μ=0.2).
- **State.** `u(t)`.
- **Truth.**
  ```
  -0.2 * u^2 * du/dx0 + 0.2 * du/dx0 + -1.0 * u = d^2u/dx0^2
  ```
  i.e. `u'' + 0.2·(u² - 1)·u' + u = 0`.
- **Data.** `pic/data/vdp/vdp_data.npy`, shape (320,). Adapter
  reconstructs the time grid as for `ode`.

## wave — 1+1D wave equation

- **Class.** 1+1D linear PDE (wave speed c = 0.2, c² = 0.04).
- **State.** `u(t, x)`.
- **Truth.**
  ```
  0.04 * d^2u/dx1^2 = d^2u/dx0^2
  ```
- **Data.** `pic/data/wave/wave_sln_80.csv`, shape (81, 81), generator
  `pic/data/wave/wave.py`.

## Notes on canonical-token notation

- `{power: N}` is the per-factor exponent. `1.0` is the default; the
  YAMLs spell it out so canonical matching is unambiguous when EPDE
  rediscovers a term with an integer-cast power.
- `{dim: D}` selects which grid axis a grid token is bound to:
  `dim:0 = t`, `dim:1 = x` (1+1D PDE) or `dim:1 = y, dim:2 = x` (NS).
- `x_0`, `x_1` are grid tokens registered via the `grid_tokens:
  max_power: 2` block in `defaults.yaml`; `x{dim: D}` is the same token
  spelled in the canonical form EPDE emits.
- Multi-symbol factors like `cos(t)sin(x)` are atomic product tokens
  built by the adapter — they are not decomposed by `TrigonometricTokens`.

## Coupling between configs and aggregator output

The system names in the summary table above match the row order of
`new_vs_legacy.md` (alphabetical) and the figure names under
`figures/`. Each rep's discovered tokens are canonicalised against the
`truth_equations` (primary) plus `truth_alternatives` (analytic
identities valid on the dataset). A rep credits as "success" if any
canonical permutation matches.
