# Data sets in projects/pic/data

Generated from the data-set loaders; edit the loaders, not this file. The name in the first column is what the command line, the scripts and the notebooks use to refer to a record.

Suites: **core** -- synthetic with a known law; the main benchmark tables; **extended** -- synthetic with a known law, slower or less standard; **real** -- measurements; **other** -- loads, but is not part of the benchmark.

Equations are written in EPDE text form: `dx0` is time, `dx1`, `dx2`, ... are the space axes in the order given under "axes"; `x{power: p, dim: k}` is the k-th coordinate to the power p. Only the set of terms is scored; coefficients are for reference.

| name | suite | class | source | shape | variables | law | title |
|---|---|---|---|---|---|---|---|
| `ode` | core | ODE (one equation) | synthetic | 320 | u | known | Forced oscillator with time-dependent damping |
| `vdp` | core | ODE (one equation) | synthetic | 320 | u | known | Van der Pol oscillator (mu = 0.2) |
| `duffing` | core | ODE (one equation) | synthetic | 1001 | u | known | Forced Duffing oscillator |
| `lv` | core | system of ODEs | synthetic | 301 | u, v | known (+1 alt.) | Lotka-Volterra predator-prey |
| `lorenz` | core | system of ODEs | synthetic | 1041 | u, v, w | known | Lorenz-63 system |
| `ac` | core | PDE, 1 space dimension | synthetic | 51 x 128 | u | known | Allen-Cahn equation |
| `burgers` | core | PDE, 1 space dimension | synthetic | 101 x 256 | u | known | Viscous Burgers equation (PDE-FIND data) |
| `burgers_inviscid` | core | PDE, 1 space dimension | synthetic | 101 x 101 | u | known (+2 alt.) | Inviscid Burgers equation |
| `wave` | core | PDE, 1 space dimension | synthetic | 81 x 81 | u | known (+4 alt.) | Wave equation |
| `kdv` | core | PDE, 1 space dimension | synthetic | 201 x 512 | u | known (+3 alt.) | Korteweg-de Vries equation (PDE-FIND data) |
| `kdv_cossin` | core | PDE, 1 space dimension | synthetic | 81 x 81 | u | known | KdV with a cos(t)sin(x) source |
| `ks` | core | PDE, 1 space dimension | synthetic | 251 x 1024 | u | known | Kuramoto-Sivashinsky equation |
| `pde_compound` | core | PDE, 1 space dimension | synthetic | 251 x 100 | u | known | Nonlinear diffusion u_t = (u u_x)_x |
| `pde_divide` | core | PDE, 1 space dimension | synthetic | 251 x 100 | u | known | PDE with a 1/x coefficient |
| `ns` | core | PDE, 2 space dimensions | synthetic | 50 x 20 x 36 | u, v, p | known | Navier-Stokes, cylinder wake (Re = 100) |
| `kdv_homogen` | extended | PDE, 1 space dimension | synthetic | 120 x 480 | u | known | KdV, homogeneous, x in [-3, 3] |
| `kdv_sga` | extended | PDE, 1 space dimension | synthetic | 201 x 512 | u | known | KdV, SGA-PDE record (u_t = -u u_x - 0.0025 u_xxx) |
| `jhtdb_plane` | extended | PDE, 2 space dimensions | synthetic | 40 x 48 x 48 | u, v | known | Isotropic turbulence, 2-D slice of JHTDB |
| `pend_single` | real | ODE (one equation) | measured | 5000 | theta | known (+1 alt.) | Real single pendulum (encoder, HardwareX rig) |
| `dp_encoder` | real | system of ODEs | measured | 6000 | theta1, theta2 | unknown | Real double pendulum (encoder, HardwareX rig) |
| `robot_arm` | real | ODE (one equation) | measured | 1024 | y | unknown | DaISy flexible robot arm (input torque -> acceleration) |
| `ballbeam` | real | ODE (one equation) | measured | 1000 | y | unknown | DaISy ball and beam (beam angle -> ball position) |
| `heat_solar_1d` | other | PDE, 1 space dimension | synthetic | 576 x 51 | u | unknown | Heat in soil under solar forcing, 1-D |
| `heat_solar_2d` | other | PDE, 2 space dimensions | synthetic | 144 x 51 x 51 | u | unknown | Heat in soil under solar forcing, 2-D |
| `heat_laser` | other | PDE, 3 space dimensions | synthetic | 20 x 51 x 51 x 3 | u | unknown | Heat equation with a moving laser source, 3-D |
| `dp_sim` | other | system of ODEs | synthetic | 1001 | theta1, theta2 | unknown | Simulated double pendulum |
| `dp_video` | other | system of ODEs | measured | 38827 | theta1, theta2 | unknown | Real double pendulum (video tracking) |
| `sst` | other | PDE, 2 space dimensions | measured | 90 x 268 x 384 | T | unknown | Sea surface temperature, ESA CCI L4 (Jan-Mar 2025) |
| `darcy` | other | PDE, 2 space dimensions | synthetic | - | - | no files | Darcy flow -div(nu grad u) = 1 (data files missing) |

## Details

### `ode` -- Forced oscillator with time-dependent damping

- **class:** ODE (one equation); **source:** synthetic; **suite:** core
- **axes:** t; **shape:** 320; **variables:** u
- **law:**
  - `-4.0 * u{power: 1.0} + -1.0 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + 1.5 * x{power: 1.0, dim: 0.0} = d^2u/dx0^2{power: 1.0}`
- **notes:** u'' + sin(2t) u' + 4u = 1.5t on t in [0, 16), dt = 0.05.

### `vdp` -- Van der Pol oscillator (mu = 0.2)

- **class:** ODE (one equation); **source:** synthetic; **suite:** core
- **axes:** t; **shape:** 320; **variables:** u
- **law:**
  - `-0.2 * u{power: 2.0} * du/dx0{power: 1.0} + 0.2 * du/dx0{power: 1.0} + -1.0 * u{power: 1.0} = d^2u/dx0^2{power: 1.0}`
- **notes:** u'' = 0.2 (1 - u^2) u' - u on t in [0, 16), dt = 0.05.

### `duffing` -- Forced Duffing oscillator

- **class:** ODE (one equation); **source:** synthetic; **suite:** core
- **axes:** t; **shape:** 1001; **variables:** u
- **law:**
  - `-0.20000000298023224 * du/dx0{power: 1.0} + -1.0 * u{power: 1.0} + -1.0 * u{power: 3.0} + 0.30000001192092896 * cos{power: 1.0, freq: 1.0, dim: 0.0} = d^2u/dx0^2{power: 1.0}`
- **notes:** u'' + delta u' + alpha u + beta u^3 = gamma cos(omega t); the parameters are stored in the data file.

### `lv` -- Lotka-Volterra predator-prey

- **class:** system of ODEs; **source:** synthetic; **suite:** core
- **axes:** t; **shape:** 301; **variables:** u, v
- **law:**
  - `20.0 * u{power: 1.0} + -20.0 * u{power: 1.0} * v{power: 1.0} = du/dx0{power: 1.0}`
  - `20.0 * u{power: 1.0} * v{power: 1.0} + -20.0 * v{power: 1.0} = dv/dx0{power: 1.0}`
  - also accepted: `-1.0 * dv/dx0{power: 1.0} + 20.0 * u{power: 1.0} + -20.0 * v{power: 1.0} = du/dx0{power: 1.0}`; `20.0 * du/dx0{power: 1.0} + -20.0 * dv/dx0{power: 1.0} + -1.0 * d^2u/dx0^2{power: 1.0} = d^2v/dx0^2{power: 1.0}`
- **notes:** alpha = beta = gamma = delta = 20, all 301 samples: with only the first half of the record the system is not identifiable.

### `lorenz` -- Lorenz-63 system

- **class:** system of ODEs; **source:** synthetic; **suite:** core
- **axes:** t; **shape:** 1041; **variables:** u, v, w
- **law:**
  - `10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} = du/dx0{power: 1.0}`
  - `28.0 * u{power: 1.0} + -1.0 * u{power: 1.0} * w{power: 1.0} + -1.0 * v{power: 1.0} = dv/dx0{power: 1.0}`
  - `1.0 * u{power: 1.0} * v{power: 1.0} + -2.6666666666666665 * w{power: 1.0} = dw/dx0{power: 1.0}`
- **notes:** sigma = 10, rho = 28, beta = 8/3. Window t in [20.0, 25.2] of the stored run, every 5th sample: on the attractor, not the initial transient.

### `ac` -- Allen-Cahn equation

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** core
- **axes:** t, x; **shape:** 51 x 128; **variables:** u
- **law:**
  - `0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} = du/dx0{power: 1.0}`
- **notes:** u_t = 1e-4 u_xx + 5u - 5u^3. The diffusion term is tiny, which makes it hard to separate from noise.

### `burgers` -- Viscous Burgers equation (PDE-FIND data)

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** core
- **axes:** t, x; **shape:** 101 x 256; **variables:** u
- **law:**
  - `-1.0 * u{power: 1.0} * du/dx1{power: 1.0} + 0.1 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}`
- **notes:** u_t = -u u_x + 0.1 u_xx, periodic in x.

### `burgers_inviscid` -- Inviscid Burgers equation

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** core
- **axes:** t, x; **shape:** 101 x 101; **variables:** u
- **law:**
  - `-1.0 * u{power: 1.0} * du/dx1{power: 1.0} = du/dx0{power: 1.0}`
  - also accepted: `1.0 * u{power: 1.0} = x{power: 1.0, dim: 1.0} * du/dx1{power: 1.0}`
  - also accepted: `0.5 * x{power: 1.0, dim: 0.0} * u{power: 1.0} + -0.5 * x{power: 1.0, dim: 1.0} = du/dx1{power: 1.0} * x{power: 1.0, dim: 1.0}`
- **notes:** The record is the similarity solution u = x / (t + c), so two identities hold as well as the PDE and count as correct.

### `wave` -- Wave equation

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** core
- **axes:** t, x; **shape:** 81 x 81; **variables:** u
- **law:**
  - `0.04 * d^2u/dx1^2{power: 1.0} = d^2u/dx0^2{power: 1.0}`
  - also accepted: `48.67869238111131 * d^2u/dx1^2{power: 1.0} * d^2u/dx0^2{power: 1.0} + -591.9797844706457 * d^2u/dx0^2{power: 2.0} = d^2u/dx1^2{power: 2.0}`
  - also accepted: `0.04012 * d^2u/dx1^2{power: 1.0} + 1.08338 * d^2u/dx0^2{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + -0.04347 * d^2u/dx1^2{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} = d^2u/dx0^2{power: 1.0}`
  - also accepted: `0.04012 * d^2u/dx1^2{power: 1.0} + 0.68737 * du/dx0{power: 1.0} * d^2u/dx0^2{power: 1.0} + -0.02751 * d^2u/dx1^2{power: 1.0} * du/dx0{power: 1.0} = d^2u/dx0^2{power: 1.0}`
  - also accepted: `0.46769 * u{power: 1.0} * d^2u/dx0^2{power: 1.0} + -0.01878 * u{power: 1.0} * d^2u/dx1^2{power: 1.0} + 0.04029 * d^2u/dx1^2{power: 1.0} = d^2u/dx0^2{power: 1.0}`
- **notes:** u_tt = 0.04 u_xx. The alternatives are the wave equation multiplied by another factor; they are accepted as in the group's former benchmark.

### `kdv` -- Korteweg-de Vries equation (PDE-FIND data)

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** core
- **axes:** t, x; **shape:** 201 x 512; **variables:** u
- **law:**
  - `-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} = du/dx0{power: 1.0}`
  - also accepted: `-1.0 * u{power: 3.0} + 1.0 * du/dx1{power: 2.0} = u{power: 1.0} * d^2u/dx1^2{power: 1.0}`
  - also accepted: `-3.0 * u{power: 2.0} * du/dx1{power: 1.0} + 1.0 * du/dx1{power: 1.0} * d^2u/dx1^2{power: 1.0} = d^3u/dx1^3{power: 1.0} * u{power: 1.0}`
  - also accepted: `-0.3333333333 * u{power: 1.0} * du/dx0{power: 1.0} + -0.3333333333 * du/dx1{power: 1.0} * d^2u/dx1^2{power: 1.0} = u{power: 2.0} * du/dx1{power: 1.0}`
- **notes:** u_t = -6 u u_x - u_xxx. The record is a soliton family, so three identities of it are also exact and accepted.

### `kdv_cossin` -- KdV with a cos(t)sin(x) source

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** core
- **axes:** t, x; **shape:** 81 x 81; **variables:** u
- **law:**
  - `-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} + 1.0 * cos(t)sin(x){power: 1.0} = du/dx0{power: 1.0}`
- **extra tokens:** cos(t)sin(x)
- **notes:** The source enters as one product token cos(t)sin(x).

### `ks` -- Kuramoto-Sivashinsky equation

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** core
- **axes:** t, x; **shape:** 251 x 1024; **variables:** u
- **law:**
  - `-1.0 * u{power: 1.0} * du/dx1{power: 1.0} + -1.0 * d^2u/dx1^2{power: 1.0} + -1.0 * d^4u/dx1^4{power: 1.0} = du/dx0{power: 1.0}`
- **notes:** u_t = -u u_x - u_xx - u_xxxx; chaotic, needs a 4th derivative.

### `pde_compound` -- Nonlinear diffusion u_t = (u u_x)_x

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** core
- **axes:** t, x; **shape:** 251 x 100; **variables:** u
- **law:**
  - `1.0 * du/dx1{power: 2.0} + 1.0 * d^2u/dx1^2{power: 1.0} * u{power: 1.0} = du/dx0{power: 1.0}`
- **notes:** u_t = u_x^2 + u u_xx on t in [0, 0.5], x in [1, 2].

### `pde_divide` -- PDE with a 1/x coefficient

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** core
- **axes:** t, x; **shape:** 251 x 100; **variables:** u
- **law:**
  - `-2.0 * du/dx1{power: 1.0} + 0.5 * d^2u/dx1^2{power: 1.0} * x{power: 1.0, dim: 1.0} = du/dx0{power: 1.0} * x{power: 1.0, dim: 1.0}`
- **notes:** x u_t = -2 u_x + 0.5 x u_xx, i.e. u_t = -2 u_x / x + 0.5 u_xx (needs the x token).

### `ns` -- Navier-Stokes, cylinder wake (Re = 100)

- **class:** PDE, 2 space dimensions; **source:** synthetic; **suite:** core
- **axes:** t, y, x; **shape:** 50 x 20 x 36; **variables:** u, v, p
- **law:**
  - `-1.0 * u{power: 1.0} * du/dx2{power: 1.0} + -1.0 * v{power: 1.0} * du/dx1{power: 1.0} + -1.0 * dp/dx2{power: 1.0} + 0.01 * d^2u/dx2^2{power: 1.0} + 0.01 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}`
  - `-1.0 * u{power: 1.0} * dv/dx2{power: 1.0} + -1.0 * v{power: 1.0} * dv/dx1{power: 1.0} + -1.0 * dp/dx1{power: 1.0} + 0.01 * d^2v/dx2^2{power: 1.0} + 0.01 * d^2v/dx1^2{power: 1.0} = dv/dx0{power: 1.0}`
  - `-1.0 * dv/dx1{power: 1.0} = du/dx2{power: 1.0}`
- **notes:** Axes (t, y, x): dx1 = d/dy, dx2 = d/dx. Two momentum equations (nu = 0.01) and continuity. By default a subset of about 36 thousand points; the full window has 250 thousand points per variable.

### `kdv_homogen` -- KdV, homogeneous, x in [-3, 3]

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** extended
- **axes:** t, x; **shape:** 120 x 480; **variables:** u
- **law:**
  - `-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} = du/dx0{power: 1.0}`
- **notes:** Same law on a short interval, as in the group's original experiments.

### `kdv_sga` -- KdV, SGA-PDE record (u_t = -u u_x - 0.0025 u_xxx)

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** extended
- **axes:** t, x; **shape:** 201 x 512; **variables:** u
- **law:**
  - `-1.0 * du/dx1{power: 1.0} * u{power: 1.0} + -0.0025 * d^3u/dx1^3{power: 1.0} = du/dx0{power: 1.0}`
- **notes:** The record used in the symbolic genetic algorithm study.

### `jhtdb_plane` -- Isotropic turbulence, 2-D slice of JHTDB

- **class:** PDE, 2 space dimensions; **source:** synthetic; **suite:** extended
- **axes:** t, y, x; **shape:** 40 x 48 x 48; **variables:** u, v
- **law:**
  - `-1.0 * u{power: 1.0} * du/dx2{power: 1.0} + -1.0 * v{power: 1.0} * du/dx1{power: 1.0} + -1.0 * w{power: 1.0} * u_z{power: 1.0} + -1.0 * p_x{power: 1.0} + 1.0 * nu_lap_u{power: 1.0} = du/dx0{power: 1.0}`
  - `-1.0 * u{power: 1.0} * dv/dx2{power: 1.0} + -1.0 * v{power: 1.0} * dv/dx1{power: 1.0} + -1.0 * w{power: 1.0} * v_z{power: 1.0} + -1.0 * p_y{power: 1.0} + 1.0 * nu_lap_v{power: 1.0} = dv/dx0{power: 1.0}`
- **extra tokens:** p_x, p_y, nu_lap_u, nu_lap_v, w, u_z, v_z
- **derivatives:** supplied with the data and passed to EPDE as given
- **notes:** 48 x 48 slice (stride 8 of the DNS grid), 40 frames. Too coarse for numerical derivatives (aliased), so the exact server-side gradients are passed instead; out-of-plane and pressure terms enter as exact tokens. Artificial noise is unsupported with these supplied derivatives; use noise=0.

### `pend_single` -- Real single pendulum (encoder, HardwareX rig)

- **class:** ODE (one equation); **source:** measured; **suite:** real
- **axes:** t; **shape:** 5000; **variables:** theta
- **law:**
  - `-64.8 * theta{power: 1.0} + -0.65 * dtheta/dx0{power: 1.0} = d^2theta/dx0^2{power: 1.0}`
  - also accepted: `-64.8 * theta{power: 1.0} = d^2theta/dx0^2{power: 1.0}`
- **notes:** Small swing about the hanging position, angle centred (theta - pi): a damped linear oscillator, -64.8 = -g/l. The undamped form also counts (damping is weak).

### `dp_encoder` -- Real double pendulum (encoder, HardwareX rig)

- **class:** system of ODEs; **source:** measured; **suite:** real
- **axes:** t; **shape:** 6000; **variables:** theta1, theta2
- **extra tokens:** sin_th1, sin_th2, cos_delta, sin_delta
- **notes:** Coupled Lagrangian equations with cos/sin of the angle difference; their exact coefficients are not known, so the run is not scored. sin(theta) and the coupling factors enter as tokens.

### `robot_arm` -- DaISy flexible robot arm (input torque -> acceleration)

- **class:** ODE (one equation); **source:** measured; **suite:** real
- **axes:** t; **shape:** 1024; **variables:** y
- **extra tokens:** u_in
- **notes:** Unknown truth (a ~5th-order flexible structure). dt is not distributed with the file; 0.01 s is assumed (coefficients scale with it, structure does not).

### `ballbeam` -- DaISy ball and beam (beam angle -> ball position)

- **class:** ODE (one equation); **source:** measured; **suite:** real
- **axes:** t; **shape:** 1000; **variables:** y
- **extra tokens:** u_in
- **notes:** Unknown truth; idealised physics is y'' proportional to the beam angle. Sampling period 0.1 s per the DaISy description.

### `heat_solar_1d` -- Heat in soil under solar forcing, 1-D

- **class:** PDE, 1 space dimension; **source:** synthetic; **suite:** other
- **axes:** t, x; **shape:** 576 x 51; **variables:** u
- **notes:** Simulated soil temperature with a periodic surface flux. Expected law: the heat equation u_t = a u_xx in the interior (coefficient not stored), so no truth is scored. The data also hold the time derivative.

### `heat_solar_2d` -- Heat in soil under solar forcing, 2-D

- **class:** PDE, 2 space dimensions; **source:** synthetic; **suite:** other
- **axes:** t, x, y; **shape:** 144 x 51 x 51; **variables:** u
- **notes:** 2-D version of heat_solar_1d (576 x 51 x 51 before striding in t).

### `heat_laser` -- Heat equation with a moving laser source, 3-D

- **class:** PDE, 3 space dimensions; **source:** synthetic; **suite:** other
- **axes:** t, x, y, z; **shape:** 20 x 51 x 51 x 3; **variables:** u
- **extra tokens:** L
- **notes:** Only 3 points along z and 20 in t, so z- and t-derivatives are crude. The laser source is supplied as a field in (t, x, y), assumed uniform in z. No truth is scored.

### `dp_sim` -- Simulated double pendulum

- **class:** system of ODEs; **source:** synthetic; **suite:** other
- **axes:** t; **shape:** 1001; **variables:** theta1, theta2
- **notes:** Used by the PINN studies in dp/. The equations of motion need coupling tokens and are not written in token form here.

### `dp_video` -- Real double pendulum (video tracking)

- **class:** system of ODEs; **source:** measured; **suite:** other
- **axes:** t; **shape:** 38827; **variables:** theta1, theta2
- **notes:** Mean link angles from video markers. Noisier than the encoder record, which is the preferred source.

### `sst` -- Sea surface temperature, ESA CCI L4 (Jan-Mar 2025)

- **class:** PDE, 2 space dimensions; **source:** measured; **suite:** other
- **axes:** t, lat, lon; **shape:** 90 x 268 x 384; **variables:** T
- **notes:** Daily analysed sea surface temperature (K), 90 days. By default the largest rectangular ocean region finite on every day is used; the original box with land masked can be loaded for plotting only.

### `darcy` -- Darcy flow -div(nu grad u) = 1 (data files missing)

- **status:** the data files are missing; loading and search are not verified.
