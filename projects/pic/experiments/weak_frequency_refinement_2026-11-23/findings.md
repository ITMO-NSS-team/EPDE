# Weak-residual frequency refinement before discovery

**The native and refined libraries recovered the same 19 of 45 complete systems on the held-out front.** This held at both 3,000 total fits and 3,000 discovery evaluations. No paired system changed on the primary endpoint. The separate harmonic oscillator control was recovered on all three seeds by both arms. The rule missed the positive-gain threshold for confirmation. [Paired endpoints and front sizes](by_system.csv) · [Protocol](protocol.md)

## What the frequency tolerance measures

The benchmark compares the frequency in a discovered term with the registered frequency. A relative error of at most 1% counts as a frequency match. For Duffing the registered value is 1, giving an accepted interval of 0.99–1.01. On seed 923301 the original peak was 1.07648 and the installed refinement was 0.96884. The absolute relative error fell from 7.65% to 3.12%. The refined term still missed the frequency criterion. Held-out WAPE separately measures the residual of the equation with its fitted coefficients and frequency frozen. [Frequency and residual measurements](known_topology_diagnostic.csv)

| Dataset | Seed | Registered | Native peak | Installed peak | Installed relative error | Correct term pattern scored by refined search¹ |
|---|---:|---:|---:|---:|---:|---|
| Duffing | 923301 | 1 | 1.07648 | 0.96884 | 3.12% | No |
| Duffing | 923302 | 1 | 1.07727 | 0.97164 | 2.84% | No |
| Duffing | 923303 | 1 | 1.07970 | 0.97350 | 2.65% | No |
| Forced oscillator | 923301 | 2 | 2.05404 | 2.05404 | 2.70% | Yes |
| Forced oscillator | 923302 | 2 | 2.06247 | 2.26197 | 13.10% | Yes |
| Forced oscillator | 923303 | 2 | 2.05942 | 2.16412 | 8.21% | Yes |

¹ Pattern matching here ignores the frequency value and retains the sine or cosine family, axis, factors and powers. Both grids missed the frequency criterion on all six cases. The forced oscillator's accepted interval is 1.98–2.02. [Support and frequency audit](forcing.csv)

## Why the training residual matters

On all three Duffing seeds the calibration selected a first-derivative target with a cosine and a quadratic time trend. Its residual favored a slightly slower oscillation than the catalog's second-order driven system. The held-out observations and catalog structure played no role in selecting the grid. [Selected calibration equations and residuals](profiles.json)

After discovery, a separate diagnostic supplied the correct equation terms and profiled only the frequency and linear coefficients on the original training weak windows. The best frequency entered the 1% band on one Duffing seed and two forced-oscillator seeds. Held-out WAPE improved relative to using the native peak on **all six**. For example, on Duffing seed 923302 the diagnostic moved the frequency from 1.07727 to 0.97729. This still missed the 1% structural cutoff, while held-out WAPE fell from 0.1850 to 0.1009. The diagnostic uses catalog information and earns no discovery credit. It motivates testing refinement conditional on equations actually visited during search, particularly on the forced oscillator where search repeatedly visited the correct pattern. [Known-topology diagnostic](known_topology_diagnostic.csv)

## Cost and checks

Across the primary suite, 31 of 156 spectral peaks moved, affecting 22 of the 45 searches. Calibration used 360 background fits and 1,782 frequency-profile fits. Charging all 2,142 left a median of 2,948 discovery evaluations at equal total cost. Under that budget, both arms retained 32 of 60 individual laws. Refinement gained Lorenz's second law and lost Navier–Stokes' first law on seed 923301. Training-front system recovery fell from 15 to 14 with a loss on Van der Pol seed 923302. Among 39 laws scored by both arms at equal cost, median first discovery was at fit 396 with native search and 560 with refinement. [Work](work.csv) · [Laws](laws.csv) · [Systems](systems.csv) · [Summary](summary.json)

The screen reused 48 verified native archives and completed 48 refined runs. Pair checks covered archive hashes, data hashes, settings, window geometry, original peaks and installed grids. Frozen-coefficient replay reproduced eligible training WAPE within 1.67 × 10⁻¹⁵. Direct joint least squares reproduced the profiled squared residuals at 24 original and proposed frequencies on the forcing cases within 3.7 × 10⁻¹⁵ relative error. Training windows overlap, so the BIC penalty is a selection heuristic. The paired held-out fronts supply the recovery endpoint. [Run manifest](manifest.json)

Run `frequency_trial.py`, `evaluate.py` and `forcing_diagnostic.py` in that order using the repository research Python environment. All frequency changes are confined to the research harness.
