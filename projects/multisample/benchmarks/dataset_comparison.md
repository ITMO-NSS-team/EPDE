# Joint multisample fitting on held-out trajectories

## Discovery results

Held-out relative L2 errors are given as percentages. The two seeds produced the same displayed value except on joint Duffing. The support columns count the seeds for which the exact registered structure appeared anywhere on the reported front.

| Dataset | Mean error | Joint error | Mean exact support on front | Joint exact support on front |
|---|---:|---:|---:|---:|
| Reaction–diffusion | 55.40% | 6.0 × 10⁻¹⁴% | 0/2 | 2/2 |
| Homogeneous KdV | 0.915% | 0.117% | 2/2 | 2/2 |
| Duffing, seeds 0 and 1 | 0.981%, 0.981% | 10.178%, 0.871% | 2/2 | 1/2 |
| Robot motion, median of three held-out recordings | 85.903% | 85.895% | Unknown | Unknown |

Reaction–diffusion has a rank-one physical design in each training record. The two records together distinguish reaction from diffusion. The joint front contained the exact law. The discrepancy-selected joint equation also retained an advection coefficient of magnitude 1.4 × 10⁻¹⁷.

For homogeneous KdV, the registered coefficients of `u u_x` and `u_xxx` are −6 and −1. Mean selected −5.9434 and −0.9775. Joint selected −5.9830 and −0.9972, together with an extra `u_x` term of −0.00549. A single soliton also obeys a speed-dependent advection law, so held-out residual and structural recovery answer different questions on this test.

Duffing has forcing frequency 1 rad per unit time. Mean selected the four physical terms and estimated 1.00250 at both seeds. Joint seed 1 recovered all four and estimated 1.00203. Joint seed 0 omitted damping, kept a quadratic displacement term, and estimated 0.95492. Only the final reported front was inspected for structural recovery.

| Held-out robot recording | Mean | Joint | Repeating the previous velocity |
|---|---:|---:|---:|
| N30_PWM10_D41 | 85.162% | 85.156% | 110.621% |
| N40_PWM10_D40 | 87.514% | 87.507% | 111.640% |
| N40_PWM10_D41 | 85.903% | 85.895% | 108.072% |

## Fixing the physical terms

The following fits use the same training records and native EPDE measurement rows as discovery. The diagnostic supplies the physical terms, omits the intercept, and estimates their coefficients. Duffing also estimates the cosine frequency. The error is evaluated on the original held-out trajectory.

| Model with supplied terms | Mean error | Joint error | What the fit identified |
|---|---:|---:|---|
| Reaction–diffusion | Individually rank deficient | 1.3 × 10⁻¹³% | Each two-column training design had rank one. Pooling recovered −0.2 and 0.1. |
| Homogeneous KdV | 0.915% | 0.00190% | The joint dispersion coefficient was −0.999978. The mean fit gave −0.977474. |
| Duffing | 0.981% | 0.871% | On the same four terms, the two modes found frequencies 1.00250 and 1.00203. The large seed-0 discovery error arose with a different selected structure. |

### Forced KdV and the dispersion coefficient

This diagnostic supplied the three physical terms `u u_x`, `u_xxx`, and `cos(ω_t t) sin(ω_x x)` to EPDE’s native weak form. It estimated both frequencies and all three coefficients. The true dispersion coefficient is −1 and both true frequencies are 1. The single-record arm used the original PIC `kdv/data.csv`. The paired arms added a periodic solution of the same forced PDE with a different initial profile. All three arms were scored against a third, clean forced trajectory.

| Training noise | Fit | Held-out relative L2 | Fitted coefficient of `u_xxx` |
|---|---|---:|---:|
| Clean | Original record only | 0.04654% | −0.999715 |
| Clean | Mean, two records | 0.01936% | −1.000096 |
| Clean | Joint, two records | 0.00317% | −1.000026 |
| 5%, median of three draws | Original record only | 54.70% | −0.541 |
| 5%, median of three draws | Mean, two records | 45.47% | −0.583 |
| 5%, median of three draws | Joint, two records | 20.97% | −0.788 |

The held-out errors for the three noisy draws were 95.62%, 42.76%, and 54.70% for the original record, 7,446.03%, 24.09%, and 45.47% for mean, and 41.35%, 2.61%, and 20.97% for joint. On the first draw, mean averaged forcing amplitudes of −0.078 and 1,079.032. Joint fitted a common amplitude of 1.178. Even joint misplaced the spatial frequency on that draw (0.509) and retained an attenuated dispersion coefficient (−0.599).

Noise was independent Gaussian observation noise with standard deviation equal to 5% of each training field’s standard deviation. Draws 2401, 2402, and 2403 were paired across fit modes. They repeat noise on the same underlying trajectories. The two generated periodic profiles used distinct pairs of Fourier modes, (1, 2) and (2, 3). A dealiased Fourier solver advanced the forced PDE with integrating-factor RK4. Time-step refinement checked the field and its third spatial derivative. The holdout used spectral derivatives on the clean target grid. This comparison measures coefficient and frequency recovery conditional on the supplied equation structure.

## Protocol

Each discovery run trained EPDE on two complete, separately preprocessed trajectories. Mean mode averaged the per-record coefficient estimates. Joint mode fitted one coefficient vector with equal total weight for each record. Both modes used the same input records, token library, seeds (0 and 1), plain NSGA-II, population setting of 8, and maximum of 3 epochs. Fill-and-prune was disabled in both arms because mean mode skips that move with multiple records. The small reaction–diffusion library exhausted unique population members in some runs.

The selected equation minimizes training discrepancy on its variable’s reported Pareto front. Ties use term count, then equation text. Every held-out score uses the frozen coefficients and frequencies. Relative L2 error is the norm of the prediction error divided by the norm of the held-out derivative or forward rate. Predicting zero gives 100%. The score measures local predictions at observed states. Exact support counts every retained nonzero coefficient, including roundoff-size terms, and requires zero intercept for the three known discovery laws.

| Dataset | Fit records | Held-out evaluation |
|---|---|---|
| Reaction–diffusion control | Two analytic single spatial modes from the multisample example with exact derivative channels | A different grid and a mixture of modes 1 and 3, scored on analytic derivatives |
| Homogeneous KdV | The stored PIC PDE-FIND and homogeneous KdV arrays. The first used its first 41 times and points within `[-15, 15]`. The second used every third time and spatial point. Strong rows used finite differences. Four samples were excluded at each edge, leaving two spatial layers using the edge stencil. | A separately generated soliton with `k=0.91`, scored on analytic derivatives |
| Duffing | Two independently integrated initial states from the stored PIC parameters. Observations were velocity and displacement. Candidate predictors included displacement, its square and cube, velocity, and fitted sine and cosine terms. | The original stored PIC trajectory, scored on velocity increments divided by elapsed time. Discovery estimated forcing frequencies from the training observations. |
| Robot motion | Two of three archived PWM10 camera recordings. The lowest common physical robot ID was 11. Inputs were relative camera-x position, sine and cosine of measured heading, and the preceding x velocity. | The entire remaining recording, rotating over all three. The target was the next 0.2-second camera-x increment divided by 0.2 seconds. |

The two search seeds reuse the same held-out data. The two synthetic ODE training records and the generated PDE controls are simulations. The robot data come from physical recordings. The empty `projects/pic/data/trajectories-2` file was excluded.

## Reproduction

Run these commands from the repository root with the EPDE environment active. The runner enforces the existing 2% Windows host CPU cap, single CPU affinity, and single numerical-library thread. Discovery resumes from completed entries in `dataset_runs.json`.

```powershell
$env:PYTHONHASHSEED='0'
python projects/multisample/benchmarks/dataset_comparison.py
python projects/multisample/benchmarks/dataset_comparison.py --fixed --forced-kdv
```

The individual Pareto candidates and held-out scores are in [dataset_runs.json](dataset_runs.json). The coefficient diagnostics are in [fixed_support.json](fixed_support.json) and [forced_kdv_support.json](forced_kdv_support.json).
