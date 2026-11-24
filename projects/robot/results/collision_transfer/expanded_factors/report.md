# Expanded robot collision search

Five recording-level holdouts predict the mean two-dimensional velocity over the next 0.4 s. The four training recordings supply separate 75% calibration and 25% validation segments. Each segment rebuilds history from its own start. The first 50 frames provide history, and unresolved initial contact histories are masked. Each recording receives equal weight.

## Search and errors

The 185 translation candidates in the workbook, 81 three-scalar pair products, and 33 robot-modulated contact vectors give 299 columns. The generated pair products multiply distance, a closing/release/drive factor and a signed or geometric factor before summation. The two scalar robot modifiers are smooth wall proximity and absolute focal rim speed. The 14 yaw candidates have a different target. Eight further candidates need instrumentation.

The native EPDE NSGA-II jobs used one seed (923701), 16 individuals, 12 generations, up to three native factors and ten RHS terms. Each holdout ran both a broad pool and a focused 11-vector pool. Each physical RHS term contains one contact or control vector. The two optional modifiers are available for the 11 designated contact vectors. The directional equations have zero scalar intercept. Native receipts archive screen fits and final populations.

The deterministic companion search ranks one-term extensions by training squared error with a beam of eight and up to ten extensions per parent. It includes every single term, the four-term benchmark with every single addition, and focused searches on restricted libraries. EPDE scores candidate structures on 512 stratified component rows per training recording using WAPE and dyad coefficient instability. Full training segments supply the coefficients used for the validation and transfer tables. Native screen-fit errors are also available separately. Validation selects each fold’s model before its held-out score is used.

The search evaluated 24,542 fold–structure combinations. The merged Pareto front minimizes the two mean EPDE training objectives over the 1,317 eligible structures independently evaluated in all five folds. It contains 10 structures. The finite search covers a subset of possible equations.

| Model | Mean R² | Worst R² | Velocity RMSE | Pair-normal RMSE | Pair-tangent RMSE |
|---|---:|---:|---:|---:|---:|
| Four-term benchmark | 0.274 | 0.045 | 61.16 | 68.06 | 86.37 |
| Common five-term law | 0.377 | 0.269 | 57.22 | 58.11 | 80.65 |
| Catalog contact terms | 0.404 | 0.286 | 56.09 | 55.40 | 78.88 |
| Expanded contact terms | 0.404 | 0.286 | 56.07 | 55.56 | 78.76 |
| Contact and robot controls | 0.420 | 0.317 | 55.38 | 55.74 | 78.27 |
| Focused grammar, 1 factor cap | 0.279 | 0.058 | 60.96 | 67.21 | 86.20 |
| Focused grammar, 2 factor cap | 0.283 | 0.060 | 60.81 | 67.38 | 85.85 |
| Focused grammar, 3 factor cap | 0.283 | 0.060 | 60.79 | 67.39 | 85.81 |

## Held-out velocity R² by recording

| Omitted recording | Four terms | Common five terms | Expanded contact | With robot controls |
|---|---:|---:|---:|---:|
| N30_PWM10_D41 | 0.620 | 0.623 | 0.623 | 0.620 |
| N40_PWM10_D40 | 0.045 | 0.292 | 0.346 | 0.368 |
| N40_PWM10_D41 | 0.277 | 0.423 | 0.453 | 0.473 |
| N40_PWM20_D40 | 0.194 | 0.277 | 0.313 | 0.323 |
| N40_PWM20_D41 | 0.233 | 0.269 | 0.286 | 0.317 |

## Merged Pareto front

Each row gives one common equation structure. WAPE and dyad are minimized. Held-out R² and velocity RMSE describe transfer. The term dictionary supplies the vector and weight for every ID. The full evaluation table includes fitted coefficients for each recording holdout.

| Terms | WAPE | Dyad | Mean held-out R² | Worst held-out R² | RMSE (px/s) |
|---|---:|---:|---:|---:|---:|
| `C012+C108+C109+C111` | 0.8247 | 2.34 | 0.384 | 0.266 | 56.96 |
| `C012+C108+C109` | 0.8320 | 0.654 | 0.366 | 0.263 | 57.62 |
| `C006+C108+C109` | 0.8321 | 0.651 | 0.368 | 0.265 | 57.57 |
| `C003+C108+C109` | 0.8325 | 0.65 | 0.368 | 0.263 | 57.56 |
| `C108+C109+C002*Gspin` | 0.8333 | 0.632 | 0.370 | 0.269 | 57.46 |
| `C102+C105` | 0.8451 | 0.408 | 0.355 | 0.223 | 58.24 |
| `C108+C109` | 0.8495 | 0.32 | 0.342 | 0.222 | 58.73 |
| `C002+C108` | 0.8627 | 0.293 | 0.309 | 0.127 | 59.91 |
| `C005+C108` | 0.8630 | 0.277 | 0.307 | 0.124 | 59.98 |
| `C109` | 0.8654 | 0.0843 | 0.322 | 0.209 | 59.56 |

RMSE values are in camera pixels per second. Velocity RMSE and R² use stacked x and y components. Pair errors project relative velocity on the current contact basis for neighbors inside 253 px. The compact law won the single-addition validation comparison in all five folds. Its cross-validation scores use fold-specific training coefficients. The pooled fit for deployment uses the first 75% of all five recordings.

## Common law

\[\widehat{\mathbf v}_i=\mathbf v_{i,\mathrm{base}}+V\sum_{j\,:\,0<r_{ij}<253}\left(0.30664q_{ij}+0.06130c_{ij}+0.00031h_i+0.30925D_j+0.50327D_i\right)\mathbf n_{ij}.\]

The normal points from neighbor j to focal robot i. The dimensionless quantities are cutoff closeness q, approach speed c, inward focal heading h, and the components of each calibrated free propulsion vector directed into the contact, D_i and D_j. V is the cap diameter divided by 0.4 s. The additive base contains calibrated free propulsion, velocity persistence and the wall term.

Past-hit terms use completed 0.12 s pre/post onset velocity differences with 0.4 s decay and a 1.2 s cutoff. These are kinematic proxies for influences at a contact onset. The fixed dimensionless scales are q0=c0=s0=z0=0.1, x0=1 and eps=0.01. Setup-level generalization is measured across these five recordings, several of which share robot IDs.

## Reproduction

```powershell
python projects/robot/collision/run.py --stage expanded
```
