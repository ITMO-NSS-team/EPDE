# Coefficient fits and localized weak holdouts

The domain-refactor plain search prints the native weak-row weighted least-squares (WLS) fit for the evolved target and active terms. The experimental first objective estimates a homogeneous coefficient vector independently on each training fold. Dyad measures coefficient variation using its own slab fits.

## Results on frozen candidates

The table counts observations on which the canonical reference equation was nondominated against the union of the two saved fronts. Each system contributed three discovery seeds and 32 fresh 5% noise recordings per seed. Candidates, token frequencies and dyad were fixed. Only variants with dyad at most the reference's best allowed dyad can remove it from this front. The signed readout used the same training-only normalization scheme for every coefficient source.

| Discrepancy coefficients and holdouts | ode_simple | Van der Pol | Wave | Allen–Cahn |
|---|---:|---:|---:|---:|
| Original implicit fit, broad | 65 | 7 | 38 | 1 |
| Original implicit fit, localized | 62 | 5 | 32 | 1 |
| Full-record native WLS, broad | 65 | 64 | 53 | 4 |
| Full-record unit-column total least squares, broad | 65 | 63 | 51 | 3 |
| Training-fold native WLS, broad | 69 | 15 | 26 | 0 |
| Training-fold unit-column total least squares, broad | 49 | 18 | 0 | 0 |

Each localized bank replaced the broad Fourier moments with 9 overlapping half-span time windows for ODEs or a 5 by 5 grid of half-span space-time windows for PDEs. All windows in a fold shared one frozen coefficient vector. On the original discovery observations, localization alone put the canonical reference on the pooled front in 2/12 cases, versus 4/12 with broad moments.

Van der Pol shows the influence of the rival fits. With broad holdouts, changing just the reference from the implicit to its native fit raised survival from 7 to 18. Changing just the rivals raised it from 7 to 44. Changing both raised it to 64. For wave the implicit fits gave a wave coefficient between 0.0436 and 0.0479 across nine folds. Native WLS gave 0.04358, 0.03943 and 0.03884 against the physical value of 0.04.

For Allen–Cahn, supplying the physical coefficients to the full law and scoring the rivals at their native weights gave 17/96 front survivals. Using native WLS on both sides favored the diffusion-bearing candidate over reaction alone on 47/96 positive recordings and 34/96 matched zero-diffusion recordings. These pairwise counts compare first objectives. The zero fixture has the same initial profile, analytically known reaction dynamics and the same standardized noise draws as the positive fixture.

## Interpretation and checks

The full-record WLS comparison uses the entire discovery field and is evaluated on independent recordings. In the training-fold variant each fit uses only every third time sample, and its native window is selected again from that subsampled field. The difference therefore includes sample count, time spacing and window choice. For the short ode_simple training groups the full dyad row floor was infeasible. Only the coefficient fit used a WLS row floor of at least the maximum equation width plus one. All reported dyad values come from the original runs.

The original broad scores were replayed on the discovery fields. The coefficient study independently reproduced the localized and broad scores on every fresh recording and on the clean diagnostic. Native fitted coefficients reproduced archived printed coefficients among eligible rivals to a maximum scaled difference below 7e-12. The counts apply to repeated measurements of four fixed trajectories. A fresh evolutionary search and transfer to other trajectories are outside this experiment.

The per-record scores, coefficient vectors, eligible variants, training geometries and paired controls are in `local_holdouts.json` and `coefficient_audit.json`. The corresponding Python scripts reproduce them with `PYTHONHASHSEED=0`.
