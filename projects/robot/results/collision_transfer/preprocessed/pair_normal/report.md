# Close-pair normal velocity over 0.4 seconds

The target is $u^+_{ij}=[(\Delta\mathbf x_i-\Delta\mathbf x_j)/\Delta t]\cdot\mathbf n_{ij}(t)$, positive for separation. Both displacements and the origin normal use the measured centers. Projected centers supply kinematics and clearance, and the original measured gap supplies contact evidence. The 253 px neighbor cutoff, forecast origins and availability masks match the previous vector-law experiment. All models are tested on the same 174,120 unordered pair observations.

Each holdout uses the first 75% of the other four recordings to fit coefficients and their final 25% to choose the equation or tree capacity. Each recording has equal weight. The merged common equation uses the union of proposals across the five training folds and is exploratory. Its coefficients for deployment use all five training prefixes. Rows marked coherent use one velocity per robot, including at simultaneous contacts.

| Prediction | Mean normal R² | Worst recording R² | Normal RMSE (px/s) |
|---|---:|---:|---:|
| Previous vector law | 0.1027 | -0.1415 | 55.65 |
| Direct three-term law | -0.0177 | -0.0311 | 60.01 |
| Direct sparse equation | 0.1662 | 0.1239 | 54.30 |
| Direct boosted pair model | 0.2024 | 0.1529 | 53.10 |
| Three-term correction | 0.2419 | 0.1770 | 51.71 |
| Fold-selected sparse correction, coherent | 0.2904 | 0.2164 | 50.00 |
| Common sparse correction, coherent | 0.2943 | 0.2213 | 49.85 |
| Boosted correction, pair prediction | 0.3177 | 0.2341 | 49.06 |
| Boosted correction, coherent | 0.3203 | 0.2348 | 48.97 |

## A compact common correction

Let $u_0$ be the previous vector law projected onto the observed line of centers, $g=\max(r_{\rm projected}-D,0)$ be the clearance in pixels, and $[x]_+=\max(x,0)$. The pooled three-term correction gives

$$\hat u = 4.844 +0.6925u_0 +1.0135[-g/\Delta t-u_0]_+.$$

The hinge activates when the predicted inward displacement exceeds the available radial clearance over the forecast interval. The intercept has units of px/s. Spin, lateral motion, neighbor effects and measurement noise can change observed motion along the origin axis over this interval.

## Sparse correction by recording

| Held-out recording | Pairs | Previous normal R² | Updated normal R² | Previous RMSE | Updated RMSE |
|---|---:|---:|---:|---:|---:|
| N30_PWM10_D41 | 19,172 | 0.326 | 0.424 | 54.09 | 50.00 |
| N40_PWM10_D40 | 40,716 | -0.141 | 0.245 | 50.80 | 41.31 |
| N40_PWM10_D41 | 37,073 | 0.069 | 0.326 | 51.30 | 43.66 |
| N40_PWM20_D40 | 41,633 | 0.113 | 0.241 | 58.71 | 54.33 |
| N40_PWM20_D41 | 35,526 | 0.146 | 0.216 | 63.33 | 60.68 |

To obtain one velocity for each robot in a frame, the update minimizes $\|A\delta v-(\hat u-A v_0)\|^2+0.1\|\delta v\|^2$. Each row of $A$ contains the two opposite unit normals for a pair. This preserves the sum of velocity updates within each connected contact group. Tangential and whole-robot errors are recomputed after this update.

Mean tangential R² changes from 0.409 to 0.448. Mean whole-robot Cartesian R² changes from 0.421 to 0.458.
The paired 95% interval for the sparse update’s mean R² gain is 0.182 to 0.194. It resamples ten-second time blocks within these five recordings.

| State at the forecast origin | Pairs | Previous normal RMSE | Updated normal RMSE |
|---|---:|---:|---:|
| approaching | 98,061 | 55.02 | 48.85 |
| separating or stationary | 76,059 | 56.34 | 51.28 |
| isolated pair | 2,697 | 67.01 | 63.61 |
| multiple neighbors | 171,423 | 55.34 | 49.63 |
| observed inside cap | 4,404 | 57.27 | 51.04 |
| observed outside cap | 169,716 | 55.64 | 49.99 |

## Search and outputs

Two native EPDE NSGA-II searches ran per holdout, one on the pair speed and one on the previous law’s residual. Each used seed 923701, 16 individuals, 12 generations, up to ten RHS terms and up to three factors per term. Deterministic beam searches also proposed supports. For the correction, 5,348 distinct proposals yielded 5,176 common eligible structures and 36 nondominated structures on mean residual WAPE and dyad instability. The validation-error versus size front is also flagged. Rank-deficient fits have no unique coefficients.

The main comparison is the sparse correction selected independently within each fold. The nonlinear reference uses histogram gradient boosting and selects its leaf count on the same validation tails. Forecasts use past pose windows of 2, 3, 6, 10 and 20 frames. All forecasts are conditional on a close pair at the origin. Force, impulse and event-time restitution are separate quantities.

[Full common equations and pooled coefficients](correction/pareto_equations.csv) · [Term dictionary](term_dictionary.csv) · [Prior vector law and calibration](../search/cross_recording/pooled_common_law.json) · [Every candidate’s fold errors](correction/candidate_errors.csv.gz) · [Vector scores](vector_scores.csv) · [Matched score plot](normal_transfer.png)
