# Predicting normal speed in the first 40 ms after a measured crossing

The primary comparison uses exactly the 1,888 held-out nominal entry events from the earlier restitution study. Raw measured center displacements supply both the target and its sign. Eligibility includes the original looser third-robot clearance, wall and tracking checks. All errors are in px/s.

| Model | Entry RMSE | Entry mean R² | Entry MAE | Matched entry RMSE | Next interval RMSE |
|---|---:|---:|---:|---:|---:|
| One training mean | 166.54 | -0.097 | 129.82 | 163.27 | 153.33 |
| Pure restitution | 174.48 | -0.206 | 134.08 | 170.96 | 155.38 |
| Mean by PWM | 160.35 | -0.021 | 125.56 | 158.33 | 147.89 |
| PWM + incoming speed | 160.54 | -0.024 | 125.59 | 158.72 | 147.06 |
| PWM + pre-entry history | 158.15 | 0.005 | 122.65 | 156.61 | 147.29 |
| PWM + incoming speed + entry depth | 160.63 | -0.025 | 125.58 | 158.82 | 147.04 |
| PWM + incoming speed + entry geometry | 160.64 | -0.025 | 125.29 | 158.86 | 146.88 |
| PWM + history + entry geometry | 158.17 | 0.005 | 122.43 | 156.57 | 147.06 |
| Selected on other recordings’ validation tails | 158.13 | 0.006 | 122.51 | 156.61 | 147.08 |

## Contributions to prediction error

| Compared model | Reference | RMSE reduction | Within-recording 95% interval |
|---|---|---:|---:|
| Mean by PWM | One training mean | 6.20 | 3.89 to 8.42 |
| PWM + pre-entry history | Mean by PWM | 2.20 | 0.50 to 3.99 |
| PWM + incoming speed + entry depth | PWM + incoming speed | -0.09 | -0.25 to 0.06 |
| PWM + incoming speed + entry geometry | Mean by PWM | -0.29 | -0.74 to 0.17 |
| Selected on other recordings’ validation tails | Mean by PWM | 2.22 | 0.48 to 4.04 |
| Selected on other recordings’ validation tails | Pure restitution | 16.35 | 13.05 to 19.74 |

## Variation across recordings

| Recording | Events | Mean observed speed | Observed standard deviation | History-model bias (prediction minus observation) | History-model RMSE |
|---|---:|---:|---:|---:|---:|
| N30_PWM10_D41 | 539 | 84.78 | 154.28 | +16.38 | 150.94 |
| N40_PWM10_D40 | 196 | 72.19 | 148.89 | +27.87 | 148.19 |
| N40_PWM10_D41 | 200 | 103.20 | 131.44 | -25.52 | 134.59 |
| N40_PWM20_D40 | 444 | 181.99 | 186.85 | -33.38 | 187.19 |
| N40_PWM20_D41 | 509 | 152.03 | 172.17 | +15.72 | 169.84 |

## Evaluation and interpretation

Each fold trains on the first three quarters of the other four recordings, with equal total weight per recording. Validation on their last quarter chooses ridge strength (0, 0.03 or 0.3 on standardized features) for each history model and selects the model for the last row. Every other row evaluates the named feature set. RMSE and R² are computed within each held-out session and averaged over the five. The paired intervals resample 10-second blocks in each recording 2,000 times, retaining equal recording weights and fixed fitted models.

PWM is the known motor setting (10 or 20). All history features end one frame before entry. Slopes use backwards differences over 40, 120 and 400 ms, shortened at the recording start or a tracking flag. The remaining predictors are absolute tangential speed, spin magnitude, signed tangential speed times summed spin, relative heading along the incoming pair axis and proximity to a third robot. The earlier restitution baseline uses the original entry axis and its published eligibility.

Entry geometry adds the depth below one diameter at the first threshold crossing and the speed of the crossing interval. The entry position is available at forecast time and also starts the measured velocity interval. The delayed check therefore refits all models to the next 40 ms interval, keeping the original normal fixed. Its position difference shares neither endpoint with the entry geometry. Both matched columns use the same events with available, quality-screened later measurements. Their physical timing differs.

The model predicts an observed event outcome conditional on a sampled crossing. A measured PWM association can include differences between sessions. Formal verification of instantaneous restitution requires independently timed physical contacts, enough samples on both sides of contact to estimate velocity uncertainty, and evaluation on new recordings.

[Per-recording scores](by_recording.csv) · [Fold models and fitted coefficients](fold_models.csv) · [Event predictions](predictions.csv.gz) · [Paired comparisons](paired_differences.csv)
