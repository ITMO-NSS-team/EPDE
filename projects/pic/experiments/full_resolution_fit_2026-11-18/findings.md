# Full-resolution weak coefficient fitting across 15 systems

Preserving the weak window improved single-record front retention with both estimators. Choosing the holdout axis from grid geometry eliminated the unresolved blocked fits on twelve fresh recordings per system. Homogeneous total least squares (TLS) gave a modest structural gain over written-target weighted least squares (WLS). Its wave coefficients and Allen–Cahn controls still leave reasons to keep WLS as the production baseline.

## Matched coefficient comparison

Both estimators used the same raw full-resolution weak rows, constant columns, training normalization, test observations, candidate panel and directional dyad. The screen and confirmation each comprise six noise anchors per system, 20 equation slots and 120 slot–anchor cases. A hit means that a registered equation or equivalent survived the whole nondominated front of its slot. The two test recordings used for cross-products have independent measurement noise on the same trajectory as the training recording.

| Test | Screen WLS | Screen TLS | Confirmation WLS | Confirmation TLS |
|---|---:|---:|---:|---:|
| Full training recording, independent test recordings | 96 | 98 | 97 | 103 |
| One recording, disjoint time halves, held-out squared energy | 91 | 95 | 93 | 96 |
| Fit one time half, cross-product on the other half of independent recordings | 91 | 90 | 91 | 92 |

On the full-record confirmation comparison, TLS gained eight cases and lost two relative to WLS. Six gains came from nonlinear diffusion, (u_t=u_x^2+u u_{xx}). The median fitted coefficients were (0.011, 0.235) for WLS and (1.359, 1.003) for TLS, against (1, 1). The full law moved from 0/6 to 6/6. On five WLS anchors an extension by a cosine term dominated the reference. Across all 120 cases, the median reference coefficient angle in clean weak-column units was 0.323° for WLS and 0.368° for TLS. The registered support and the accuracy of its coefficients are separate outcomes.

On the time-half confirmation comparison TLS gained four cases and lost one using squared energy. It also retained 137 unregistered one-term extensions, versus 112 with WLS. The thin time-window lattices made one time/field candidate unresolved on Allen–Cahn, wave and inviscid Burgers for WLS (18 candidate instances) and on Allen–Cahn and wave for TLS (12). On inviscid Burgers the formally unique TLS fit assigned negligible normalized weight to the written target in all 24 half fits across the two cohorts. A short block can admit a relation among the time columns alone. Bounding the missing scores at both extremes gave the same registered recovery counts.

## Choosing the split axis while preserving the original window

An exploratory replay on the first twelve anchors traced a wave-coefficient error to shortening the time window for the half-block experiment. On the six confirmation anchors the full-record median wave coefficient changed from 0.0663 to 0.0395 for WLS and from 0.2614 to 0.0441 for TLS when using the original training window. The physical coefficient is 0.04.

A subsequent comparison used twelve new single-record anchors per system. Before processing them we fixed a rule that keeps the original window and stride, splits along a feasible grid axis, and maximizes the smallest number of distinct window centers across dimensions in either half. Ties use row count and then axis order. The rule chose an axis for all 15 systems.

| Single-record fit and validation | Registered / 240 | Exact / 240 | Added supports on fronts | Unresolved candidate instances |
|---|---:|---:|---:|---:|
| WLS, time split with window cap | 187 | 162 | 227 | 36 |
| WLS, split preserving window | 195 | 171 | 247 | 0 |
| TLS, time split with window cap | 195 | 170 | 258 | 24 |
| TLS, split preserving window | 200 | 176 | 251 | 0 |

With the geometry-based split, TLS gained eight cases relative to WLS and lost three. Relative to time splitting, WLS gained eight with no losses. TLS gained six and lost one. For wave, both methods retained the law on all 12 fresh recordings under every split. Holding the spatial split fixed, restoring the original time window moved the median coefficient across the 24 training halves from 0.0876 to 0.0396 for WLS and from 0.3356 to 0.0470 for TLS.

On Allen–Cahn both estimators favored retaining diffusion on 7/12 diffusion-bearing recordings with the preserved window. WLS also favored it on 3/12 matched zero-diffusion recordings and TLS on 4/12. This is a head-to-head discrepancy comparison, distinct from front survival. The forced KdV reference or equivalent survived on 1/12 single-record WLS fronts and 0/12 TLS fronts under the geometry rule. Both remain open cases.

## Scope and reproducibility

Repeated recordings change observation noise on the existing trajectories. The search spaces here are frozen benchmark panels. The method has yet to face fresh initial conditions or the competitors produced by a live evolutionary run. The weak plans are taken from the archived seed-0 panel, which is distinct from the live search's per-record window selection. During the matched comparison the code replayed the original weak matrix and directional reference dyad for each default system. Independent calculations reproduced 26,040 scores, checked 1,440 slot/method/split Pareto summaries and verified 36 constant-bearing `ode_simple` identity cases. The separate axis experiment also checked every finite held-out block energy by a second residual calculation.

The first comparison uses `study.py`, `pareto_report.py`, and `verify.py` in that order. `window_control.py` replays the changed windows on the original anchors. `axis_test.py` runs the fresh single-record comparison. Protocols, per-candidate coefficient vectors and scores, slot outcomes, control margins and source-recording hashes are stored alongside this report.
