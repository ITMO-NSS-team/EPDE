# Coefficient sensitivity as a weak-window selection rule

The selector chose a second geometry on 27 of 84 fresh recordings from the fourteen transfer systems. The registered law reached the front on one additional Allen–Cahn recording. Repeatability improved for four of the nineteen equation slots, worsened for seven and stayed the same for eight. The observed benefit is too narrow to justify changing the default.

Each system supplied one regular-grid trajectory. Six archived and six fresh 5% noise draws gave 180 paired recordings including forced KdV. Each candidate-independent window decision used training fields and the allowed derivative orders. Every equation candidate had identical row counts across the two fits, and frozen coefficients were evaluated on identical disjoint validation rows. A hit means a registered equation or equivalent from the fixed candidate panel survived on the held-out-error and training-dyad Pareto front.

| Fourteen transfer systems | Archived native | Archived rank rule | Archived sensitivity rule | Fresh native | Fresh sensitivity rule |
|---|---:|---:|---:|---:|---:|
| Second geometry used, out of 84 | 0 | 4 | 32 | 0 | 27 |
| Registered law on front, out of 114 | 109 | 109 | 110 | 110 | 111 |
| Literal law on front, out of 114 | 104 | 104 | 105 | 103 | 105 |
| Unregistered one-term additions on fronts | 170 | 171 | 161 | 156 | 157 |
| Median scaled coefficient spread over 19 slots | 0.02365 | 0.02365 | 0.02303 | 0.02561 | 0.02509 |
| Median scaled coefficient error over 19 slots | 0.01475 | 0.01475 | 0.01475 | 0.01390 | 0.01283 |

Both additional registered hits were Allen–Cahn, one in each cohort. The other thirteen transfer systems had identical registered-hit counts for the native and sensitivity rules. All 114 reference fits in each cohort were available. Among the fresh draws, truth-relative coefficient error improved on six slots and worsened on four. The sensitivity rule replaced 2,243 of 13,446 fresh fitting rows. It used 6.81 process-seconds across the fresh transfer recordings, including selection, versus 2.88 for native fitting. Both timing totals exclude their shared native-plan build. The worker used a 2% host CPU hard cap, one logical CPU and one numerical thread.

## Forced KdV

The sensitivity rule selected a second geometry on four of six fresh draws. Both arms retained the law once. On archived draws the sensitivity rule lost the native arm's one hit. The catalogued forcing function was supplied to the fixed-panel fit.

To locate the source of the dispersion error, replay the six fresh fits on their frozen training windows with clean benchmark values for selected columns. The true coefficient of `u_xxx` is −1. The values below are medians of physical coefficients across the six draws.

| Measurements in the fit | Native geometry | Sensitivity-selected geometry |
|---|---:|---:|
| All noisy | −0.0211 | −0.0216 |
| Clean `u_xxx`, remaining columns noisy | −1.0033 | −1.0015 |
| Noisy `u_xxx`, remaining columns clean | −0.0189 | −0.0190 |
| All clean | −0.9998 | −0.9998 |

Replacing only the dispersion column put the estimate within 0.008 of −1 on every fresh draw using the sensitivity-selected geometry. After projecting off the other clean predictors, the median norm of the dispersion-column error was 49.5 times the norm of the remaining clean dispersion signal. One draw had a ratio below one. This isolates a measurement problem for the weak dispersion column on this support, trajectory and noise level. The broad probe-stability criterion accepted windows that left that problem unresolved. The clean swap is an oracle diagnostic and requires benchmark information unavailable during discovery.

[Protocol](protocol.md) · [Selection code](uncertainty.py) · [Paired results](summary.json) · [KdV column-swap results](kdv_diagnostic.json)
