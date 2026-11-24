# Fit-aware pair additions on nonlinear diffusion

The archived pair-add operator (PVPA) recovered one late diffusion support that native search and residual-guided single additions (PVRA) missed. A guided single addition completed that support. It appeared on the held-out archive front and was dominated on the training front. Across the six paired noisy fields, changing the ordinary addition had little effect on returned fronts.

## Matched discovery runs

The comparison used identical raw observations, weak geometries, plain-mode coefficient fitting, fourth-order differences and native fill-and-prune in each seed and direction. Ten epochs produced at least 5,875 objective calls in all 36 runs. This shortest run set the common budget under the [prospective protocol](pair_protocol.md). The last column reports the actual returned training front at ten epochs. The archive columns include all eligible candidates scored through call 5,875. Held-out scoring freezes training coefficients and dyad, and recomputes WAPE on the disjoint half.

| Discovery half | Ordinary addition | Generated | Training archive front | Held-out archive front | Returned training front |
|---|---|---:|---:|---:|---:|
| Early | Native | 4/6 | 4/6 | 1/6 | 4/6 |
| Early | PVRA | 4/6 | 4/6 | 1/6 | 4/6 |
| Early | PVPA | 4/6 | 4/6 | 1/6 | 4/6 |
| Late | Native | 0/6 | 0/6 | 0/6 | 0/6 |
| Late | PVRA | 0/6 | 0/6 | 0/6 | 0/6 |
| Late | PVPA | 1/6 | 0/6 | 1/6 | 0/6 |

All arms found the same four early supports and retained the same seed on the held-out front. Within those four discoveries, median first scoring calls were 292.5 for native, 187.5 for PVRA and 155 for PVPA. Across all twelve full PVPA searches, 3,864 pair insertions required 14,283,360 pair-by-intercept-state gain evaluations. In 3,053 insertions at least one chosen term fell outside the single-term top eight. The median paired elapsed-time ratio to native was 1.26. Timing is indicative since the PVPA batch ran separately. The six seeds are noise realizations of one benchmark trajectory, and the two directions reuse the fields. [Per-run results](pair_results.csv)

## What produced the extra discovery

On late seed 923104 the last logged guided mutation before call 5,077 had target $u_t$, predictor $u_x^2$, and added the single term $u u_{xx}$. The following call scored $u_t=0.9372\,u u_{xx}+0.9036\,u_x^2$. A source-front equation of the form $u_t=c u^2$ dominated it. Their training WAPEs were 0.1313 for diffusion and 0.1296 for the reaction equation, and their dyads were 0.0658 and 0.0285. On the earlier half their frozen WAPEs were 0.0759 and 0.1210. This ordering allowed both candidates to survive on the held-out archive front. The native final population contained the diffusion equation below the first front.

Before the common cutoff there were 1,189 guided double additions with target $u_t$ on the late halves. None inserted the two physical predictors together. On the early halves 29 of 1,279 selected the physical pair, each alongside at least one existing predictor. The conditional pair ranking was also replayed from a target-only parent. Both physical terms passed the historical top-64 single screen on all twelve halves. Their joint rank was first on the four successful early halves, 459 and 488 on the other two, and 167 to 1,569 on the six late halves. The chooser samples only the first eight pairs. With the most frequently observed single off-law predictor for each late seed the joint rank ranged from 265 to 1,440. The target-only parent is a diagnostic construction. It did not occur in the logged double additions with this target. Direct weighted least-squares fits independently reproduced the joint gain of each screened physical pair. [Pair shortlist replay](pair_shortlist.csv)

## Scoring the known supports separately

After discovery we scored the supplied diffusion support and the observed reaction competitor with the native search width, weak windows, least-squares fit and dyad. The scores reproduced archived values wherever those supports had been scored. Comparing each diagnostic point with the existing PVPA fronts separates the effect of the objectives from whether a support was proposed.

| Discovery half | Fixed equation support | Nondominated against training archive | Nondominated against held-out archive |
|---|---|---:|---:|
| Early | Diffusion | 4/6 | 1/6 |
| Early | Reaction | 6/6 | 6/6 |
| Late | Diffusion | 0/6 | 3/6 |
| Late | Reaction | 6/6 | 6/6 |

On every late field, using the clean benchmark values on the exact same windows gave the diffusion equation both lower WAPE and lower dyad than the reaction competitor. With the known unit coefficients the median diffusion WAPE was 0.00052 on clean observations and 0.192 on noisy observations. Fitting those two coefficients to the noisy measurements gave median WAPE 0.170. Fitting each support on the clean field gave median dyads of 0.00042 for diffusion and 0.00946 for reaction. On the noisy fields these medians were 0.140 and 0.0119. Each comparison fixes the window geometry selected from the noisy training field. The clean comparison examines these two supports only. [Reference fits and front comparisons](reference_front.csv)

The diffusion and reaction supports provide a diagnostic for noisy derivative products. A change to EPDE's default requires a rule that runs without access to either equation and transfers across the default ODE and PDE suite. The [all-system follow-up](../general_weak_holdout_2026-11-23/protocol.txt) chooses geometry before reading candidate equations, then tests residual normalizations on the complete panels of 15 systems.

## Reproduction

Run `operator_trial.py --pair`, `evaluate_trial.py --pair`, `probe_pair.py` and `reference_front.py` from the experiment environment with the two earlier arms available. The archived pair chooser had SHA-256 `a3c2da045e8b104894aa89fca8333badac8931d99993368b12c333b9bad3514c`. The 36 source WAPE replays had maximum absolute error below 1.8e-15. The reference diagnostic loads the discovery configuration because the widest permitted equation controls the dyadic partitions.
