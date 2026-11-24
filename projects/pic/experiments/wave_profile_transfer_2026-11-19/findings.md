# A changed initial wave profile separates several single-record competitors

The six saved confirmation searches per objective supplied all candidate fits. Each of the 18 live searches had scored the wave law, and nine retained it. In seven misses the final front dominated every recorded orientation. In two misses a previously scored orientation was undominated by the final front and absent. The saved logs establish this comparison at the end of the search. They do not record the generation in which an orientation disappeared. For native WAPE the affine spatial relation between the second spatial derivative and the field dominated the time-target wave equation on seeds 917402 and 917403. A relation between the first and third spatial derivatives dominated it on seed 917405.

Each cell counts observations on which the wave support survived on the pooled front after scoring fresh data. Each pool includes the three saved final fronts and both orientations of the known wave law. Each row has 96 paired observations from six fitted pools and sixteen noise patterns per pool. The initial displacement was changed by adding a standing mode to the benchmark. Original coefficient fits, target choices, frequencies, source dyads and window geometry were fixed throughout.

| Wave profile | WAPE | Blocked WLS | Blocked TLS | WAPE with blocked-WLS eligibility check |
|---|---:|---:|---:|---:|
| Original with fresh noise | 34 | 64 | 64 | 34 |
| Added third mode at benchmark speed | 96 | 81 | 79 | 96 |
| Added fourth mode at benchmark speed | 96 | 96 | 96 | 96 |
| Third mode at double speed | 63 | 0 | 0 | 0 |
| Fourth mode at double speed | 64 | 0 | 0 | 0 |

The last column is an exploratory combination tested after inspecting the three individual scores. It applies the existing blocked-WLS cutoff of one to every candidate, then takes the WAPE–dyad front among the eligible candidates. Both orientations of the reference exceeded the blocked-WLS cutoff on all 192 mixed-speed observations. Its WAPE eligibility held in every observation. The negative controls are deliberately strong. Neither their size nor this relative ranking establishes sensitivity to small changes in physical speed.

With only the candidates originally returned by the WAPE search, WAPE with the blocked check retained the wave support on 48/96 observations for each compatible profile and zero for each incompatible profile. Only three of six WAPE fronts contained the support to begin with. The other three require a candidate source upstream of the final front for a validation step to recover them. The next live test should record every distinct evaluated support and target with its weights, validate those candidates on an independent profile, and compare against validating only the final front on the same runs. Report the entire surviving nondominated set, including rival equations and its size. Fresh seeds and smaller changes of speed would test generalization.

No differential equation solver was used. The altered profiles use the known benchmark wave law to build validation controls. An application would need a separately observed trajectory to make the same cross-profile comparison. The paired noise repeats are conditional on six original fits. The in-recording control reproduced the earlier candidate-level scores for all 192 pooled support–target variants across those fits.
