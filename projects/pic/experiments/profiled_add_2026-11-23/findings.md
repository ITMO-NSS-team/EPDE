# Duffing forcing-term addition replay

**Making standalone trigonometric terms available for addition put the missing cosine first on every genuine one-addition opportunity.** Optimizing all competing frequencies before ranking kept the cosine first on all seven seeds, reduced its mean probability of selection from 18.95% to 14.98%, and increased the total coefficient-fit count from 308 to 2,906. The matched frequency-profiling comparison missed the prespecified selection criterion. The experiment ended at replay. [Protocol](protocol.md) · [Paired rankings and probabilities](parent_comparisons.csv)

| Addition method | Missing cosine in top eight | Ranked first | Mean draw probability | Shortlisted within 1% of frequency 1 | Coefficient fits |
|---|---:|---:|---:|---:|---:|
| Ordinary enumerated addition library | 0/7 | 0/7 | 0% | 0/7 | 280 |
| Standalone sine and cosine allowed, training peaks fixed | 7/7 | 7/7 | 18.95% | 0/7 | 308 |
| Same candidates, frequencies fitted before ranking | 7/7 | 7/7 | 14.98% | 4/7 | 2,906 |

EPDE had already generated standalone trigonometric terms through its broader search. Its ordinary discrete addition library omitted them because a trigonometric token could not start a term. Allowing it to start a term added four standalone candidates on each pure parent, one sine and cosine at each training peak. Both matched arms competed all 43 admissible additions on each of these parents. The gain-proportional sampler gives every positive-gain candidate in its top eight a share of the probability. Profiling increased the relative weight of competing terms. The cosine's draw probability fell on every primary seed. [Library and ranker](ranking.py) · [Machine-readable summary](replay_summary.json)

## Frequency accuracy and held-out error

Each primary parent had the acceleration pivot and exactly the displacement, cubic displacement and velocity terms. Appending the first-ranked cosine at its fixed peak gave a median held-out weak WAPE of **19.61%**. Fitting its frequency and linear coefficients on training data and freezing them gave **2.85%**. Held-out WAPE fell on all seven seeds. An independent refit of the fixed shortlist's first-ranked candidate reproduced the seven frequencies from full-library profiling within 0.000003. The corresponding fit count, including ranking, was **392**. With the original stochastic sampler the expected count for profiling only its single draw was 369.15 across the seven parents, based on the actual shortlist probabilities and trial counts. These are conditional proposal results on actual archived parents. [Per-seed coefficients, frequencies, costs and held-out errors](frozen_proposals.csv)

For each parent, 250 independent draws from the benchmark's 5% observation-noise model were applied to the underlying stored signal. The trained window geometry, search bounds and proposed term structure were held fixed while the frequency and coefficients were refitted. Depending on the parent, **76–100 of 250** estimates fell within 1% of the registered frequency. The seven conditional central 95% ranges all included 1. The same noise seeds were used across parents, so the seven sets of draws are paired. [Conditional frequency distributions](conditional_noise_draws.csv)

## Scope and checks

All 12 native runs from seeds 923701–923712 reproduced the archived settings, training hashes, weak-window plans, 3,000 evaluations, labels, points, actual system combinations and both budget clocks. Seven runs supplied a genuine one-addition completion. Nine supplied a parent with the correct mechanical predictors and at most one distractor. Across the 22 distinct distractor-bearing parents, the cosine reached the top eight in both matched arms on all 22. Such parents still carry the extra term. The other three runs had no parent in either cohort. The retrospective catalog filter identifies opportunities. Proposal ranking uses training data alone. [Seed coverage](opportunities.csv) · [Replay implementation](replay.py)

The fit counts include the parent fit and each distinct conditional addition fit, including every tried frequency. The audit archived all candidate gains and charged frequency trials in each seed's compressed rankings file. Across 29 distinct parents, another 261 direct least-squares fits verified the projected shortlist gains. The held-out and noise diagnostics ran after ranking. To reproduce, run `python replay.py` and then `python validate_proposals.py` in this directory using the EPDE research environment. [Validation code](validate_proposals.py)
