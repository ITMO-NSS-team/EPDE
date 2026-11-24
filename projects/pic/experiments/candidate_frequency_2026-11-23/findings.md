# Refining frequencies in discovered equations

**The frozen rule recovered 79/180 complete systems on 12 fresh paired seeds per default dataset, compared with 72/180 for native search.** All seven gains came on the forced oscillator. There were zero paired losses, and the result held at both budget endpoints. The separate oscillator control scored 11/12 in each arm. The additional initial-condition test yielded 3/6 for each arm. These results support using the refinement when a discovered equation contains an uncertain trigonometric frequency. The evidence for a recovery gain is concentrated on one trajectory, so the implementation stays in the research harness. [Confirmation by dataset](confirmation/by_system.csv) · [New initial conditions](initial_conditions/systems.csv)

## Screening

On the three paired screening seeds, refining a frequency inside a discovered equation recovered **21 of 45 complete systems**, compared with **19 of 45** for native search. The same two pairs improved at 3,000 completed evaluations and 3,000 total fits. Both gains came on the forced oscillator. The other 14 datasets kept their native recovery counts, and both arms recovered the separate oscillator control on all three seeds. [Paired results and front sizes](by_system.csv)

| Forced oscillator seed | Refined frequency in the discovered law | Held-out complete recovery |
|---|---:|---|
| 923301 | 1.95570 | No |
| 923302 | 1.98260 | Yes |
| 923303 | 1.98797 | Yes |

The registered frequency is 2. The benchmark accepts 1.98–2.02, so 1.95570 misses by more than the allowed 1%. On all three seeds the correct term pattern appeared in a fitted parent by the checkpoint. Duffing reached no such parent on these seeds. In the earlier experiment, moving frequencies for the entire library before discovery yielded 19/45. [Forcing audit](forcing.csv) · [Pre-search refinement](../weak_frequency_refinement_2026-11-23/findings.md)

On the seed that missed the frequency criterion, the frozen equation's held-out WAPE still fell from 11.08% to 6.14%. On the two recovered seeds it fell from 13.57% to 2.83% and from 22.61% to 18.06%. These comparisons use the same pivot and term pattern in each actual parent and child. [Measured parent and child errors](forcing_profiles.csv)

With the failed seed's discovered structure fixed, 250 new draws of 5% measurement noise produced a central 95% frequency range of 1.9465–2.0305. The original 1.9557 estimate is inside that range. A separate nested-grid simulation quantified the clean discretization effect and the change in noisy precision at a fixed physical window width. [Measurement and discretization study](frequency_uncertainty.md)

The search chose parents from the first 2,400 completed evaluations using training WAPE and dyad within each slot and predictor count. It profiled 492 parent equations across the 45 default runs, using 6,518 extra weak coefficient fits. Native scoring admitted 786 parent-specific child variants to the archive. The original population continued from its native parents. All 48 screen comparisons passed the training-data, window, starting-grid and native-prefix checks. Independent reconstruction of eligible training WAPE agreed within 1.67 × 10⁻¹⁵. [Protocol](protocol.md) · [Work](work.csv) · [Archive verification](manifest.json)

Run `candidate_trial.py` and then `evaluate_candidates.py` from this directory in the repository's research Python environment. Both accept `--confirmation` for the frozen 12-seed comparison. Each run's compressed archive records the selected parent, exact frequency and bounds, trained residuals, completed fit count and the actual child systems. The assessment also records held-out front membership, time to first scored law, coefficient error and the observed dominators of each dominated law. [Law-level results](laws.csv)

## Fresh-seed confirmation

The forced oscillator improved from 0/12 to 7/12. Its correct term pattern appeared in an actual parent by the checkpoint on 11 seeds and reached the shortlist on nine. Duffing supplied zero matching parents by the checkpoint. Across the default suite, individual held-out laws rose from 120/240 to 128/240. The rule profiled 1,984 actual parent equations, charged 26,626 auxiliary fits, and admitted 3,035 child variants. All 192 pairs, including the control, passed prefix, training-data and weak-window verification. Independent reconstruction of training WAPE agreed within 2.11 × 10⁻¹⁵. [Forcing audit](confirmation/forcing.csv) · [Run-level budgets and fronts](confirmation/systems.csv) · [Verification](confirmation/manifest.json)

## New initial conditions

Both arms recovered the forced oscillator on all three seeds starting at displacement 1 and velocity 0, and on zero of three starting at displacement 0 and velocity 1. The two new, independently simulated trajectories therefore yielded **3/6 for each method** at either budget. On the second trajectory, the correct term pattern appeared in a fitted parent by the checkpoint on one seed. The shortlist passed over that parent. Neither of the other two seeds supplied such a parent before the checkpoint. [Held-out recovery and front sizes](initial_conditions/systems.csv) · [Parent-selection audit](initial_conditions/forcing.csv)

The other roadmap comparisons are documented with their own paired results. Guided replacement tied native swap, two-objective addition lost three complete recoveries against gain-guided addition, and donor-block crossover tied native crossover. [Guided replacement](../guided_swap_2026-11-23/findings.md) · [Two-objective addition](../two_objective_add_2026-11-23/findings.md) · [Coupled parent and offspring traces](../coupled_lineage_2026-11-23/findings.md)
