# Fit-aware single additions in live discovery across the default 15 systems

Keep native addition as the default at this budget. Changing ordinary addition to the historical residual-guided operator (PVRA) gained one held-out law across 60 paired law–seed cases and lost one complete system across 45 dataset–seed cases. The same operator accelerated some discoveries and delayed others. [Paired results by dataset](by_system.csv)

The 15-system comparison isolated PVRA. Earlier experiments also used exact two-objective additions (PVXA), instability-weighted removals (PVXD), forward–backward moves (PVFB), joint pair additions (PVPA), and local search on front equations (PVLS). PVLS already implemented a guided swap by choosing the outgoing term by instability and ranking replacements after removing it. The open experiment is to isolate that swap as an ordinary mutation on the current pipeline. [Historical campaign table](C:/Users/NSSLab/AppData/Local/Temp/claude/C--Users-NSSLab-PycharmProjects-EPDE/c5a71b1a-1530-4755-b8a8-0feb3924d241/scratchpad/auto/results_all.txt) and [archived PVLS swap](C:/Users/NSSLab/AppData/Local/Temp/claude/C--Users-NSSLab-PycharmProjects-EPDE/c5a71b1a-1530-4755-b8a8-0feb3924d241/scratchpad/auto/arm_pvls.py)

## Live searches

Both arms used plain-mode NSGA-II, fourth-order preprocessing, native weak least-squares weights, WAPE and dyad. Native beam fill and pruning stayed active in both searches. PVRA sampled among the eight largest admissible single-term residual gains conditional on the current equation, with 10% ordinary exploration. The arms received identical noisy fields, seeds, weak plans and frequency settings. Three seeds per dataset and 3,000 completed equation fitness calls per arm gave 270,000 calls in the primary suite. The separate ode_simple control contributed another 18,000 calls.

Each search saw only the first half along the longest sampled axis. All candidate equations used the preceding study's training-only window plan. The opposite half used frozen coefficients, exact token frequencies and the same weak geometry. Its two Pareto coordinates were held-out WAPE and training dyad. Each held-out WAPE comparison required native source eligibility and held-out WAPE at most one. The reporting archive includes every scored active relation and separately records complete systems the plain fitter handed back. Reconstructed fronts cover the fixed evaluation prefix. They measure what was available by that budget, including candidates that left the population. [Prospective protocol](protocol.txt)

| Fixed-budget endpoint | Native addition | Fit-aware addition | Paired gains | Paired losses |
|---|---:|---:|---:|---:|
| Individual law scored | 40/60 | 41/60 | 2 | 1 |
| Individual law on training front | 28/60 | 28/60 | 2 | 2 |
| Individual law on held-out front | 32/60 | 33/60 | 2 | 1 |
| Complete system assembled | 27/45 | 26/45 | 0 | 1 |
| Complete system on training front | 15/45 | 13/45 | 0 | 2 |
| Complete system on held-out front | 19/45 | 18/45 | 0 | 1 |

The individual-law front is computed within each EPDE slot. A benchmark law receives one hit if it survives in an allowed orientation in any slot. A complete system requires the distinct laws to occur together in one actual candidate. Its front uses ordinary componentwise dominance across all slots. For a scalar equation, the plain fitter can score a version and then discard it before handing back a candidate. This also accounts for differences between individual-law and assembled-system counts. The endpoints on this suite had the same totals under literal and registered-equivalent support matching. [Law outcomes](laws.csv) and [assembled systems](systems.csv)

The guided arm gained the second Lorenz equation and the Navier–Stokes x-momentum equation on seed 923302. It lost inviscid Burgers on seed 923301. All three changes carried through to the held-out individual-law front. Neither arm assembled the full Lorenz or Navier–Stokes system on any seed. Both assembled Lotka–Volterra on all three and retained it on all three held-out system fronts.

Both arms found 39 of the same law–seed cases. Among them, median first discovery was call 470 with native addition and 378 with guided addition. Guidance was earlier in 19 cases, later in 16, and tied in four. The guide installed 10,554 additions and screened 2,870,017 extra term columns across intercept states. The median paired runtime ratio was 1.13. Elapsed time is approximate because searches ran concurrently and the Duffing retries had a different concurrent load. [Summary](summary.json)

## Where the remaining misses came from

- **Proposal.** Of 19 guided cases in which the law was never scored, seven had every required term observed somewhere and a scored equation one term replacement from a registered law. These were inviscid Burgers on three seeds, the second Lorenz equation on two, Navier–Stokes y-momentum on one and nonlinear diffusion on one. Four also had a one-addition neighbor and three had a one-deletion neighbor. This counts algebraic edits among scored supports. The search-time probability of selecting an eligible parent and the rank of the useful replacement still need measurement. Forced KdV was two edits away on all three seeds, although every required term appeared individually. [Reachability diagnostics](reachability.csv)
- **Frequency representation.** On all six guided Duffing and forced-oscillator cases, the training-derived frequency grid missed the registered forcing frequency by more than the 1% relative matching tolerance. Duffing's nearest peaks were 1.076–1.080 for a unit frequency. The forced oscillator's were 2.054–2.062 for frequency 2. Every forced-oscillator run in the guided arm did score the correct term pattern when frequency was ignored. Constant and frequency parameters were never re-fitted on the held-out observations.
- **Ranking.** The guided arm scored 41 laws and retained 33 on the held-out front. Seven were dominated and one exceeded the held-out WAPE cutoff. Both arms generated Allen–Cahn and wave on all three seeds. All three Allen–Cahn laws and two wave laws were dominated on validation, and the remaining wave law exceeded the cutoff. More proposals cannot by themselves improve the measured objectives of those already-scored supports.

## Convergence of complete systems across operators

The same 45 dataset–seed pairs at 5% noise have traced handover costs under four operators. At each checkpoint the cells show complete systems assembled / complete systems on the held-out first archive front. At 1,000 total objective fits residual-guided addition had assembled four more systems than native and retained one more on that front. At 2,000 fits native was ahead on both endpoints. Among the 26 pairs in which both arms assembled the system by 3,000 fits, residual guidance was earlier in 15, later in eight and tied in three. The median first-assembly costs within that common set were 659.5 fits with native addition and 301.5 with guidance. These are conditional times. The checkpoint counts include all 45 cases. [Paired first assembly](system_convergence.csv) · [Paired front membership](system_front_convergence.csv)

| Operator | 500 fits | 1,000 fits | 2,000 fits | 3,000 fits |
|---|---:|---:|---:|---:|
| Native | 13 / 11 | 17 / 15 | 26 / 19 | 27 / 19 |
| Residual-guided addition (PVRA) | 16 / 12 | 21 / 16 | 24 / 18 | 26 / 18 |
| Guided swap | 12 / 10 | 16 / 14 | 21 / 18 | 25 / 19 |
| Two-objective addition (PVXA) | 10 / 8 | 12 / 9 | 19 / 14 | 22 / 15 |

Run `system_convergence.py` to reproduce the counts from the matched archives. Each timestamp uses the first handover of the actual complete candidate. The total-fit clock charges auxiliary objective evaluations. The projection work used to screen individual columns adds elapsed time outside that clock.

## Separate oscillator control and checks

The catalog's forced oscillator, `ode`, is a different dataset from `ode_simple`. Both arms recovered the literal simple harmonic oscillator on the training and held-out fronts in all three additional runs. The structural check allowed the registered phase and energy identities and explicit additive constants. Symbolic evaluation also verified an identity at the literal frequencies on every control front.

The Duffing file stores its uniform time coordinates at float32 precision. Its rounded steps tripped the native uniform-grid check. Reconstructing the uniform grid from its endpoints reproduced every stored value on conversion back to float32 and changed a coordinate by at most 9.16e-7 time units. Both arms used that grid. The observed field values were preserved. All 96 archives reached exactly 3,000 fitness calls. Independent weak-matrix replay reproduced every eligible source WAPE to a maximum absolute error below 1.8e-15.

These are short searches on three noise and search seeds per trajectory. The structural advantage from single additions is too small here to justify their extra screening as a general default. The next comparison should change only the replacement move. Rank candidate replacements conditional on the retained terms, use native training data for proposal, and measure generation and complete-system validation across the same 15 systems. The seven one-replacement misses give a diagnostic for the mechanism. Separate frequency refinement would address a different observed failure.

Run `live.py`, `assess.py`, `misses.py` and `summarize.py` in order to reproduce the tables. The archived chooser has SHA-256 `9b2fc7c4261485f9e1afffd78c3e944b956154df19c6926f7defebb7c5fdc0b7` and can be located through `EPDE_HISTORICAL_OPERATOR_DIR`.
