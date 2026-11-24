# Experiment A · Guided replacement in live EPDE discovery

**Native and guided swaps each recovered 19 of 45 complete systems on the held-out nondominated front.** Guidance gained Lorenz on seed 923301 and lost `pde_divide` on seed 923303. The same pairs changed at 3,000 full discovery evaluations and at 3,000 total objective fits. The confirmation rule required more gains than losses, so the guided swap stays in the research harness and the three-seed screen closes Experiment A. [Paired system results](systems.csv) · [Protocol](protocol.md)

| Endpoint at 3,000 total fits | Native | Guided | Paired gains | Paired losses |
|---|---:|---:|---:|---:|
| Complete system on held-out front | 19/45 | 19/45 | 1 | 1 |
| Complete system on training front | 15/45 | 15/45 | 2 | 2 |
| Complete system assembled | 27/45 | 25/45 | 1 | 3 |
| Individual law on held-out front | 32/60 | 31/60 | 2 | 3 |
| Individual law on training front | 28/60 | 26/60 | 2 | 4 |
| Individual law scored | 40/60 | 38/60 | 3 | 5 |

These aggregate counts also held at the full discovery budget. Each coupled-system hit required the distinct registered laws in one actual candidate handed back by the fitter. Fronts include all eligible nondominated candidates discovered by the budget, including those that later left the population. The separate `ode_simple` control succeeded on all three seeds in both arms under both budgets. Each control system front also contained an identity verified symbolically at its literal frequencies. The registry permits phase and energy identities with explicit constants. [Per-system results and front sizes](by_system.csv) · [Control](control.csv)

## What happened on the two discordant pairs

**Lorenz, seed 923301.** Native search scored two of three laws. Guided search scored all three, assembled the system and retained it on a held-out system front of 63 candidates. Its second law was dominated in its own slot. One observed competitor retained the same target, substituted the second derivative of *u* for the linear *v* term, and reduced training WAPE from 0.01780 to 0.01576, held-out WAPE from 0.02360 to 0.02232, and dyad from 0.16956 to 0.13803. The full Lorenz candidate survived the comparison across all three slots. The useful replacement reached the shortlist at three swap opportunities and was selected zero times. [Law outcomes](laws.csv) · [Observed laws, coefficients and dominators](domination_replay.jsonl.gz)

**`pde_divide`, seed 923303.** Native search scored the law at call 1,149 and retained it on validation. Guided search visited two suitable swap parents. It removed the needed outgoing term once. The completing replacement ranked first with a draw probability of 0.144, and the draw selected another term. Guided search finished without scoring the law. [Swap trace](swap_trace.jsonl.gz)

## What the swap trace explains

Across the seven one-replacement misses identified in the earlier PVRA campaign, only three guided-swap runs actually applied an ordinary swap to a suitable parent within the total-fit budget. A suitable parent here has a determined, retained target and can reach a registered support by replacing one admissible feature. The native fit considers the allowed intercept states after mutation. [Seven historical misses, matched by law and seed](historical_misses.csv)

| Prior miss | Seed | Suitable parents | Correct removals | Useful shortlists | Selected completions | Held-out law |
|---|---:|---:|---:|---:|---:|---|
| Inviscid Burgers | 923301 | 26 | 26 | 26 | 6 | On front |
| Inviscid Burgers | 923302 | 3 | 3 | 3 | 0 | Never scored |
| Inviscid Burgers | 923303 | 0 | 0 | 0 | 0 | Never scored |
| Lorenz, second law | 923301 | 3 | 3 | 3 | 0 | Dominated |
| Lorenz, second law | 923303 | 0 | 0 | 0 | 0 | Never scored |
| Navier–Stokes, y-momentum | 923303 | 0 | 0 | 0 | 0 | Never scored |
| Nonlinear diffusion | 923303 | 0 | 0 | 0 | 0 | Never scored |

Across all 60 guided law–seed cases at equal total fits, 22 laws were never scored. Fourteen had no suitable swap opportunity, four reached the correct removal but the replacement missed the positive top eight, and four reached the shortlist without selection. Seven further laws were scored and then dominated on validation. The trace identifies the stage each case reached. It records shortlist ranks and gains and does not assign ranks below eighth. [All law and swap outcomes](laws.csv)

## Discovery time, accuracy and cost

Among 35 law–seed cases scored by both arms, median time to first discovery was 289 total objective fits with native swap and 282 with guidance. Guidance was earlier in 14 pairs, later in 18 and tied in three. Median relative coefficient error at first discovery was 0.64% in each arm. This compares the fitted coefficient vector with the catalog equation after orienting both on the fitted target, over cases with a shared discovery. [Law-level times and coefficient errors](laws.csv)

Guidance completed 135,000 discovery evaluations across the primary suite and used 8,949 additional native objective evaluations for parent fits. At equal total fits, it completed a median of 2,809 discovery evaluations per run, with a range of 2,714–2,873. Across the full runs it screened 3,532,491 candidate columns counted across intercept states, reused 5,086 fitted instability vectors and 3,015 cached parent-version lookups, and hit the weak-column cache on 3,314,413 of 3,337,438 requests. Median paired elapsed time was 1.23 times native in concurrent runs. [Per-run work](work.csv) · [Budget comparisons](summary.json)

## Reproduction and checks

Run `swap_trial.py`, `assess_swap.py`, then `report.py` with the repository's research Python environment. The runner uses the historical operator directory resolved by the preceding study, configurable via `EPDE_HISTORICAL_OPERATOR_DIR`. Each worker starts a fresh process. Guidance uses the slot's training weak columns and the archived instability and residual-gain rules. Every auxiliary native WAPE–dyad evaluation is charged, including evaluations of parent versions that are subsequently discarded. The equal-fit system front uses the cost at the first actual handover of each combination.

All 48 traced native runs reproduced their archived settings, data hashes, weak plans, frequency selection, scored equations and actual system combinations. All 96 searches reached 3,000 full discovery evaluations. Independent weak-matrix replay matched eligible training WAPEs to a maximum absolute error of 1.67 × 10⁻¹⁵. Held-out evaluation froze coefficients and token frequencies. The three paired seeds vary observational noise and search randomness on each existing trajectory. Transfer to new initial conditions remains a separate endpoint. [Run and operator hashes](manifest.json)

The compressed swap trace contains one term dictionary per run followed by its events. Each event records the slot, full parent, determined target when available, admissible parent features, outgoing term, ranked shortlist with gains, selected term, resulting child, and both budget clocks. Native random swaps have an empty shortlist. The domination replay stores exact factor parameters, coefficients, targets, both WAPEs and training dyad for each lost law and all its observed held-out dominators. [Swap trace](swap_trace.jsonl.gz) · [Domination replay](domination_replay.jsonl.gz)
