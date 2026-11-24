# Experiment B · Does fitting the addition shortlist pay?

**At 3,000 total objective fits, PVRA recovered 18 of 45 complete systems on the held-out nondominated front and PVXA recovered 15.** PVXA gained zero paired cases and lost three. This falls short of the prespecified threshold for twelve-seed confirmation. Both arms retained the harmonic oscillator control on all three seeds. [Paired systems](systems.csv) · [Control](control.csv) · [Protocol](protocol.md)

| Endpoint | Budget per run | PVRA | PVXA | Paired gains | Paired losses |
|---|---|---:|---:|---:|---:|
| Complete system on held-out front | 3,000 total fits | 18/45 | 15/45 | 0 | 3 |
| Complete system on held-out front | 3,000 discovery evaluations | 18/45 | 17/45 | 1 | 2 |
| Complete system assembled | 3,000 total fits | 26/45 | 22/45 | 0 | 4 |
| Individual law on held-out front | 3,000 total fits | 33/60 | 29/60 | 0 | 4 |
| Individual law on held-out front | 3,000 discovery evaluations | 33/60 | 30/60 | 1 | 4 |

PVRA samples among the eight highest positive residual gains in an ordinary addition. PVXA fits those shortlisted supports and samples terms with a version on the local WAPE–dyad front. Both use the same training-only shortlisting rule, 10% ordinary exploration, native weak least squares, fill-and-prune and search settings. Each run supplies its own sequence of parents. All 48 pairs produced identical first nonempty shortlists and gains. The traced PVRA runs reproduced the earlier comparator's data hashes, settings, weak plans, frequency grids, scored equations and actual system combinations. [Run manifest](manifest.json)

## Cost and recovery

PVXA made 98,692 auxiliary objective evaluations in addition to 135,000 discovery evaluations across the main suite. Of 83,400 shortlisted candidates, 54,332 required a fresh fit attempt, 17,254 reused a previous shortlist fit and 11,814 reused an ordinary discovery fit. Under the total-fit cap, PVXA completed a median of 1,668 discovery evaluations per run, ranging from 1,367 to 1,992. Median elapsed time to finish the full discovery budget was 1.47 times PVRA in concurrent runs. [Projection, cache and timing records](work.csv)

Three paired losses at equal cost explain the primary result.

- **Lotka–Volterra, seed 923303.** PVXA retained both individual laws on their held-out slot fronts. Their first discoveries cost 191 and 169 total fits. The search never handed back a candidate containing both, even at the full discovery budget.
- **`pde_divide`, seed 923301.** PVRA scored the law at fit 549. PVXA reached it at discovery call 1,793, which cost 3,411 total fits. The law survived on the held-out front at the larger budget.
- **`pde_divide`, seed 923303.** PVRA scored the law at fit 181. PVXA never scored it. No ordinary addition received a parent one admissible addition away from the reference support.

The one full-budget gain was inviscid Burgers on seed 923301. Its first PVXA discovery cost 3,618 total fits. Per-dataset results include held-out front sizes and all paired regressions. [Results by dataset](by_system.csv) · [Law-level outcomes](laws.csv)

## What the second stage changed

The highest-gain term lay on 9,022 of 10,412 nonempty PVXA local fronts. Across those same proposals, a gain-proportional draw would have assigned an average of 37.1% of its probability to the terms PVXA kept eligible. The local front therefore changed which edits could be selected. It used enough additional evaluations to reduce discovery opportunities substantially.

At equal total fits PVXA never scored 24 law–seed cases. Nineteen had no suitable addition parent, four had a suitable screened parent whose completing term missed the shortlist, and one had a selectable completion that was never drawn. Seven further laws were scored and then dominated on validation. Among the 36 laws scored by both arms, median cost to first discovery was 371 fits for PVRA and 565.5 for PVXA. Median relative coefficient error on these paired first discoveries was 0.65% in both arms. [Addition traces with local scores and draw probabilities](addition_trace.jsonl.gz) · [Summary](summary.json)

All 96 runs reached their discovery budget. Frozen weak-matrix replay reproduced eligible training WAPE to a maximum absolute error of 1.78 × 10⁻¹⁵. These are three seeds on existing trajectories. The next planned experiment with a directly observed failure is training-only frequency refinement. The configured libraries missed the registered forcing frequency on every Duffing and forced-oscillator seed. Test the same refinement rule across all 15 systems and freeze the refined frequency and coefficients before held-out scoring. [Earlier frequency audit](../live_operator_suite_2026-11-23/findings.md)
