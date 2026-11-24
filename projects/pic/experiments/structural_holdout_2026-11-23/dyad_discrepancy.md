# Cross-validated accuracy on the same slabs as stability

The tables on this page describe 5% added measurement noise. The clean reruns used the same search seeds and budget. [Clean and noisy results by dataset](fronts_by_noise.md)

At 5% added measurement noise, on 15 systems with 12 paired confirmation seeds each, the complete system appeared on the entire first Pareto front in **100/180** runs under ordinary full-record discrepancy and stability, two-slab CV and stability, and all-scale CV and stability. The respective median front sizes were **13, 7.5 and 7**. Against ordinary discrepancy on identical candidate pools, all-scale CV gained two front recoveries and lost two. Halves and all scales recovered the same runs. The contiguous held-out pipeline recovered **79/180** with a median front size of seven, using a separately discovered pool. [Results by dataset](dyad_discrepancy_by_dataset.csv) · [Holdout results](summary_confirmation.csv)

The full-record archive contained a correct complete system in 124 runs. Under all-scale CV it reached the front in 100 and fell outside it in 24. The front counts out of 12 per system were 12 each for viscous Burgers, KdV, Kuramoto–Sivashinsky and the reciprocal-coefficient PDE, 11 for Lotka–Volterra, 10 each for inviscid Burgers, the forced oscillator and Van der Pol, eight for nonlinear diffusion, and three for wave. Allen–Cahn, Duffing, forced KdV, Lorenz and Navier–Stokes each scored zero. The separate oscillator control had a correct identity on the front in 12/12 runs. [Confirmation returns](dyad_rows_confirmation.csv)

## How the score works

Build the usual weak-equation matrix once from the full record and reuse dyad's exact row ordering and slab boundaries on every supported grid axis. For each candidate, freeze the terms, target and frequencies. At each scale, hold out one slab, fit the linear coefficients on all remaining rows, and evaluate the weak equation on the omitted rows. A four-slab partition therefore produces four fits, each trained on three slabs. The discrepancy for each omitted slab is the sum of the absolute residuals divided by the sum of the absolute weak target values. Average equally over slabs, then scales, then axes. For the stability comparison, use the candidate's existing full-record dyad score as the second objective. [Protocol](dyad_discrepancy_protocol.md) · [Implementation](dyad_discrepancy.py)

## Front membership and supplementary shortlist counts

The multiobjective endpoint uses the complete first nondominated front. The first-choice and six-candidate columns below use an extra experimental ordering within Pareto layers.

| Ranking rule | Correct first | Correct in six | Correct on first Pareto front | Median front size |
|---|---:|---:|---:|---:|
| Ordinary full-record discrepancy | 0 | 0 | 0 | 1 |
| Two-slab CV discrepancy | 2 | 3 | 2 | 1 |
| All-scale CV discrepancy | 2 | 3 | 2 | 1 |
| Ordinary full-record discrepancy and stability | 15 | 46 | 100 | 13 |
| Two-slab CV discrepancy and stability | 36 | 83 | 100 | 7.5 |
| All-scale CV discrepancy and stability | 34 | 84 | 100 | 7 |
| Current contiguous holdout and stability, separate discovery | 58 | 74 | 79 | 7 |

The first six rows use identical candidates and discovery budgets. A correct complete system was available in 124 of those 180 runs. Cross-validation with stability gained 40 six-candidate recoveries and lost two on the same archives relative to ordinary discrepancy with stability. First-front recovery counted 100 runs under each of those scorers, and the median front size fell from 13 to seven. In the contiguous pipeline, the candidate pool contained a correct system in 111 runs. Against that pipeline, all-scale CV with stability gained 23 six-candidate recoveries and lost 13. These comparisons use the existing eligibility rules and Pareto ordering over per-equation objectives. Every run returned six candidates. The separate oscillator identity control scored 10/12 for either CV rule with stability, 5/12 for ordinary full-record discrepancy with stability and 11/12 for the contiguous pipeline. [Confirmation returns](dyad_rows_confirmation.csv)

## How much the finer scales contributed

The weak lattices supported only halves in 116 runs. Of the remaining 64, the finest native partition had four slabs in 52 and eight slabs in 12. With stability, correct-in-six counts on those 64 went from 29 with halves to 30 across all scales. Four paired gains and three losses explain the difference. Viscous Burgers gained one and inviscid Burgers gained two. KdV and wave each lost one. The reciprocal-coefficient PDE gained one seed and lost another. In the three-seed screen, halves scored 21/45 and all scales 19/45. There was no consistent gain from adding finer scales across the two cohorts. [Screen returns](dyad_rows_screen.csv) · [Confirmation by dataset](dyad_discrepancy_by_dataset.csv)

Evaluating all eligible scales took 1,758,356 coefficient refits across the 180 confirmation runs, compared with 830,976 for halves. These counts cover distinct target-and-predictor designs in the returned search archive. Each search had the same 3,000 fit-equivalent discovery budget. Fold refits are additional offline scoring work. Full-record discovery produced zero correct candidate systems on Duffing, forced KdV, Lorenz and Navier–Stokes in these runs.

The folds exclude weak rows during coefficient fitting. Structures, frequencies and windows came from full-record discovery, and adjacent weak rows can share underlying raw observations. These results measure ranking on the observed trajectories under repeated noise and search seeds. A separate check on a new noise draw over the same trajectory gave median leading-candidate WAPEs of 6.02% for ordinary discrepancy with stability, 6.37% for halves and 6.45% across all scales. The experiment used the research harness for post-search scoring. [Numerical checks against direct least squares](check_dyad_discrepancy.py) · [Report generator](dyad_report.py)
