# Choosing a hold-out strategy for equation discovery

The paired 0% and 5% comparison reports first-front recovery for the 15 core datasets separately at each noise level. It also reconciles the current suite with the earlier 13-system universal-search benchmark. [Clean and noisy results](fronts_by_noise.md)

The complete system appeared on the first nondominated front in 100/180 full-record searches at 5% added measurement noise with ordinary discrepancy and in 100/180 with cross-validated discrepancy on the exact stability slabs. The median front size fell from 13 to seven. The contiguous held-out pipeline recovered 79/180 on its front. [Native-slab discrepancy experiment and full results](dyad_discrepancy.md)

## What the current validation measures

The benchmark selects weak integration windows on the training observations, discovers candidate equations there, and freezes their coefficients and frequencies. On the disjoint validation grid it measures the residual of each equation at the observed field values. For a coupled system it retains only combinations that actually occurred together during search. The existing selection objectives are held-out weighted absolute percentage error (WAPE) for each equation and coefficient instability across training blocks (dyad). The experiment uses the frequency-refinement research harness. [Discovery implementation](../live_operator_suite_2026-11-23/live.py) · [Validation implementation](../live_operator_suite_2026-11-23/assess.py)

The current split holds out half the samples on the longest axis, choosing time in a tie. Eleven of the fifteen systems split along time. Allen–Cahn, viscous Burgers, KdV and Kuramoto–Sivashinsky split along space. The benchmark's standard validation score is an equation residual. A forecast error requires evolving the fitted model and comparing trajectories.

The multiobjective success criterion is the presence of a complete registered system anywhere on the entire first nondominated front. The experimental shortlist adds an ordering within Pareto levels and draws from deeper levels when the first contains fewer than six systems. Its first-choice and six-candidate counts describe that additional readout. [Protocol](protocol.md) · [Front computation and optional ordering](compare.py)

## Ranking the same candidate pool

All four rows use the same 180 contiguous-half searches across 15 datasets, 12 seeds per dataset. Each search had 3,000 total fit equivalents. The archive contained an actual correct system in 111 runs. The control is scored separately.

| Residual used with training dyad | First candidate correct | Correct in six | Correct on unrestricted first front | Median first-front size | Median fresh-measurement WAPE of first candidate |
|---|---:|---:|---:|---:|---:|
| Held-out block | 58/180 | 74/180 | 79/180 | 7 | 7.02% |
| Training block | 8/180 | 32/180 | 66/180 | 13 | 10.56% |
| Mean of the two block WAPEs | 44/180 | 72/180 | 86/180 | 8 | 7.94% |
| Worse of the two block WAPEs | 53/180 | 70/180 | 79/180 | 7.5 | 7.22% |

Compared with held-out ranking, training alone gained three six-candidate recoveries and lost 45. Mean WAPE gained four and lost six. Worst-block WAPE gained zero and lost four. Registered and literal primary term structures gave the same fixed-six counts in this comparison. For the first structurally correct system in each list, the median largest relative coefficient-vector error across slots was 0.45% with held-out ranking, conditional on its 74 recoveries. Post-selection coefficient errors use the catalog equation oriented on the archived target. [Per-run returns](contiguous_confirmation.csv) · [Coefficient audit](coefficients_contiguous_confirmation.csv)

The fresh-measurement column uses one independent 5% Gaussian noise realization on the original second half and evaluates all candidates on that same grid with the same weak window per dataset and seed. It measures the leading candidate's equation consistency on a repeat observation of the same dynamics. Scores average equally across slots within a candidate. They depend on the fitted target used for normalization.

## Changing which data discovery sees

The screen used three paired searches on each of the 15 datasets. Each alternative reran discovery, including the selection of weak windows and trigonometric frequencies. Alternating samples give both partitions coverage of the observed range, with twice the grid spacing on the split axis. Each derivative and weak residual uses only its partition. Full-record discovery uses all samples at original spacing and ranks by its own training residual and dyad. All arms were assessed at 3,000 total fit equivalents.

| Three-seed screen | Correct structure assembled | First candidate correct | Correct in six | Correct on unrestricted front | Median front size | Median fresh-measurement WAPE of first candidate |
|---|---:|---:|---:|---:|---:|---:|
| Contiguous split, held-out WAPE | 29/45 | 15/45 | 18/45 | 21/45 | 7 | 6.99% |
| Alternating split, held-out WAPE | 28/45 | 13/45 | 21/45 | 24/45 | 10 | 5.23% |
| Full record, training WAPE | 31/45 | 2/45 | 9/45 | 24/45 | 14 | 5.57% |

Full-record discovery gained two six-candidate hits and lost eleven relative to contiguous hold-out. Its unrestricted front contained more correct systems and was twice as large at the median. The alternating split gained four and lost one. The separate oscillator control scored 3/3 for both split designs and 2/3 for the full-record design. Median recorded search times were 19.5, 17.0 and 19.3 seconds, respectively, for contiguous, alternating and full-record search. These timings include concurrent jobs under different machine loads. [Paired screening results by dataset](summary_screen.csv) · [Full-record returns](full_screen.csv)

## Supplementary experiment with raw-grid slabs and new weak windows

Full-record discovery supplies the same actual candidates for three additional scorers. Each feasible axis is partitioned into halves, quarters and eighths before constructing weak measurements. The frozen-coefficient control evaluates the original fitted vector separately in every slab. The same-scale rule refits the archived support on the other slabs of that size and scores the one left out. The cross-scale rule fits outside a coarser parent and tests each of the parent's two children with their finer weak windows. A fit outside the first half, for example, is evaluated separately on the first and second quarters. Both fits and evaluations rotate over all supported locations. The scorer averages across locations, then scales, then axes. [Implementation](slabs.py) · [Raw-sample isolation and direct-solve check](check_slabs.py)

| Slab screen on the full-record candidates | First candidate correct | Correct in six | Complete first-front recovery |
|---|---:|---:|---:|
| Frozen coefficients across slabs | 10/45 | 19/45 | 22/45 |
| Refit outside each slab, evaluate at the same scale | 14/45 | 19/45 | 21/45 |
| Refit outside the parent, evaluate on its children | 14/45 | 22/45 | 23/45 |

Cross-scale validation gained six six-candidate hits and lost two versus the original contiguous hold-out. It gained two wave cases. The 45 primary searches required 861,836 linear fits across folds and distinct term designs for the full slab diagnostic. The two refitted readouts reused the same coefficient solutions. The splits protect raw observations during the linear refits and residual checks. The earlier full-record discovery stage had access to the whole field when selecting support and frequencies. [Per-run slab counts and candidate lists](slabs_screen.csv)

On 12 confirmation seeds, frozen scoring on raw slabs reached 79/180, refitting at the same scale reached 84/180, and fitting outside the parent and scoring its children reached 91/180. The last rule gained 22 and lost five against contiguous holdout. These arms changed the integration windows when constructing each raw slab. [Per-dataset results](summary_confirmation.csv) · [Comparison figure](recovery_by_dataset.png)

## Does a hold-out remove identifying information?

In Duffing, the contiguous training half spans 1.59 drive cycles. Alternating training points span 3.18 cycles with the same 500 samples. Using all 1,001 samples also spans 3.18 cycles. We reran the earlier signal-only estimator on each actual allocation with the same 250 noise draws. With the optional lag proposal the median candidate nearest the registered frequency of 1 was 1.080 on the first half, 1.056 on the last half, 1.095 on alternating training points and 1.095 on the full record. All four scored 0/250 within 1%. The three discovery designs each assembled zero correct Duffing equations in the three-seed screen. Here the missing frequency persisted even when every sample was available. This is an empirical limitation of the tested proposals on this trajectory. [Frequency check](duffing_summary.csv) · [Data-only estimator](../signal_only_frequency_2026-11-23/signal_frequency.py)

## Twelve-seed split confirmation

Alternating observations tied the current contiguous split at **74/180** complete systems in six candidates, with 13 paired gains and 13 losses. First-choice recovery fell from 58 to 52. Unrestricted first-front recovery rose from 79 to 94 as median front size rose from 7 to 11. Both original physical support frequencies and the benchmark's registered equivalences were scored under the same rules. The forced oscillator gained five six-candidate hits (7/12 to 12/12). Inviscid Burgers lost three (6/12 to 3/12) and nonlinear diffusion lost two (3/12 to 1/12). The separate control went from 11/12 to 10/12. [Per-run returns](alternate_confirmation.csv) · [Paired results](summary_confirmation.csv)

## Four-fold cross-validation of the search

Each outer fold starts discovery from scratch with three contiguous raw quarters as separate training trajectories. The omitted quarter enters only after search. In one Duffing fold, replacing every measurement in the omitted quarter left all 606 fitted points and 112 actual candidate systems exactly unchanged. Single-trajectory fill-and-prune falls back to the drawn mutation on multiple trajectories, so an additional matched comparator uses that same fallback on the contiguous training half. Native EPDE averages the three training-trajectory coefficient estimates in each fold. The complete-record consensus retains an actual fold representative with its coefficients and frequency frozen. [Protocol](blocked_protocol.md) · [Exclusion check](blocked_exclusion_check.json)

All 180 outer folds on the 15 default datasets completed. At four searches of 750 evaluations each, consensus returned a correct structure in six in 2/45 runs. The contiguous comparator using the same mutation fallback returned 5/45 at 3,000 evaluations. The short oscillator control had no feasible weak window on its 50-sample quarters. [Four-fold results by dataset](blocked_by_dataset.csv)
