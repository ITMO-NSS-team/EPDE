# Live window-preserving fitting in EPDE

The blocked TLS score retained a registered equation in 10/15 live searches, compared with 9/15 for native WAPE and 7/15 for blocked WLS. Its largest gain over WAPE was on wave. On six further wave seeds, each of the three objectives retained the wave equation in 3/6 runs. These results support retaining native WAPE as the default and native WLS for the printed coefficients. Blocked scoring has a more repeatable wave ranking on a frozen pool of candidates.

## Initial live comparison

Each cell counts a registered equation or accepted equivalent on the entire final nondominated front. The three arms had identical fields, initial random seeds, ten NSGA-II epochs, a seven-term maximum, fourth-order finite differences, native weak rows and native directional dyad. The search ran in plain mode with zero VWSR calls. The fitting objective used no term-count preference.

| System | Native WAPE | Blocked WLS | Blocked TLS |
|---|---:|---:|---:|
| Nonlinear diffusion | 2/3 | 2/3 | 2/3 |
| Wave | 1/3 | 2/3 | 3/3 |
| Van der Pol | 3/3 | 1/3 | 3/3 |
| Allen–Cahn | 1/3 | 1/3 | 1/3 |
| ode_simple, registered oscillator or equivalent | 2/3 | 1/3 | 1/3 |
| Total | 9/15 | 7/15 | 10/15 |

Relative to WAPE, blocked TLS gained two wave cases and lost one ode_simple case under the registered endpoint. For nonlinear diffusion, WAPE and blocked TLS failed to score the exact structure on the same third seed. Blocked WLS scored it and a final-front rival dominated it. The true coefficients of $u_t=u u_{xx}+u_x^2$ are both one. Native WLS fitted (0.994, 0.988) and (0.999, 1.000) on the two successful fields when expressed with the time derivative as target. The 0/6 to 6/6 TLS gain in the earlier fixed-panel experiment did not recur under the live native window selection.

For Allen–Cahn, reaction alone beat the complete law on both objectives in each of the two missed TLS runs. In a separate refit of the full and reaction-only structures on 48 matched positive and 48 zero-diffusion recordings, blocked WLS and TLS each favored diffusion 16 times when present and 10 times when absent. Native WAPE favored it 36 and 43 times respectively. These are first-objective preferences with the time derivative as target. Each positive and zero pair used the same standardized noise and the same native window geometry. The zero fixture uses the analytic reaction-only trajectory with the original initial profile.

For ode_simple, an exact symbolic check addressed valid laws outside the finite registry. At the literal proposed token frequencies, full-support identities appeared on 2/3 WAPE, 1/3 WLS and 1/3 TLS fronts. Allowing a frequency within the existing 1% unit-frequency tolerance to take its ideal value gave at least one analytic identity on all nine fronts. For example the unregistered support $u'=a u+b\sin(t)$ is valid on this trajectory with $a=10/13$ and $b=-269/130$. One printed fit gave 0.8114 and −2.1091 with frequency 1.00377. The symbolic test requires every proposed term to admit a nonzero coefficient. It retains explicit constants and tests structural compatibility separately from the printed coefficient error.

## Six further wave seeds

All 18 wave searches scored the exact support. Each arm retained it on three final fronts. The source seeds giving successful recovery were not identical across arms. In two blocked-TLS misses an actual final competitor dominated every recorded reference point. In the third, at least one evaluated reference point was undominated by the final front and absent from it. The saved data cannot establish precisely when it left the population. A recurring rival was the spatial relation $u_{xx}\approx-0.430u^3$.

Across the six confirmation fields, forcing the standard wave support and time target gave median full-record coefficients of 0.04058 with native WLS and 0.04206 with full-record unit-column TLS. The physical value is 0.04. The experimental search always printed native WLS weights.

The separate ranking check pooled the three discovered fronts per seed, inserted the canonical reference with eligible native derivative targets, and froze coefficients, token frequencies, training scales and source dyads. Sixteen independent 5% noise recordings per seed supplied new residual observations. On the discovery recordings, the inserted law survived on 2/6 pooled fronts with WAPE and 4/6 with each blocked score. On fresh recordings the corresponding counts were 34/96, 64/96 and 64/96. The repeated observations share six fitted pools on one trajectory. They measure repeatability of a fixed ranking and do not constitute additional evolutionary searches.

The wave native grids gave the two possible split axes the same geometric rank on all nine seeds. The declared tie rule chose time. As an exploratory control, we evaluated the spatial split and the equal mean of the two split energies on those same frozen supports. On the six confirmation pools, fresh survival was 64/96 for either time-split score, 32/96 for spatial WLS and 31/96 for spatial TLS. Averaging the two axes gave 45/96 and 32/96. The spatial alternative did not improve the fixed-pool confirmation result.

## Verification and scope

The experiment completed 63 live searches. The objective was replayed for all 740 final-front rows. Independent reconstruction from the printed equation and original fields verified the first objective and native dyad for every saved front across both cohorts. The pooled check covered 418 support–target variants across the original 15 cases and 192 across the six new wave cases, counting a variant once per pool. The five-system screen targets previously studied scalar systems. Earlier fixed-panel counts from the 15-system suite are a separate experiment. Both live cohorts used short budgets and repeated measurements of the same trajectories. The split validates coefficients conditional on the native window selection and full-record data-derived token frequencies.

The main tables are in `outcomes.csv`, `wave_confirmation.csv`, `ac_controls.csv` and `wave_axes.csv`. The per-search records contain every front equation, both objectives, the native weights, a diagnostic TLS refit and the evaluated reference points. To reproduce the work, run `batch.py`, `transfer.py`, `ode_identities.py`, `ac_control.py`, `summarize.py`, `wave_confirmation.py`, `confirm_report.py` and `wave_axes.py` in that order.
