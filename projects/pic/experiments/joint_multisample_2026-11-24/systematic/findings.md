# Joint multisample Pareto discovery

5 of 100 searches verified. Each front contains every nondominated system among the eligible fits evaluated in its run. All supplied recordings contributed to discovery.

The design uses 5% correlated measurement noise (AR(1), lag-one correlation 0.8), up to five matched noise draws per arm, and one optimizer seed per benchmark. The [protocol](../protocol.md) describes the shared frequency fit and Pareto comparison.

## Paired search policies

Each pair uses identical recording hashes, optimizer seed, epoch count and working population. The revised policy fits every resolved frequency minimum and generates structural alternatives independently of residual rank. Recovery requires a catalog match on the complete Pareto front.

These 5 pairs were selected from searches that missed the catalog law on the previous front.

| System | Pairs | Catalog terms generated, old → revised | Front recoveries, old → revised | Gained | Lost | Median search seconds, old → revised |
|---|---:|---:|---:|---:|---:|---:|
| duffing | 5 | 3 → 5 | 0 → 4 | 4 | 0 | 17.4 → 181.2 |

The catalog law appears on 3/5 combined fronts after pooling all eligible systems from both policies on each matched draw. The [complete combined fronts](combined_fronts.csv.gz) list every surviving archive reference, including ties and membership in both archives.

New structures received frequency fits in the final epoch in 5/5 searches. The profiler reused previously fitted structures 32,337 times. Search times include fitting and optimization. [Matched runs](paired_policies.csv)

## Recovery by recording configuration

| System | Recordings | Law generated | Law on front | Median front size | Structures on at least two fronts |
|---|---|---:|---:|---:|---:|
| duffing | one | 3/3 | 2/3 | 20 | 6 |
| duffing | repeat3 | 1/1 | 1/1 | 153 | 0 |
| duffing | repeat5 | 1/1 | 1/1 | 380 | 0 |

The arm names give the recording count. Repeat arms use independent measurement noise on the same initial condition. Diverse arms also vary the initial conditions. Generation counts a catalog system anywhere in the evaluated pool. Recovery counts a catalog system on the complete front. A structure contributes at most once per draw to recurrence.

Coefficient and frequency ranges use all aligned fits of each recurring structure. Physical parameter errors are conditional on recovering a catalog structure on the full front. The five draws describe measurement sensitivity at one fixed optimizer seed per system.

[Structure recurrence](structure_recurrence.csv.gz) · [Parameter ranges](coefficient_ranges.csv.gz) · [Benchmark parameter errors](benchmark_parameter_errors.csv) · [Individual searches](runs.csv)

## Catalog parameter ranges

Each interval spans all front fits that recovered the indicated catalog system, over the draws where it appeared. The pivot coefficient is fixed at −1 in the zero-form equation. Angular frequencies use the original time units.

| System | Arm | Pivot | Term | Catalog value | Fitted range | Draws |
|---|---|---|---|---:|---:|---:|
| duffing | one | `d^2u/dx0^2` | `cos[0]` | 0.3 | 0.29448 to 0.32576 | 2 |
| duffing | one | `d^2u/dx0^2` | `cos[0]:omega1` | 1 | 0.99994 to 1.0091 | 2 |
| duffing | one | `d^2u/dx0^2` | `du/dx0` | -0.2 | -0.25253 to -0.18863 | 2 |
| duffing | one | `d^2u/dx0^2` | `u` | -1 | -0.98853 to -0.91838 | 2 |
| duffing | one | `d^2u/dx0^2` | `u^3` | -1 | -1.1738 to -1.0014 | 2 |
| duffing | repeat3 | `d^2u/dx0^2` | `cos[0]` | 0.3 | 0.28349 to 0.30717 | 1 |
| duffing | repeat3 | `d^2u/dx0^2` | `cos[0]:omega1` | 1 | 0.99472 to 1.004 | 1 |
| duffing | repeat3 | `d^2u/dx0^2` | `du/dx0` | -0.2 | -0.21619 to -0.16752 | 1 |
| duffing | repeat3 | `d^2u/dx0^2` | `u` | -1 | -1.0449 to -1.0158 | 1 |
| duffing | repeat3 | `d^2u/dx0^2` | `u^3` | -1 | -0.94559 to -0.88985 | 1 |
| duffing | repeat5 | `d^2u/dx0^2` | `cos[0]` | 0.3 | 0.29285 to 0.2988 | 1 |
| duffing | repeat5 | `d^2u/dx0^2` | `cos[0]:omega1` | 1 | 0.997 to 0.99954 | 1 |
| duffing | repeat5 | `d^2u/dx0^2` | `du/dx0` | -0.2 | -0.19916 to -0.18613 | 1 |
| duffing | repeat5 | `d^2u/dx0^2` | `u` | -1 | -1.0106 to -1.0031 | 1 |
| duffing | repeat5 | `d^2u/dx0^2` | `u^3` | -1 | -0.99301 to -0.97889 | 1 |

## Paired changes in recovery

Each comparison uses only completed pairs. A gain means the catalog system appeared on the second front alone. A loss means it appeared on the first front alone. Counts pool the four systems.

| First arm | Second arm | Paired searches | Gain | Loss | On both | On neither |
|---|---|---:|---:|---:|---:|---:|
| one | repeat3 | 1 | 1 | 0 | 0 | 0 |
| one | repeat5 | 0 | 0 | 0 | 0 | 0 |
| repeat3 | diverse3 | 0 | 0 | 0 | 0 | 0 |
| repeat5 | diverse5 | 0 | 0 | 0 | 0 | 0 |
| repeat3 | repeat5 | 0 | 0 | 0 | 0 | 0 |
| diverse3 | diverse5 | 0 | 0 | 0 | 0 | 0 |

## Where recovery was lost

The counts below include all arms. A frequency miss means the search generated the full system of catalog terms, but never generated those terms with every frequency within the catalog tolerance. A front exclusion means the search generated a catalog match and every matching fit was dominated.

| System | Searches | Catalog terms never generated | Frequency miss | Front exclusion | Recovered |
|---|---:|---:|---:|---:|---:|
| ode | 0 | 0 | 0 | 0 | 0 |
| vdp | 0 | 0 | 0 | 0 | 0 |
| duffing | 5 | 0 | 0 | 1 | 4 |
| lv | 0 | 0 | 0 | 0 | 0 |

## Compute budget

The worker logs record 250 epochs and 500 generations. Recorded host CPU load at admission ranged from 25.3% to 56.7% across 5 starts. The two-worker controller uses a six-core affinity mask and single-threaded numerical libraries.

The profiler searched 27,128 equation-specific frequency bands and proposed 57,303 frequencies. 24,024 bands yielded multiple minima. The search evaluated 55,115 distinct systems containing a profiled fit, of which 114 appeared on the final fronts.
