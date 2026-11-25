# Joint multisample Pareto discovery

34 of 100 searches verified. Each front contains every nondominated system among the eligible fits evaluated in its run. All supplied recordings contributed to discovery.

The design uses 5% correlated measurement noise (AR(1), lag-one correlation 0.8), up to five matched noise draws per arm, and one optimizer seed per benchmark. The [protocol](../protocol.md) describes the shared frequency fit and Pareto comparison.

## Paired frequency policies

Each pair uses identical recording hashes, optimizer seed, epoch count, working population and search options. The revised policy also allows standalone sine and cosine terms. Recovery requires a catalog match on the complete Pareto front.

These 34 completed pairs come from the 34 original forced-oscillator and Duffing generation failures. This selection measures recovery of those misses. An overall recovery rate requires the other benchmark runs.

| System | Pairs | Catalog terms generated, old → revised | Front recoveries, old → revised | Gained | Lost | Median search seconds, old → revised |
|---|---:|---:|---:|---:|---:|---:|
| ode | 9 | 9 → 9 | 0 → 9 | 9 | 0 | 15.5 → 33.6 |
| duffing | 25 | 3 → 23 | 0 → 20 | 20 | 0 | 13.8 → 31.8 |

New structures received frequency fits in the final epoch in 34/34 searches. The profiler reused previously fitted structures 24,633 times. Search times include fitting and optimization. [Matched runs](paired_policies.csv)

## Recovery by recording configuration

| System | Recordings | Law generated | Law on front | Median front size | Structures on at least two fronts |
|---|---|---:|---:|---:|---:|
| ode | diverse3 | 4/4 | 4/4 | 104.5 | 24 |
| ode | diverse5 | 5/5 | 5/5 | 267 | 150 |
| duffing | one | 4/5 | 2/5 | 13 | 7 |
| duffing | repeat3 | 4/5 | 4/5 | 52 | 19 |
| duffing | diverse3 | 5/5 | 5/5 | 61 | 33 |
| duffing | repeat5 | 4/5 | 4/5 | 89 | 38 |
| duffing | diverse5 | 5/5 | 5/5 | 123 | 71 |

The arm names give the recording count. Repeat arms use independent measurement noise on the same initial condition. Diverse arms also vary the initial conditions. Generation counts a catalog system anywhere in the evaluated pool. Recovery counts a catalog system on the complete front. A structure contributes at most once per draw to recurrence.

Coefficient and frequency ranges use all aligned fits of each recurring structure. Physical parameter errors are conditional on recovering a catalog structure on the full front. The five draws describe measurement sensitivity at one fixed optimizer seed per system.

[Structure recurrence](structure_recurrence.csv.gz) · [Parameter ranges](coefficient_ranges.csv.gz) · [Benchmark parameter errors](benchmark_parameter_errors.csv) · [Individual searches](runs.csv)

## Catalog parameter ranges

Each interval spans all front fits that recovered the indicated catalog system, over the draws where it appeared. The pivot coefficient is fixed at −1 in the zero-form equation. Angular frequencies use the original time units.

| System | Arm | Pivot | Term | Catalog value | Fitted range | Draws |
|---|---|---|---|---:|---:|---:|
| ode | diverse3 | `d^2u/dx0^2` | `du/dx0*sin[0]` | -1 | -1.0256 to -0.89029 | 4 |
| ode | diverse3 | `d^2u/dx0^2` | `du/dx0*sin[0]:omega1` | 2 | 1.9884 to 2.0115 | 4 |
| ode | diverse3 | `d^2u/dx0^2` | `u` | -4 | -4.0485 to -3.9838 | 4 |
| ode | diverse3 | `d^2u/dx0^2` | `x[0]` | 1.5 | 1.4985 to 1.523 | 4 |
| ode | diverse5 | `d^2u/dx0^2` | `du/dx0*sin[0]` | -1 | -1.0856 to -0.89736 | 5 |
| ode | diverse5 | `d^2u/dx0^2` | `du/dx0*sin[0]:omega1` | 2 | 1.9924 to 2.0177 | 5 |
| ode | diverse5 | `d^2u/dx0^2` | `u` | -4 | -4.0411 to -3.953 | 5 |
| ode | diverse5 | `d^2u/dx0^2` | `x[0]` | 1.5 | 1.4891 to 1.5224 | 5 |
| duffing | one | `d^2u/dx0^2` | `cos[0]` | 0.3 | 0.30695 to 0.30825 | 2 |
| duffing | one | `d^2u/dx0^2` | `cos[0]:omega1` | 1 | 1.0007 to 1.001 | 2 |
| duffing | one | `d^2u/dx0^2` | `du/dx0` | -0.2 | -0.21373 to -0.21016 | 2 |
| duffing | one | `d^2u/dx0^2` | `u` | -1 | -1.0018 to -0.98248 | 2 |
| duffing | one | `d^2u/dx0^2` | `u^3` | -1 | -1.0446 to -1.0085 | 2 |
| duffing | repeat3 | `d^2u/dx0^2` | `cos[0]` | 0.3 | 0.29371 to 0.31205 | 4 |
| duffing | repeat3 | `d^2u/dx0^2` | `cos[0]:omega1` | 1 | 0.99842 to 1.0028 | 4 |
| duffing | repeat3 | `d^2u/dx0^2` | `du/dx0` | -0.2 | -0.21789 to -0.19184 | 4 |
| duffing | repeat3 | `d^2u/dx0^2` | `u` | -1 | -1.0164 to -0.96672 | 4 |
| duffing | repeat3 | `d^2u/dx0^2` | `u^3` | -1 | -1.0828 to -0.95724 | 4 |
| duffing | diverse3 | `d^2u/dx0^2` | `cos[0]` | 0.3 | 0.29499 to 0.30453 | 5 |
| duffing | diverse3 | `d^2u/dx0^2` | `cos[0]:omega1` | 1 | 0.994 to 1.0029 | 5 |
| duffing | diverse3 | `d^2u/dx0^2` | `du/dx0` | -0.2 | -0.21437 to -0.17932 | 5 |
| duffing | diverse3 | `d^2u/dx0^2` | `u` | -1 | -1.0008 to -0.97374 | 5 |
| duffing | diverse3 | `d^2u/dx0^2` | `u^3` | -1 | -1.0498 to -0.98603 | 5 |
| duffing | repeat5 | `d^2u/dx0^2` | `cos[0]` | 0.3 | 0.29455 to 0.30768 | 4 |
| duffing | repeat5 | `d^2u/dx0^2` | `cos[0]:omega1` | 1 | 0.99892 to 1.0019 | 4 |
| duffing | repeat5 | `d^2u/dx0^2` | `du/dx0` | -0.2 | -0.21347 to -0.19012 | 4 |
| duffing | repeat5 | `d^2u/dx0^2` | `u` | -1 | -1.0382 to -0.98142 | 4 |
| duffing | repeat5 | `d^2u/dx0^2` | `u^3` | -1 | -1.0463 to -0.90516 | 4 |
| duffing | diverse5 | `d^2u/dx0^2` | `cos[0]` | 0.3 | 0.29483 to 0.30808 | 5 |
| duffing | diverse5 | `d^2u/dx0^2` | `cos[0]:omega1` | 1 | 0.99872 to 1.0061 | 5 |
| duffing | diverse5 | `d^2u/dx0^2` | `du/dx0` | -0.2 | -0.20527 to -0.19156 | 5 |
| duffing | diverse5 | `d^2u/dx0^2` | `u` | -1 | -1.0086 to -0.95754 | 5 |
| duffing | diverse5 | `d^2u/dx0^2` | `u^3` | -1 | -1.0418 to -1.0095 | 5 |

## Paired changes in recovery

Each comparison uses only completed pairs. A gain means the catalog system appeared on the second front alone. A loss means it appeared on the first front alone. Counts pool the four systems.

| First arm | Second arm | Paired searches | Gain | Loss | On both | On neither |
|---|---|---:|---:|---:|---:|---:|
| one | repeat3 | 5 | 2 | 0 | 2 | 1 |
| one | repeat5 | 5 | 3 | 1 | 1 | 0 |
| repeat3 | diverse3 | 5 | 1 | 0 | 4 | 0 |
| repeat5 | diverse5 | 5 | 1 | 0 | 4 | 0 |
| repeat3 | repeat5 | 5 | 1 | 1 | 3 | 0 |
| diverse3 | diverse5 | 9 | 0 | 0 | 9 | 0 |

## Where recovery was lost

The counts below include all arms. A frequency miss means the search generated the full system of catalog terms, but never generated those terms with every frequency within the catalog tolerance. A front exclusion means the search generated a catalog match and every matching fit was dominated.

| System | Searches | Catalog terms never generated | Frequency miss | Front exclusion | Recovered |
|---|---:|---:|---:|---:|---:|
| ode | 9 | 0 | 0 | 0 | 9 |
| vdp | 0 | 0 | 0 | 0 | 0 |
| duffing | 25 | 2 | 1 | 2 | 20 |
| lv | 0 | 0 | 0 | 0 | 0 |

For `duffing`, the frequency profiler reached the catalog term set in 1 of 1 frequency misses.

## Compute budget

The worker logs record 1,700 epochs and 3,400 generations. Recorded host CPU load at admission ranged from 24.0% to 61.3% across 34 starts. The two-worker controller uses a six-core affinity mask and single-threaded numerical libraries.

The profiler fitted a shared frequency 40,178 times. The search evaluated 38,319 distinct systems containing a profiled fit, of which 914 appeared on the final fronts.
