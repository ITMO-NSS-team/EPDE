# Joint multisample Pareto discovery

100 of 100 searches verified. Each front contains every nondominated system among the eligible fits evaluated in its run. All supplied recordings contributed to discovery.

| System | Recordings | Law generated | Law on front | Median front size | Structures on at least two fronts |
|---|---|---:|---:|---:|---:|
| ode | one | 5/5 | 2/5 | 12 | 6 |
| ode | repeat3 | 5/5 | 5/5 | 92 | 35 |
| ode | diverse3 | 0/5 | 0/5 | 93 | 41 |
| ode | repeat5 | 5/5 | 5/5 | 254 | 126 |
| ode | diverse5 | 0/5 | 0/5 | 177 | 133 |
| vdp | one | 5/5 | 2/5 | 15 | 2 |
| vdp | repeat3 | 5/5 | 5/5 | 100 | 30 |
| vdp | diverse3 | 5/5 | 5/5 | 139 | 59 |
| vdp | repeat5 | 4/5 | 4/5 | 146 | 50 |
| vdp | diverse5 | 5/5 | 5/5 | 230 | 135 |
| duffing | one | 0/5 | 0/5 | 11 | 4 |
| duffing | repeat3 | 0/5 | 0/5 | 27 | 11 |
| duffing | diverse3 | 0/5 | 0/5 | 78 | 72 |
| duffing | repeat5 | 0/5 | 0/5 | 31 | 23 |
| duffing | diverse5 | 0/5 | 0/5 | 81 | 74 |
| lv | one | 2/5 | 2/5 | 199 | 2 |
| lv | repeat3 | 5/5 | 5/5 | 620 | 10 |
| lv | diverse3 | 5/5 | 5/5 | 922 | 36 |
| lv | repeat5 | 5/5 | 5/5 | 878 | 20 |
| lv | diverse5 | 5/5 | 5/5 | 1864 | 74 |

The arm names give the recording count. Repeat arms use independent measurement noise on the same initial condition. Diverse arms also vary the initial conditions. Generation counts a catalog system anywhere in the evaluated pool. Recovery counts a catalog system on the complete front. A structure contributes at most once per draw to recurrence.

Coefficient and frequency ranges use all aligned fits of each recurring structure. Physical parameter errors are conditional on recovering a catalog structure on the full front. The five draws describe measurement sensitivity at one fixed optimizer seed per system.

[Structure recurrence](structure_recurrence.csv.gz) · [Parameter ranges](coefficient_ranges.csv.gz) · [Benchmark parameter errors](benchmark_parameter_errors.csv) · [Individual searches](runs.csv)

## Catalog parameter ranges

Each interval spans all front fits that recovered the indicated catalog system, over the draws where it appeared. The pivot coefficient is fixed at −1 in the zero-form equation. Angular frequencies use the original time units.

| System | Arm | Pivot | Term | Catalog value | Fitted range | Draws |
|---|---|---|---|---:|---:|---:|
| ode | one | `d^2u/dx0^2` | `du/dx0*sin[0]` | -1 | -1.1198 to -0.85348 | 2 |
| ode | one | `d^2u/dx0^2` | `du/dx0*sin[0]:omega1` | 2 | 1.9965 to 2.0006 | 2 |
| ode | one | `d^2u/dx0^2` | `u` | -4 | -4.0293 to -4.0236 | 2 |
| ode | one | `d^2u/dx0^2` | `x[0]` | 1.5 | 1.4949 to 1.5446 | 2 |
| ode | repeat3 | `d^2u/dx0^2` | `du/dx0*sin[0]` | -1 | -1.0987 to -0.84132 | 5 |
| ode | repeat3 | `d^2u/dx0^2` | `du/dx0*sin[0]:omega1` | 2 | 1.996 to 2.0073 | 5 |
| ode | repeat3 | `d^2u/dx0^2` | `u` | -4 | -4.0463 to -3.9594 | 5 |
| ode | repeat3 | `d^2u/dx0^2` | `x[0]` | 1.5 | 1.4976 to 1.5246 | 5 |
| ode | repeat5 | `d^2u/dx0^2` | `du/dx0*sin[0]` | -1 | -1.1076 to -0.81309 | 5 |
| ode | repeat5 | `d^2u/dx0^2` | `du/dx0*sin[0]:omega1` | 2 | 1.996 to 2.0105 | 5 |
| ode | repeat5 | `d^2u/dx0^2` | `u` | -4 | -4.0426 to -3.9615 | 5 |
| ode | repeat5 | `d^2u/dx0^2` | `x[0]` | 1.5 | 1.5044 to 1.5284 | 5 |
| vdp | one | `d^2u/dx0^2` | `du/dx0` | 0.2 | 0.22265 to 0.24394 | 2 |
| vdp | one | `d^2u/dx0^2` | `du/dx0*u^2` | -0.2 | -0.27794 to -0.2232 | 2 |
| vdp | one | `d^2u/dx0^2` | `u` | -1 | -1.0049 to -0.99986 | 2 |
| vdp | repeat3 | `d^2u/dx0^2` | `du/dx0` | 0.2 | 0.13349 to 0.24472 | 5 |
| vdp | repeat3 | `d^2u/dx0^2` | `du/dx0*u^2` | -0.2 | -0.26389 to -0.098015 | 5 |
| vdp | repeat3 | `d^2u/dx0^2` | `u` | -1 | -1.0069 to -0.99946 | 5 |
| vdp | diverse3 | `d^2u/dx0^2` | `du/dx0` | 0.2 | 0.16819 to 0.23519 | 5 |
| vdp | diverse3 | `d^2u/dx0^2` | `du/dx0*u^2` | -0.2 | -0.24202 to -0.15763 | 5 |
| vdp | diverse3 | `d^2u/dx0^2` | `u` | -1 | -1.0039 to -0.99547 | 5 |
| vdp | repeat5 | `d^2u/dx0^2` | `du/dx0` | 0.2 | 0.14166 to 0.2412 | 4 |
| vdp | repeat5 | `d^2u/dx0^2` | `du/dx0*u^2` | -0.2 | -0.25903 to -0.11248 | 4 |
| vdp | repeat5 | `d^2u/dx0^2` | `u` | -1 | -1.0043 to -1.0004 | 4 |
| vdp | diverse5 | `d^2u/dx0^2` | `du/dx0` | 0.2 | 0.16756 to 0.21739 | 5 |
| vdp | diverse5 | `d^2u/dx0^2` | `du/dx0*u^2` | -0.2 | -0.21327 to -0.1593 | 5 |
| vdp | diverse5 | `d^2u/dx0^2` | `u` | -1 | -1.0043 to -0.99753 | 5 |
| lv | one | `du/dx0` | `u` | 20 | 19.855 to 20.176 | 2 |
| lv | one | `du/dx0` | `u*v` | -20 | -20.231 to -20.021 | 2 |
| lv | one | `dv/dx0` | `u*v` | 20 | 19.585 to 19.877 | 2 |
| lv | one | `dv/dx0` | `v` | -20 | -19.624 to -19.3 | 2 |
| lv | repeat3 | `du/dx0` | `u` | 20 | 19.281 to 20.217 | 5 |
| lv | repeat3 | `du/dx0` | `u*v` | -20 | -20.247 to -19.502 | 5 |
| lv | repeat3 | `dv/dx0` | `u*v` | 20 | 19.366 to 20.145 | 5 |
| lv | repeat3 | `dv/dx0` | `v` | -20 | -20.274 to -19.473 | 5 |
| lv | diverse3 | `du/dx0` | `u` | 20 | 19.436 to 20.451 | 5 |
| lv | diverse3 | `du/dx0` | `u*v` | -20 | -20.242 to -19.395 | 5 |
| lv | diverse3 | `dv/dx0` | `u*v` | 20 | 19.417 to 19.88 | 5 |
| lv | diverse3 | `dv/dx0` | `v` | -20 | -19.86 to -19.299 | 5 |
| lv | repeat5 | `du/dx0` | `u` | 20 | 19.455 to 20.179 | 5 |
| lv | repeat5 | `du/dx0` | `u*v` | -20 | -20.058 to -19.402 | 5 |
| lv | repeat5 | `dv/dx0` | `u*v` | 20 | 19.524 to 20.053 | 5 |
| lv | repeat5 | `dv/dx0` | `v` | -20 | -20.322 to -19.846 | 5 |
| lv | diverse5 | `du/dx0` | `u` | 20 | 19.517 to 20.308 | 5 |
| lv | diverse5 | `du/dx0` | `u*v` | -20 | -20.226 to -19.492 | 5 |
| lv | diverse5 | `dv/dx0` | `u*v` | 20 | 19.454 to 19.857 | 5 |
| lv | diverse5 | `dv/dx0` | `v` | -20 | -19.779 to -19.391 | 5 |

## Paired changes in recovery

Each comparison uses only completed pairs. A gain means the catalog system appeared on the second front alone. A loss means it appeared on the first front alone. Counts pool the four systems.

| First arm | Second arm | Paired searches | Gain | Loss | On both | On neither |
|---|---|---:|---:|---:|---:|---:|
| one | repeat3 | 20 | 9 | 0 | 6 | 5 |
| one | repeat5 | 20 | 8 | 0 | 6 | 6 |
| repeat3 | diverse3 | 20 | 0 | 5 | 10 | 5 |
| repeat5 | diverse5 | 20 | 1 | 5 | 9 | 5 |
| repeat3 | repeat5 | 20 | 0 | 1 | 14 | 5 |
| diverse3 | diverse5 | 20 | 0 | 0 | 10 | 10 |

## Where recovery was lost

The counts below include all arms. A frequency miss means the search generated the full system of catalog terms, but never generated those terms with every frequency within the catalog tolerance. A front exclusion means the search generated a catalog match and every matching fit was dominated.

| System | Searches | Catalog terms never generated | Frequency miss | Front exclusion | Recovered |
|---|---:|---:|---:|---:|---:|
| ode | 25 | 0 | 10 | 3 | 12 |
| vdp | 25 | 1 | 0 | 3 | 21 |
| duffing | 25 | 20 | 5 | 0 | 0 |
| lv | 25 | 3 | 0 | 0 | 22 |

## Compute budget

The worker logs record 7,500 epochs and 15,000 generations. Recorded host CPU load at admission ranged from 19.4% to 57.5% across 100 starts. The two-worker controller uses a six-core affinity mask and single-threaded numerical libraries.
