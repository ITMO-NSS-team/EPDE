# Campaign `baseline_clean_2026-10-09`

Runs: 150 (ok: 130, timeout: 10, unsupported: 10).
Success: the true structure (or an accepted equivalent form) is on the final Pareto front; "pick": the equation chosen from the front without knowing the truth (compromise) is correct. Cells: k/n (rate, 95 % Wilson interval). Errors and timeouts count as misses. pending: planned, not yet run; unsupported: the method cannot represent the problem. These two statuses are excluded from the denominators.

## Noise 0 %

### Truth on the Pareto front

| dataset | default | pysindy |
|---|---|---|
| ac | 4/5 (80%, 38-96) | 0/5 (0%, 0-43) |
| burgers | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| burgers_inviscid | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| duffing | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| kdv | 2/5 (40%, 12-77) | 0/5 (0%, 0-43) |
| kdv_cossin | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| ks | 0/5 (0%, 0-43) | 0/5 (0%, 0-43) |
| lorenz | 0/5 (0%, 0-43) | 5/5 (100%, 57-100) |
| lv | 2/5 (40%, 12-77) | 5/5 (100%, 57-100) |
| ns | 0/5 (0%, 0-43) |  |
| ode | 4/5 (80%, 38-96) | 5/5 (100%, 57-100) |
| pde_compound | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| pde_divide | 5/5 (100%, 57-100) |  |
| vdp | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| wave | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |

### Compromise pick is correct

| dataset | default | pysindy |
|---|---|---|
| ac | 0/5 (0%, 0-43) | 0/5 (0%, 0-43) |
| burgers | 3/5 (60%, 23-88) | 5/5 (100%, 57-100) |
| burgers_inviscid | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| duffing | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| kdv | 2/5 (40%, 12-77) | 0/5 (0%, 0-43) |
| kdv_cossin | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| ks | 0/5 (0%, 0-43) | 0/5 (0%, 0-43) |
| lorenz | 0/5 (0%, 0-43) | 5/5 (100%, 57-100) |
| lv | 2/5 (40%, 12-77) | 5/5 (100%, 57-100) |
| ns | 0/5 (0%, 0-43) |  |
| ode | 3/5 (60%, 23-88) | 5/5 (100%, 57-100) |
| pde_compound | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| pde_divide | 5/5 (100%, 57-100) |  |
| vdp | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |
| wave | 5/5 (100%, 57-100) | 5/5 (100%, 57-100) |

## Best variant per problem class

Mean over the data sets of the class supported by every variant of the per-set success rate; a data set is left out of the ranking when any variant is unsupported on it or has no finished runs yet. Ties are broken by the compromise pick, then by speed.

| class | noise, % | best variant | success (front) | success (pick) | median time, s |
|---|---|---|---|---|---|
| ODE | 0 | pysindy | 100% | 100% | 1 |
| ODE system | 0 | pysindy | 100% | 100% | 2 |
| PDE 1-D | 0 | default | 78% | 62% | 94 |

## All data sets together (mean success on the front)

| variant | 0.0 |
|---|---|
| pysindy | 77% |
| default | 72% |

## Median search time, s

| dataset | default | pysindy |
|---|---|---|
| ac | 39.7 | 9.0 |
| burgers | 62.1 | 23.7 |
| burgers_inviscid | 106.5 | 10.9 |
| duffing | 34.6 | 1.8 |
| kdv | 214.6 | 154.8 |
| kdv_cossin | 94.5 | 7.1 |
| ks |  | 225.2 |
| lorenz | 176.6 | 2.3 |
| lv | 42.8 | 1.6 |
| ode | 12.3 | 1.2 |
| pde_compound | 144.1 | 33.3 |
| pde_divide | 284.0 |  |
| vdp | 26.4 | 1.5 |
| wave | 63.7 | 7.7 |

## Failed runs

- ks / default / noise 0 / seed 0: timeout 
- ks / default / noise 0 / seed 1: timeout 
- ks / default / noise 0 / seed 2: timeout 
- ks / default / noise 0 / seed 3: timeout 
- ks / default / noise 0 / seed 4: timeout 
- ns / default / noise 0 / seed 0: timeout 
- ns / default / noise 0 / seed 1: timeout 
- ns / default / noise 0 / seed 2: timeout 
- ns / default / noise 0 / seed 3: timeout 
- ns / default / noise 0 / seed 4: timeout 

## Pending and unsupported runs

- ns / pysindy / noise 0 / seed 0: unsupported the baseline cannot represent the spatial continuity equation; one temporal equation per variable would substitute p_t for continuity
- ns / pysindy / noise 0 / seed 1: unsupported the baseline cannot represent the spatial continuity equation; one temporal equation per variable would substitute p_t for continuity
- ns / pysindy / noise 0 / seed 2: unsupported the baseline cannot represent the spatial continuity equation; one temporal equation per variable would substitute p_t for continuity
- ns / pysindy / noise 0 / seed 3: unsupported the baseline cannot represent the spatial continuity equation; one temporal equation per variable would substitute p_t for continuity
- ns / pysindy / noise 0 / seed 4: unsupported the baseline cannot represent the spatial continuity equation; one temporal equation per variable would substitute p_t for continuity
- pde_divide / pysindy / noise 0 / seed 0: unsupported the baseline cannot represent x*u_t on the left-hand side and has no inverse-x coefficient token
- pde_divide / pysindy / noise 0 / seed 1: unsupported the baseline cannot represent x*u_t on the left-hand side and has no inverse-x coefficient token
- pde_divide / pysindy / noise 0 / seed 2: unsupported the baseline cannot represent x*u_t on the left-hand side and has no inverse-x coefficient token
- pde_divide / pysindy / noise 0 / seed 3: unsupported the baseline cannot represent x*u_t on the left-hand side and has no inverse-x coefficient token
- pde_divide / pysindy / noise 0 / seed 4: unsupported the baseline cannot represent x*u_t on the left-hand side and has no inverse-x coefficient token
