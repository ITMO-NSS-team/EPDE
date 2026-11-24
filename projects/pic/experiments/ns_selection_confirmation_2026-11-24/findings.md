# Complete-system selection on clean Navier–Stokes

Each cell counts search seeds on the same clean flow. A hit requires both momentum equations and continuity together. The global front checks all admitted candidates against one another on WAPE and dyadic coefficient instability.

## Fresh seeds at equal fit budget

| Method | Complete system assembled | On global first front |
|---|---:|---:|
| Native | 0/3 | 0/3 |
| Joint reassignment | 0/3 | 0/3 |
| Exchange only (inherited target) | 0/3 | 0/3 |
| Exchange with target choice for the copy | 2/3 | 0/3 |
| Exchange and joint reassignment | 2/3 | 1/3 |

| Seed | Method | Native fits | Assembly refits | Total fits | Complete candidates | Nondominated complete candidates |
|---:|---|---:|---:|---:|---:|---:|
| [24112401](results/24112401/comparison.json) | Native | 273,797 | 0 | 273,797 | 0 | 0 |
| [24112401](results/24112401/comparison.json) | Joint reassignment | 273,683 | 114 | 273,797 | 0 | 0 |
| [24112401](results/24112401/comparison.json) | Exchange only (inherited target) | 272,837 | 960 | 273,797 | 0 | 0 |
| [24112401](results/24112401/comparison.json) | Exchange with target choice for the copy | 271,207 | 2,590 | 273,797 | 4 | 0 |
| [24112401](results/24112401/comparison.json) | Exchange and joint reassignment | 269,802 | 3,995 | 273,797 | 6 | 0 |
| [24112402](results/24112402/comparison.json) | Native | 164,010 | 0 | 164,010 | 0 | 0 |
| [24112402](results/24112402/comparison.json) | Joint reassignment | 164,003 | 7 | 164,010 | 0 | 0 |
| [24112402](results/24112402/comparison.json) | Exchange only (inherited target) | 163,646 | 364 | 164,010 | 0 | 0 |
| [24112402](results/24112402/comparison.json) | Exchange with target choice for the copy | 163,322 | 688 | 164,010 | 0 | 0 |
| [24112402](results/24112402/comparison.json) | Exchange and joint reassignment | 163,062 | 948 | 164,010 | 0 | 0 |
| [24112403](results/24112403/comparison.json) | Native | 152,244 | 0 | 152,244 | 0 | 0 |
| [24112403](results/24112403/comparison.json) | Joint reassignment | 152,203 | 41 | 152,244 | 0 | 0 |
| [24112403](results/24112403/comparison.json) | Exchange only (inherited target) | 151,967 | 277 | 152,244 | 0 | 0 |
| [24112403](results/24112403/comparison.json) | Exchange with target choice for the copy | 151,498 | 746 | 152,244 | 3 | 0 |
| [24112403](results/24112403/comparison.json) | Exchange and joint reassignment | 151,074 | 1,170 | 152,244 | 6 | 2 |

EPDE’s native end-of-run archive contained a complete system on 0/3 seeds.

### Equation placement on seeds gained by joint reassignment

The second method in the table can move any equation to another field slot and choose its target. Within each method, all placements face the same global pool of rivals.

| Seed | Method | u slot | v slot | p slot | Complete | Nondominated |
|---:|---|---|---|---|---:|---:|
| 24112403 | Exchange with target choice for the copy | X momentum | Continuity | Y momentum | 3 | 0 |
| 24112403 | Exchange and joint reassignment | Continuity | Y momentum | X momentum | 2 | 2 |
| 24112403 | Exchange and joint reassignment | X momentum | Continuity | Y momentum | 4 | 0 |

### Equations available at the assembly checkpoint

This diagnostic uses benchmark labels after selection. Each count is the number of complete parent candidates containing the indicated law or combination, including registered identities.

| Seed | Source | Candidates | X momentum | Y momentum | Continuity | Continuity with either momentum |
|---:|---|---:|---:|---:|---:|---:|
| 24112401 | Eligible native handovers | 109,767 | 4,286 | 4,132 | 2,765 | 195 |
| 24112401 | Assembly parents | 1,219 | 438 | 292 | 78 | 52 |
| 24112402 | Eligible native handovers | 81,707 | 0 | 28 | 2,969 | 0 |
| 24112402 | Assembly parents | 2,653 | 0 | 0 | 207 | 0 |
| 24112403 | Eligible native handovers | 65,445 | 5,400 | 0 | 2,488 | 167 |
| 24112403 | Assembly parents | 1,582 | 389 | 0 | 118 | 29 |

For correct equations seen during search and absent from the assembly parents, the table counts the fewest distinct competing supports that dominated a matching equation on its own slot’s two objectives.

| Seed | Missing law | Distinct matching fitted equations | Minimum dominating supports |
|---:|---|---:|---:|
| 24112402 | Y momentum | 1 | 1 |

### Reusing equations across parents and epochs

All rows reuse the same 150-epoch search budget per seed. The two cross-parent rules differ only in which epoch-end fronts can supply equations. The counts below are complete candidates, with registered identities accepted.

| Seed | Cross-parent donor pool | Distinct donor structures | Refits | Total fits | Complete | Unbeaten by native candidates | On global first front |
|---:|---|---:|---:|---:|---:|---:|---:|
| [24112401](results/24112401/donor_history.json) | Epoch 100 | 180 | 3,995 | 273,797 | 12 | 0 | 0 |
| [24112401](results/24112401/donor_history.json) | Epochs 1–100 | 811 | 16,827 | 273,797 | 12 | 0 | 0 |
| [24112402](results/24112402/donor_history.json) | Epoch 100 | 72 | 948 | 164,010 | 0 | 0 | 0 |
| [24112402](results/24112402/donor_history.json) | Epochs 1–100 | 611 | 8,345 | 164,010 | 3 | 3 | 3 |
| [24112403](results/24112403/donor_history.json) | Epoch 100 | 52 | 1,170 | 152,244 | 6 | 2 | 2 |
| [24112403](results/24112403/donor_history.json) | Epochs 1–100 | 553 | 11,358 | 152,244 | 6 | 2 | 2 |

## Original seeds with extra fits after epoch 150

| Method | Complete system assembled | On global first front |
|---|---:|---:|
| Native | 1/3 | 1/3 |
| Joint reassignment | 1/3 | 1/3 |
| Exchange only (inherited target) | 1/3 | 1/3 |
| Exchange with target choice for the copy | 3/3 | 1/3 |
| Exchange and joint reassignment | 3/3 | 2/3 |

| Seed | Method | Native fits | Assembly refits | Total fits | Complete candidates | Nondominated complete candidates |
|---:|---|---:|---:|---:|---:|---:|
| [923701](results/923701/comparison.json) | Native | 167,152 | 0 | 167,152 | 0 | 0 |
| [923701](results/923701/comparison.json) | Joint reassignment | 167,152 | 19 | 167,171 | 0 | 0 |
| [923701](results/923701/comparison.json) | Exchange only (inherited target) | 167,152 | 167 | 167,319 | 0 | 0 |
| [923701](results/923701/comparison.json) | Exchange with target choice for the copy | 167,152 | 456 | 167,608 | 2 | 0 |
| [923701](results/923701/comparison.json) | Exchange and joint reassignment | 167,152 | 701 | 167,853 | 6 | 4 |
| [923702](results/923702/comparison.json) | Native | 299,931 | 0 | 299,931 | 0 | 0 |
| [923702](results/923702/comparison.json) | Joint reassignment | 299,931 | 188 | 300,119 | 0 | 0 |
| [923702](results/923702/comparison.json) | Exchange only (inherited target) | 299,931 | 1,415 | 301,346 | 0 | 0 |
| [923702](results/923702/comparison.json) | Exchange with target choice for the copy | 299,931 | 4,108 | 304,039 | 4 | 0 |
| [923702](results/923702/comparison.json) | Exchange and joint reassignment | 299,931 | 6,411 | 306,342 | 6 | 0 |
| [923703](results/923703/comparison.json) | Native | 175,385 | 0 | 175,385 | 4 | 4 |
| [923703](results/923703/comparison.json) | Joint reassignment | 175,385 | 431 | 175,816 | 12 | 8 |
| [923703](results/923703/comparison.json) | Exchange only (inherited target) | 175,385 | 2,768 | 178,153 | 4 | 4 |
| [923703](results/923703/comparison.json) | Exchange with target choice for the copy | 175,385 | 8,377 | 183,762 | 4 | 4 |
| [923703](results/923703/comparison.json) | Exchange and joint reassignment | 175,385 | 12,842 | 188,227 | 12 | 8 |

EPDE’s native end-of-run archive contained a complete system on 1/3 seeds.

### Equation placement on seeds gained by joint reassignment

The second method in the table can move any equation to another field slot and choose its target. Within each method, all placements face the same global pool of rivals.

| Seed | Method | u slot | v slot | p slot | Complete | Nondominated |
|---:|---|---|---|---|---:|---:|
| 923701 | Exchange with target choice for the copy | Continuity | Y momentum | X momentum | 2 | 0 |
| 923701 | Exchange and joint reassignment | Continuity | Y momentum | X momentum | 2 | 0 |
| 923701 | Exchange and joint reassignment | X momentum | Continuity | Y momentum | 4 | 4 |

[Independent three-seed confirmation of donor retention](retention_confirmation.md)

[Experiment protocol](protocol.md)
