# Independent confirmation of donor retention on clean Navier–Stokes

Three previously unused search seeds share the original full 200 × 50 × 100 measurement record. For each seed the two assembly rules use the same native trajectory, weak rows, seven-term cap, beam width of 1,000 and total fit budget from 150 native epochs. Each permits cross-parent mixing, one field/axis exchange and joint choice of equation slots and derivative targets. The only selection change is which fronts supply equations at epoch 100.

| Donor pool | Seeds with complete system | Seeds with complete system on global first front |
|---|---:|---:|
| Epoch 100 | 2/3 | 2/3 |
| Epochs 1–100 | 2/3 | 2/3 |

Historical retention gained 0 and lost 0 paired first-front recoveries. The other 3 pairs agreed.

| Seed | Donor pool | Donor structures | Assembly refits | Total fits | Complete candidates | On global first front | Still unbeaten against full native run |
|---:|---|---:|---:|---:|---:|---:|---:|
| [24112404](results/24112404/donor_history.json) | Epoch 100 | 76 | 990 | 254,702 | 0 | 0 | 0 |
| [24112404](results/24112404/donor_history.json) | Epochs 1–100 | 619 | 8,843 | 254,702 | 0 | 0 | 0 |
| [24112405](results/24112405/donor_history.json) | Epoch 100 | 35 | 714 | 156,577 | 6 | 4 | 4 |
| [24112405](results/24112405/donor_history.json) | Epochs 1–100 | 395 | 9,423 | 156,577 | 9 | 1 | 1 |
| [24112406](results/24112406/donor_history.json) | Epoch 100 | 393 | 5,669 | 173,549 | 6 | 4 | 4 |
| [24112406](results/24112406/donor_history.json) | Epochs 1–100 | 1,115 | 19,222 | 173,549 | 9 | 1 | 1 |

A complete candidate contains both momentum equations and continuity in one system, including registered equivalent forms. The global front compares WAPE and dyadic instability per slot against the admitted native and assembled candidates. The last column also tests survivors against native candidates beyond each arm’s fit cutoff.

## Availability on seeds where both rules missed assembly

| Seed | Law | First native fit (epoch) | First epoch-end front | Front at epoch 100 | Front at epoch 125 |
|---:|---|---:|---:|---|---|
| 24112404 | X momentum | — | — | No | No |
| 24112404 | Y momentum | 118 | 118 | No | Yes |
| 24112404 | Continuity | 1 | 1 | Yes | Yes |

## Elapsed time

| Seed | Donor pool | Full native run (min) | Native through allotted fit (min) | Assembly and assessment (min) | Sum of allotted native and assembly time (min) |
|---:|---|---:|---:|---:|---:|
| 24112404 | Epoch 100 | 51.5 | 51.3 | 0.4 | 51.7 |
| 24112404 | Epochs 1–100 | 51.5 | 50.2 | 1.1 | 51.3 |
| 24112405 | Epoch 100 | 31.3 | 31.2 | 0.3 | 31.5 |
| 24112405 | Epochs 1–100 | 31.3 | 29.6 | 1.4 | 31.0 |
| 24112406 | Epoch 100 | 35.5 | 34.5 | 0.9 | 35.4 |
| 24112406 | Epochs 1–100 | 35.5 | 31.7 | 2.0 | 33.8 |

Elapsed time uses a monotonic clock. The two arms have separate weak-column caches. Each sum includes native execution through the last charged fit and the timed assembly and assessment. Shared archive I/O and scorer setup are excluded. The three seed workers ran concurrently.

[Protocol](protocol.md) · [Earlier seeds](findings.md) · [Pressure-slot dominance on seed 24112401](dominance_24112401.md)
