# When to assemble the clean Navier–Stokes system

Delaying one assembly round from epoch 100 to epoch 125 recovered one additional complete system across three paired search seeds on the same clean flow. Both donor rules improved from 2/3 seeds to 3/3 on the global first Pareto front, with one gain and zero losses each. At epoch 125 the current front supplied enough equations on every seed. Pooling every prior front took 43,008 assembly fits across seeds, compared with 8,365 for the current front.

| Seed | Donor equations | Complete candidates at 100 → 125 | Complete candidates on global front at 100 → 125 | Assembly fits at 100 → 125 | Total fits per arm |
|---:|---|---:|---:|---:|---:|
| [24112404](results/24112404/assembly_timing.json) | Current front | 0 → 6 | 0 → 4 | 990 → 1,385 | 254,702 |
| 24112404 | All fronts to assembly | 0 → 6 | 0 → 4 | 8,843 → 11,369 | 254,702 |
| [24112405](results/24112405/assembly_timing.json) | Current front | 6 → 6 | 4 → 4 | 714 → 966 | 156,577 |
| 24112405 | All fronts to assembly | 9 → 9 | 1 → 1 | 9,423 → 9,807 | 156,577 |
| [24112406](results/24112406/assembly_timing.json) | Current front | 6 → 6 | 4 → 4 | 5,669 → 6,014 | 173,549 |
| 24112406 | All fronts to assembly | 9 → 9 | 1 → 1 | 19,222 → 21,832 | 173,549 |

On seed 24112404, continuity was present at epoch 100. Y momentum first appeared at epoch 118. At epoch 125 the current front had two registered Y-momentum structures, one continuity structure and zero X-momentum structures. Swapping the two velocity fields and the two spatial axes maps Y momentum to X momentum, which supplied the missing equation. The search tried those exchanges using the observed term vocabulary. Benchmark identities were used only to check the resulting systems. [Native donor history](retention_confirmation.md)

All complete candidates in the table came from assembly. Each contains both momentum equations and continuity or a registered equivalent in a single system. Each comparison admits all native candidates up to its fit cutoff and all assembly candidates. It compares WAPE and dyadic coefficient instability per equation slot. Every first-front survivor in the table also survived a check against all remaining native candidates through epoch 150.

The replay used the saved fronts and weak rows from the full 200 × 50 × 100 record. Each seed’s original 150-epoch fit budget covered every requested assembly fit, with the balance allocated to native search. Replaying epoch 100 reproduced the original counts and charges for every arm. Epoch 125 was chosen after inspecting the late arrival of Y momentum on seed 24112404, so the gain is exploratory. [Replay script](assembly_timing.py) · [Protocol](protocol.md)
