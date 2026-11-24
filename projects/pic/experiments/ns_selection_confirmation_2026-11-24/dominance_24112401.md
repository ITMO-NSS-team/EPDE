# Why the 12 complete systems lost on seed 24112401

Each complete system had 2–5 dominators among 153,971 native handovers at the historical-donor arm’s fit cutoff. Every candidate had a rival sharing the same two velocity-slot equation structures. In the pressure slot the rival fitted a pressure–Poisson relation and improved both objectives. EPDE skips the objectives of structurally identical slots in this comparison.

| Candidate fit IDs (u, v, p) | Momentum law in pressure slot | Rival pressure fit ID | Candidate WAPE | Rival WAPE | Candidate instability | Rival instability |
|---|---|---:|---:|---:|---:|---:|
| 1821, 444, 105528 | Y momentum | 789 | 0.000479961 | 0.000173223 | 0.00506258 | 0.000138035 |
| 1821, 444, 105710 | Y momentum | 789 | 0.000479076 | 0.000173223 | 0.00511811 | 0.000138035 |
| 1894, 3638, 103676 | X momentum | 1220 | 0.000307563 | 0.000144421 | 0.00453618 | 7.71664e-05 |
| 1894, 3638, 104530 | X momentum | 1220 | 0.000310365 | 0.000144421 | 0.0039694 | 7.71664e-05 |
| 1894, 6006, 103676 | X momentum | 1220 | 0.000307563 | 0.000144421 | 0.00453618 | 7.71664e-05 |
| 1894, 6006, 104530 | X momentum | 1220 | 0.000310365 | 0.000144421 | 0.0039694 | 7.71664e-05 |
| 4415, 444, 105528 | Y momentum | 789 | 0.000479961 | 0.000173223 | 0.00506258 | 0.000138035 |
| 4415, 444, 105710 | Y momentum | 789 | 0.000479076 | 0.000173223 | 0.00511811 | 0.000138035 |
| 5338, 444, 105528 | Y momentum | 1220 | 0.000479961 | 0.000144421 | 0.00506258 | 7.71664e-05 |
| 5338, 444, 105710 | Y momentum | 1220 | 0.000479076 | 0.000144421 | 0.00511811 | 7.71664e-05 |
| 6147, 444, 105528 | Y momentum | 1220 | 0.000479961 | 0.000144421 | 0.00506258 | 7.71664e-05 |
| 6147, 444, 105710 | Y momentum | 1220 | 0.000479076 | 0.000144421 | 0.00511811 | 7.71664e-05 |

The table selects the native rival with the fewest changed structures and then the earliest handover. Each column below gives the coefficients of a fitted pressure-slot relation whose weighted sum of terms approximates zero. Subscripts denote derivatives.

| Term | Fit 789 | Fit 1220 |
|---|---:|---:|
| pᵧᵧ | -1.000000 | -1.000000 |
| pₓₓ | -0.999959 | -1.000082 |
| uᵧ vₓ | -2.000085 | -2.000290 |
| uₓ vᵧ | — | 2.000371 |
| (uₓ)² | -1.999806 | — |

The two supports agree under incompressibility, which gives v_y = −u_x. The pressure–Poisson equation follows by taking the spatial divergence of momentum. Replacing one momentum equation by this relation can lose information about acceleration. For example, with spatially constant pressure and velocity (u(t), 0), continuity, Y momentum and pressure–Poisson all hold, even when u changes in time and the X-momentum equation fails. This is an equation-level example without additional boundary conditions.

[All candidate and witness scores](results/24112401/donor_dominance.json)
