# Coupled-system recovery in Lorenz and Navier–Stokes

Across six native searches, **none of the 74 mating populations contained a pair of parents that jointly covered all three registered laws in compatible slots**. Lorenz scored two of its three laws on each seed. Navier–Stokes scored two on one seed and one on the other two. Bringing existing parents together could therefore recover at most two laws on these trajectories. [Run-level audit](by_run.csv)

In 14 populations, two surviving individuals carried correct laws in different slots that no member carried together. Native selection actually mated complementary parents four times, all on Lorenz. The assembled child's fitted descendants retained both laws in three of those matings. The within-equation crossover child's descendants retained both in zero. On Navier–Stokes, four populations offered a partial combination and selection drew none of the available complementary pairs. [Actual matings](matings.csv) · [Live populations](populations.csv)

All six instrumented searches reproduced the archived native points, coefficients, handovers, training-window plans and both evaluation clocks. The per-run compressed archive contains the selected parents, post-crossover structures, post-mutation fit inputs and scored outcomes, including sibling variants. Empty descendant lists represent children or fit attempts that had no recorded outcome before the 3,000-evaluation cutoff. [Trace implementation](trace_lineage.py) · [Protocol](protocol.md)

## Donor-block crossover

The repeated partial combinations triggered the archived donor-block comparison. Native and donor-block crossover recovered **the same 19 of 45 complete systems** at each budget. Both recovered all three Lotka–Volterra runs and zero Lorenz or Navier–Stokes runs. At equal total fits, individual held-out law recovery fell from 32/60 to 30/60. The donor-block rule missed the threshold for confirmation. [Paired results and front sizes](mix_screen/by_system.csv) · [Law-level outcomes](mix_screen/laws.csv)

The nine active coupled searches tried mixing on 6,312 equation slots and accepted 3,533 edits, at a cost of 3,641 extra native objective evaluations. Scalar searches follow the same native path in both arms because they have no per-variable mating plan. Their archived results enter the comparison once per arm. Both arms recovered the separate scalar control on all three seeds. [Run-level costs](mix_screen/work.csv) · [Comparator hashes](mix_screen/manifest.json)
