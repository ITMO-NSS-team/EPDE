# Complementary weak windows

Forced KdV motivated the design. The transfer panel comprises the other 14 systems from the existing disjoint-block weak-form benchmark. The first six archived iid 5% noise draws per trajectory form the screen, and the remaining six form the confirmation. The selector and scoring rules are identical in both cohorts.

## Training

Start from the existing data-selected plan for the training half of each observation. Verify its data hash. Probe the training field with the measured variables, coordinate columns and single derivatives in the benchmark library. The selector accepts the maximum order on each axis as input. In this comparison the library allows up to order two in time and four on each spatial axis.

Try halving or doubling the width along one axis or moving centers by half a stride on that axis. The smoothness, edge cut and lattice shape come from the primary plan. A second geometry measures the odd cells of a checkerboard wherever its full support fits. The primary geometry covers all remaining cells. This preserves the row count and the spatial ordering used by the dyad objective.

Rank each design by the number of column combinations passing EPDE's existing signal-over-noise criterion. Estimate noise from the training fields and use the same perturbation across candidate geometries. Accept an increase in count only when a second synthetic noise draw also gives a strict increase over the primary plan. Ties among improving candidates favor the lower summed noise energy in the retained directions. Select once before reading the equation panel. All fits within a recording share the decision.

## Measurement

Fit the same fixed panel of candidate equations in each arm. Evaluate the frozen coefficients on the disjoint validation half using the same primary windows. Form the nondominated front of held-out WAPE and training dyad. Count slots retaining any registered law or equivalent and the unregistered one-term additions surviving on those fronts. For the literal law, compare fitted coefficient vectors across independent noise draws on the same trajectory. Scale each coefficient by its held-out column RMS divided by target RMS, averaged between the two draws, and report the median pairwise Euclidean change per system and equation slot.

Both arms have the same CPU ceiling of 30 process-seconds per recording after loading the shared archived plan and data. Record actual CPU consumption including selection for the complementary arm. One Windows worker runs with a 2% host CPU hard cap, one logical CPU and one numerical thread. The experiment replays the baseline WAPE and dyad against the archive before reporting results.

Run `python -B run.py`, then `python -B run.py --confirm`, then `python -B summarize.py` from this directory in the EPDE environment.
