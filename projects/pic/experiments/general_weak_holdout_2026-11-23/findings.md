# Equation-independent weak validation across the default 15 systems

EPDE can choose weak measurements before it sees a proposed equation, fit each candidate once, and compare its frozen coefficients on disjoint observations. A screen on the default 15 systems gave a small apparent gain when the held-out residual was normalized by every term in the equation. The gain disappeared in the twelve-seed confirmation. The fully term-scaled score produced larger nondominated fronts.

## Rule under test

Split the grid along its longest axis by sample count. Fit on the first half and reserve the second half for validation. Choose one window plan for all candidates and all equation slots using only training fields, coordinate columns and the configured derivative orders. Use the native weak-window selector with the median over the derivative targets from every measured field. Keep the selected window shape, stride and smoothness on the held-out block. This experiment used the same two time and four spatial derivative orders for all systems, with the existing half-power gate and a dyad row floor based on configured width seven.

For a candidate written as a homogeneous relation with weak columns $a_j$ and fitted weights $c_j$, compare two ways to scale the mean absolute held-out residual. Held-out WAPE divides by the mean absolute held-out target column, with its coefficient fixed to minus one. The alternative divides by $\sum_j |c_j|\,\mathrm{mean}_{\rm train}|a_j|$. For a fixed relation the alternative is invariant to reordering terms, changing their units and multiplying the equation by a nonzero number. A change to the written target followed by a new least-squares fit can change the relation. Directional dyad also depends on the target. Both fronts here use the same frozen weights and training dyad. Explicit constants enter as ordinary columns. The benchmark labels enter only when composing the fixed evaluation panel and reporting recovery. [Protocol](protocol.txt) and [implementation](screen.py)

## Confirmation on fresh measurement noise

The confirmation used 180 noisy recordings, covering 20 equation slots and 12 seeds per slot. Each comparison used the entire nondominated front of finite candidates in its slot. A registered hit includes the benchmark's accepted equivalent identities. Literal support requires the exact terms of the listed equation. A front membership counts one candidate on one slot and recording.

| Discrepancy, with training dyad | Registered hit / 240 | Literal hit / 240 | One-term extensions on fronts | Total front memberships |
|---|---:|---:|---:|---:|
| Training WAPE | 220 | 208 | 456 | 920 |
| Held-out WAPE | 220 | 208 | 339 | 745 |
| Held-out residual divided by training target magnitude | 220 | 208 | 333 | 744 |
| Held-out residual divided by training magnitude of all terms | 220 | 207 | 349 | 834 |

The training-target control isolates the effect of using multiple terms in the denominator. Normalizing by all terms gained one registered forced KdV case and lost a different one relative to held-out WAPE. The other 238 paired outcomes agreed. Allen–Cahn was 3/12 and forced KdV was 2/12 under both held-out scores. Comparing training WAPE with held-out WAPE also gives 220 hits in each arm, with six gains and six losses across individual cases. Held-out WAPE retained fewer unregistered one-term extensions overall. These counts describe support membership, not a complexity objective. [Per-slot confirmation results](confirmation/slots.csv)

The three-seed screen had given 55/60 registered hits with all-term scaling and 53/60 with held-out WAPE, gaining one Allen–Cahn case and one nonlinear diffusion case. Neither initial gain persisted as an aggregate gain in confirmation. [Screen results](slots.csv)

## Coverage and interpretation

All 225 screen and confirmation fields produced a weak plan meeting the half-power gate. All 1,050 screen fits and 4,199 of 4,200 confirmation fits were resolved. In inviscid Burgers on seed 923212, an unregistered competitor had linearly dependent predictors. Excluding that entire slot leaves both held-out rules at 219/239. Independent front reconstruction from the candidate scores reproduced registered hits and total membership for both held-out rules in both cohorts.

The 15 systems include scalar and coupled ODEs and PDEs on regular grids, with repeated measurement noise on existing trajectories. The panel contains the registered equations, equivalents, deletions and competing structures. These results measure ranking conditional on the structures being available. They do not establish discovery rates in live search, transfer to new initial conditions or recovery from uninformative observations.

Keep native weak WLS with WAPE and dyad as the search comparator. The evidence supports evaluating frozen candidate equations on disjoint observations as a common diagnostic across systems. Replacing WAPE with the all-term normalization would add competing equations to the confirmation fronts for the same registered recovery. The next live comparison is the ordinary native addition versus the fit-aware single addition across all 15 systems, with the same scoring budget, the same validation rule and separate counts for generation, training-front retention and validation-front retention.

Run `python screen.py` for the three-seed screen and `python screen.py --confirm` for the twelve-seed confirmation.
