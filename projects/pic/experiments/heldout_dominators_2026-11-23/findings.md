# Why correct systems disappeared on held-out data

At the 3,000-total-fit checkpoint in the 15-system confirmation, 111 of 180 runs assembled a correct system and 79 retained one on the held-out front. Of the 32 that assembled and lost it, **31 had an eligible correct system beaten on the two objectives, and one wave run exceeded the held-out error cutoff**. Of those 31, Allen–Cahn accounted for 11, wave for 10, nonlinear diffusion for seven, the PDE with a reciprocal coefficient for two, and Van der Pol for one. [Confirmation handovers](../candidate_frequency_2026-11-23/confirmation/systems.csv) · [Replay by run](runs.csv)

The analysis includes every Allen–Cahn and wave confirmation run plus every other assembled-but-lost system. Each comparison uses actual system handovers from the frequency-refinement arm. Discovery used 5% measurement noise. WAPE is the sum of absolute weak residuals divided by the sum of absolute target values on that data split. Dyad measures coefficient instability across training blocks. A rival dominates a correct equation when it scores at least as well on both measures and strictly better on at least one. Coefficients and dyad were frozen throughout the replay. [Audit code](audit.py)

## Allen–Cahn

All 11 assembled correct systems had a same-target rival with better or equal scores on both objectives. For each seed, we took the eligible correct equation with the lowest held-out WAPE, then identified its dominating held-out front member with the lowest WAPE. **All 11 rivals retained both reaction terms, $u$ and $u^3$, and omitted the linear diffusion term.** Their median paired improvement was only 0.23 percentage points of held-out WAPE. The reaction-only equation beat the selected correct candidate on both measures with the same target in five runs. In seed 923701, dropping diffusion changed held-out WAPE from 5.229% to 5.182% and dyad from 0.821 to 0.0124. [Correct equations](correct_points.csv) · [Rival equations and coefficients](witnesses.csv)

To measure how large the missing term was, we evaluated the published law on the noise-free held-out field using the archived windows. The median absolute diffusion contribution was **0.30%** of the absolute $u_t$ target, using sums over the weak windows. With the published coefficients, median held-out WAPE was 0.0017% on the clean field and 3.17% on the noisy field. [Clean and noisy measurements](runs.csv)

## Wave

The correct system appeared in 11 runs. Ten lost to eligible competitors. On seed 923704 the correct candidate had held-out WAPE 101.99%, above the 100% cutoff. In the remaining run the search never scored the correct law. Among the ten dominated runs, the lowest-WAPE dominating front member always predicted spatial curvature, $u_{xx}$. The best eligible correct equation per seed had median held-out WAPE 92.7%, compared with 14.8% for these selected rivals. [Run fates](runs.csv) · [Rival objectives](witnesses.csv)

With the **published** coefficient in $u_{tt}=0.04u_{xx}$, median held-out WAPE was **0.21% on the clean data and 92.8% on the noisy data**. We also measured each weak derivative's change under added noise, dividing the sum of absolute changes by the sum of its absolute clean values. The median change was 264% for $u_{tt}$ and 3.5% for $u_{xx}$. This identifies a measurement problem specific to the time derivative on these windows. An exact coefficient still leaves a large residual. [Clean and noisy measurements](runs.csv)

## Effect of comparing equations with the same target

We recomputed the held-out fronts within each prediction target and kept their union. This retrospective check used every eligible archived candidate and the same frozen scores. Across all 12 wave runs, it retained the correct law on **5/12**, up from 0/12, while increasing the median candidate count from **6 to 28**. Allen–Cahn stayed at **0/12**, and the median count rose from 5.5 to 16.5. These counts quantify the role of cross-target comparisons. A method comparison with a fixed number of returned candidates would be needed to measure an improvement in selection accuracy. [Front sizes and recovery by seed](runs.csv)

Run `python audit.py` from this directory in the repository research environment to regenerate the tables. The script checks the original held-out front size and recovery for each included run and compares reconstructed training WAPE to the archive. The maximum absolute discrepancy was $2.11\times10^{-15}$. Correct equations and ground-truth coefficients enter only in this post-search diagnosis.
