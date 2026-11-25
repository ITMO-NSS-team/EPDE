# Independent-trajectory confirmation

Use full-record 5% correlated-noise discovery with three fresh seeds (924801–924803) on the forced oscillator, Van der Pol, Duffing, and the coupled Lotka–Volterra system. Reuse 923701–923703 as a separate retrospective cohort with clean, Gaussian and correlated observations. Discovery, window selection, epoch budget and training stability use the original noise campaign. Two single-threaded discovery jobs can run concurrently inside its six-core affinity mask, with below-normal priority and the original host-load admission check.

Score all actual archived systems in four ways: original-record WAPE, mean WAPE on two new recordings of the same trajectory, mean WAPE on two trajectories with new initial states, and single-record two-fold validation with disjoint raw supports. Compare the last method with and without an additional 14-sample gap. The gap is specified for the benchmark's lag-one correlation of 0.8 (0.8^14 is about 0.044). The folds use the archived window geometry and refit coefficients on one side before predicting the other. Candidate supports, frequencies and training stability already saw the original recording, so the fold results assess coefficient validation conditional on the discovered library.

For multi-record replay freeze target, structure, coefficients, frequencies and original dyadic instability. Both fresh-data arms receive the same number of samples, noise seeds and field-relative RMS convention. Initial states are (1.25, −0.75) and (−0.75, 1.25) for the three second-order systems, and (1.4, 0.7) and (0.65, 1.35) for Lotka–Volterra. An additional evaluation pair uses (0.6, 1.7) and (−1.5, −0.3), or (1.6, 1.15) and (0.85, 0.55) for Lotka–Volterra. Validation and evaluation draws use distinct seeds. Preserve the time origin and forcing. The benchmark equations generate these synthetic recordings. The scorer consumes field measurements and uses the catalog only after selection to assess structural recovery.

Report catalog generation, complete Pareto-front membership and front size. Store held-out WAPE for every member of every reported front, indexed by archive system, recording and equation. An undefined held-out ratio is recorded as JSON null. Compare the results within each dataset and seed, including generation failures.

## Forced-KdV measurement check

Fit the fixed physical support, including the supplied forcing token, on 12 new Gaussian and 12 new correlated 5% noise cases. Rotate through the three original correlated-search window geometries in seed order. Compare one recording, the average of two independently perturbed recordings of the same field, and smoothing just the weak dispersion column of their average. Select the two smoothing degrees by predicting each recording's column from the other on the weak-row grid. Evaluate dispersion-column error, coefficient error relative to −1, and WAPE with and without dispersion on both the clean field and a third independently perturbed field. Only evaluation uses clean measurements. This check measures coefficient recovery conditional on the supplied support.

## Single-record check with a genuinely unseen segment

For the forced oscillator and Van der Pol, repeat the three fresh 50-epoch searches using only the first 153 of 320 observations. Generate the 5% perturbation on the full record before splitting, preserving the original paired draw and noise normalization. Compute the Pareto front of the frozen candidates on the training prefix, the next 153 samples (start index 153), and the last 153 samples (start index 167). The last arm leaves 14 raw samples between discovery and validation. Every bank computes its derivatives, windows and coordinate factors within its own segment and uses absolute time. The two validation segments overlap each other. Their comparison includes a shift in time coverage as well as the gap. This separately checks whether catalog structures are generated with a shorter training trajectory.

## KdV follow-up with an independent calibration trajectory

After the repeat-measurement experiment left the dispersion coefficient near zero, test the earlier staged
forced/unforced idea on 12 different noise seeds per family (936811–936822). Match the raw sample count:
each fit gets two 81 × 81 recordings. Sample the unforced soliton record at time indices 20, 22, …, 180 and
space indices 15, 21, …, 495 before adding noise. Use the same window geometry in grid steps on both fields.
Compare averaging two forced recordings, joint forced/unforced regression with equal relative quadratic error
per record, and a staged fit of the two intrinsic coefficients on the unforced data followed by the forcing
amplitude on the forced field. Evaluate all fits on the same clean forced field and third noisy forced
recording. The two dynamics have different physical grids, excitation and forcing. Fixed physical support and a
shared intrinsic law are assumed.

## Multisample Pareto selection

Replay the archived candidates with their fitted coefficients and frequencies frozen. Keep the same
candidate pool as the mean-error comparison. Each equation must meet the original training validity
checks and its mean validation WAPE must be at most 1. Use one WAPE objective per recording per
equation and one training-instability objective per equation. Minimize all objectives and retain all
nondominated systems, including ties. The reported front is the primary output.

Evaluate candidate systems by Pareto dominance on the complete objective vector. Every nondominated member is retained, including exact ties. Front order follows archive indices for lookup. Each method’s held-out evaluation uses the two additional trajectories after the front has been determined.

## Reproduction and use

Run these commands from this directory with the EPDE Python environment.

```powershell
python discover.py
python validate_trajectories.py retrospective
python validate_trajectories.py fresh
python holdout_time.py
python check_simulators.py
python kdv_measurement.py
python kdv_calibration.py
python ns_donors.py
python report_trajectories.py
python test_validation.py
```

To score a completed full-record archive against independent measurements, use
`python score_archive.py ARCHIVE_DIRECTORY recording1.npz recording2.npz`.
Each recording contains arrays named for the measured fields and one one-dimensional coordinate vector
per axis (`axis0`, `axis1`, etc.). Supply the discovery grid, variable units, time reference and forcing
for every record. The JSON output gives indices into the archive's `systems` array and the equation-level
scores of the returned members. Both `front` and `multisample.front` contain the full multisample front. `mean_error_front` contains the control front using mean recording WAPE per equation alongside training instability. `wape_by_recording` follows the listed recording order, with one column per equation.

The current replay writes `pareto_scores/`, `pareto_runs.csv`, and `time_results/*/*/pareto.json`. Earlier scalar-shortlist experiments are archived in `scores/`, `runs.csv`, and `comparison.json`.
