# Prospective collision tests

## Frozen shared interaction test at 40, 80 and 400 ms

Use the first independently acquired compatible recording and register its identity, sampling rate, motor setting and spatial calibration before examining its outcomes. One population-sweep ZIP counts as one recording across all occupancy stages. The model was developed on five 25 Hz swarms and three 50 Hz sweeps. Replays and excerpts of those sessions are development material. A new session is needed to test independent replication.

The [version 1 coefficient file](shared_pair_v1.json) freezes the validation-selected, horizon-specific EPDE equations with recent-velocity persistence as the primary model. The secondary compact equations use two corrections per direction. The endpoints with historical bidirectional gains are normal motion at 40 and 80 ms and sideways motion at 40, 80 and 400 ms. Report normal motion at 400 ms as a secondary endpoint. All speeds and prediction errors in the scorer output use mm/s. Evaluate each component and horizon against the strongest of five frozen forecasts: recent-velocity persistence, its clearance clamp, a velocity-history regression, the learned isolated-motion forecast and its clearance clamp. The same gate events contribute to all forecasts.

From the repository root, score one acquisition into a new directory.

```powershell
python projects/robot/collision/shared_interaction_score.py "D:/data/new_session.npz" --output "projects/robot/results/shared_interaction/prospective/session_01"
python projects/robot/collision/shared_interaction_score.py "D:/data/new_sweep.zip" --calibration "D:/data/new_sweep_scale.csv" --output "projects/robot/results/shared_interaction/prospective/sweep_01"
```

- A calibrated NPZ contains `raw_pos[frame,robot,xy]`, unwrapped `phi[frame,robot]` in radians, per-pose `quality_flags` (zero is usable), unique `ids`, contiguous zero-based `frame`, `recording`, `fps`, and `pwm`. Supply `arena=[center_x,center_y,center_radius]` and `diameter` or `cap_px` in the units of the positions. The radius describes the outer envelope of **robot centers**. The command also accepts `--arena CX CY R` and `--diameter D`. The physical robot diameter is 48.7 mm.
- A 50 Hz sweep ZIP uses the existing paired `data_N=…` and `time_N=…` stage exports. The stage calibration CSV contains `N,center_x,center_y,radius,diameter_archive_units`. A shared CSV can include `experiment` matching the ZIP stem. A single independently measured calibration can instead be passed with `--arena` and `--diameter`. The filename encodes PWM, or the operator can pass the recorded value with `--pwm`.
- Keep every eligible inward crossing of 1.10 diameters after clearance of 1.30 diameters, including passes and sustained encounters. Use the gate normal throughout. The scorer applies the study's 440 ms history, 400 ms follow-up, tracking-quality and wall masks. It scores all primary and secondary endpoints on the same crossings.
- Keep the receipt with the coverage, forecasts and scores. It records input and model fingerprints, known overlap, calibration and whether scoring code matches the freeze. A previously unseen file still needs an independent acquisition history. A recording with zero eligible crossings produces coverage and a receipt.
- Report RMSE, MAE and within-recording (R^2) for every forecast. For each endpoint report the full and compact improvements over the lowest reference RMSE, including their paired 95% intervals. Each of 2,000 draws resamples occupied ten-second blocks and reselects the strongest frozen reference. Intervals describe within-recording uncertainty and are reported per endpoint. Keep negative gains and inconclusive intervals. The outcome concerns finite-window motion after approach.

The [cross-family comparison](../../results/shared_interaction/compact/transfer.csv) records the development results. [Replay checks](../../results/shared_interaction/compact/replay.csv) compare native acquisition inputs against the archived encounter features.

## Independent swarm recording

Use the first independent, technically valid swarm recording collected under the existing 25 Hz acquisition geometry, with PWM 10 or 20 and a 40 or 41 cm arena. The five development recordings and their excerpts remain development data. A new session can repeat a setup. Record the session ID, physical robot IDs, run duration, PWM, camera coordinate system and arena before inspecting trajectory outcomes. Record six minutes if practical, or the previous 330-second duration. Report the actual duration and the overlap in robot IDs with training. Preserve the original tracks.

The package in [frozen_v1.json](frozen_v1.json) contains the pooled shared normal-velocity equation, the original Cartesian law, robot and arena calibration, the affine baseline, the compact equation, constants and implementation hashes. The five development training prefixes have equal weight. A new robot ID uses the frozen mean at its PWM. Camera position and scale must be registered to the training coordinates using an external calibration if the camera moved.

The primary endpoint is the next 0.4 s of measured relative velocity along the observed normal at the forecast origin. Use one unordered pair per eligible origin, the established warmup and 253 px cutoff, and one joint velocity per robot. The primary comparison is the shared correction against the affine baseline with the **same** simultaneous-contact update. Report their paired difference in R², RMSE and pair count. The supplied interval resamples occupied 10-second time blocks 2,000 times and describes uncertainty within this recording. A positive point estimate favors the shared model. An interval spanning zero leaves the direction uncertain. A single new recording measures transfer to that recording.

Secondary scores include the original, clearance-only and compact models, independent pair predictions, and isolated versus multiple-neighbor strata at the origin. Preserve all prespecified scores. An isolated pair at a forecast origin is a different criterion from continuous isolation throughout an encounter. Document capture failures, missing IDs, unscorable intervals and counts before interpreting results.

From the repository root, score a trusted lab pickle with the standard swarm filename or a NumPy archive.

```powershell
python projects/robot/collision/prospective_collision.py score 'C:/data/new_session.npz'
```

The archive fields are `raw_pos` (frames × robots × 2 in camera px), `phi` (unwrapped heading radians), `ids` (unique integer IDs), `frame` (contiguous indices from zero), `recording` (session label), `arena_cm`, `pwm`, `fps` and `cap_px`. Keep the measurement calibration of 219.54929004593717 px per 48.7 mm diameter. Preserve the original left/right and heading conventions. The scorer exports origin-level predictions, scores and an input/model hash receipt into a track-specific folder under `results/collision_transfer/preprocessed/prospective`. The `verify` subcommand checks the implementation against archived development-prefix inputs and predictions.

## Two-robot mechanism experiment

**Question.** For the same incoming motion, does contact change relative normal speed and signed transverse speed? The primary spin comparison uses grazing contacts and near misses. Head-on and oblique collisions characterize how the normal response varies with incidence.

Save a working copy of the [randomized trial sheet](two_robot_trials.csv) for logging measurements. Four balanced blocks each contain 32 settings. The four signed spin assignments use the sign of the unwrapped heading derivative in tracking coordinates. The targets for the magnitude of the relative incoming velocity are 150 and 290 px/s, approximately the historical lower and upper quartiles in short approaches. Use each block as a separate session when feasible. The 128 trials form an initial balanced pilot. Precision across sessions will determine the sample size of a subsequent confirmatory study.

1. Calibrate the camera with the stationary arena and both robot diameters. Measure each robot's free speed and spin before its block. Reuse physical pairs across conditions where hardware permits. If spin direction is fixed to each robot, rotate multiple pairs through each feasible condition, record their identities, and treat differences between spin groups as partly confounded by hardware.
2. Keep the encounter in the arena interior. Set parallel or opposing launch headings as appropriate to realize the assigned relative speed and signed impact parameter. Here the impact parameter is the lateral offset of the extrapolated incoming relative trajectory, divided by the robot diameter. The targets are 0, 0.65, 0.97 and 1.03 diameters, with mirrored nonzero offsets balanced across blocks. Use independent releases and identical actuation rules through the encounter. Log assigned geometry and actual closest approach separately. A missed target remains in the trial record.
3. Capture full poses for at least two seconds before the approach and one second after closest approach. Start outside the interaction neighborhood so histories can resolve. Use synchronized high-speed video if available, retaining a calibrated 25 Hz stream for direct comparison. Mark visible physical contact independently of the kinematic outcomes. Record ambiguous frames and the camera timing uncertainty.
4. Define a common gate at the first inward crossing of 1.3 diameters. Measure incoming velocity over the preceding 0.2 s and retain the gate normal and tangent for all projections. Report change in normal and tangential relative velocity from before the gate to the first 0.4 s after closest approach. Also report outcomes at a fixed 0.4 s from the gate and the velocity change in the preceding 0.2 s as timing and pretrend checks.
5. Compare assigned grazing-contact and near-miss trials within speed, spin assignment, block and mirror orientation. Report actual contact rates in each assignment, signed tangential changes separately for ++ and --, and both mixed-spin controls. Estimate uncertainty across independent sessions and physical pairs. Keep the full assigned sample for the primary comparison and label analyses by verified contact status as secondary.

The independent swarm recording tests predictive transfer. The controlled pair trials test the contact mechanism under deliberately varied geometry and spin. The two-robot trials have their own sampling distribution and are reported separately.

## High-speed head-on restitution measurement

The [48-run acquisition sheet](high_speed_head_on_trials.csv) crosses three relative approach speeds (approximately 33, 49 and 64 mm/s) with four signed spin assignments in each of four randomized session blocks. The low and high speeds reuse the original 32 head-on settings. The 16 middle-speed settings extend that design. Preserve the original trial reference when the same physical run supplies both analyses. Distribute physical pairs across sessions and rotate which robot takes each starting side.

1. Before starting the randomized blocks, adjust the camera on a few practice passes. Try 240 fps if the hardware supports it. Inspect the native timestamps, exposure blur and rim visibility at the fastest speed. Increase the frame rate or light level as needed to resolve several sharp, separated body-center positions immediately before and after impact. Check localization repeatability on stationary and freely spinning robots. Keep practice runs separate. Log the actual native frame rate and exposure.
2. Register the image to the arena plane with fixed fiducials and the measured 48.7 mm diameter. Estimate centers from the circular body outlines and check their agreement with the marker trajectories during free spin. Save both raw footage and calibrated tracks. Keep the drive setting fixed across the approach and release. Log actual spin signs, speeds, impact parameter and any deviation from the target geometry.
3. Review the rims without displaying velocity outcomes. Enter `single`, `multiple`, `miss` or `uncertain` in `observed_contact`. For a single contact, mark the last clear frame before touch, first and last visibly touching frames, and first clear frame after release. Use zero-based native frame indices and actual capture timestamps. A second reviewer can check the brackets independently. Record disagreements and unresolved cases in the notes.
4. Export a CSV with columns `trial_id,frame,time_s,x_a_mm,y_a_mm,x_b_mm,y_b_mm`. Each row contains synchronized arena-registered body centers. The scorer uses the contact-frame normal and fits a line to three clear positions per side. It also exports five-position estimates when available, entry and release timing uncertainty, and all outgoing signs. Review the timing brackets and sign changes across windows before interpreting a ratio. Unresolved trials stay in the audit.
5. Run the scorer from the repository root using the working acquisition sheet and actual pose file.

```powershell
python projects/robot/collision/prospective/high_speed_rebound.py filled_trials.csv calibrated_poses.csv scored_trials
```

After at least two sessions contribute approaching events, the scorer fits one shared coefficient and a same/opposite-spin interaction with equal session weights, and holds out one session at a time. The constant training mean supplies a baseline. Compare signed residuals by assigned speed, physical pair and spin direction. Check whether any improvement repeats across held-out sessions and across timing windows. The pilot measures precision for deciding how many additional sessions to collect.
