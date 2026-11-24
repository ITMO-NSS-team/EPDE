# Contact entries and matched near misses

Five swarm recordings at 25 Hz. A contact entry is inferred from measured center separation below one calibrated diameter (219.55 px). A near miss remains at least 2 px outside that diameter. Video confirmation of physical contact was unavailable.

| Third-neighbor criterion | Contact entries | Near misses | Matched comparisons | Co-rotating matches | Contributing recordings |
|---|---:|---:|---:|---:|---:|
| None within 253 px throughout | 6 | 13 | 0 | 0 | 0 |
| None within 1.03 diameters throughout | 30 | 181 | 2 | 2 | 1 |
| None within 253 px immediately before and at entry | 22 | 75 | 3 | 2 | 1 |
| Within 253 px at some point | 3,885 | 11,786 | 1,728 | 781 | 5 |

The strict and third-neighbor groups partition the encounters. The other two rows are overlapping sensitivity analyses. The strictly isolated events all came from the 30-robot recording. No matched estimate was available for continuous strict isolation.

## Normal and tangential velocity

Each entry below is the contact-minus-near-miss difference in velocity change, in px/s. Positive normal velocity points toward separation along the entry axis. The spin-aligned tangent uses minus the common spin sign times signed tangential velocity and includes co-rotating pairs only. Closest and entry columns average velocity over the 0.4 s following the named frame and subtract the same incoming velocity. Lead columns compare two incoming velocity windows before entry.

| Recording with third neighbors | Matches | Normal, closest | Normal, entry | Normal, lead | Tangent, closest | Tangent, lead | Spin-aligned, closest | Spin-aligned, lead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N30_PWM10_D41 | 192 | +5.21 | -27.22 | +5.80 | -1.45 | +27.62 | +38.25 | +2.94 |
| N40_PWM10_D40 | 59 | +11.81 | -32.54 | -0.43 | +6.88 | +17.63 | -13.50 | -54.11 |
| N40_PWM10_D41 | 46 | +4.70 | -28.77 | -12.84 | -18.23 | +10.23 | -25.96 | -41.46 |
| N40_PWM20_D40 | 847 | +12.89 | -21.91 | +5.27 | +9.78 | +10.14 | +9.04 | +9.18 |
| N40_PWM20_D41 | 584 | +14.03 | -25.89 | -11.35 | +30.29 | -6.22 | -11.36 | -7.25 |
| Equal weight per recording | 1,728 | +9.73 | -27.27 | -2.71 | +5.45 | +11.88 | -0.71 | -18.14 |

Normal contrasts switch sign with the timing reference in every recording. Closest approach is selected retrospectively from each trajectory. Matching also leaves differences in pre-entry tangential motion. These observations limit attribution to contact or spin.

## Sensitivity to the measured contact boundary

Offsets shift both thresholds. The 2 px gap between contact and near miss is retained. Effects are equal-weight means over contributing recordings in the third-neighbor group.

| Boundary offset (px) | Matches | Normal, closest | Normal, entry | Spin-aligned, closest |
|---:|---:|---:|---:|---:|
| -2 | 969 | +18.90 | -29.21 | +29.34 |
| +0 | 1,728 | +9.73 | -27.27 | -0.71 |
| +2 | 2,253 | +16.53 | -20.70 | +3.37 |

## Selection and matching

An encounter starts at the first inward crossing of diameter + 12 px following clearance of 253 px. It must clear 253 px again within 25 frames. Closest approach is the minimum measured separation before that clearance. The analysis excludes nonapproaching pairs, wall encounters and flagged tracks. Overlapping observation windows for the same pair are screened in chronological order. This restricts inference to short, completed passes.

We use the axis from the first robot to the second at entry. Relative velocity is the second robot’s velocity minus the first’s. Incoming velocity runs from frame −5 to −1 relative to entry. The earlier window runs from −10 to −6. Continuous isolation and quality checks span frame −10 through the later of release or 10 frames after closest approach. The entry-only isolation check spans frames −5 through 0.

Matches require the same recording, robot identities and individual spin signs. Maximum differences are 50 px/s in incoming closing speed, 30 degrees in signed approach angle, 2 rad/s for each angular velocity and half a diameter in wall clearance. We maximize the number of matches without reuse, then minimize the sum of squared differences scaled by these tolerances. Repeated encounters can share robots. Equal recording weights prevent the largest recording from dominating the reported contrasts.

The [per-recording results](by_recording.csv) include covariate differences and all outcomes. The [matches](matches.csv.gz) give frame indices for each contact and near miss. The [encounters](encounters.csv.gz) include eligible events before matching.
