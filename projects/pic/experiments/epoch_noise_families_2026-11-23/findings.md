# Structural recovery under three measurement noise models

Complete. 144/144 new searches and 96/96 controls have verified measurements and four scored fronts.

Each cell counts runs with a complete catalog match or verified equivalent system on the final native first front. The core suite has 45 runs per condition and the oscillator has three.

| Dataset | Clean | Gaussian 5% | Spikes 5% | Correlated 5% | Signal-dependent 5% |
|---|---:|---:|---:|---:|---:|
| Core suite | 37/45 | 29/45 | 29/45 | 15/45 | 23/45 |
| ac | 3/3 | 0/3 | 0/3 | 0/3 | 1/3 |
| burgers | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| burgers_inviscid | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| duffing | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| kdv | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| kdv_cossin | 3/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| ks | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| lorenz | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| lv | 3/3 | 2/3 | 3/3 | 1/3 | 2/3 |
| ns | 1/3 | 2/3 | 1/3 | 0/3 | 0/3 |
| ode | 3/3 | 3/3 | 2/3 | 0/3 | 2/3 |
| pde_compound | 3/3 | 1/3 | 2/3 | 0/3 | 1/3 |
| pde_divide | 0/3 | 3/3 | 3/3 | 0/3 | 2/3 |
| vdp | 3/3 | 2/3 | 3/3 | 0/3 | 0/3 |
| wave | 3/3 | 1/3 | 0/3 | 0/3 | 0/3 |
| ode_simple | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |

## Generation and retention across fronts

Counts use the same completed searches across all four front definitions.

| Group | Noise family | Searches | Generated | Native | Ordinary | Two-slab CV | Multiscale CV |
|---|---|---:|---:|---:|---:|---:|---:|
| Core suite | clean | 45 | 40 | 37 | 34 | 34 | 34 |
| Core suite | gaussian | 45 | 37 | 29 | 29 | 29 | 29 |
| Core suite | spikes | 45 | 36 | 29 | 29 | 28 | 28 |
| Core suite | correlated | 45 | 32 | 15 | 15 | 14 | 14 |
| Core suite | signal_dependent | 45 | 36 | 23 | 24 | 24 | 24 |
| Oscillator | clean | 3 | 3 | 3 | 3 | 3 | 3 |
| Oscillator | gaussian | 3 | 3 | 3 | 3 | 3 | 3 |
| Oscillator | spikes | 3 | 3 | 3 | 3 | 3 | 3 |
| Oscillator | correlated | 3 | 3 | 2 | 3 | 3 | 3 |
| Oscillator | signal_dependent | 3 | 3 | 3 | 3 | 3 | 3 |

## Catalog recovery on the native front

Core counts use the structural registry and exclude historical Lorenz identity credit. The oscillator retains its original identity rule. Each comparison pairs the same dataset and seed.

| Group | Noise family | Catalog hits / runs | Catalog generated | Gains vs clean | Losses vs clean | Gains vs Gaussian | Losses vs Gaussian |
|---|---|---:|---:|---:|---:|---:|
| Core suite | clean | 34/45 | 39 | 0 | 0 | 12 | 4 |
| Core suite | gaussian | 26/45 | 36 | 4 | 12 | 0 | 0 |
| Core suite | spikes | 28/45 | 36 | 6 | 12 | 7 | 5 |
| Core suite | correlated | 12/45 | 30 | 0 | 22 | 0 | 14 |
| Core suite | signal_dependent | 21/45 | 36 | 3 | 16 | 3 | 8 |
| Oscillator | clean | 3/3 | 3 | 0 | 0 | 0 | 0 |
| Oscillator | gaussian | 3/3 | 3 | 0 | 0 | 0 | 0 |
| Oscillator | spikes | 3/3 | 3 | 0 | 0 | 0 | 0 |
| Oscillator | correlated | 2/3 | 3 | 0 | 1 | 0 | 1 |
| Oscillator | signal_dependent | 3/3 | 3 | 0 | 0 | 0 | 0 |

## Paired changes in native-front recovery

A gain means only the new noise family recovered the system. A loss means only the control did. The archive-hit column counts pairs where the new search generated a registered system at any epoch.

| Group | Noise family | Control | Pairs | Gains | Losses | Archive hits |
|---|---|---|---:|---:|---:|---:|
| Core suite | spikes | clean | 45 | 4 | 12 | 36 |
| Core suite | spikes | gaussian | 45 | 5 | 5 | 36 |
| Core suite | correlated | clean | 45 | 0 | 22 | 32 |
| Core suite | correlated | gaussian | 45 | 0 | 14 | 32 |
| Core suite | signal_dependent | clean | 45 | 2 | 16 | 36 |
| Core suite | signal_dependent | gaussian | 45 | 2 | 8 | 36 |
| Oscillator | spikes | clean | 3 | 0 | 0 | 3 |
| Oscillator | spikes | gaussian | 3 | 0 | 0 | 3 |
| Oscillator | correlated | clean | 3 | 0 | 1 | 3 |
| Oscillator | correlated | gaussian | 3 | 0 | 1 | 3 |
| Oscillator | signal_dependent | clean | 3 | 0 | 0 | 3 |
| Oscillator | signal_dependent | gaussian | 3 | 0 | 0 | 3 |

The per-run table includes all four fronts, archive recovery, realized per-field RMS, and search cost. A catalog match or a verified complete-system identity counts once per run. The Lorenz audit shows the identity evidence included in those counts.

[Per-run results](runs.csv) · [Lorenz identity credit](historical_lorenz.csv) · [Protocol](protocol.md)
