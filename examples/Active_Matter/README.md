# Data-Driven Symbolic Equation Discovery of Multi-Robot Dynamics

This repository presents a physics-informed machine learning pipeline for the analysis of multi-robot trajectory data and the discovery of interpretable mathematical models describing their dynamics.

The workflow combines:
* Trajectory preprocessing
* Hyperparameter optimization with **Optuna**
* Symbolic regression via **EPDE**.

## Project Goal

The goal of this work is to develop a hyperparameter optimization with **Optuna** for identifying accurate, compact, and interpretable systems of ordinary differential equations (ODEs) governing the motion of robots based on experimental trajectory data.

---
##  Data Description

* **Pickle file** contains raw experimental data extracted from video tracking.
* For each robot and timestep:
  * robot ID,
  * 2D coordinates `(x, y)`.
* Supported robot shapes:
  * circle
  * oval

---
## Components

`DataProcessor`
* Extracts coordinates
* Normalizes trajectories (MinMax scaling)

`discovery_science.ipynb`
* Splits trajectories into segments
* Runs EPDE for symbolic regression
* Uses Optuna for hyperparameter tuning
* Outputs differential equations

---
## Hyperparameter optimization

For each trajectory segment of `n_parts` independently:

* **Optuna** optimizes EPDE parameters:

  * polynomial window,
  * smoothing sigma,
  * boundary,
  * population size.

* **Optuna** optimizes TEDEOUS parameters:

  * the number and frequency of Fourier transform embeddings,
  * the number of layers
  
* Objective balances:

  * equation residuals,
  * model complexity,
  * reconstruction stability.
---

## Repository Structure

```
.
├── discovery_science_mezo.ipynb
├── data_process.py
├── {circle|oval}/
│   ├── data/
│       └── {circle_data_00_330_[30_bots_PWM_10_15cw_15ccw_D_41cm].MP4.pickle | oval_data_[30_bots_PWM_1_exp_1].pickle}
│   ├── levels_robots_ids.json
│   ├── EPDE_output_micro/
│       ├── robot_{id}/
│           └── {n}_parts/
│               └── {with|without}_force/
│                   ├── part_{nk}_system_best_params.json
│                   ├── part_{nk}_system_history_plot.html
│                   ├── part_{nk}_system_importances_plots.html
│                   ├── system_{nk}.csv
│                   └── part_{nk}_obj_func_and_equations.txt
│   └── EPDE_output_meso
│       ├── robot_{ids}
│           └── ...
└── README.md
```
---

## Dependencies

Core libraries:
* numpy, pandas, scipy

Machine Learning
* scikit-learn

Optimization & Discovery
* epde
* optuna

Utility
* dill (pickle)
* pathlib
* json

---
## Research Outcomes
* Developed a unified pipeline for data-driven discovery of dynamical systems

---

## Limitations and Assumptions

* Robot ID < 100
* Only `oval` and `circle` supported
* derivatives up to second order
* EPDE sensitive to noise and hyperparameters

---

For more information, see the repository https://github.com/20saaa02/Active_Matter






