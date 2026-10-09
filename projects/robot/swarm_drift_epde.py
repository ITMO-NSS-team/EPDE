"""Interaction (drift) laws of the swarm with EPDE's forward-increment preprocessor.

Contact repulsion and wall confinement are gradient-type drifts. For stochastic data a time-symmetric
derivative cannot see them (E[u' g(u)] = 0 at stationarity), so the default 'poly' preprocessor fits
them with coefficient ~0. 'forward' differentiates with forward increments over ``lag`` samples,
which estimate the drift. The robots' velocity persists for ~0.18 s, so the lag must be several
times that (the signal check peaks at 0.4-1 s).

Every robot is one trajectory of one shared law, with its own interaction tokens (see
swarm_data.contact_features): push_*, wall_*, gear_*. Positions are centred on the arena.

    python swarm_drift_epde.py --file N40_PWM20_D41 --robots 10 --lag 10
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# insert, not append: a stale pip-installed epde would otherwise shadow the repo
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '../..')))
sys.path.insert(0, HERE)

import numpy as np

from epde.interface.interface import EpdeSearch
from epde import CacheStoredTokens

from swarm_data import (swarm_files, swarm_meta, load_swarm, arena_envelope, spin_rates,
                        contact_features)

LABELS = ["push_x", "push_y", "wall_x", "wall_y", "gear_x", "gear_y"]


def pooled_ols(x, y, feats, robots, lag, dt):
    """Reference coefficients: forward increments of the chosen robots on all six features."""
    rows = []
    for k in robots:
        for comp, pos in (("x", x[:, k]), ("y", y[:, k])):
            inc = (pos[lag:] - pos[:-lag]) / (lag * dt)
            A = np.column_stack([feats[lab][:-lag, k] for lab in LABELS])
            rows.append((inc, A))
    inc = np.concatenate([r[0] for r in rows]); A = np.vstack([r[1] for r in rows])
    A1 = np.column_stack([A, np.ones(len(inc))])
    c, *_ = np.linalg.lstsq(A1, inc, rcond=None)
    r2 = 1 - np.var(inc - A1 @ c) / np.var(inc)
    return dict(zip(LABELS + ["const"], c)), r2


def drift_discovery(file_key="N40_PWM20_D41", robots=10, lag=10, popsize=8, epochs=10,
                    terms=5, preprocessor="forward", sparsity="knee", kinds=("push", "wall", "gear"),
                    instability=None):
    """``sparsity``: the default 'vwsr' derives each term's penalty from its coefficient stability
    over sliding windows. For drift estimation from stochastic data every window is noise-dominated
    (per-sample R^2 ~0.1 at best), so it can prune every term, leaving the equation degenerate.
    'knee' selects terms by the shape of the RSS-vs-support curve instead.
    ``kinds``: which interaction token pairs to offer (push, wall, gear)."""
    labels = [lab for lab in LABELS if lab.split('_')[0] in kinds]
    path = next(p for p in swarm_files() if swarm_meta(p)["tag"] == file_key)
    t, x, y, phi, ids = load_swarm(path)
    dt = t[1] - t[0]
    xc, yc, r_env = arena_envelope(x, y)
    feats = contact_features(x, y, spin_rates(t, phi), xc, yc, r_env)
    chosen = np.linspace(0, x.shape[1] - 1, robots).astype(int)
    ref, r2 = pooled_ols(x, y, feats, chosen, lag, dt)
    print(f"{file_key}: robots {ids[chosen].tolist()}, lag {lag} ({lag*dt:.2f}s)")
    print("pooled OLS reference (forward increments): R2 %.3f; " % r2 +
          ", ".join(f"{k} {v:+.2f}" for k, v in ref.items()))

    prep_kwargs = {'lag': lag} if preprocessor == "forward" else {'polynomial_window': 2 * lag + 1,
                                                                    'poly_order': 3, 'mp_poolsize': 1}
    # instability: the second objective's estimator (None = config default, 'chi2')
    search = EpdeSearch(multiobjective_mode=True, use_solver=False, sparsity_cls=sparsity,
                        **({'instability_metric': instability} if instability else {}),
                        # covers the forward tail (lag) and the poly window's half-width (lag + 1)
                        boundary_width=lag + 1,
                        default_preprocessor_type=preprocessor, preprocessor_kwargs=prep_kwargs,
                        population_size=popsize, training_epochs=epochs,
                        verbose_params={'show_iter_idx': True})
    _, domain = search.createDomain((t,), ID=0)
    # tensors differ per robot: declared per trajectory, not with the family
    interaction = CacheStoredTokens(token_type='interaction', token_labels=labels, token_tensors=None,
                                    params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                                    meaningful=True)
    trajectories = []
    for i, k in enumerate(chosen):
        _, traj = search.createTrajectory(
            {'x': x[:, k] - xc, 'y': y[:, k] - yc}, domain, cache_id=i,
            cached_token_tensors={interaction: {lab: feats[lab][:, k] for lab in labels}})
        trajectories.append(traj)
    search.fit(data=trajectories, max_deriv_order=(1,), data_fun_pow=1,
               equation_terms_max_number=terms, additional_tokens=[interaction],
               equation_factors_max_number={'factors_num': [1], 'probas': [1.0]})
    search.equations(only_print=True, num=1)
    return search


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="N40_PWM20_D41")
    ap.add_argument("--robots", type=int, default=10)
    ap.add_argument("--lag", type=int, default=10)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--preprocessor", default="forward", choices=["forward", "poly"])
    ap.add_argument("--sparsity", default="knee", choices=["knee", "vwsr", "lasso"])
    ap.add_argument("--kinds", default="push,wall,gear", help="comma-separated subset of push,wall,gear")
    ap.add_argument("--instability", default=None, help="instability metric, e.g. cv (default: chi2)")
    a = ap.parse_args()
    drift_discovery(a.file, a.robots, a.lag, a.popsize, a.epochs, preprocessor=a.preprocessor,
                    sparsity=a.sparsity, kinds=tuple(a.kinds.split(',')), instability=a.instability)
