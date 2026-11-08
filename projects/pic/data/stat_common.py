#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""System-independent pieces of the weight-free PINN loss, shared across
the per-system experiment directories.

The implementations live in ``dp/cv_metric.py`` -- single source of
truth, pinned by ``tests/unit/test_dp_chi2_torch_parity.py`` and
``tests/unit/test_dp_autoscale_loss.py``. Nothing is copied here.

Why the explicit path load: EVERY system directory (dp, duffing, wave,
ns, ...) has its own ``cv_metric.py``, so a plain ``from cv_metric
import ...`` inside e.g. duffing resolves to duffing's module. Loading
the dp one under a distinct module name sidesteps the collision without
touching any existing import.

Exported (all take (target, features, ...) and are truth-free):
    global_ols      one guarded weighted OLS -> (theta, residual)
    tls_fit         target-free TLS on the scaled design -> (lam, theta, w0)
    tls_scales      its column yardsticks (RMS per column) from a design
    chi2_per_term   raw Nyblom-Hansen score-path constancy, per term
    het_per_window  DerSimonian-Laird calibrated heterogeneity, per term
    bounded         x/(1+x): [0, inf) -> [0, 1), order-preserving
    max_corr        max|A^T W y|, the sparsity.py scale anchor
    observation_loss  mean((pred-obs)^2)/Var(obs), the data term
    anchor_penalty  per-term coefficient pull in signal units
    anchor_scales   its yardsticks (mean(A_j^2), Var(y)) from a design
    central_diff    centred FD of a chosen order of accuracy (edges dropped)
    fd_core         slice aligning a channel with central_diff's output
    spectral_diff   Fourier derivative on a periodic grid
    combine_loss    the total loss from its terms: 'sum' | 'log'
    grad_norms      per-term ||grad||: who actually drives training
    HardICWrapper   exact initial condition, so the IC term disappears
"""

import importlib.util
import os

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "dp", "cv_metric.py")
_spec = importlib.util.spec_from_file_location("_dp_cv_metric", _SRC)
_dp_cv_metric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dp_cv_metric)

EPS = _dp_cv_metric.EPS
global_ols = _dp_cv_metric.global_ols
tls_fit = _dp_cv_metric.tls_fit
tls_scales = _dp_cv_metric.tls_scales
chi2_per_term = _dp_cv_metric.chi2_per_term
het_per_window = _dp_cv_metric.het_per_window
bounded = _dp_cv_metric.bounded
max_corr = _dp_cv_metric.max_corr
observation_loss = _dp_cv_metric.observation_loss
anchor_penalty = _dp_cv_metric.anchor_penalty
anchor_scales = _dp_cv_metric.anchor_scales
central_diff = _dp_cv_metric.central_diff
fd_core = _dp_cv_metric.fd_core
spectral_diff = _dp_cv_metric.spectral_diff
combine_loss = _dp_cv_metric.combine_loss
grad_norms = _dp_cv_metric.grad_norms
HardICWrapper = _dp_cv_metric.HardICWrapper

__all__ = ["EPS", "global_ols", "tls_fit", "tls_scales",
           "chi2_per_term", "het_per_window",
           "bounded", "max_corr", "observation_loss", "grad_norms",
           "anchor_penalty", "anchor_scales", "central_diff", "fd_core",
           "spectral_diff", "combine_loss",
           "HardICWrapper"]
