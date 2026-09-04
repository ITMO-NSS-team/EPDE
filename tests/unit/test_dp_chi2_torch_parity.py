#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parity of the DP PINN testbed's torch ``chi2_per_term``
(``projects/pic/data/dp/cv_metric.py``) with the canonical numpy
``survival.chi2_scores(..., fit_intercept=False)`` --
plus the torch-only contracts: NaN-free gradients through the exact-fit
floor and the dead-column 0/0 route, and a LIVE detach boundary on the
calibration denominator D."""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir, os.pardir))
_DP_DIR = os.path.join(_REPO_ROOT, 'projects', 'pic', 'data', 'dp')
if _DP_DIR not in sys.path:
    sys.path.insert(0, _DP_DIR)

# ``cv_metric`` is the DP PINN testbed's torch port and is not part of the
# package; skip rather than error at COLLECTION when it is absent, which
# otherwise takes the whole suite down with it and forces an --ignore flag.
chi2_per_term = pytest.importorskip(
    "cv_metric",
    reason="projects/pic/data/dp/cv_metric.py is absent from the tree",
).chi2_per_term  # noqa: E402
from epde.operators.common.survival import chi2_scores  # noqa: E402


def _problem(n=600):
    """Drifting coefficient on the second column keeps every term off the
    exact-fit floor and gives all three a genuinely nonzero score."""
    x = np.linspace(0.5, 3.0, n)
    f1 = np.sin(5.0 * x)
    f2 = np.cos(7.0 * x)
    f3 = np.sin(11.0 * x + 0.3)
    A = np.column_stack([f1, f2, f3])
    y = 2.0 * f1 + x * f2 - 3.0 * f3
    return A, y


class TestNumpyParity:

    def test_parity_float64_no_ridge(self):
        # ridge=0.0 leaves only the absolute float64 tiny (~2.2e-16) in
        # the torch solve vs numpy's unridged _solve_gram -- negligible.
        A, y = _problem()
        want = chi2_scores(A, y, None, (y.size,), fit_intercept=False)
        got = chi2_per_term(torch.tensor(y, dtype=torch.float64),
                            torch.tensor(A, dtype=torch.float64),
                            ridge=0.0)["score"].numpy()
        assert np.all(want > 0)
        assert got == pytest.approx(want, rel=1e-8)

    def test_parity_float32_default_ridge(self):
        # Realistic training dtype + the default adaptive ridge: loose
        # agreement documents that the statistic survives float32.
        A, y = _problem()
        want = chi2_scores(A, y, None, (y.size,), fit_intercept=False)
        got = chi2_per_term(torch.tensor(y, dtype=torch.float32),
                            torch.tensor(A, dtype=torch.float32),
                            )["score"].numpy()
        assert got == pytest.approx(want, rel=1e-3)

    def test_weighted_parity(self):
        A, y = _problem()
        w = 0.5 + np.linspace(0.0, 1.0, y.size)
        want = chi2_scores(A, y, w, (y.size,), fit_intercept=False)
        got = chi2_per_term(torch.tensor(y, dtype=torch.float64),
                            torch.tensor(A, dtype=torch.float64),
                            weights=torch.tensor(w, dtype=torch.float64),
                            ridge=0.0)["score"].numpy()
        assert got == pytest.approx(want, rel=1e-8)


class TestTorchContracts:

    def test_exact_fit_floors_to_zero_with_finite_grads(self):
        A_np, _ = _problem()
        A = torch.tensor(A_np, dtype=torch.float64).requires_grad_(True)
        c_true = torch.tensor([2.0, -1.5, 0.7], dtype=torch.float64)
        y = A @ c_true
        out = chi2_per_term(y, A, ridge=0.0)
        assert torch.all(out["score"] == 0.0)
        out["score"].sum().backward()
        assert A.grad is not None
        assert torch.all(torch.isfinite(A.grad))

    def test_dead_column_scores_zero_with_finite_grads(self):
        A_np, y_np = _problem()
        A_np = A_np.copy()
        A_np[:, 2] = 0.0                       # dead column: D_j = 0
        y_np = y_np + 3.0 * np.sin(11.0 * np.linspace(0.5, 3.0, y_np.size)
                                   + 0.3)     # keep y off the floor
        A = torch.tensor(A_np, dtype=torch.float64).requires_grad_(True)
        y = torch.tensor(y_np, dtype=torch.float64)
        out = chi2_per_term(y, A, ridge=0.0)
        assert float(out["score"][2].detach()) == 0.0
        assert torch.all(torch.isfinite(out["score"]))
        out["score"].sum().backward()
        assert torch.all(torch.isfinite(A.grad))

    def test_no_floor_keeps_live_scores_at_exact_fit(self):
        # use_floor=False: an exact fit no longer snaps to the zero
        # cliff -- the score stays the (round-off-tiny) live functional
        # with finite gradients. Off the floor the flag changes nothing.
        A_np, y_np = _problem()
        A = torch.tensor(A_np, dtype=torch.float64).requires_grad_(True)
        c_true = torch.tensor([2.0, -1.5, 0.7], dtype=torch.float64)
        y_exact = (A @ c_true).detach()
        floored = chi2_per_term(y_exact, A, ridge=0.0, use_floor=True)
        live = chi2_per_term(y_exact, A, ridge=0.0, use_floor=False)
        assert torch.all(floored["score"] == 0.0)
        assert torch.all(torch.isfinite(live["score"]))
        live["score"].sum().backward()
        assert torch.all(torch.isfinite(A.grad))
        # Off-floor invariance of the flag.
        y = torch.tensor(y_np, dtype=torch.float64)
        a = chi2_per_term(y, torch.tensor(A_np, dtype=torch.float64),
                          ridge=0.0, use_floor=True)["score"]
        b = chi2_per_term(y, torch.tensor(A_np, dtype=torch.float64),
                          ridge=0.0, use_floor=False)["score"]
        assert torch.equal(a, b)

    def test_denom_none_is_the_raw_path_energy(self):
        # denom='none' drops the D_j normalization: off the exact-fit
        # floor the two forms differ by exactly that factor.
        A_np, y_np = _problem()
        A = torch.tensor(A_np, dtype=torch.float64)
        y = torch.tensor(y_np, dtype=torch.float64)
        std = chi2_per_term(y, A, ridge=0.0)
        raw = chi2_per_term(y, A, ridge=0.0, denom="none")
        assert torch.all(std["D"] > 0)
        assert torch.allclose(raw["score"], std["score"] * std["D"],
                              rtol=1e-10)

    def test_calibration_is_detached(self):
        A_np, y_np = _problem()
        y = torch.tensor(y_np, dtype=torch.float64)

        def grads(detach):
            A = torch.tensor(A_np, dtype=torch.float64).requires_grad_(True)
            out = chi2_per_term(y, A, ridge=0.0,
                                detach_calibration=detach)
            assert not out["D"].requires_grad     # diagnostic always detached
            out["score"].sum().backward()
            return A.grad

        g_detached = grads(True)
        g_attached = grads(False)
        assert torch.all(torch.isfinite(g_detached))
        assert torch.all(torch.isfinite(g_attached))
        # The boundary is LIVE: routing gradient through D changes it.
        assert not torch.allclose(g_detached, g_attached)
