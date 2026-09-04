#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contracts for the weight-free ("no hyperparameter") DP PINN loss
pieces in ``projects/pic/data/dp/cv_metric.py``:

  * ``global_ols``  -- ONE solve shared by the statistic and by any loss
    term that reads the net's own data-driven coefficients,
  * ``bounded``     -- order-preserving map onto [0, 1) so an unbounded
    statistic needs no weight,
  * ``max_corr``    -- the ``sparsity.PhysicsInformedLasso`` scale anchor
    (``active_thresholds = active_cv * max_corr``), carried as a
    diagnostic,
  * ``HardICWrapper`` -- exact initial condition, i.e. an IC constraint
    with no weight at all,
  * ``observation_loss`` -- the data term, divided by the observed
    field's own variance so it too carries no weight,
  * scale-freedom of the statistics, which is what makes weight 1
    meaningful in the first place.
"""

import ast
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

# Optional testbed module living under ``projects/``, not in the package.
# Skip rather than error at COLLECTION when it is absent -- a collection
# error takes the whole suite down and forces an --ignore flag.
_cv_metric = pytest.importorskip(
    "cv_metric",
    reason="projects/pic/data/dp/cv_metric.py is absent from the tree")
HardICWrapper = _cv_metric.HardICWrapper
bounded = _cv_metric.bounded
chi2_per_term = _cv_metric.chi2_per_term
global_ols = _cv_metric.global_ols
het_per_window = _cv_metric.het_per_window
max_corr = _cv_metric.max_corr
observation_loss = _cv_metric.observation_loss


def _problem(n=600):
    x = np.linspace(0.5, 3.0, n)
    f1 = np.sin(5.0 * x)
    f2 = np.cos(7.0 * x)
    f3 = np.sin(11.0 * x + 0.3)
    A = np.column_stack([f1, f2, f3])
    y = 2.0 * f1 + x * f2 - 3.0 * f3
    return (torch.tensor(A, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64))


class TestGlobalOls:

    def test_is_the_solve_chi2_uses(self):
        # One loss must not contain two different theta_hats.
        A, y = _problem()
        c, r = global_ols(y, A, ridge=0.0)
        out = chi2_per_term(y, A, ridge=0.0)
        assert torch.allclose(c, out["theta"], rtol=1e-12, atol=0.0)
        assert torch.allclose(r, y - A @ c, rtol=1e-12, atol=0.0)

    def test_weighted_and_gradient_bearing(self):
        A, y = _problem()
        w = 0.5 + torch.linspace(0.0, 1.0, y.numel(), dtype=torch.float64)
        A = A.clone().requires_grad_(True)
        c, r = global_ols(y, A, weights=w, ridge=0.0)
        assert c.requires_grad and r.requires_grad
        (r ** 2).sum().backward()
        assert torch.all(torch.isfinite(A.grad))

    def test_parked_field_does_not_raise(self):
        # AtA -> 0 (a saturated net / LBFGS line-search probe) must give a
        # guarded solve, never a LinAlgError.
        A = torch.zeros(50, 3, dtype=torch.float64)
        y = torch.zeros(50, dtype=torch.float64)
        c, r = global_ols(y, A)
        assert torch.all(torch.isfinite(c)) and torch.all(torch.isfinite(r))


class TestBounded:

    def test_maps_to_unit_interval_monotonically(self):
        s = torch.tensor([0.0, 1e-6, 1e-3, 1.0, 1e3, 1e8],
                         dtype=torch.float64)
        b = bounded(s)
        assert float(b[0]) == 0.0
        assert torch.all(b >= 0.0) and torch.all(b < 1.0)
        assert torch.all(b[1:] > b[:-1])

    def test_preserves_order(self):
        # Order preservation is the whole justification: bounding chi may
        # not change WHICH term is judged least constant.
        rng = np.random.default_rng(0)
        s = torch.tensor(np.abs(rng.standard_normal(64)) * 1e4)
        assert torch.equal(torch.argsort(s), torch.argsort(bounded(s)))


class TestMaxCorr:

    def test_matches_the_sparsity_anchor(self):
        A, y = _problem()
        w = 0.5 + torch.linspace(0.0, 1.0, y.numel(), dtype=torch.float64)
        want = np.max(np.abs(A.numpy().T @ (w.numpy() * y.numpy())))
        assert float(max_corr(y, A, weights=w)) == pytest.approx(want,
                                                                 rel=1e-12)
        want_uw = np.max(np.abs(A.numpy().T @ y.numpy()))
        assert float(max_corr(y, A)) == pytest.approx(want_uw, rel=1e-12)


class TestHardIC:

    def test_reproduces_state_and_velocity_at_t0(self):
        torch.manual_seed(0)
        inner = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(),
                                    torch.nn.Linear(32, 2)).double()
        th0 = torch.tensor([[0.7, -1.3]], dtype=torch.float64)
        om0 = torch.tensor([[2.5, -0.4]], dtype=torch.float64)
        net = HardICWrapper(inner, 0.0, 8.0, th0, om0).double()

        t = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
        out = net(t)
        assert torch.allclose(out, th0, atol=1e-12)
        ones = torch.ones_like(out[:, 0])
        w1 = torch.autograd.grad(out[:, 0], t, ones, create_graph=True)[0]
        w2 = torch.autograd.grad(out[:, 1], t, ones)[0]
        assert float(w1.detach()) == pytest.approx(float(om0[0, 0]), abs=1e-9)
        assert float(w2) == pytest.approx(float(om0[0, 1]), abs=1e-9)

    def test_inner_parameters_are_exposed(self):
        inner = torch.nn.Linear(1, 2)
        net = HardICWrapper(inner, 0.0, 1.0, torch.zeros(1, 2),
                            torch.zeros(1, 2))
        assert list(net.parameters()) == list(inner.parameters())
        assert any(k.startswith("inner.") for k in net.state_dict())


class TestStatisticsAreScaleFree:
    """Why unit weights are legitimate: a common rescaling of the
    equation (y, A -> a*y, a*A) leaves both statistics unchanged, so
    neither needs a yardstick -- only its unbounded RANGE needed fixing.
    """

    @pytest.mark.parametrize("a", [1e-3, 1e3])
    def test_chi_invariant_under_common_scaling(self, a):
        A, y = _problem()
        base = chi2_per_term(y, A, ridge=0.0)["score"]
        scaled = chi2_per_term(a * y, a * A, ridge=0.0)["score"]
        assert scaled.numpy() == pytest.approx(base.numpy(), rel=1e-8)

    @pytest.mark.parametrize("a", [1e-3, 1e3])
    def test_het_invariant_under_common_scaling(self, a):
        A, y = _problem()
        n = y.numel()
        lo = torch.linspace(0, n - 60, 12).round()
        mask = torch.zeros(12, n, dtype=torch.float64)
        for i, s in enumerate(lo):
            mask[i, int(s):int(s) + 60] = 1.0
        base = het_per_window(y, A, mask, ridge=0.0)["score"]
        scaled = het_per_window(a * y, a * A, mask, ridge=0.0)["score"]
        assert scaled.numpy() == pytest.approx(base.numpy(), rel=1e-6)


class TestObservationLoss:
    """The data term is admissible in a weight-free loss only because
    dividing by the observed field's own variance makes it dimensionless
    -- exactly the argument ``TestStatisticsAreScaleFree`` makes for the
    statistics. These pin that argument."""

    def test_zero_when_the_field_is_reproduced(self):
        obs = torch.randn(200, 2, dtype=torch.float64)
        assert float(observation_loss(obs.clone(), obs, 1.7)) == 0.0

    @pytest.mark.parametrize("a", [1e-3, 1e3])
    def test_invariant_under_a_common_rescaling_of_the_field(self, a):
        # Rescaling the field by `a` scales its variance -- the yardstick
        # -- by a**2. If the ratio did NOT hold fixed, l_data would need a
        # weight to stay commensurate with l_phys, and the whole
        # no-hyperparameter construction would fail at this term.
        torch.manual_seed(0)
        obs = torch.randn(300, dtype=torch.float64)
        pred = obs + 0.05 * torch.randn(300, dtype=torch.float64)
        norm = float(obs.var(unbiased=False))
        base = float(observation_loss(pred, obs, norm))
        scaled = float(observation_loss(a * pred, a * obs, a * a * norm))
        assert scaled == pytest.approx(base, rel=1e-9)

    def test_per_channel_norm_divides_inside_the_mean(self):
        # dp passes IC_NORM_TH, a (2,) vector: theta1 and theta2 explore
        # different ranges, so one shared scalar would let the wider
        # channel dominate. duffing/wave pass a scalar; both must work.
        pred = torch.tensor([[1.0, 4.0], [3.0, 8.0]], dtype=torch.float64)
        obs = torch.zeros(2, 2, dtype=torch.float64)
        norm = torch.tensor([1.0, 4.0], dtype=torch.float64)
        want = float(((pred ** 2) / norm).mean())
        assert float(observation_loss(pred, obs, norm)) == pytest.approx(want)
        # and a scalar norm is the ordinary mean/norm
        assert float(observation_loss(pred, obs, 2.0)) == pytest.approx(
            float((pred ** 2).mean()) / 2.0)

    def test_carries_gradient_to_the_prediction(self):
        obs = torch.randn(50, dtype=torch.float64)
        pred = torch.zeros(50, dtype=torch.float64, requires_grad=True)
        observation_loss(pred, obs, 1.0).backward()
        assert pred.grad is not None and float(pred.grad.abs().sum()) > 0.0


class TestDataTermWiring:
    """The three experiment scripts must gate the term on DATA_TERM,
    route through the ONE shared formula, and agree on the default --
    duffing's mode block says "IDENTICAL defaults to the DP script,
    that is the point". Structural, because importing a script would
    start a training run."""

    SCRIPTS = ("dp", "duffing", "wave")

    def _tree(self, system):
        path = os.path.join(_REPO_ROOT, "projects", "pic", "data", system,
                            "pinn_test_autoscale.py")
        with open(path, encoding="utf-8") as fh:
            return ast.parse(fh.read())

    def _flag(self, system, name):
        flags = [n for n in self._tree(system).body
                 if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == name for t in n.targets)]
        assert len(flags) == 1, f"{system}: {len(flags)} {name} assignments"
        return flags[0].value.value

    def test_the_three_systems_share_one_default(self):
        # The August 2026 sweep made ols+data the default on all three.
        # Whichever way a future sweep moves it, it must move TOGETHER --
        # a per-system default silently makes the systems incomparable.
        data = {s: self._flag(s, "DATA_TERM") for s in self.SCRIPTS}
        coef = {s: self._flag(s, "COEF_SOURCE") for s in self.SCRIPTS}
        assert len(set(data.values())) == 1, f"DATA_TERM disagrees: {data}"
        assert len(set(coef.values())) == 1, f"COEF_SOURCE disagrees: {coef}"
        assert data["dp"] is True and coef["dp"] == "ols", (data, coef)

    @pytest.mark.parametrize("system", SCRIPTS)
    def test_gated_and_uses_the_shared_formula(self, system):
        tree = self._tree(system)
        fn = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "data_loss"]
        assert len(fn) == 1, f"{system}: no single data_loss"
        body = ast.dump(fn[0])
        assert "DATA_TERM" in body, f"{system}: data_loss is not gated"
        assert "observation_loss" in body, f"{system}: not the shared formula"

    @pytest.mark.parametrize("system", SCRIPTS)
    def test_summed_into_total_loss_with_no_coefficient(self, system):
        tree = self._tree(system)
        fn = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "total_loss"]
        assert len(fn) == 1
        ret = [n for n in ast.walk(fn[0]) if isinstance(n, ast.Return)]
        assert len(ret) == 1
        # every operand is a bare name/subscript -- no Mult anywhere, which
        # is what "no weights" means operationally
        assert not any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)
                       for n in ast.walk(ret[0])), f"{system}: a weight crept in"
