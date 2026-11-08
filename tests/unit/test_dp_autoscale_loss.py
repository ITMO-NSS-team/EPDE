#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contracts for the weight-free ("no hyperparameter") DP PINN loss
pieces in ``projects/pic/data/dp/cv_metric.py``:

  * ``global_ols``  -- ONE solve shared by the statistic and by any loss
    term that reads the net's own data-driven coefficients,
  * ``bounded``     -- order-preserving map onto [0, 1), the OPT-IN form
    (``STAT_BOUND=True``); the statistics enter the loss raw by default,
  * ``het_per_window``'s ``score_raw`` -- the unbounded ``tau2/theta_bar^2``
    whose ``bounded`` image is exactly the bounded het score,
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
anchor_penalty = _cv_metric.anchor_penalty
anchor_scales = _cv_metric.anchor_scales
central_diff = _cv_metric.central_diff
fd_core = _cv_metric.fd_core
spectral_diff = _cv_metric.spectral_diff
combine_loss = _cv_metric.combine_loss
tls_fit = _cv_metric.tls_fit
tls_scales = _cv_metric.tls_scales


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
    neither needs a yardstick -- raw or bounded, only the RANGE differs.
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
        base = het_per_window(y, A, mask, ridge=0.0)
        scaled = het_per_window(a * y, a * A, mask, ridge=0.0)
        for key in ("score", "score_raw"):
            assert scaled[key].numpy() == pytest.approx(base[key].numpy(),
                                                        rel=1e-6), key


def _windows(n, n_win=12, width=60):
    lo = torch.linspace(0, n - width, n_win).round()
    mask = torch.zeros(n_win, n, dtype=torch.float64)
    for i, s in enumerate(lo):
        mask[i, int(s):int(s) + width] = 1.0
    return mask


class TestHetRaw:
    """The unbounded het is the SAME statistic as the bounded one, only
    un-squashed: tau2/(tau2 + m^2) == r/(1 + r) with r = tau2/m^2."""

    def test_bounded_image_is_the_bounded_score(self):
        # _problem's middle coefficient drifts (x * f2), so tau2 > 0 there
        # and the identity is exercised off the trivial zero.
        A, y = _problem()
        out = het_per_window(y, A, _windows(y.numel()), ridge=0.0)
        assert float(out["score_raw"].max()) > 0.0
        assert bounded(out["score_raw"]).numpy() == pytest.approx(
            out["score"].numpy(), rel=1e-12, abs=1e-15)

    def test_is_tau2_over_squared_level(self):
        A, y = _problem()
        out = het_per_window(y, A, _windows(y.numel()), ridge=0.0)
        want = out["tau2"] / out["theta_bar"] ** 2
        assert out["score_raw"].numpy() == pytest.approx(want.numpy(),
                                                         rel=1e-12)

    def test_parked_field_stays_finite(self):
        # A collapsed net (A == 0, theta_bar == 0) must not put inf/nan
        # into the loss.
        A = torch.zeros(600, 3, dtype=torch.float64)
        y = torch.zeros(600, dtype=torch.float64)
        out = het_per_window(y, A, _windows(600))
        assert torch.all(torch.isfinite(out["score_raw"]))

    def test_gradient_does_not_fade_with_inconsistency(self):
        # The reason to want it raw: d score/d tau2 = m^2/(tau2+m^2)^2
        # vanishes as tau2 grows, d score_raw/d tau2 = 1/m^2 does not.
        m2 = torch.tensor(1.0, dtype=torch.float64)
        for t in (1e-2, 1.0, 1e2):
            tau2 = torch.tensor(t, dtype=torch.float64, requires_grad=True)
            (g_b,) = torch.autograd.grad(tau2 / (tau2 + m2), tau2)
            tau2r = torch.tensor(t, dtype=torch.float64, requires_grad=True)
            (g_r,) = torch.autograd.grad(tau2r / m2, tau2r)
            assert float(g_r) == pytest.approx(1.0)
            assert float(g_b) == pytest.approx(1.0 / (1.0 + t) ** 2)


class TestStatScoresDefaultsToRaw:
    """``pinn_common.stat_scores`` is the route dp and duffing take into
    the loss; its default is the raw statistic."""

    @pytest.fixture(scope="class")
    def stat_scores(self):
        data_dir = os.path.dirname(_DP_DIR)
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)
        return pytest.importorskip("pinn_common").stat_scores

    def test_het(self, stat_scores):
        A, y = _problem()
        mask = _windows(y.numel())
        out = het_per_window(y, A, mask, ridge=1e-12)
        raw, _ = stat_scores(y, A, mask, "het")
        bnd, _ = stat_scores(y, A, mask, "het", bound=True)
        assert torch.equal(raw, out["score_raw"])
        assert torch.equal(bnd, out["score"])

    def test_chi(self, stat_scores):
        A, y = _problem()
        mask = _windows(y.numel())
        out = chi2_per_term(y, A, ridge=1e-12)
        raw, _ = stat_scores(y, A, mask, "chi")
        bnd, _ = stat_scores(y, A, mask, "chi", bound=True)
        assert torch.equal(raw, out["score"])
        assert torch.equal(bnd, bounded(out["score"]))


class TestFiniteDifferences:
    """THETA_FD is only as good as the derivative behind it; these pin
    the stencils' accuracy order and the edge trimming the scripts rely
    on to keep channels aligned."""

    @pytest.mark.parametrize("deriv", [1, 2])
    @pytest.mark.parametrize("order", [2, 4, 6])
    def test_converges_at_its_order(self, deriv, order):
        errs = []
        # Coarse on purpose: at n=161 the 6th-order 2nd derivative is already
        # at the eps/h^2 roundoff floor and the measured rate is noise.
        for n in (21, 41):
            x = np.linspace(0.0, 1.0, n)
            h = x[1] - x[0]
            f = np.sin(3.0 * x)
            exact = (3.0 * np.cos(3.0 * x) if deriv == 1
                     else -9.0 * np.sin(3.0 * x))[fd_core(order)]
            errs.append(np.abs(central_diff(f, h, deriv, order) - exact).max())
        rate = np.log2(errs[0] / errs[1])
        assert rate == pytest.approx(order, abs=0.3), (errs, rate)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_exact_on_polynomials_it_can_represent(self, order):
        x = np.linspace(-1.0, 2.0, 40)
        h = x[1] - x[0]
        f = x ** order
        want = order * x ** (order - 1)
        got = central_diff(f, h, 1, order)
        assert got == pytest.approx(want[fd_core(order)], rel=1e-7, abs=1e-7)

    def test_trims_the_differentiated_axis_only(self):
        f = np.random.default_rng(0).standard_normal((30, 50))
        out = central_diff(f, 0.1, 2, 6, axis=1)
        assert out.shape == (30, 50 - 6)
        out0 = central_diff(f, 0.1, 1, 4, axis=0)
        assert out0.shape == (30 - 4, 50)

    def test_periodic_wraps_and_keeps_length(self):
        x = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        h = x[1] - x[0]
        out = central_diff(np.sin(x), h, 2, 6, periodic=True)
        assert out.shape == x.shape
        assert np.abs(out + np.sin(x)).max() < 1e-7

    def test_order_two_matches_np_gradient_interior(self):
        # FD_ORDER=2 in the scripts keeps np.gradient itself; the stencil
        # must agree with it where both are centred.
        x = np.linspace(0.0, 1.0, 101)
        f = np.exp(np.sin(4.0 * x))
        assert central_diff(f, x[1] - x[0], 1, 2) == pytest.approx(
            np.gradient(f, x)[1:-1], rel=1e-12)

    def test_spectral_is_exact_on_band_limited_periodic_fields(self):
        x = np.linspace(-1.0, 1.0, 64, endpoint=False)
        h = x[1] - x[0]
        f = np.sin(3.0 * np.pi * x) + 0.5 * np.cos(np.pi * x)
        d2 = -(3.0 * np.pi) ** 2 * np.sin(3.0 * np.pi * x) \
            - 0.5 * np.pi ** 2 * np.cos(np.pi * x)
        assert np.abs(spectral_diff(f, h, 2) - d2).max() < 1e-9
        g = np.stack([f, 2.0 * f], axis=1)          # axis handling
        assert np.abs(spectral_diff(g, h, 2, axis=0)[:, 1] - 2.0 * d2).max() < 1e-8

    def test_rejects_unknown_stencils(self):
        with pytest.raises(ValueError):
            central_diff(np.zeros(10), 0.1, 3, 4)
        with pytest.raises(ValueError):
            central_diff(np.zeros(10), 0.1, 1, 5)


class TestAnchorPenalty:
    """The anchor measures a coefficient miss by the residual it would
    leave on the observed design, as a share of Var(y) -- the same
    1 - R^2 form l_phys has, which is what licenses weight 1."""

    def test_is_the_share_of_target_variance_the_miss_leaves(self):
        A, y = _problem()
        col_ms, y_var = anchor_scales(y, A)
        a = torch.tensor([2.0, 1.0, -3.0], dtype=torch.float64)
        c = a + torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
        per = anchor_penalty(c, a, "signal", col_ms, y_var)
        want = [float(((c[j] - a[j]) * A[:, j]).pow(2).mean()
                      / y.var(unbiased=False)) for j in range(3)]
        assert per.numpy() == pytest.approx(want, rel=1e-6)

    @pytest.mark.parametrize("s", [1e-4, 1e4])
    def test_invariant_under_rescaling_one_column(self, s):
        # A unit change on one feature moves its coefficient by 1/s and its
        # column energy by s^2 -- the penalty must not notice.
        A, y = _problem()
        a = torch.tensor([2.0, 1.0, -3.0], dtype=torch.float64)
        c = a + torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
        base = anchor_penalty(c, a, "signal", *anchor_scales(y, A))
        A2, a2, c2 = A.clone(), a.clone(), c.clone()
        A2[:, 0] *= s; a2[0] /= s; c2[0] /= s
        scaled = anchor_penalty(c2, a2, "signal", *anchor_scales(y, A2))
        assert scaled.numpy() == pytest.approx(base.numpy(), rel=1e-6)

    @pytest.mark.parametrize("k", [1e-3, 1e3])
    def test_invariant_under_common_rescaling(self, k):
        A, y = _problem()
        a = torch.tensor([2.0, 1.0, -3.0], dtype=torch.float64)
        c = a + torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
        base = anchor_penalty(c, a, "signal", *anchor_scales(y, A))
        scaled = anchor_penalty(c, a, "signal", *anchor_scales(k * y, k * A))
        assert scaled.numpy() == pytest.approx(base.numpy(), rel=1e-6)

    def test_relative_form_is_signal_over_energy_share(self):
        # The defect, pinned: relative = signal / (a_j^2 mean(A_j^2)/Var y),
        # so a weak term is pulled harder exactly by how weak it is.
        A, y = _problem()
        col_ms, y_var = anchor_scales(y, A)
        a = torch.tensor([2.0, 1.0, -3.0], dtype=torch.float64)
        c = a * 1.1
        share = a ** 2 * col_ms / y_var
        rel = anchor_penalty(c, a, "relative")
        sig = anchor_penalty(c, a, "signal", col_ms, y_var)
        assert rel.numpy() == pytest.approx((sig / share).numpy(), rel=1e-9)

    def test_weak_term_is_not_inflated(self):
        # Allen-Cahn shape: a real but WEAK term (tiny coefficient, share
        # ~1e-5) next to a dominant one, same 10% miss on each. Relative
        # calls them equal; signal ranks them by what they do to the fit.
        n = 800
        x = torch.linspace(0.0, 6.0, n, dtype=torch.float64)
        A = torch.stack([torch.sin(3.0 * x), torch.cos(2.0 * x)], dim=1)
        a = torch.tensor([1e-3, 5.0], dtype=torch.float64)
        y = A @ a
        c = a * 1.1
        rel = anchor_penalty(c, a, "relative")
        sig = anchor_penalty(c, a, "signal", *anchor_scales(y, A))
        assert float(rel[0]) == pytest.approx(float(rel[1]), rel=1e-9)
        assert float(sig[0]) < 1e-6 * float(sig[1])

    def test_gradient_bearing_and_needs_its_scales(self):
        A, y = _problem()
        a = torch.tensor([2.0, 1.0, -3.0], dtype=torch.float64)
        c = (a + 0.1).clone().requires_grad_(True)
        anchor_penalty(c, a, "signal", *anchor_scales(y, A)).sum().backward()
        assert torch.all(torch.isfinite(c.grad)) and float(c.grad.abs().sum()) > 0
        with pytest.raises(ValueError):
            anchor_penalty(c, a, "signal")
        with pytest.raises(ValueError):
            anchor_penalty(c, a, "absolute")


class TestCombineLoss:
    """The total loss, with no coefficients in either form."""

    def _terms(self):
        torch.manual_seed(0)
        p = torch.randn(5, dtype=torch.float64, requires_grad=True)
        phys = (p ** 2).sum() * 1e-3
        data = ((p - 1.0) ** 2).mean()
        return p, phys, data

    def test_sum_is_the_original_line_exactly(self):
        _, phys, data = self._terms()
        zero = torch.zeros((), dtype=torch.float64)
        got = combine_loss([phys, zero, 0.0, zero, data], "sum")
        assert torch.equal(got, phys + zero + 0.0 + zero + data)

    def test_log_is_the_sum_of_logs_of_active_terms(self):
        _, phys, data = self._terms()
        zero = torch.zeros((), dtype=torch.float64)       # a disabled term
        got = combine_loss([phys, zero, 0.0, data], "log").detach()
        want = (torch.log(phys) + torch.log(data)).detach()
        assert float(got) == pytest.approx(float(want), rel=1e-12)

    @pytest.mark.parametrize("k", [1e-6, 1e6])
    def test_log_gradient_ignores_a_constant_factor_on_any_term(self, k):
        # The licence for dropping yardsticks: rescaling a term by a
        # constant only shifts the log-loss, never its gradient.
        p, phys, data = self._terms()
        (g1,) = torch.autograd.grad(combine_loss([phys, data], "log"), p,
                                    retain_graph=True)
        (g2,) = torch.autograd.grad(combine_loss([k * phys, data], "log"), p)
        assert g2.numpy() == pytest.approx(g1.numpy(), rel=1e-9)

    def test_log_balances_relative_progress(self):
        # grad of log(t) is grad(t)/t: a term 1000x smaller in value gets
        # the same pull as a large one with the same relative slope.
        p = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        small, large = 1e-3 * p ** 2, 1.0 * p ** 2
        (gs,) = torch.autograd.grad(combine_loss([small], "log"), p)
        (gl,) = torch.autograd.grad(combine_loss([large], "log"), p)
        assert float(gs) == pytest.approx(float(gl), rel=1e-12)

    def test_rejects_unknown_forms_and_empty_log(self):
        _, phys, _ = self._terms()
        with pytest.raises(ValueError):
            combine_loss([phys], "mean")
        with pytest.raises(ValueError):
            combine_loss([torch.zeros(()), 0.0], "log")


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

    SCRIPTS = ("dp", "duffing", "wave", "ac")

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
        bound = {s: self._flag(s, "STAT_BOUND") for s in self.SCRIPTS}
        assert set(bound.values()) == {False}, f"STAT_BOUND: {bound}"
        order = {s: self._flag(s, "FD_ORDER") for s in self.SCRIPTS}
        assert set(order.values()) == {4}, f"FD_ORDER: {order}"
        form = {s: self._flag(s, "LOSS_FORM") for s in self.SCRIPTS}
        assert set(form.values()) == {"sum"}, f"LOSS_FORM: {form}"
        phys = {s: self._flag(s, "PHYS_TERM") for s in self.SCRIPTS}
        assert set(phys.values()) == {True}, f"PHYS_TERM: {phys}"

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
        # and the terms go through the ONE shared combiner, keyed on the flag
        call = ret[0].value
        assert (isinstance(call, ast.Call) and getattr(call.func, "id", None)
                == "combine_loss"), f"{system}: total_loss bypasses combine_loss"
        assert any(isinstance(a, ast.Name) and a.id == "LOSS_FORM"
                   for a in call.args), f"{system}: LOSS_FORM not passed"

    @pytest.mark.parametrize("system", SCRIPTS)
    def test_tls_physics_is_wired_through_the_shared_helper(self, system):
        # COEF_SOURCE='tls' must reach l_phys through the ONE tls_fit, with
        # the column yardsticks built by tls_scales from the FD design --
        # never a per-script reimplementation of the eigenproblem.
        tree = self._tree(system)
        fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "physics_and_stat_loss"]
        assert len(fn) == 1
        body = ast.dump(fn[0])
        assert "'tls'" in body and "tls_fit" in body and "TLS_SCALE" in body, \
            f"{system}: tls arm not routed through tls_fit"
        scale = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "TLS_SCALE"
                         for t in n.targets)]
        assert len(scale) == 1 and "tls_scales" in ast.dump(scale[0]), \
            f"{system}: TLS_SCALE not built by tls_scales"
        src = ast.unparse(tree)
        assert "'tls': '_tls'" in src, f"{system}: tls arm has no artifact suffix"


def _exact_problem(n=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, 3, generator=g, dtype=torch.float64)
    c = torch.tensor([2.0, -0.5, 3.0], dtype=torch.float64)
    return A, A @ c, c


def _noisy_problem(n=500, seed=1, sigma=0.05):
    A, y, c = _exact_problem(n, seed)
    g = torch.Generator().manual_seed(seed + 100)
    noise = sigma * torch.randn(n, 4, generator=g, dtype=torch.float64)
    return A + noise[:, 1:], y + noise[:, 0], c


class TestTls:
    """``tls_fit``: the target-free physics term (COEF_SOURCE='tls') --
    smallest eigenvalue of the column-scaled [target | features] Gram."""

    def test_exact_relation_scores_zero_and_recovers_the_coefficients(self):
        A, y, c = _exact_problem()
        lam, theta, w0 = tls_fit(y, A, tls_scales(y, A))
        assert float(lam) < 1e-25
        assert torch.allclose(theta, c, rtol=1e-10, atol=0.0)
        assert w0 > 0.1

    def test_lam_is_the_smallest_eigenvalue_of_the_scaled_gram(self):
        A, y, _ = _noisy_problem()
        s = tls_scales(y, A)
        lam, _, _ = tls_fit(y, A, s)
        Z = (torch.cat([y.unsqueeze(1), A], dim=1) / s).numpy()
        ref = np.linalg.eigvalsh(Z.T @ Z / Z.shape[0])[0]
        assert float(lam) == pytest.approx(ref, rel=1e-10)

    def test_scales_are_the_uncentred_column_rms(self):
        A, y, _ = _noisy_problem()
        A = A + 3.0                                   # a non-zero mean
        Z = np.column_stack([y.numpy(), A.numpy()])
        ref = np.sqrt((Z ** 2).mean(axis=0))
        assert np.allclose(tls_scales(y, A).numpy(), ref, rtol=1e-14)

    def test_scales_floor_a_dead_column_at_tiny_not_eps(self):
        A, y, _ = _exact_problem()
        A[:, 1] = 0.0
        s = tls_scales(y, A)
        assert float(s[2]) == torch.finfo(torch.float64).tiny

    def test_never_above_the_ols_loss_in_the_same_units(self):
        # OLS pins the target's coefficient at 1; TLS minimises over every
        # direction, so in the same scaled units it can only be lower.
        A, y, _ = _noisy_problem()
        s = tls_scales(y, A)
        lam, _, _ = tls_fit(y, A, s)
        c, r = global_ols(y, A, ridge=0.0)
        ols = float((r ** 2).mean()) / float(s[0]) ** 2
        assert float(lam) < ols

    def test_target_free_swapping_the_target_changes_nothing(self):
        A, y, _ = _noisy_problem()
        s = tls_scales(y, A)
        lam, _, _ = tls_fit(y, A, s)
        # column 1 of A becomes the "target", y moves into the library
        A2 = torch.stack([y, A[:, 0], A[:, 2]], dim=1)
        s2 = torch.stack([s[2], s[0], s[1], s[3]])
        lam2, _, _ = tls_fit(A[:, 1], A2, s2)
        assert float(lam2) == pytest.approx(float(lam), rel=1e-12)

    def test_invariant_under_rescaling_the_equation(self):
        # The column yardsticks come from the same data, so a common or
        # per-column rescale moves nothing: weight 1 stays meaningful.
        A, y, _ = _noisy_problem()
        lam, theta, _ = tls_fit(y, A, tls_scales(y, A))
        a, b = 1e-3, torch.tensor([1.0, 7.0, 1.0], dtype=torch.float64)
        y2, A2 = a * y, a * A * b
        lam2, theta2, _ = tls_fit(y2, A2, tls_scales(y2, A2))
        assert float(lam2) == pytest.approx(float(lam), rel=1e-10)
        assert torch.allclose(theta2, theta / b, rtol=1e-10, atol=0.0)

    def test_gradient_is_the_eigenvalue_derivative(self):
        # v is detached; by the envelope theorem the gradient through Z
        # alone IS d lambda_min -- checked against central differences.
        A, y, _ = _noisy_problem(n=200)
        s = tls_scales(y, A)
        yv = y.clone().requires_grad_(True)
        Av = A.clone().requires_grad_(True)
        lam, _, _ = tls_fit(yv, Av, s)
        lam.backward()
        g = torch.Generator().manual_seed(7)
        dy = torch.randn(y.shape, generator=g, dtype=torch.float64)
        dA = torch.randn(A.shape, generator=g, dtype=torch.float64)
        h = 1e-6
        up = float(tls_fit(y + h * dy, A + h * dA, s)[0])
        dn = float(tls_fit(y - h * dy, A - h * dA, s)[0])
        fd = (up - dn) / (2 * h)
        an = float((yv.grad * dy).sum() + (Av.grad * dA).sum())
        assert an == pytest.approx(fd, rel=1e-6)

    def test_an_identity_without_the_target_drives_w0_to_zero(self):
        # The failure mode OLS cannot have: the design's null direction
        # does not involve the target, so theta is undefined -- w0 says so.
        g = torch.Generator().manual_seed(3)
        f = torch.randn(400, 2, generator=g, dtype=torch.float64)
        A = torch.stack([f[:, 0], f[:, 1], f[:, 0] + f[:, 1]], dim=1)
        y = torch.randn(400, generator=g, dtype=torch.float64)
        lam, _, w0 = tls_fit(y, A, tls_scales(y, A))
        assert float(lam) < 1e-25
        assert w0 < 1e-8

    def test_float32_design_keeps_its_dtype_and_stays_finite(self):
        A, y, c = _noisy_problem()
        A32, y32 = A.float(), y.float()
        s = tls_scales(y32, A32)
        lam, theta, w0 = tls_fit(y32, A32, s)
        assert lam.dtype == torch.float32 and theta.dtype == torch.float64
        lam64, theta64, _ = tls_fit(y, A, tls_scales(y, A))
        assert float(lam) == pytest.approx(float(lam64), rel=1e-4)
        assert torch.allclose(theta, theta64, rtol=1e-4, atol=0.0)
