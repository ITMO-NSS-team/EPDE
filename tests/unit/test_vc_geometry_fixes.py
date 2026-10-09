"""Grid geometry of the two Gram-backed instability estimators.

``'vcoef'`` (``VaryingCoefSetup``) and ``'cv'`` (``GramSetup``) are the only
instability metrics whose construction depends on the grid's shape beyond the
flat sample count: vcoef builds a per-axis cosine basis whose resolution ``K_d``
comes from the main variable's Taylor microscale, and cv builds sliding windows
along every axis. Four geometry defects lived there:

* the resolved ``K`` was cached per ``(grid_shape, main_var)`` with no
  trajectory key, so every same-shape trajectory reused whichever one was
  resolved first;
* ``K_d`` was never clamped to the axis length -- a length-``n`` DCT-II has
  exactly ``n`` modes, so ``K_d > n_d`` adds round-off / aliased columns and
  the scores of even the exact truth exploded to 1e10-1e12;
* a size-1 grid axis crashed both (``np.gradient`` in the microscale,
  ``window_size = 0`` in the windows) and aborted a live search;
* a failed basis resolution fell back to ``K = 3`` per axis without a word.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import epde
import epde.globals as global_var
from conftest import using_config
from epde.globals import EPDEUsageWarning
from epde.operators.common import stability as stab
from epde.operators.common.sparsity import PhysicsInformedLasso


class _Fields:
    """The one samples-manager call the basis resolver makes,
    ``get((var, (1.0,)), 0)[sample_key]``: trajectory key -> flat field."""

    def __init__(self, fields):
        self._fields = fields

    def get(self, label, *args, **kwargs):
        return dict(self._fields)


class _Broken:
    """A samples manager whose field lookup fails."""

    def get(self, label, *args, **kwargs):
        raise KeyError(label)


@pytest.fixture
def fields(monkeypatch):
    """Install a stub samples manager holding ``{key: flat u}`` (``None`` for
    no manager at all) behind an empty basis cache, as ``init_caches`` leaves
    it at the start of a search."""
    def install(per_key):
        manager = None if per_key is None else (
            per_key if isinstance(per_key, _Broken) else _Fields(per_key))
        monkeypatch.setattr(global_var, 'samples_manager', manager,
                            raising=False)
        global_var.vc_modes_cache.clear()
    yield install
    global_var.vc_modes_cache.clear()


def _wave(n0, n1, rough):
    """``u`` on a (t, x) grid with exact x-derivatives. ``rough`` varies fast
    along both axes (Taylor K hits the cap); the smooth one sits on a large
    offset, so its microscale is long and K stays at the floor."""
    t = np.linspace(0.0, 1.0, n0)
    x = np.linspace(0.0, 2 * np.pi, n1, endpoint=False)
    T, XG = np.meshgrid(t, x, indexing='ij')
    if rough:
        u = np.sin(3 * XG - 4 * T) + 0.5 * np.sin(5 * XG + 2 * T)
        ux = 3 * np.cos(3 * XG - 4 * T) + 2.5 * np.cos(5 * XG + 2 * T)
        uxx = -9 * np.sin(3 * XG - 4 * T) - 12.5 * np.sin(5 * XG + 2 * T)
    else:
        u = 5.0 + np.sin(XG - T)
        ux = np.cos(XG - T)
        uxx = -np.sin(XG - T)
    return u.reshape(-1), ux.reshape(-1), uxx.reshape(-1)


def _truth(n0, n1, rough=True):
    """``y = -u_x + 0.1 u_xx`` exactly: a constant-coefficient truth, so a
    sound varying-coefficient score is round-off small."""
    u, ux, uxx = _wave(n0, n1, rough)
    X = np.column_stack([ux, uxx])
    return X, -1.0 * ux + 0.1 * uxx, np.ones(ux.size), u


def _line(n=40):
    """The 1-D problem of ``test_keeprule_metric``: u_tt = -u plus two
    hitchhikers, on ``n`` points."""
    t = np.linspace(0, 4 * np.pi, n)
    u = np.sin(t) + 0.3 * np.cos(t)
    X = np.column_stack([u, np.cos(3 * t), t / t.max()])
    return X, -u, np.ones(n), u


class TestBasisCacheIsPerTrajectory:
    """Two same-shape trajectories with different microscales must each get
    their own ``K``, whichever is resolved first."""

    GRID = (32, 40)

    def _two(self):
        return {0: _wave(*self.GRID, rough=False)[0],
                1: _wave(*self.GRID, rough=True)[0]}

    def _alone(self, fields, key):
        fields(self._two())
        return stab.resolve_vc_modes_from_input(self.GRID, 'u', sample_key=key)

    def test_the_two_trajectories_really_differ(self, fields):
        assert self._alone(fields, 0) != self._alone(fields, 1)

    @pytest.mark.parametrize('first, second', [(0, 1), (1, 0)])
    def test_each_trajectory_keeps_its_own_K_in_either_order(
            self, fields, first, second):
        expected = self._alone(fields, second)
        fields(self._two())
        stab.resolve_vc_modes_from_input(self.GRID, 'u', sample_key=first)
        got = stab.resolve_vc_modes_from_input(self.GRID, 'u',
                                               sample_key=second)
        assert got == expected

    def test_the_objective_scores_trajectory_one_on_its_own_basis(self, fields):
        X, y, w, _ = _truth(*self.GRID)
        y = y + 1e-3 * np.sin(5 * np.arange(y.size))  # a non-zero score
        fields(self._two())
        alone = stab.vc_stability_total_lr(X, y, w, self.GRID, main_var='u',
                                           sample_key=1)
        fields(self._two())
        stab.vc_stability_total_lr(X, y, w, self.GRID, main_var='u',
                                   sample_key=0)
        after = stab.vc_stability_total_lr(X, y, w, self.GRID, main_var='u',
                                           sample_key=1)
        assert after == alone

    def test_the_super_gram_resolves_each_trajectory_on_its_own(self, fields):
        """The keep-rule's RPS path (precompute_super) and the direct
        construction agree on every trajectory's basis size."""
        X, _, w, _ = _truth(*self.GRID)
        fields(self._two())
        direct = {key: stab.VaryingCoefSetup(X[:, :1], X[:, 1], w, self.GRID,
                                             main_var='u', sample_key=key).B
                  for key in (1, 0)}
        fields(self._two())
        sup = stab.VaryingCoefSetup.precompute_super(
            {0: X, 1: X}, {0: w, 1: w}, {0: self.GRID, 1: self.GRID},
            main_var='u')
        assert {key: sup['per_sample'][key]['B'] for key in (0, 1)} == direct
        assert direct[0] != direct[1]


class TestShortAxisIsClamped:
    """``K_d <= n_d``: no mode index may reach the axis length."""

    @pytest.mark.parametrize('n0', [2, 3, 4, 5])
    def test_the_resolved_basis_never_exceeds_the_axis(self, fields, n0):
        X, y, w, u = _truth(n0, 64)
        fields({0: u})
        modes = stab.VaryingCoefSetup._resolve_modes(None, (n0, 64), 'u', None,
                                                     sample_key=0)
        assert all(K <= n for K, n in zip(modes, (n0, 64))), modes

    @pytest.mark.parametrize('n0', [2, 3, 4, 5])
    @pytest.mark.parametrize('source', ['taylor', 'pinned', 'fallback'])
    @pytest.mark.filterwarnings('ignore::epde.globals.EPDEUsageWarning')
    def test_the_exact_truth_no_longer_explodes(self, fields, n0, source):
        """Was 1e10-1e12 on every n0 <= 5 (the search then pruned every
        feature, the truth included); a constant-coefficient truth is
        round-off small, as it is on a long axis."""
        X, y, w, u = _truth(n0, 64)
        fields(None if source == 'fallback' else {0: u})
        setup = stab.VaryingCoefSetup(
            X, y, w, (n0, 64), main_var='u', sample_key=0,
            modes=(6, 6) if source == 'pinned' else None, fit_intercept=False)
        scores = setup.score()
        assert np.all(np.isfinite(scores))
        assert np.max(scores) < 1e-12, scores


class TestSizeOneAxisIsAbsent:
    """A size-1 axis carries no variation: it contributes no modes / no
    windows, so ``(n, 1)`` and ``(1, n)`` score exactly like ``(n,)``."""

    GRIDS = [(40, 1), (1, 40)]

    @pytest.mark.parametrize('grid', GRIDS)
    def test_vcoef_objective(self, fields, grid):
        X, y, w, u = _line()
        fields({0: u})
        line = stab.vc_stability_total_lr(X, y, w, (40,), main_var='u',
                                          sample_key=0)
        got = stab.vc_stability_total_lr(X, y, w, grid, main_var='u',
                                         sample_key=0)
        assert np.isfinite(got)
        np.testing.assert_allclose(got, line, rtol=1e-12)

    @pytest.mark.parametrize('grid', GRIDS)
    def test_cv_objective(self, grid):
        X, y, w, _ = _line()
        line = stab.cv_scores(stab.calculate_weights(X, y, w, (40,)))
        got = stab.cv_scores(stab.calculate_weights(X, y, w, grid))
        np.testing.assert_allclose(got, line, rtol=1e-12)

    @pytest.mark.parametrize('grid', GRIDS)
    def test_cv_instability_divides_by_the_windowed_axes_only(self, grid):
        # Instability's cv branch divides by the number of window batches the
        # stack holds; a size-1 axis contributes none, so (n, 1) must not be
        # halved against (n,).
        from epde.operators.common.objectives import FitContext, Instability

        X, y, w, _ = _line()

        class _Eq:
            weights_internal = np.r_[np.ones(X.shape[1]), 0.0]
            main_var_to_explain = 'u'
            _cached_vc_score = _cached_sw_weights = _cached_alt_instability = None

            def evaluate(self, active_only=False):
                return {0: y}, {0: X}

        def _value(shape):
            ctx = FitContext(g_fun_vals={0: w}, data_shape={0: np.asarray(shape)},
                             penalty_coeff=0.2)
            with using_config(instability_metric='cv'):
                return Instability().compute(_Eq(), ctx)

        np.testing.assert_allclose(_value(grid), _value((40,)), rtol=1e-12)

    @pytest.mark.parametrize('grid', GRIDS)
    @pytest.mark.parametrize('metric', ['vcoef', 'cv'])
    def test_keep_rule(self, fields, grid, metric):
        X, y, w, u = _line()
        fields({0: u})
        with using_config(instability_metric=metric):
            line = PhysicsInformedLasso(main_var='u').fit(
                X, y, sample_key=0, grid_shape=(40,), sample_weights=w)
            line_coef = line.full_coef_.copy()
            got = PhysicsInformedLasso(main_var='u').fit(
                X, y, sample_key=0, grid_shape=grid, sample_weights=w)
        np.testing.assert_allclose(got.full_coef_, line_coef, rtol=1e-12)
        if metric == 'vcoef':
            np.testing.assert_allclose(got.cached_vc_score_,
                                       line.cached_vc_score_, rtol=1e-12)
        else:
            np.testing.assert_allclose(got.cached_weights_,
                                       line.cached_weights_, rtol=1e-12)

    @pytest.mark.parametrize('grid', GRIDS)
    @pytest.mark.parametrize('metric', ['vcoef', 'cv'])
    def test_rps_super_gram(self, fields, grid, metric):
        """The keep-rule's super-Gram path (EqRPS sweep). Compared with the
        same path on ``(n,)``: super and direct construction already differ by
        path round-off on this nearly collinear design, independent of the
        grid's shape."""
        X, y, w, u = _line()
        Z = np.column_stack([y, X])
        fields({0: u})

        def view(shape):
            if metric == 'vcoef':
                return stab.VaryingCoefSetup.from_full(
                    stab.VaryingCoefSetup.precompute_super(
                        {0: Z}, {0: w}, {0: shape}, main_var='u'),
                    0)[0].score()
            return stab.GramSetup.from_full(
                stab.GramSetup.precompute_super({0: Z}, {0: w}, {0: shape}),
                0)[0].solve()

        np.testing.assert_allclose(view(grid), view((40,)), rtol=1e-12)

    @pytest.mark.filterwarnings('ignore::epde.globals.EPDEUsageWarning')
    def test_no_usable_axis_leaves_the_existing_degenerate_paths(self, fields):
        """Every axis size 1: vcoef keeps only its constant column (the
        no-mode solve), cv has no window at all -- neither raises."""
        X, y, w, u = (a[:1] for a in _line())
        fields({0: u})
        setup = stab.VaryingCoefSetup(X, y, w, (1, 1), main_var='u',
                                      sample_key=0)
        assert setup.B == 1
        assert setup.score().shape == (X.shape[1] + 1,)
        assert stab.GramSetup(X, y, w, (1, 1)).solve().shape == (
            0, X.shape[1] + 1)


class TestFallbackIsVisible:
    """An unresolvable basis still falls back to ``K = 3`` per axis, but says
    so -- once per trajectory/grid/variable, not once per candidate."""

    def _build(self, key=0, grid=(40,)):
        X, y, w, _ = _line()
        return stab.VaryingCoefSetup(X, y, w, grid, main_var='u',
                                     sample_key=key)

    @pytest.mark.parametrize('manager', [None, _Broken()])
    def test_warns_once_per_cache_key(self, fields, manager):
        fields(manager)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self._build(0)
            self._build(0)
            self._build(1)
        ours = [w for w in caught if issubclass(w.category, EPDEUsageWarning)]
        assert len(ours) == 2, [str(w.message) for w in ours]
        assert self._build(0).B == 3        # the fallback itself is unchanged

    def test_a_resolved_basis_is_silent(self, fields):
        fields({0: _line()[3]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self._build(0)
        assert not [w for w in caught
                    if issubclass(w.category, EPDEUsageWarning)]

    def test_the_warning_survives_the_process_wide_ignore(self):
        """``init_verbose`` installs ``filterwarnings('ignore')``; only the
        EPDE categories get through, and pytest's own filter handling would
        hide a plain ``UserWarning`` slipping through here."""
        script = '; '.join((
            'import numpy as np, epde.globals as gv',
            'from epde.operators.common.stability import VaryingCoefSetup',
            'gv.init_verbose()',
            'gv.samples_manager = None',
            'gv.vc_modes_cache.clear()',
            't = np.linspace(0, 4 * np.pi, 40)',
            'VaryingCoefSetup(np.column_stack([np.sin(t), np.cos(3 * t)]), '
            '-np.sin(t), np.ones(40), (40,), main_var="u", sample_key=0)'))
        done = subprocess.run([sys.executable, '-c', script],
                              capture_output=True, text=True,
                              cwd=str(Path(epde.__file__).parent.parent))
        assert done.returncode == 0, done.stderr[-2000:]
        assert 'EPDEUsageWarning' in done.stderr, done.stderr[-2000:]
