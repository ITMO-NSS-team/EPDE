"""The opt-in evaluation row budget (``domain.evaluation_stride``, per axis).

Derivatives are computed on the full grid; under a budget only a regular
strided sub-lattice of the interior box becomes cache rows. A Domain hands out
the rows (``g_func_mask``), the weights (``gFunc('dm')``) and the shape
(``inner_shape``) from one stride, so every scorer sees the same lattice. With
no budget -- or all-ones strides -- nothing may change at all. Each axis is
thinned only by its own stride (there is no total-row-count option: it would
trade time against space).

What is pinned here:

* the pure stride rule (``resolve_evaluation_stride``) and its refusals;
* the default path is today's formulas, bit for bit;
* the lattice geometry, and that a kept point carries exactly the value the
  unstrided build has there (the derivatives are never taken on the lattice);
* the guards: the test function's support, the row floors, the solver
  refusals, the budget/domain mismatch in both directions (H1);
* the vcoef microscale comes from the full box, rescaled by the stride;
* the warnings, the immutability of the stride, and the pool rebuild.

The fit-start row floor (H6) refuses a lattice too short for the configured
instability metric. Every fixture that reaches it is sized to clear the floor
of EVERY shipped metric at the default width (``TestFixturesClearEveryFloor``),
so the tests pin the row budget whichever metric is the default; only the
row-floor tests themselves use a lattice some metric refuses, on purpose.
"""
import inspect
import itertools
import pathlib
import warnings

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.globals import EPDEUsageWarning
from epde.interface.interface import EpdeSearch
from epde.interface.prepared_tokens import (ControlVarTokens, ExternalDerivativesTokens,
                                            GridTokens, TrigonometricTokens)
from epde.interface.search_config import METRIC_MENUS, check_evaluation_budget, load_search_config
from epde.operators.common import survival
from epde.operators.common.sparsity import instability_scores
from epde.operators.common.stability import resolve_vc_modes_from_input, taylor_microscale
from epde.structure.domain import (Domain, LATTICE_MIN_LEN, _lattice_len,
                                   resolve_evaluation_stride)

EPDE_ROOT = pathlib.Path(epde.__file__).resolve().parent


# ---------------------------------------------------------------------------
# fixture sizes: derived from the row floors, not from one metric
# ---------------------------------------------------------------------------

#: The width a default search judges the fit-start row floor at.
WIDTH = survival.configured_width(load_search_config().search_space)


def _refused_by(lattice, width=WIDTH):
    """The shipped instability metrics whose row floor refuses ``lattice``."""
    return [metric for metric in METRIC_MENUS['instability_metric']
            if survival.row_floor(metric, lattice, width) is not None]


#: The fewest 1-D rows no shipped metric refuses (het's three blocks of
#: 4 * (WIDTH + 1) rows: 84 at width 6).
ROWS_1D = next(n for n in itertools.count(1) if not _refused_by((n,)))
#: The coarsest stride this module puts on the default 1-D trajectory.
COARSEST_1D = 4
#: The default 1-D trajectory: a margin of BW_1D and an interior box of which
#: every stride up to COARSEST_1D keeps at least ROWS_1D rows.
BW_1D = 10
BOX_1D = COARSEST_1D * ROWS_1D
SHAPE_1D = (BOX_1D + 2 * BW_1D,)


#: The 2-D strided search of the consistency checks: a (18, 7) lattice -- 7
#: rows per axis-0 level, a divisor of the 4 * (6 + 1) block at width 6, so
#: tile's and het's level-aligned slabs (their count taken from the axis-0
#: levels) never fall below a block.
SHAPE_2D, BW_2D, STRIDE_2D = (40, 25), 2, (2, 3)


def _kept_1d(values, stride):
    """The points of a default-trajectory array a 1-D ``stride`` keeps."""
    return values[BW_1D:BW_1D + BOX_1D:stride]


def _resolved_lattice(shape, bw, budget):
    """The lattice shape a budget gives a grid of ``shape`` with margin ``bw``."""
    bw = (bw,) * len(shape) if np.isscalar(bw) else tuple(bw)
    stride = resolve_evaluation_stride(shape, bw, budget.get('evaluation_stride')) or (1,) * len(shape)
    return tuple(_lattice_len(n - 2 * b, k) for n, b, k in zip(shape, bw, stride))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _search(prep='FD', **kwargs):
    kwargs.setdefault('verbose_params', {'show_iter_idx': False})
    search = epde.EpdeSearch(**kwargs)
    search.set_preprocessor(default_preprocessor_type=prep, preprocessor_kwargs={})
    return search


def _grids(shape):
    axes = [np.linspace(0.0, 1.0 + axis, n) for axis, n in enumerate(shape)]
    return axes[0] if len(shape) == 1 else np.meshgrid(*axes, indexing='ij')


def _field(grids):
    if isinstance(grids, np.ndarray):
        return np.sin(3 * grids) + 0.5 * grids ** 2
    return np.sin(2 * grids[0]) * np.cos(3 * grids[1]) + 0.3 * sum(g ** 2 for g in grids)


def _domain(search, shape, bw, domain_id=0, gfunction=None):
    return search.createDomain(_grids(shape), boundary_width=bw, ID=domain_id,
                               gfunction=gfunction)[1]


def _trajectory(search, shape=SHAPE_1D, bw=BW_1D, cache_id=0, data=None, **kwargs):
    domain = _domain(search, shape, bw, domain_id=cache_id)
    data = _field(_grids(shape)) if data is None else data
    return search.createTrajectory({'u': data}, domain, cache_id=cache_id, **kwargs)[1]


def _quiet(fn, *args, **kwargs):
    """Run ``fn`` with the budget's usage warning silenced (it is tested on its own)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', EPDEUsageWarning)
        return fn(*args, **kwargs)


def _no_search(search, monkeypatch):
    """Count pool builds and skip the evolution itself."""
    builds = []
    original = search._build_pool

    def counted(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(search, '_build_pool', counted)
    monkeypatch.setattr(search, '_run_optimization', lambda *args, **kwargs: None)
    return builds


def _registered():
    return list(global_var.samples_manager._traj.values())


def _cached(cache_id=0):
    """Copies of every tensor in a trajectory's subcache (the next search resets the caches)."""
    return {label: np.array(tensor, copy=True)
            for label, tensor in global_var.tensor_cache.memory_default[cache_id].items()}


def _build(shape, bw, prep='FD', **budget):
    """One search, one trajectory, the pool built: the domain and a copy of the cache."""
    search = _quiet(_search, prep=prep, **budget)
    trajectory = _quiet(_trajectory, search, shape, bw)
    _quiet(search.create_pool, data=[trajectory], max_deriv_order=(1,) * len(shape))
    return trajectory, _cached()


def _box(domain):
    return tuple(int(n) for n in domain.box_shape)


def _lattice(stride):
    """Slices of the box that keep every ``k``-th level from its first point."""
    return tuple(slice(None, None, int(k)) for k in stride)


def _usage_warnings(records):
    return [w for w in records if issubclass(w.category, EPDEUsageWarning)]


# ---------------------------------------------------------------------------
# T1: the stride rule
# ---------------------------------------------------------------------------

class TestResolver:

    def test_an_int_strides_every_axis(self):
        assert resolve_evaluation_stride((20, 30), (2, 2), stride=2) == (2, 2)

    def test_a_tuple_or_list_passes_through(self):
        assert resolve_evaluation_stride((20, 30), (2, 2), stride=(2, 3)) == (2, 3)
        assert resolve_evaluation_stride((20, 30), (2, 2), stride=[2, 3]) == (2, 3)
        assert resolve_evaluation_stride((20, 30), (2, 2), stride=(np.int64(2), 3)) == (2, 3)

    @pytest.mark.parametrize('stride', [(2,), (2, 2, 2), 0, -1, True, (2, 0), (2, -3),
                                        (True, 2), 2.0, (2, 2.0), '2'])
    def test_a_bad_stride_raises(self, stride):
        with pytest.raises(ValueError, match='positive integer'):
            resolve_evaluation_stride((20, 30), (2, 2), stride=stride)

    def test_there_is_no_row_count_parameter(self):
        """Only per-axis strides: a total row count would mix time and space."""
        assert list(inspect.signature(resolve_evaluation_stride).parameters) == [
            'shape', 'boundary_per_axis', 'stride']

    def test_a_boundary_per_axis_of_the_wrong_length_raises(self):
        with pytest.raises(ValueError, match='boundary widths'):
            resolve_evaluation_stride((20, 30), (2,), stride=2)

    def test_all_ones_is_no_budget(self):
        assert resolve_evaluation_stride((20, 30), (2, 2), stride=1) is None
        assert resolve_evaluation_stride((20, 30), (2, 2), stride=(1, 1)) is None

    def test_nothing_requested_is_no_budget(self):
        assert resolve_evaluation_stride((20, 30), (2, 2)) is None

    def test_the_boundary_anchors_the_box(self):
        """The lattice lives on the interior box, not on the grid."""
        assert resolve_evaluation_stride((140,), (20,), stride=2) == (2,)
        with pytest.raises(ValueError, match='fewer than 3 levels'):
            resolve_evaluation_stride((140,), (20,), stride=50)    # box 100: 2 levels

    def test_an_explicit_stride_below_three_levels_raises(self):
        assert LATTICE_MIN_LEN == 3
        assert resolve_evaluation_stride((100,), (0,), stride=40) == (40,)       # 3 levels
        with pytest.raises(ValueError, match='fewer than 3 levels'):
            resolve_evaluation_stride((100,), (0,), stride=50)                    # 2 levels

    def test_a_box_shorter_than_three_keeps_all_its_levels(self):
        assert resolve_evaluation_stride((2, 100), (0, 0), stride=(1, 4)) == (1, 4)
        with pytest.raises(ValueError, match='fewer than 3 levels'):
            resolve_evaluation_stride((2, 100), (0, 0), stride=(2, 4))


# ---------------------------------------------------------------------------
# T2: the default is today's code
# ---------------------------------------------------------------------------

class TestDefaultIdentity:

    SHAPE, BW = (40, 30), (2, 3)

    def _check_default(self, domain):
        g = domain.g_func
        assert domain.evaluation_stride is None
        assert np.array_equal(domain.g_func_mask, g != 0)
        assert domain.g_func_mask.dtype == (g != 0).dtype
        expected = np.array(domain.initial_shape) - 2 * np.array(self.BW)
        assert np.array_equal(domain.inner_shape, expected)
        assert domain.inner_shape.dtype == expected.dtype
        full = [np.array(grid, copy=True) for grid in domain.getGrids('full')]
        solver = domain.getGrids('solver')
        assert all(np.array_equal(s, f[2:-2, 3:-3]) for s, f in zip(solver, full))

    def test_no_budget(self):
        search = _search()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            domain = _domain(search, self.SHAPE, self.BW)
        assert not _usage_warnings(records)
        self._check_default(domain)
        trajectory = search.createTrajectory({'u': _field(_grids(self.SHAPE))}, domain, cache_id=0)[1]
        g = domain.g_func
        assert np.array_equal(trajectory.gFuncs('m'), g != 0)
        assert np.array_equal(trajectory.gFuncs('dm'), g[g != 0])
        assert np.array_equal(trajectory.gFuncs('dmf'), g[g != 0].reshape(-1))

    @pytest.mark.parametrize('budget', [dict(evaluation_stride=1),
                                        dict(evaluation_stride=(1, 1))])
    def test_a_budget_that_resolves_to_none(self, budget):
        """All-ones strides normalise to the default path."""
        search = _search(**budget)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            domain = _domain(search, self.SHAPE, self.BW)
        assert not _usage_warnings(records)
        self._check_default(domain)

    def test_a_budget_that_resolves_to_none_passes_the_fit_check(self, monkeypatch):
        search = _search(evaluation_stride=1)
        _no_search(search, monkeypatch)
        trajectory = _trajectory(search)
        search.fit(data=[trajectory], max_deriv_order=(1,))
        assert _registered() == [trajectory]
        assert trajectory.get(('u', (1.0,))).shape == (BOX_1D,)


# ---------------------------------------------------------------------------
# T3: the lattice geometry
# ---------------------------------------------------------------------------

def _support(grids):
    """A non-constant positive test function (BoundaryExclusion zeroes the margin)."""
    grids = grids if isinstance(grids, (list, tuple)) else [grids]
    return 1.0 + sum((axis + 1) * g ** 2 for axis, g in enumerate(grids))


class TestLatticeGeometry:

    CASES = [((120,), 10, 3, (slice(10, 110, 3),)),
             ((40, 30), (2, 3), (2, 3), (slice(2, 38, 2), slice(3, 27, 3))),
             ((12, 10, 9), 2, (2, 1, 2), (slice(2, 10, 2), slice(2, 8, 1), slice(2, 7, 2)))]

    @pytest.mark.parametrize('shape, bw, stride, slices', CASES)
    def test_rows_weights_and_shape(self, shape, bw, stride, slices):
        search = _search(evaluation_stride=stride)
        domain = _quiet(_domain, search, shape, bw, gfunction=_support)
        assert domain.evaluation_stride == tuple(s.step for s in slices)
        mask = domain.g_func_mask
        assert int(mask.sum()) == int(np.prod(domain.inner_shape))
        assert tuple(int(n) for n in domain.inner_shape) == tuple(
            len(range(s.start, s.stop, s.step)) for s in slices)
        assert domain.inner_shape.dtype == domain.box_shape.dtype

        data = np.random.default_rng(0).standard_normal(shape)
        assert np.array_equal(data[mask], data[slices].ravel())

        # the weights follow the same mask, in the same order
        g = domain.g_func
        assert len(np.unique(g[g != 0])) > 1                 # non-constant
        trajectory = search.createTrajectory({'u': data}, domain, cache_id=0)[1]
        assert np.array_equal(trajectory.gFuncs('dm'), g[slices].ravel())
        assert np.array_equal(trajectory.gFuncs('dmf'), g[slices].ravel())

    @pytest.mark.parametrize('shape, bw, stride, slices', CASES)
    def test_no_solver_grid(self, shape, bw, stride, slices):
        search = _search(evaluation_stride=stride)
        domain = _quiet(_domain, search, shape, bw)
        with pytest.raises(ValueError, match='sub-lattice'):
            domain.getGrids('solver')
        assert len(domain.getGrids('full')) == len(shape)    # the full grid is still there

    def test_a_direct_domain_takes_a_stride(self):
        _search()
        domain = _quiet(Domain, _grids((120,)), global_var.grid_cache, 0, 3, boundary=10,
                        evaluation_stride=3)
        domain.g_func = EpdeSearch._get_g_func(None, 10)
        domain.validateTestFunction()
        assert domain.evaluation_stride == (3,)
        assert domain.evaluation_request == 3
        assert int(domain.g_func_mask.sum()) == 34 == int(np.prod(domain.inner_shape))
        with pytest.raises(TypeError, match='evaluation_max_rows'):
            Domain(_grids((120,)), global_var.grid_cache, 0, 4, boundary=10, evaluation_max_rows=40)

    def test_the_box_slices_are_the_whole_box(self):
        search = _search(evaluation_stride=2)
        domain = _quiet(_domain, search, (40, 30), (2, 3))
        assert domain.box_slices() == (slice(2, 38, 1), slice(3, 27, 1))
        assert _box(domain) == (36, 24)
        assert domain.evaluation_request == 2


# ---------------------------------------------------------------------------
# T4: a kept point carries the unstrided value
# ---------------------------------------------------------------------------

#: (shape, margin, preprocessor, budget) of the bitwise kept-value checks. The
#: lattices: (112,), (18, 7), (4, 4, 7), (12, 7), (84,), (84,), (12, 7) at
#: width 6 -- 2-D and 3-D ones keep 7 or 28 rows per axis-0 level, a divisor of
#: the 4 * (6 + 1) block, so tile's and het's level-aligned slabs (their count
#: taken from the axis-0 levels) never fall below a block.
KEPT_CASES = [
    (SHAPE_1D, BW_1D, 'FD', dict(evaluation_stride=3)),
    ((40, 27), (2, 3), 'FD', dict(evaluation_stride=(2, 3))),
    ((12, 12, 18), 2, 'FD', dict(evaluation_stride=2)),
    ((40, 18), 2, 'FD', dict(evaluation_stride=(3, 2))),
    (SHAPE_1D, BW_1D, 'poly', dict(evaluation_stride=COARSEST_1D)),
    (SHAPE_1D, BW_1D, 'spectral', dict(evaluation_stride=COARSEST_1D)),
    ((40, 18), 2, 'poly', dict(evaluation_stride=(3, 2))),
]


class TestKeptValues:

    @pytest.mark.parametrize('shape, bw, prep, budget', KEPT_CASES)
    def test_every_cached_tensor_is_the_sliced_unstrided_one(self, shape, bw, prep, budget):
        """Bitwise: the derivatives come from the full grid, then rows are dropped."""
        whole, full = _build(shape, bw, prep)
        box = _box(whole._domain)
        strided, kept = _build(shape, bw, prep, **budget)
        stride = strided._domain.evaluation_stride
        assert stride is not None
        assert set(kept) == set(full) and len(kept) >= len(shape) + 1
        for label, tensor in kept.items():
            assert np.array_equal(tensor, full[label].reshape(box)[_lattice(stride)].ravel()), label

    def test_evaluated_tokens_are_the_full_grid_values_at_the_kept_points(self):
        """Grid and trig tokens are evaluated on the full grid, then masked (CustomEvaluator)."""
        search = _search(evaluation_stride=3)
        trajectory = _quiet(_trajectory, search)
        trig = TrigonometricTokens(freq_values=[2.0], dimensionality=0)
        coordinate = GridTokens(['t'], dimensionality=0, max_power=1)
        search.create_pool(data=[trajectory], max_deriv_order=(1,), additional_tokens=[trig, coordinate])
        grid = _grids(SHAPE_1D)
        np.random.seed(0)
        _, sine = trig._token_family.create(label='sin', all_vars=['u'])
        _, time = coordinate._token_family.create(label='t', all_vars=['u'])
        assert np.array_equal(sine.evaluate()[0], _kept_1d(np.sin(2.0 * grid), 3))
        assert np.array_equal(time.evaluate()[0], _kept_1d(grid, 3))

    def test_stored_tokens_are_masked_to_the_lattice(self):
        search = _search(evaluation_stride=2)
        trajectory = _quiet(_trajectory, search)
        values = np.cos(np.arange(float(SHAPE_1D[0])))
        search.create_pool(data=[trajectory], additional_tokens=[_stored(values)], max_deriv_order=(1,))
        assert np.array_equal(trajectory.get(('f', (1.0,))), _kept_1d(values, 2))


# ---------------------------------------------------------------------------
# T5: the guards
# ---------------------------------------------------------------------------

class TestTestFunctionSupport:

    def _domain(self, search, support, **budget):
        domain = Domain(_grids((120,)), global_var.grid_cache, 0, 7, boundary=10, **budget)
        g = np.zeros(120)
        g[support] = 1.0
        domain.g_func = g
        return domain

    def test_a_shifted_box_with_the_right_count_passes_only_unstrided(self):
        """The count alone lets a shifted box through; on a lattice the support is checked too."""
        search = _search()
        self._domain(search, slice(11, 111)).validateTestFunction()
        strided = _quiet(self._domain, search, slice(11, 111), evaluation_stride=2)
        with pytest.raises(ValueError, match='interior box'):
            strided.validateTestFunction()

    def test_a_wider_support_with_the_lattice_count_raises(self):
        search = _search()
        strided = _quiet(self._domain, search, slice(9, 110), evaluation_stride=2)
        assert int(strided.g_func_mask.sum()) == int(np.prod(strided.inner_shape))
        with pytest.raises(ValueError, match='interior box'):
            strided.validateTestFunction()

    def test_the_box_itself_passes(self):
        search = _search()
        _quiet(self._domain, search, slice(10, 110), evaluation_stride=2).validateTestFunction()


class TestRowFloor:

    W = 6                       # a block of 4 * (6 + 1) = 28 rows

    @pytest.mark.parametrize('metric', ['het', 'het_raw'])
    def test_het_needs_three_blocks(self, metric):
        assert survival.row_floor(metric, (83,), self.W) is not None
        assert survival.row_floor(metric, (84,), self.W) is None
        assert survival.row_floor(metric, (12, 7), self.W) is None

    @pytest.mark.parametrize('metric', ['tile', 'dyad'])
    def test_tile_and_dyad_need_two_slabs(self, metric):
        assert survival.row_floor(metric, (55,), self.W) is not None
        assert survival.row_floor(metric, (56,), self.W) is None

    @pytest.mark.parametrize('metric', ['chi2', 'chi2_centered'])
    def test_chi2_needs_three_levels_on_some_axis(self, metric):
        assert survival.row_floor(metric, (2, 2), self.W) is not None
        assert survival.row_floor(metric, (3, 1), self.W) is None

    @pytest.mark.parametrize('metric', ['vcoef', 'cv', 'survival', None])
    def test_other_metrics_have_no_floor(self, metric):
        assert survival.row_floor(metric, (2,), self.W) is None

    def test_the_reason_names_the_shortfall(self):
        assert '84' in survival.row_floor('het', (50,), self.W) and '50' in survival.row_floor('het', (50,), self.W)

    @pytest.mark.parametrize('metric', ['tile', 'dyad'])
    def test_the_slabs_are_cut_at_whole_axis0_levels(self, metric):
        """60 rows are two blocks of 28 in total, but 3 levels of 20 rows cut
        into slabs of 20 and 40: the estimators cut whole levels of axis 0."""
        reason = survival.row_floor(metric, (3, 20), self.W)
        assert reason is not None and '20 rows' in reason
        assert survival.row_floor(metric, (60,), self.W) is None

    def test_a_short_level_count_reachable_from_the_resolver(self):
        """Box (9, 50) under stride (1, 6) is a (9, 9) lattice; with a width
        of 9 (blocks of 40) its two slabs hold 36 and 45 rows."""
        stride = resolve_evaluation_stride((9, 50), (0, 0), stride=(1, 6))
        assert stride == (1, 6)
        lattice = tuple(_lattice_len(n, k) for n, k in zip((9, 50), stride))
        assert lattice == (9, 9)
        assert survival.row_floor('tile', lattice, 9) is not None
        assert survival.row_floor('dyad', lattice, 9) is not None

    def test_het_blocks_are_cut_at_whole_axis0_levels(self):
        """Lattice (4, 21): 84 rows, three blocks of 28 in total, but whole
        21-row levels hold only two blocks of 28 (two levels each)."""
        assert survival.row_floor('het', (4, 21), self.W) is not None
        assert survival.row_floor('het', (84,), self.W) is None

    @pytest.mark.parametrize('metric', ['tile', 'het', 'het_raw'])
    def test_a_coarse_axis0_is_counted_in_levels(self, metric):
        """(13, 3, 3): 117 rows are four blocks of 28 in total, and four slabs
        of nine-row levels are 27, 27, 27 and 36 rows. The estimators count
        their slabs in levels (four to a block) and cut three of 36, 36 and 45,
        so the grid passes."""
        assert survival.row_floor(metric, (13, 3, 3), self.W) is None

    @pytest.mark.parametrize('metric, start', [('tile', 1), ('dyad', 1), ('het', 2), ('het_raw', 2)])
    @pytest.mark.parametrize('cross', [(10,), (3, 3), (40,), (100,), (3, 30)])
    def test_the_floor_is_monotone_along_axis0(self, metric, start, cross):
        """From ``start`` axis-0 levels on, once a grid passes, every grid with
        more levels passes too. (13, 3, 3) used to fail at width 6 where
        (12, 3, 3) passed. ``het`` starts at two levels: one level is cut as
        flat rows (``test_one_level_is_the_flat_grid``) and can pass where two
        levels cannot (``test_one_two_and_three_levels``)."""
        for width in (3, 6, 9):
            passes = [survival.row_floor(metric, (n0,) + cross, width) is None
                      for n0 in range(start, 80)]
            first = passes.index(True)
            assert all(passes[first:]), (width, [n0 + start for n0, ok in enumerate(passes) if not ok])

    @pytest.mark.parametrize('metric', ['tile', 'het', 'het_raw', 'dyad'])
    @pytest.mark.parametrize('n0, before, after', [(1, (), ()), (1, (3,), ()), (1, (), (3,)),
                                                   (2, (), ()), (3, (), ()),
                                                   (5, (3,), ()), (5, (), (3,))])
    def test_the_floor_is_monotone_along_the_other_axes(self, metric, n0, before, after):
        """More rows per axis-0 level -- a longer axis other than the first --
        never turns a passing grid into a refused one, a single level
        included."""
        for width in (3, 6, 9):
            passes = [survival.row_floor(metric, (n0,) + before + (k,) + after, width) is None
                      for k in range(1, 150)]
            first = passes.index(True) if True in passes else len(passes)
            assert all(passes[first:]), (width, [k + 1 for k, ok in enumerate(passes) if not ok])

    @pytest.mark.parametrize('metric', ['tile', 'het', 'het_raw', 'dyad'])
    @pytest.mark.parametrize('cross', [(10,), (3, 3), (40,), (100,), (3, 30)])
    def test_one_level_is_the_flat_grid(self, metric, cross):
        """A single axis-0 level is cut as one row per level, like the 1-D grid
        of the same rows (``survival._axis0_layout``), and the floor gives it
        that grid's verdict. ``dyad`` cuts every axis of two or more levels,
        so for it a single axis-0 level is the grid of the other axes: on
        (1, 3, 30) it cuts the three levels too, which the flat 90 rows never
        see."""
        for width in (3, 6, 9):
            same = cross if metric == 'dyad' else (int(np.prod(cross)),)
            flat = survival.row_floor(metric, same, width) is None
            assert (survival.row_floor(metric, (1,) + cross, width) is None) == flat, width

    @pytest.mark.parametrize('metric, verdicts', [('tile', [True, True, True]),
                                                  ('dyad', [True, True, True]),
                                                  ('het', [True, False, True]),
                                                  ('het_raw', [True, False, True])])
    @pytest.mark.parametrize('cross', [(100,), (3, 30), (5, 20)])
    def test_one_two_and_three_levels(self, metric, verdicts, cross):
        """At width 6 (blocks of 28) one level of 90 or 100 rows is three flat
        blocks and passes every block floor. Two whole levels of that size make
        two blocks: enough for tile's and dyad's two slabs, not for het's
        three, which three levels give again. The het estimator itself scores
        one and three levels and raises on two (``TestLevelSlabCount`` in
        test_instability_estimators.py). The row-total floor passed two levels
        on their 180 or 200 rows and het raised mid-search; the floor now
        refuses them at fit start."""
        assert [survival.row_floor(metric, (n0,) + cross, self.W) is None
                for n0 in (1, 2, 3)] == verdicts

    @pytest.mark.parametrize('metric', ['tile', 'dyad'])
    def test_one_short_level_is_refused_where_two_levels_pass(self, metric):
        """One level of 40 rows is cut as two flat slabs of 20, below 28; two
        whole 40-row levels are two slabs of 40."""
        assert survival.row_floor(metric, (1, 40), self.W) is not None
        assert survival.row_floor(metric, (2, 40), self.W) is None

    def test_the_configured_width_is_the_smaller_cap(self):
        from types import SimpleNamespace
        def space(mode, **params):
            return SimpleNamespace(equation_terms_max_number=12, system_rps_mode=mode,
                                   system_rps_params=params)
        assert survival.configured_width(space('assignment', selected_terms_max_number=None)) == 12
        assert survival.configured_width(space('assignment', selected_terms_max_number=6)) == 6
        assert survival.configured_width(space('assignment')) == 12       # the mode's default
        assert survival.configured_width(space('plain')) == 12            # no cap in plain mode



# ---------------------------------------------------------------------------
# T6: one lattice for every consumer
# ---------------------------------------------------------------------------

class TestConsistency:

    SHAPE, BW, STRIDE = SHAPE_2D, BW_2D, STRIDE_2D

    def test_rows_weights_and_shape_agree(self):
        search = _search(evaluation_stride=self.STRIDE)
        trajectory = _quiet(_trajectory, search, self.SHAPE, self.BW)
        search.create_pool(data=[trajectory], max_deriv_order=(1, 1))
        manager = global_var.samples_manager
        assert manager.evaluation_strided
        shape = tuple(int(n) for n in manager.inner_shapes[0])
        rows = int(np.prod(shape))
        assert shape == (18, 7)
        assert manager.gFunc('dm')[0].shape == (rows,)
        assert manager.gFunc('dmf')[0].shape == (rows,)
        assert int(manager.gFunc('m')[0].sum()) == rows
        cached = global_var.tensor_cache.memory_default[0]
        assert cached and all(np.shape(tensor) == (rows,) for tensor in cached.values())

    def test_the_slab_partitions_take_the_grid_branch(self):
        """With rows == prod(inner_shape) the slabs are whole levels of axis 0."""
        search = _search(evaluation_stride=self.STRIDE)
        trajectory = _quiet(_trajectory, search, self.SHAPE, self.BW)
        shape = tuple(int(n) for n in trajectory.inner_shape)
        rows = int(trajectory._domain.g_func_mask.sum())
        assert survival._axis0_layout(rows, shape) == (shape[0], rows // shape[0])
        edges = survival._slab_edges(rows, shape, 4)
        assert edges[-1] == rows and all(edge % shape[1] == 0 for edge in edges)

    @pytest.mark.parametrize('metric', ['dyad', 'tile', 'het', 'chi2'])
    def test_an_estimator_on_the_cache_is_the_estimator_on_the_sliced_box(self, metric):
        whole, full = _build(self.SHAPE, self.BW)
        box = _box(whole._domain)
        _, kept = _build(self.SHAPE, self.BW, evaluation_stride=self.STRIDE)
        manager = global_var.samples_manager
        shape = manager.inner_shapes[0]
        weights = manager.gFunc('dmf')[0]
        columns, target = [('u', (1.0,)), ('du/dx1', (1.0,))], ('du/dx0', (1.0,))
        active = np.ones(len(columns) + 1, dtype=bool)

        got = instability_scores(metric, np.column_stack([kept[c] for c in columns]),
                                 kept[target], weights, shape, active, len(columns))

        def sliced(label):
            return full[label].reshape(box)[_lattice(self.STRIDE)]
        lattice_shape = sliced(target).shape
        want = instability_scores(metric, np.column_stack([sliced(c).ravel() for c in columns]),
                                  sliced(target).ravel(), np.ones(int(np.prod(lattice_shape))),
                                  lattice_shape, active, len(columns))
        assert tuple(int(n) for n in shape) == lattice_shape
        assert np.array_equal(got, want)


# ---------------------------------------------------------------------------
# T7: the vcoef microscale
# ---------------------------------------------------------------------------

@pytest.fixture
def vc_cache():
    saved = dict(global_var.vc_modes_cache)
    global_var.vc_modes_cache.clear()
    yield global_var.vc_modes_cache
    global_var.vc_modes_cache.clear()
    global_var.vc_modes_cache.update(saved)


def _modes(shape, lam, k_min=2, k_max=100):
    return tuple(min(max(k_min, min(int(np.ceil(n / l)) + 1, k_max)), n) for n, l in zip(shape, lam))


class TestVcoefMicroscale:

    #: A (10, 28) lattice: 28 rows per axis-0 level, one 4 * (6 + 1) block.
    SHAPE, BW, STRIDE = (24, 116), 2, (2, 4)

    def _strided(self):
        search = _search(evaluation_stride=self.STRIDE)
        grids = _grids(self.SHAPE)
        data = np.sin(2 * grids[0]) * np.cos(9 * grids[1]) + 0.3 * (grids[0] ** 2 + grids[1] ** 2)
        trajectory = _quiet(_trajectory, search, self.SHAPE, self.BW, data=data)
        search.create_pool(data=[trajectory], max_deriv_order=(1, 1))
        return trajectory

    def test_the_box_field_is_the_full_resolution_field(self, vc_cache):
        trajectory = self._strided()
        box = trajectory.box_field('u')
        assert box.shape == _box(trajectory._domain)
        assert np.array_equal(box[_lattice(self.STRIDE)].ravel(), trajectory.get(('u', (1.0,))))
        assert np.array_equal(global_var.samples_manager.box_field('u', 0), box)

    def test_the_modes_use_the_box_microscale_over_the_stride(self, vc_cache):
        trajectory = self._strided()
        shape = tuple(int(n) for n in trajectory.inner_shape)
        got = resolve_vc_modes_from_input(shape, main_var='u', sample_key=0, k_max=100)

        box, box_shape = trajectory.box_field('u'), _box(trajectory._domain)
        lam_box = [taylor_microscale(box, box_shape, d) for d in range(2)]
        assert got == _modes(shape, [lam / k for lam, k in zip(lam_box, self.STRIDE)])
        # the case discriminates: neither forgetting the stride nor
        # differentiating the lattice itself gives the same modes
        assert got != _modes(shape, lam_box)
        lattice = trajectory.get(('u', (1.0,)))
        assert got != _modes(shape, [taylor_microscale(lattice, shape, d) for d in range(2)])

    def test_the_unstrided_path_is_unchanged(self, vc_cache):
        search = _search()
        trajectory = _trajectory(search, (40, 30), 2)
        search.create_pool(data=[trajectory], max_deriv_order=(1, 1))
        shape = tuple(int(n) for n in trajectory.inner_shape)
        got = resolve_vc_modes_from_input(shape, main_var='u', sample_key=0, k_max=100)
        cached = trajectory.get(('u', (1.0,)))
        assert got == _modes(shape, [taylor_microscale(cached, shape, d) for d in range(2)])

    def test_the_default_cache_key_is_unchanged(self):
        from epde.operators.common.stability import _vc_cache_key
        assert _vc_cache_key((60,), 'u', None) == ((60,), 'u', 0, 'vc')
        assert _vc_cache_key((60,), 'u', 0, ((2,), (120,))) != _vc_cache_key((60,), 'u', 0)

    def test_a_lattice_and_a_box_of_the_same_shape_do_not_share_modes(self, vc_cache):
        """Every long record thins to about the same row count, so the lattice
        shape alone does not name the field: a box of 2 R points thinned to R
        and an unthinned box of R points must each get their own modes (R the
        fewest rows every metric's floor takes)."""
        rows = ROWS_1D
        fast_shape, slow_shape = (2 * rows + 2 * BW_1D,), (rows + 2 * BW_1D,)
        search = _search(evaluation_stride=2)
        fast = _quiet(_trajectory, search, fast_shape, BW_1D, data=np.sin(40 * _grids(fast_shape)))
        search.create_pool(data=[fast], max_deriv_order=(1,))
        assert fast._domain.evaluation_stride == (2,) and tuple(int(n) for n in fast.inner_shape) == (rows,)
        k_fast = resolve_vc_modes_from_input((rows,), main_var='u', sample_key=0, k_max=50)

        # a configured stride applies to every domain of its search, so the
        # unthinned box comes from a search without a budget. Constructing it
        # clears vc_modes_cache, so the lattice's entry is put back: without
        # that the box's modes would be computed on an empty cache and a key
        # that named only the shape would go unnoticed.
        lattice_entries = dict(vc_cache)
        assert lattice_entries
        search = _search()
        vc_cache.update(lattice_entries)
        grids = _grids(slow_shape)
        slow = search.createTrajectory({'u': np.sin(2 * grids)},
                                       _domain(search, slow_shape, BW_1D, domain_id=1), cache_id=0)[1]
        search.create_pool(data=[slow], max_deriv_order=(1,))
        assert slow._domain.evaluation_stride is None and tuple(int(n) for n in slow.inner_shape) == (rows,)
        k_slow = resolve_vc_modes_from_input((rows,), main_var='u', sample_key=0, k_max=50)
        cached = slow.get(('u', (1.0,)))
        assert k_slow == _modes((rows,), [taylor_microscale(cached, (rows,), 0)], k_max=50)
        assert k_slow != k_fast


# ---------------------------------------------------------------------------
# T8: the refusals
# ---------------------------------------------------------------------------

class TestConfigRefusals:

    def test_not_with_the_solver(self):
        budget = dict(evaluation_stride=2)
        with pytest.raises(ValueError, match='use_solver'):
            EpdeSearch(use_solver=True, verbose_params={'show_iter_idx': False}, **budget)
        with pytest.raises(ValueError, match='use_solver'):
            EpdeSearch(config={'domain': budget, 'solver': {'use_solver': True}},
                       verbose_params={'show_iter_idx': False})

    def test_not_with_the_solver_in_assignment_mode(self):
        with pytest.raises(ValueError):
            EpdeSearch(use_solver=True, system_rps_mode='assignment', evaluation_stride=2,
                       verbose_params={'show_iter_idx': False})
        with pytest.raises(ValueError, match='use_solver'):
            check_evaluation_budget(2, True)

    @pytest.mark.parametrize('stride', [0, -2, True, 2.5, '2', [], [2, None], [2, 0], (True, 2),
                                        np.array([2, 2])])
    def test_a_bad_stride(self, stride):
        with pytest.raises(ValueError, match='evaluation_stride'):
            check_evaluation_budget(stride, False)

    def test_bad_values_through_the_search(self):
        with pytest.raises(ValueError, match='evaluation_stride'):
            EpdeSearch(evaluation_stride=0, verbose_params={'show_iter_idx': False})

    def test_a_row_count_is_an_unknown_parameter(self):
        """There is no total-row-count option (it would mix time and space)."""
        with pytest.raises(ValueError, match='evaluation_max_rows'):
            EpdeSearch(evaluation_max_rows=100, verbose_params={'show_iter_idx': False})

    def test_good_values_pass(self):
        for stride in [None, 2, (2, 3), [2, np.int64(3)]]:
            check_evaluation_budget(stride, False)
        check_evaluation_budget(None, True)

    def test_a_configured_stride_too_coarse_for_the_grid_fails_at_create_domain(self):
        search = _search(evaluation_stride=50)                  # box 100: 2 levels
        with pytest.raises(ValueError, match='fewer than 3 levels'):
            _domain(search, (120,), 10)

    def test_create_domain_takes_no_budget_keyword(self):
        """The budget comes from the config only (H1)."""
        parameters = inspect.signature(EpdeSearch.createDomain).parameters
        assert 'evaluation_stride' not in parameters and 'evaluation_max_rows' not in parameters

    def test_a_json_list_stride_is_the_tuple_stride(self, monkeypatch):
        search = _search(config={'domain': {'evaluation_stride': list(STRIDE_2D)}})
        _no_search(search, monkeypatch)
        trajectory = _quiet(_trajectory, search, SHAPE_2D, BW_2D)
        assert trajectory._domain.evaluation_stride == STRIDE_2D == (2, 3)
        _quiet(search.fit, data=[trajectory], max_deriv_order=(1, 1))
        assert _registered() == [trajectory]


class TestSolverRefusals:

    def _strided(self):
        search = _search(evaluation_stride=2)
        trajectory = _quiet(_trajectory, search)
        search.create_pool(data=[trajectory], max_deriv_order=(1,))
        return trajectory

    def test_the_samples_manager(self):
        self._strided()
        with pytest.raises(ValueError, match='somewhere needs the whole grid'):
            global_var.samples_manager.require_unstrided('somewhere')
        with pytest.raises(ValueError, match='sub-lattice'):
            global_var.samples_manager.grids(mode='solver')

    def test_nothing_is_refused_without_a_budget(self):
        search = _search()
        search.create_pool(data=[_trajectory(search)], max_deriv_order=(1,))
        assert not global_var.samples_manager.evaluation_strided
        global_var.samples_manager.require_unstrided('somewhere')
        assert global_var.samples_manager.grids(mode='solver')[0][0].shape == (BOX_1D,)

    def test_solver_based_fitness(self):
        """Keyed on the operator, not on the config flag (H3)."""
        from epde.operators.common.fitness import SolverBasedFitness
        self._strided()
        with pytest.raises(ValueError, match='SolverBasedFitness'):
            SolverBasedFitness.apply(object.__new__(SolverBasedFitness), None, {})

    def test_the_post_fit_solver_interface(self):
        """solver_forms / predict go through use_grids (H4)."""
        from epde.integrate.interface import SystemSolverInterface
        self._strided()
        interface = object.__new__(SystemSolverInterface)
        interface.grids = None
        with pytest.raises(ValueError, match='use_grids'):
            interface.use_grids(domain_key=0)

    def test_default_boundary_conditions_from_the_cache(self):
        """predict hands use_grids explicit full grids, so its refusal sits where
        the default boundary conditions read the cached rows as a whole field."""
        from epde.integrate.bop import PregenBOperator
        self._strided()
        operator = object.__new__(PregenBOperator)
        operator.variables = ['u']
        with pytest.raises(ValueError, match='generate_default_bc'):
            operator.generate_default_bc(required_bc_ord={'u': [1]})


class TestInvalidData:
    """A non-trajectory in the data list is reported by the type check, as
    before the budget existed -- with or without one."""

    @pytest.mark.parametrize('budget', [{}, dict(evaluation_stride=2)])
    def test_a_mixed_list(self, budget):
        search = _search(**budget)
        trajectory = _quiet(_trajectory, search)
        with pytest.raises(AssertionError, match='Individual trajectories'):
            search.create_pool(data=[trajectory, np.zeros(40)], max_deriv_order=(1,))


class TestBudgetMismatch:
    """H1: a domain carries exactly the budget the search's config states."""

    def test_a_strided_domain_from_another_search(self, monkeypatch):
        plain = _search()
        _no_search(plain, monkeypatch)
        trajectory = _quiet(_trajectory, _search(evaluation_stride=2))
        with pytest.raises(ValueError, match=r'evaluated with evaluation_stride \(2,\)'):
            plain.fit(data=[trajectory], max_deriv_order=(1,))
        with pytest.raises(ValueError, match=r'evaluated with evaluation_stride \(2,\)'):
            plain.create_pool(data=[trajectory], max_deriv_order=(1,))

    def test_a_strided_direct_domain(self, monkeypatch):
        search = _search()
        _no_search(search, monkeypatch)
        domain = _quiet(Domain, _grids((120,)), global_var.grid_cache, 0, 0, boundary=10,
                        evaluation_stride=2)
        domain.g_func = EpdeSearch._get_g_func(None, 10)
        trajectory = search.createTrajectory({'u': _field(_grids((120,)))}, domain, cache_id=0)[1]
        with pytest.raises(ValueError, match='evaluated with evaluation_stride'):
            search.fit(data=[trajectory], max_deriv_order=(1,))

    def test_a_configured_budget_with_an_unstrided_direct_domain(self, monkeypatch):
        search = _search(evaluation_stride=2)
        _no_search(search, monkeypatch)
        domain = Domain(_grids((120,)), global_var.grid_cache, 0, 0, boundary=10)
        domain.g_func = EpdeSearch._get_g_func(None, 10)
        trajectory = search.createTrajectory({'u': _field(_grids((120,)))}, domain, cache_id=0)[1]
        with pytest.raises(ValueError, match=r'evaluation_stride None.*gives \(2,\)'):
            search.fit(data=[trajectory], max_deriv_order=(1,))
        with pytest.raises(ValueError, match='evaluation_stride None'):
            search.create_pool(data=[trajectory], max_deriv_order=(1,))

    def test_another_stride_from_another_search(self, monkeypatch):
        search = _search(evaluation_stride=4)
        _no_search(search, monkeypatch)
        trajectory = _quiet(_trajectory, _search(evaluation_stride=2))
        with pytest.raises(ValueError, match=r'gives \(4,\)'):
            search.fit(data=[trajectory], max_deriv_order=(1,))

    def test_a_passed_pool_is_checked_on_the_registered_samples(self, monkeypatch):
        """The second pass, right before the optimisation (H2)."""
        plain = _search()
        _no_search(plain, monkeypatch)
        strided = _search(evaluation_stride=2)
        _quiet(strided.create_pool, data=[_quiet(_trajectory, strided)], max_deriv_order=(1,))
        with pytest.raises(ValueError, match='evaluated with evaluation_stride'):
            plain.fit(pool=strided.pool, max_deriv_order=(1,))


class TestTokenRefusals:

    def test_external_derivatives(self):
        search = _search(evaluation_stride=2)
        family = ExternalDerivativesTokens(token_type='external', time_axis=0, base_token_label=['w'],
                                           max_orders=1, params_ranges={'power': (1, 1)},
                                           params_equality_ranges=None, dimensionality=0)
        grid = _grids(SHAPE_1D)
        trajectory = _quiet(_trajectory, search, cached_token_tensors={family: {'w': np.cos(grid)}})
        with pytest.raises(ValueError, match='ExternalDerivativesTokens.*row budget'):
            search.create_pool(data=[trajectory], max_deriv_order=(1,))
        with pytest.raises(ValueError, match='row budget'):
            trajectory.uploadTokenTensors(family, {'w': np.cos(grid)})

    def test_control_variables(self):
        """Constructing ControlVarTokens needs a control net; the refusal is an
        isinstance check that runs before anything reads the family."""
        search = _search(evaluation_stride=2)
        trajectory = _quiet(_trajectory, search)
        family = object.__new__(ControlVarTokens)
        with pytest.raises(ValueError, match='ControlVarTokens'):
            search._build_pool([trajectory], [family], (1,), 1, 1)


class TestRowFloorAtFitStart:
    """H6: a lattice too small for the configured instability metric fails loudly.

    These tests name their metric: the floor is per metric, and the case is
    one lattice that one metric refuses and another takes."""

    #: A box of 100 under stride 4: a 25-row lattice, below het's floor at the
    #: default width, above chi2's three levels.
    TINY, TINY_BW, TINY_STRIDE = (120,), 10, 4

    def test_the_tiny_lattice_is_refused_by_het_only_of_the_two(self):
        lattice = _resolved_lattice(self.TINY, self.TINY_BW, dict(evaluation_stride=self.TINY_STRIDE))
        assert lattice == (25,)
        assert survival.row_floor('het', lattice, WIDTH) is not None
        assert survival.row_floor('chi2', lattice, WIDTH) is None

    def test_het_on_a_tiny_lattice(self, monkeypatch):
        search = _search(instability_metric='het', evaluation_stride=self.TINY_STRIDE)
        _no_search(search, monkeypatch)
        trajectory = _quiet(_trajectory, search, self.TINY, self.TINY_BW)   # 25 rows
        with pytest.raises(ValueError, match='keeps too few rows'):
            search.fit(data=[trajectory], max_deriv_order=(1,))
        with pytest.raises(ValueError, match='keeps too few rows'):
            search.create_pool(data=[trajectory], max_deriv_order=(1,))

    def test_the_same_lattice_under_chi2_passes(self, monkeypatch):
        search = _search(instability_metric='chi2', evaluation_stride=self.TINY_STRIDE)
        _no_search(search, monkeypatch)
        trajectory = _quiet(_trajectory, search, self.TINY, self.TINY_BW)
        search.fit(data=[trajectory], max_deriv_order=(1,))
        assert _registered() == [trajectory]

    def test_fit_judges_the_floor_by_its_own_width(self, monkeypatch):
        """A narrow search (block of 16 rows, het needs 48) on a 100-row lattice
        passes, although a wider search was constructed after it."""
        narrow = _search(instability_metric='het', evaluation_stride=2, equation_terms_max_number=3)
        _no_search(narrow, monkeypatch)
        _search(equation_terms_max_number=20)                  # publishes a width of 20
        trajectory = _quiet(_trajectory, narrow, (220,), 10)   # box 200, 100 rows
        narrow.fit(data=[trajectory], max_deriv_order=(1,))
        assert _registered() == [trajectory]

    def test_create_pool_judges_the_floor_by_its_own_width(self):
        """create_pool publishes nothing, so the width must come from this
        search's config, not from the search constructed last (width 20 would
        need 252 rows)."""
        narrow = _search(instability_metric='het', evaluation_stride=2, equation_terms_max_number=3)
        _search(equation_terms_max_number=20)
        trajectory = _quiet(_trajectory, narrow, (220,), 10)
        narrow.create_pool(data=[trajectory], max_deriv_order=(1,))
        assert _registered() == [trajectory]

    def test_create_pool_refuses_by_its_own_width_too(self):
        """The reverse order: a narrow search constructed last must not let a
        wide search's too-short lattice through."""
        wide = _search(instability_metric='het', evaluation_stride=2, equation_terms_max_number=20)
        _search(equation_terms_max_number=3)
        trajectory = _quiet(_trajectory, wide, (220,), 10)   # 100 rows, width 20 needs 252
        with pytest.raises(ValueError, match='keeps too few rows'):
            wide.create_pool(data=[trajectory], max_deriv_order=(1,))

    def test_het_without_a_budget_is_not_checked(self, monkeypatch):
        """A 40-row box het's floor would refuse, unstrided: not checked."""
        assert survival.row_floor('het', (40,), WIDTH) is not None
        search = _search(instability_metric='het')
        _no_search(search, monkeypatch)
        trajectory = _trajectory(search, (60,), 10)
        search.fit(data=[trajectory], max_deriv_order=(1,))
        assert _registered() == [trajectory]
        assert trajectory._domain.evaluation_stride is None
        assert tuple(int(n) for n in trajectory.inner_shape) == (40,)


# ---------------------------------------------------------------------------
# T9: the warnings
# ---------------------------------------------------------------------------

def _stored(values):
    from epde import CacheStoredTokens
    return CacheStoredTokens(token_type='stored', token_labels=['f'], token_tensors={'f': values},
                             params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                             dimensionality=0, meaningful=True)


def _lattice_nyquist(stride, half_turn):
    """The Nyquist frequency the warning judges a family by on the default
    trajectory under a 1-D ``stride``: ``half_turn`` (pi for an angular
    frequency, 0.5 for a cyclic one) over the lattice step."""
    step = 1.0 / (SHAPE_1D[0] - 1)                     # _grids: linspace(0, 1, n)
    return half_turn / (stride * step)


class TestWarnings:

    def test_one_per_strided_domain(self):
        search = _search(evaluation_stride=2)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            _domain(search, (120,), 10, domain_id=0)
            _domain(search, (40, 30), 2, domain_id=1)
        usage = _usage_warnings(records)
        assert len(usage) == 2
        assert 'Domain 0' in str(usage[0].message) and '50 of 100 rows' in str(usage[0].message)
        assert 'Domain 1' in str(usage[1].message) and '234 of 936 rows' in str(usage[1].message)

    def test_none_by_default(self):
        search = _search()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            _domain(search, (120,), 10, domain_id=0)
            _domain(search, (40, 30), 2, domain_id=1)
        assert not _usage_warnings(records)

    @pytest.mark.parametrize('budget, warned', [(dict(evaluation_stride=2), True), ({}, False)])
    def test_a_token_constant_on_the_lattice(self, budget, warned):
        """Zero on every even index, one on every odd: constant only on the stride-2 lattice."""
        search = _search(**budget)
        trajectory = _quiet(_trajectory, search)
        family = _stored((np.arange(SHAPE_1D[0]) % 2).astype(float))       # the box starts even
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            search.create_pool(data=[trajectory], additional_tokens=[family], max_deriv_order=(1,))
        removed = [w for w in _usage_warnings(records) if 'Constant on the row-budget lattice' in str(w.message)]
        assert len(removed) == int(warned)
        assert ('f' in family._token_family.tokens) is not warned

    @pytest.mark.parametrize('budget, factor, warned', [(dict(evaluation_stride=2), 2.0, True),
                                                        (dict(evaluation_stride=2), 0.1, False),
                                                        ({}, 2.0, False)])
    def test_a_trig_family_above_the_lattice_nyquist(self, budget, factor, warned):
        """The family's highest angular frequency at ``factor`` times the
        stride-2 lattice's Nyquist frequency (pi over the lattice step)."""
        high = factor * _lattice_nyquist(2, np.pi)
        search = _search(**budget)
        trajectory = _quiet(_trajectory, search)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            search.create_pool(data=[trajectory], max_deriv_order=(1,),
                               additional_tokens=[TrigonometricTokens(freq=(1.0, high), dimensionality=0)])
        nyquist = [w for w in _usage_warnings(records) if 'Nyquist' in str(w.message)]
        assert len(nyquist) == int(warned)

    def test_frequencies_found_in_the_data_are_judged_not_their_band(self):
        """A band reaching twice the stride-2 Nyquist frequency keeps (0, that)
        as the family's bounds, but the only frequency installed is the data's
        tone near 20 -- far below the Nyquist frequency."""
        nyquist = _lattice_nyquist(2, np.pi)
        search = _search(evaluation_stride=2)
        grids = _grids(SHAPE_1D)
        trajectory = _quiet(_trajectory, search, data=np.sin(20 * grids))
        family = TrigonometricTokens(freq=(1.0, 2 * nyquist), freq_from_data=1, dimensionality=0)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            search.create_pool(data=[trajectory], max_deriv_order=(1,), additional_tokens=[family])
        assert max(family.token_family.param_grids['freq']) < nyquist
        assert not [w for w in _usage_warnings(records) if 'Nyquist' in str(w.message)]

    @pytest.mark.parametrize('factor, warned', [(2.0, True), (0.5, False)])
    def test_a_phased_sine_is_judged_by_its_cyclic_frequency(self, factor, warned):
        """sin(2 pi f t + ...) on the stride-2 lattice aliases from f = 0.5 over
        the lattice step; the family's highest frequency sits at ``factor``
        times that -- below the angular reading's Nyquist frequency (pi over
        the step) either way, so only the cyclic reading warns."""
        from epde.interface.prepared_tokens import PhasedSine1DTokens
        high = factor * _lattice_nyquist(2, 0.5)
        assert high < _lattice_nyquist(2, np.pi)
        search = _search(evaluation_stride=2)
        trajectory = _quiet(_trajectory, search)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            search.create_pool(data=[trajectory], max_deriv_order=(1,),
                               additional_tokens=[PhasedSine1DTokens(freq=(high - 10.0, high))])
        nyquist = [w for w in _usage_warnings(records) if 'Nyquist' in str(w.message)]
        assert len(nyquist) == int(warned)


# ---------------------------------------------------------------------------
# T10: the stride never changes; another budget is another pool
# ---------------------------------------------------------------------------

class TestImmutabilityAndPool:

    def test_the_stride_has_no_setter(self):
        assert Domain.evaluation_stride.fset is None
        domain = _quiet(_domain, _search(evaluation_stride=2), (120,), 10)
        with pytest.raises(AttributeError):
            domain.evaluation_stride = (4,)
        assert domain.evaluation_stride == (2,)

    def test_another_budget_rebuilds(self, monkeypatch):
        first = _search(evaluation_stride=2)
        _no_search(first, monkeypatch)
        old = _quiet(_trajectory, first)
        first.fit(data=[old], max_deriv_order=(1,))
        old_rows = old.get(('u', (1.0,))).size

        second = _search(evaluation_stride=4)
        builds = _no_search(second, monkeypatch)
        new = _quiet(_trajectory, second)
        second.fit(data=[new], max_deriv_order=(1,))
        assert len(builds) == 1
        assert _registered() == [new]
        assert (old_rows, new.get(('u', (1.0,))).size) == (_lattice_len(BOX_1D, 2), _lattice_len(BOX_1D, 4))
        assert second._pool_built_on([new]) and not second._pool_built_on([old])


# ---------------------------------------------------------------------------
# T11: every whole-grid reader is guarded
# ---------------------------------------------------------------------------

def _modules_with(needles):
    """Modules under ``epde/`` whose code (comments and reST prose skipped) holds a needle."""
    found = set()
    for path in EPDE_ROOT.rglob('*.py'):
        for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
            code = line.split('#')[0]
            if '``' in code:
                continue
            if any(needle in code for needle in needles):
                found.add(path)
    return found


class TestStaticScan:

    def test_every_mask_reader_is_guarded(self):
        readers = _modules_with(["gFunc('m')", 'gFunc("m")'])
        readers.discard(EPDE_ROOT / 'evaluators.py')          # the rows path itself
        assert len(readers) >= 3                               # the scan does find them
        unguarded = sorted(str(path.relative_to(EPDE_ROOT)) for path in readers
                           if 'require_unstrided' not in path.read_text(encoding='utf-8', errors='ignore'))
        assert not unguarded

    def test_solver_grid_readers_go_through_get_grids(self):
        """grids(mode='solver') readers (fitness, bop, pinn) reach Domain.getGrids,
        which refuses a lattice; nothing else crops the grid for the solver."""
        readers = _modules_with(["mode='solver'", 'mode="solver"'])
        assert readers
        search = _search(evaluation_stride=2)
        trajectory = _quiet(_trajectory, search)
        with pytest.raises(ValueError, match='sub-lattice'):
            trajectory.grids(mode='solver')


# ---------------------------------------------------------------------------
# T12: the fixtures clear every metric's row floor
# ---------------------------------------------------------------------------

class TestFixturesClearEveryFloor:
    """The contract of this module's fixtures: a lattice that reaches the
    fit-start check keeps enough rows for every shipped instability metric at
    the default width, so a test reaches the guard or the value it is about,
    not the row floor -- whichever metric is the default. (TestRowFloorAtFitStart
    refuses its tiny lattice on purpose, under a named metric.)"""

    @pytest.mark.parametrize('stride', range(1, COARSEST_1D + 1))
    def test_the_default_trajectory(self, stride):
        rows = _lattice_len(BOX_1D, stride)
        assert rows >= ROWS_1D and not _refused_by((rows,))

    @pytest.mark.parametrize('shape, bw, prep, budget', KEPT_CASES)
    def test_the_kept_value_cases(self, shape, bw, prep, budget):
        assert not _refused_by(_resolved_lattice(shape, bw, budget))

    @pytest.mark.parametrize('shape, bw, budget', [
        (SHAPE_2D, BW_2D, dict(evaluation_stride=STRIDE_2D)),
        (TestVcoefMicroscale.SHAPE, TestVcoefMicroscale.BW, dict(evaluation_stride=TestVcoefMicroscale.STRIDE)),
        ((2 * ROWS_1D + 2 * BW_1D,), BW_1D, dict(evaluation_stride=2)),
    ])
    def test_the_other_strided_searches(self, shape, bw, budget):
        assert not _refused_by(_resolved_lattice(shape, bw, budget))
