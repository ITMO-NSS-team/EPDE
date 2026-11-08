"""How a trigonometric frequency is defined: continuous, or on a grid of values.

A continuous frequency range is partitioned into tolerance buckets, so two
numerically identical columns -- ``dv*cos(2.0000000006 t)`` and
``dv*cos(1.9999999946 t)`` from the range 2 +- 1e-8 -- carry different labels,
and an equation can fit its target with itself. On a grid (``freq_num``,
``freq_step``, ``freq_values``, ``freq_base``, ``freq_from_data``) the frequency
takes only the grid's values: equal values are one factor (one label, one cached
tensor) and different values are different factors. Without a grid keyword
nothing changes.
"""
import warnings

import numpy as np
import pytest

import epde
from epde.globals import EPDEUsageWarning
from epde.interface.prepared_tokens import TrigonometricTokens, PhasedSine1DTokens
from epde.interface.token_family import TokenFamily
from epde.interface.token_frequencies import dominant_frequencies, frequency_grid
from epde.structure.factor import Factor, grid_index, snap_to_grid


def _family(freq=None, **kwargs):
    if freq is not None:
        kwargs['freq'] = freq
    return TrigonometricTokens(dimensionality=kwargs.pop('dimensionality', 0), **kwargs)._token_family


def _bare_factor(family, label='sin'):
    return Factor(label, status=family.status, family_type=family.ftype,
                  latex_constructor=family.latex_constructor, variable='u', all_vars=['u'])


def _drawn(family, label='sin'):
    f = _bare_factor(family, label)
    f.set_parameters(family.token_params, family.equality_ranges, random=True,
                     param_grids=family.param_grids)
    return f


def _given(family, freq, label='sin'):
    f = _bare_factor(family, label)
    f.set_parameters(family.token_params, family.equality_ranges, random=False,
                     param_grids=family.param_grids, power=1, freq=freq, dim=0)
    return f


def _freq(factor):
    return float(factor.params[1])


LINSPACE = dict(freq=(1.0, 3.0), freq_num=5)            # 1.0, 1.5, 2.0, 2.5, 3.0
GRID = tuple(np.linspace(1.0, 3.0, 5))


# ---------------------------------------------------------------------------
# the kinds
# ---------------------------------------------------------------------------

def test_without_a_grid_the_draw_is_the_continuous_one():
    """Bit-identity of the default: the same draws, no 'grid' key, the 5% tolerance."""
    freq = (0.999, 1.001)
    fam = _family(freq)
    assert fam.param_grids == {}
    assert fam.equality_ranges['freq'] == pytest.approx((freq[1] - freq[0]) * 0.05, rel=1e-12)
    np.random.seed(5)
    f = _drawn(fam)
    np.random.seed(5)
    np.random.randint(0, 1e9)                      # Factor.__init__ draws its hash first
    assert _freq(f) == np.random.uniform(*freq)
    assert all('grid' not in d for d in f.params_description.values())
    assert f.grid_snap(1.00037, 1) == 1.00037


@pytest.mark.parametrize('kwargs, grid', [
    (dict(freq=(1.0, 3.0), freq_num=5), tuple(np.linspace(1.0, 3.0, 5))),
    (dict(freq=(1.0, 3.0), freq_num=5, freq_spacing='linear'), tuple(np.linspace(1.0, 3.0, 5))),
    (dict(freq=(0.1, 10.0), freq_num=3, freq_spacing='log'), tuple(np.geomspace(0.1, 10.0, 3))),
    (dict(freq=(0.1, 0.3), freq_step=0.1), (0.1, 0.2, 0.3)),           # 0.1 + 2*0.1 = 0.30000000000000004
    (dict(freq=(1.0, 2.9), freq_step=0.5), (1.0, 1.5, 2.0, 2.5)),
    (dict(freq_values=[2, 1, 3.5, 2]), (1.0, 2.0, 3.5)),
    (dict(freq_base=0.5, freq_num=4), (0.5, 1.0, 1.5, 2.0)),
    (dict(freq=(2, 2), freq_num=1), (2.0,)),
], ids=['linspace', 'linspace-explicit', 'geomspace', 'step-noise', 'step-off-grid-hi', 'values', 'harmonics',
        'fixed'])
def test_each_kind_gives_its_grid(kwargs, grid):
    fam = _family(**kwargs)
    assert fam.param_grids['freq'] == grid
    assert all(type(v) is float for v in fam.param_grids['freq'])
    assert fam.token_params['freq'] == (grid[0], grid[-1])
    assert all(type(v) is float for v in fam.token_params['freq'])


@pytest.mark.parametrize('kwargs, match', [
    (dict(freq_spacing='linear'), 'do not define one kind'),
    (dict(freq_num=3, freq_step=0.1), 'do not define one kind'),
    (dict(freq_num=3, freq_values=[1.0]), 'do not define one kind'),
    (dict(freq_num=3, freq_from_data=2), 'do not define one kind'),
    (dict(freq_base=1.0), 'do not define one kind'),
    (dict(freq_values=[1.0], freq_from_data=2), 'do not define one kind'),
    (dict(freq_num=3, freq_spacing='cubic'), "'linear' or 'log'"),
    (dict(freq=(1.0, 2.0), freq_values=[1.0]), 'do not pass freq'),
    (dict(freq=[1.0, 2.0], freq_base=1.0, freq_num=2), 'do not pass freq'),
    (dict(freq=(0.0, 2.0), freq_num=3), 'must be positive'),
    (dict(freq=(-1.0, 2.0), freq_step=0.5), 'must be positive'),
    (dict(freq=(0.0, 2.0), freq_num=3, freq_spacing='log'), 'must be positive'),
    (dict(freq_values=[1.0, 0.0]), 'positive finite'),
    (dict(freq_values=[]), 'positive finite'),
    (dict(freq_base=0.0, freq_num=2), 'positive finite'),
    (dict(freq=(1.0, 2.0), freq_step=0.0), 'positive finite'),
    (dict(freq=(2.0, 1.0), freq_num=3), 'low <= high'),
    (dict(freq_from_data=0), 'positive integer'),
    (dict(freq=(5.0, 5.0), freq_from_data=2), '0 <= low < high'),
])
def test_conflicting_or_bad_keywords_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _family(**kwargs)


@pytest.mark.parametrize('bad', [0, -2, 2.5, True, float('inf'), '3'])
def test_freq_num_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match='freq_num must be a positive integer'):
        _family(freq=(1.0, 2.0), freq_num=bad)


@pytest.mark.parametrize('kwargs, match', [
    (dict(freq_values=[True]), 'list of positive finite numbers'),
    (dict(freq_values=2.0), 'list of positive finite numbers'),
    (dict(freq_values=['1.0']), 'list of positive finite numbers'),
    (dict(freq=(1.0, 2.0), freq_step='0.1'), 'positive finite number'),
    (dict(freq=('1', 2.0), freq_num=2), 'low <= high'),
    (dict(freq=(2.0, 2.0), freq_num=3), 'holds one frequency'),
    (dict(freq=(2.0, 2.0), freq_num=3, freq_spacing='log'), 'holds one frequency'),
])
def test_malformed_values_raise_value_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _family(**kwargs)


def test_freq_none_is_the_same_as_not_passing_it():
    """A JSON spec with "freq": null."""
    tokens = TrigonometricTokens(freq=None, dimensionality=0, freq_values=[1.0])
    assert tokens._token_family.param_grids == {'freq': (1.0,)}
    assert TrigonometricTokens(freq=None, dimensionality=0)._token_family.token_params['freq'] == \
        (np.pi / 2., 2 * np.pi)
    assert TrigonometricTokens(freq=None, dimensionality=0, freq_from_data=1)._freq_band is None


def test_the_kinds_table_is_in_the_docstrings_at_column_zero():
    for cls in (TrigonometricTokens, PhasedSine1DTokens):
        doc = cls.__init__.__doc__
        assert 'freq_values=[f1, f2, ...]' in doc and 'freq_from_data=n' in doc
        assert '\nThe frequency is defined by ONE of these keyword sets' in doc


def test_the_default_freq_is_recognised_by_identity():
    """The same numbers passed explicitly (or as a JSON list) count as passing freq."""
    _family(freq_values=[1.0])                                       # default freq: fine
    with pytest.raises(ValueError, match='do not pass freq'):
        _family(freq=(np.pi / 2., 2 * np.pi), freq_values=[1.0])
    kind, bounds, grid = frequency_grid([1, 3], True, freq_num=3)
    assert (kind, bounds, grid) == ('linspace', (1.0, 3.0), (1.0, 2.0, 3.0))


def test_draws_land_on_the_grid_and_reach_every_value():
    fam = _family(**LINSPACE)
    np.random.seed(0)
    assert {_freq(_drawn(fam)) for _ in range(300)} == set(GRID)


def test_a_grid_value_is_its_own_label():
    fam = _family(**LINSPACE)
    labels = []
    for value in GRID:
        f = _given(fam, value)
        assert f._quantized_params()[1] == value
        labels.append(f.structural_label)
    assert len(set(labels)) == len(GRID)


def test_an_off_grid_value_cannot_be_labelled():
    """A writer that forgot to snap must fail, not share a grid point's tensor."""
    fam = _family(**LINSPACE)
    f = _given(fam, 2.0)
    f.params = np.array([1.0, 2.1, 0.0])
    with pytest.raises(ValueError, match='not one of its grid values'):
        f.structural_label


def test_equal_grid_values_are_one_factor_and_different_ones_are_not():
    fam = _family(**LINSPACE)
    a, b = _given(fam, 2.0), _given(fam, 2.0)
    assert a == b
    assert a.structural_label == b.structural_label
    assert a.cache_label == b.cache_label
    c = _given(fam, 2.5)
    assert not (a == c)
    assert a.structural_label != c.structural_label


def test_a_fine_linspace_keeps_every_value_apart():
    lo, hi, num = 0.999, 1.005, 61
    fam = _family(freq=(lo, hi), freq_num=num)
    grid = np.linspace(lo, hi, num)
    assert len({_given(fam, float(v)).structural_label for v in grid}) == num
    assert fam.equality_ranges['freq'] == pytest.approx(float(np.min(np.diff(grid))) / 2)


def test_a_single_value_grid_fixes_the_frequency():
    fam = _family(freq_values=[2.0])
    np.random.seed(1)
    factors = [_drawn(fam) for _ in range(20)]
    assert {_freq(f) for f in factors} == {2.0}
    assert len({f.structural_label for f in factors}) == 1


def test_a_self_fit_pair_is_one_structure_on_a_grid():
    """The motivating case: the continuous pin 2 +- 1e-8 labels two draws apart,
    the single-value grid cannot draw two different frequencies at all."""
    continuous = _family((2 - 1e-8, 2 + 1e-8))
    assert (_given(continuous, 2.0000000006).structural_label
            != _given(continuous, 1.9999999946).structural_label)
    gridded = _family(freq_values=[2.0])
    np.random.seed(3)
    assert _drawn(gridded).structural_label == _drawn(gridded).structural_label


def test_an_explicit_off_grid_value_is_snapped_with_a_warning():
    """A gridded parameter is labelled by its value, which must be a grid value,
    so an explicitly given 2.1 is moved to 2.0 (and says so)."""
    fam = _family(**LINSPACE)
    with pytest.warns(EPDEUsageWarning, match='not one of its grid'):
        f = _given(fam, 2.1)
    assert _freq(f) == 2.0
    assert f == _given(fam, 2.0)
    assert f.cache_label == _given(fam, 2.0).cache_label
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _freq(_given(fam, 2.5)) == 2.5                       # on the grid: no warning


def test_snap_moves_to_the_nearest_grid_value():
    assert snap_to_grid(2.2, GRID) == 2.0
    assert snap_to_grid(2.3, GRID) == 2.5
    assert snap_to_grid(-7.0, GRID) == 1.0
    assert snap_to_grid(9.0, GRID) == 3.0
    assert grid_index(2.25, GRID) == 2                              # a tie goes to the lower value


def test_the_phased_sine_takes_every_kind_and_keeps_its_phase():
    for kwargs in (dict(freq=(1.0, 3.0), freq_num=5), dict(freq_values=[0.5, 1.5]),
                   dict(freq_base=0.25, freq_num=3), dict(freq=(0.1, 0.4), freq_step=0.1)):
        fam = PhasedSine1DTokens(**kwargs)._token_family
        assert 'freq' in fam.param_grids
        assert fam.token_params['phase'] == (0., 1.)
        assert fam.equality_ranges['phase'] == 0.05
    continuous = PhasedSine1DTokens()._token_family
    assert continuous.param_grids == {}
    assert continuous.token_params['freq'] == (np.pi / 2., 2 * np.pi)
    with pytest.raises(ValueError, match='positive finite'):
        PhasedSine1DTokens(freq_values=[0.0, 1.0])


# ---------------------------------------------------------------------------
# operators keep a gridded value on its grid
# ---------------------------------------------------------------------------

class _StubTerm:
    def __init__(self, factors):
        self.structure = factors

    @property
    def factors_labels(self):
        return frozenset(f.structural_label for f in self.structure)

    def resetSavedState(self):
        pass

    def reset_occupied_tokens(self):
        pass


class _StubEquation:
    target_idx = 99

    def __init__(self, terms):
        self.structure = terms

    def _invalidate_label_cache(self):
        pass

    def reset_for_structure_change(self):
        pass


@pytest.mark.parametrize('module', ['multiobjective', 'singleobjective'])
def test_parameter_crossover_keeps_frequencies_on_the_grid(module):
    import importlib
    variation = importlib.import_module(f'epde.operators.{module}.variation')
    fam = _family(**LINSPACE)
    op = variation.TermParamCrossover(['term_param_proportion'])
    op.params = {'term_param_proportion': 0.4}
    t1, t2 = _StubTerm([_given(fam, 1.0)]), _StubTerm([_given(fam, 3.0)])
    op.apply((t1, t2), arguments={})
    assert _freq(t1.structure[0]) in GRID and _freq(t2.structure[0]) in GRID
    assert _freq(t1.structure[0]) == 2.0          # 1.0 + 0.4 * 2.0 = 1.8 -> 2.0


@pytest.mark.parametrize('module', ['multiobjective', 'singleobjective'])
def test_parameter_crossover_without_a_grid_is_the_plain_blend(module):
    import importlib
    variation = importlib.import_module(f'epde.operators.{module}.variation')
    fam = _family((1.0, 3.0))
    op = variation.TermParamCrossover(['term_param_proportion'])
    op.params = {'term_param_proportion': 0.4}
    t1, t2 = _StubTerm([_given(fam, 1.0)]), _StubTerm([_given(fam, 3.0)])
    op.apply((t1, t2), arguments={})
    assert _freq(t1.structure[0]) == 1.0 + 0.4 * (3.0 - 1.0)


@pytest.mark.parametrize('module', ['multiobjective', 'singleobjective'])
def test_parameter_mutation_keeps_frequencies_on_the_grid(module):
    import importlib
    TermParameterMutation = importlib.import_module(f'epde.operators.{module}.mutations').TermParameterMutation
    fam = _family(**LINSPACE)
    op = TermParameterMutation(['r_param_mutation', 'multiplier', 'strict_restrictions'])
    op.params = {'r_param_mutation': 1.0, 'multiplier': 0.3, 'strict_restrictions': True}
    np.random.seed(11)
    seen = set()
    for _ in range(40):
        equation = _StubEquation([_StubTerm([_given(fam, 2.0)])])
        term = op.apply((0, equation), arguments={})
        seen.add(_freq(term.structure[0]))
    assert seen <= set(GRID)
    assert len(seen) > 1                                            # it did move


# ---------------------------------------------------------------------------
# the family: validation, install_param_grid, the deferred guard
# ---------------------------------------------------------------------------

def test_power_and_dim_cannot_take_a_grid():
    fam = TokenFamily(token_type='trigonometric')
    for name in ('power', 'dim'):
        with pytest.raises(ValueError, match='cannot take a grid'):
            fam.set_params(['sin'], {'power': (1, 3), 'freq': (1.0, 2.0), 'dim': (0, 1)},
                           {'power': 0, 'freq': 0.1, 'dim': 0}, param_grids={name: [1, 2]})


def test_a_family_grid_is_validated():
    fam = TokenFamily(token_type='trigonometric')
    params, equal = {'power': (1, 1), 'freq': (1.0, 2.0)}, {'power': 0, 'freq': 0.1}
    with pytest.raises(ValueError, match='does not have'):
        fam.set_params(['sin'], params, equal, param_grids={'phase': [1.0]})
    with pytest.raises(ValueError, match='leaves its range'):
        fam.set_params(['sin'], params, equal, param_grids={'freq': [1.0, 2.5]})
    with pytest.raises(ValueError, match='finite'):
        fam.set_params(['sin'], params, equal, param_grids={'freq': [1.0, np.nan]})


def test_a_gridded_parameter_compares_at_half_the_smallest_gap():
    fam = TokenFamily(token_type='trigonometric')
    given = {'power': 0, 'freq': 0.0}
    fam.set_params(['sin'], {'power': (1, 1), 'freq': (1.0, 2.0)}, given,
                   param_grids={'freq': [1.0, 1.2, 2.0]})
    assert fam.equality_ranges['freq'] == pytest.approx(0.1)
    assert given['freq'] == 0.0                                     # the caller's dict is not touched
    assert _family(**LINSPACE).equality_ranges['freq'] == 0.25
    assert _family(freq_values=[2.0]).equality_ranges['freq'] == 0.0


def test_install_param_grid_builds_new_dicts_and_draws_nothing():
    fam = _family(freq_from_data=2)
    old_equality, old_params = fam.equality_ranges, fam.token_params
    state = np.random.get_state()[1].copy()
    grid = fam.install_param_grid('freq', [3.0, 1.0], bounds=None)
    assert grid == (1.0, 3.0)
    assert np.array_equal(np.random.get_state()[1], state)
    assert fam.equality_ranges is not old_equality and old_equality['freq'] == 0.0
    assert fam.token_params is not old_params and fam.token_params['freq'] == (1.0, 3.0)
    assert list(fam.token_params) == ['power', 'freq', 'dim']       # order = factor parameter order
    assert fam.deferred_params == set()


def test_a_deferred_family_draws_nothing_before_its_grid_is_installed():
    tokens = TrigonometricTokens(dimensionality=0, freq_from_data=2)
    with pytest.raises(RuntimeError, match='come from the fit data'):
        tokens._token_family.create(label='sin', all_vars=['u'])
    tokens._token_family.install_param_grid('freq', [1.5, 3.0])
    _, factor = tokens._token_family.create(label='sin', all_vars=['u'])
    assert _freq(factor) in (1.5, 3.0)


def test_the_family_draws_through_its_grid():
    family = _family(**LINSPACE)
    np.random.seed(6)
    drawn = set()
    for _ in range(100):
        _, factor = family.create(label='cos', all_vars=['u'])
        assert factor.params_description[1]['grid'] == GRID
        drawn.add(_freq(factor))
    assert drawn == set(GRID)


def test_config_specs_build_every_kind():
    from epde.interface.search_config import build_tokens
    built = build_tokens([
        {'family': 'trigonometric', 'freq': [1, 3], 'freq_num': 5, 'dimensionality': 0},
        {'family': 'trigonometric', 'freq_values': [1, 2, 3.5], 'dimensionality': 0},
        {'family': 'trigonometric', 'freq_from_data': 2, 'freq': [1, 5], 'dimensionality': 0},
        {'family': 'phased_sine_1d', 'freq_values': [0.5, 1.5]},
    ])
    assert built[0]._token_family.param_grids == {'freq': GRID}
    assert built[1]._token_family.param_grids == {'freq': (1.0, 2.0, 3.5)}
    assert built[2]._freq_from_data == 2 and built[2]._freq_band == (1.0, 5.0)
    assert built[3]._token_family.param_grids == {'freq': (0.5, 1.5)}


# ---------------------------------------------------------------------------
# frequencies from the data
# ---------------------------------------------------------------------------

class _Traj:
    """What dominant_frequencies reads off a Trajectory."""

    def __init__(self, grids, fields):
        self._grids, self._fields = list(grids), fields
        self.variable_names = list(fields)

    def grids(self, mode='full'):
        return self._grids

    def raw_field(self, name):
        return self._fields[name]


T = np.linspace(0, 20, 2001)
BIN = 2 * np.pi / (T.size * (T[1] - T[0]))                          # angular bin width


def test_two_tones_are_found_to_a_fraction_of_a_bin():
    traj = _Traj([T], {'u': np.sin(3 * T) + 0.5 * np.sin(7 * T)})
    angular = dominant_frequencies([traj], axes=(0,), n=2, unit='angular')
    assert np.allclose(angular, (3.0, 7.0), atol=1e-3 * BIN)
    cyclic = dominant_frequencies([traj], axes=(0,), n=2, unit='cyclic')
    assert np.allclose(cyclic, (3.0 / (2 * np.pi), 7.0 / (2 * np.pi)), atol=1e-3 * BIN / (2 * np.pi))


def test_a_meshgrid_gives_each_axis_its_tone():
    x = np.linspace(0, 12, 601)
    tt, xx = np.meshgrid(T[:801], x, indexing='ij')
    traj = _Traj([tt, xx], {'u': np.sin(3 * tt) * np.sin(5 * xx)})
    found = dominant_frequencies([traj], axes=(0, 1), n=2, unit='angular')
    assert np.allclose(found, (3.0, 5.0), atol=1e-2)


def test_variables_and_trajectories_of_different_lengths_pool():
    short = T[:1201]
    a = _Traj([T], {'u': np.sin(3 * T), 'v': np.sin(7 * T)})
    b = _Traj([short], {'u': np.sin(3 * short) + np.sin(7 * short)})
    found = dominant_frequencies([a, b], axes=(0,), n=2, unit='angular')
    assert np.allclose(found, (3.0, 7.0), atol=1e-3)


def test_a_band_restricts_the_peaks_and_a_narrow_one_warns():
    traj = _Traj([T], {'u': np.sin(3 * T) + 0.5 * np.sin(7 * T)})
    assert np.allclose(dominant_frequencies([traj], axes=(0,), n=1, unit='angular', band=(5, 10)), (7.0,),
                       atol=1e-3)
    with pytest.warns(EPDEUsageWarning, match='only 1'):
        found = dominant_frequencies([traj], axes=(0,), n=3, unit='angular',
                                     band=(7 - 1.5 * BIN, 7 + 1.5 * BIN))
    assert np.allclose(found, (7.0,), atol=1e-3)
    with pytest.raises(ValueError, match='no spectral peak inside the band'):
        dominant_frequencies([traj], axes=(0,), n=1, unit='angular', band=(100.0, 200.0))


def test_a_clean_tone_gives_one_peak_and_says_so():
    traj = _Traj([T], {'u': np.sin(3 * T)})
    with pytest.warns(EPDEUsageWarning, match='only 1'):
        assert np.allclose(dominant_frequencies([traj], axes=(0,), n=3, unit='angular'), (3.0,), atol=1e-3)


def test_a_record_shorter_than_two_cycles_has_no_peak():
    short = np.linspace(0, 1, 200)
    with pytest.raises(ValueError, match='too short'):
        dominant_frequencies([_Traj([short], {'u': np.sin(3 * short)})], axes=(0,), n=1, unit='angular')


def test_trig_from_data_reads_every_axis_up_to_its_dimensionality():
    # Each axis keeps the peaks of its own spectrum: 5 is not a frequency of t.
    x = np.linspace(0, 12, 601)
    tt, xx = np.meshgrid(T[:801], x, indexing='ij')
    traj = _Traj([tt, xx], {'u': np.sin(3 * tt) * np.sin(5 * xx)})
    tokens = TrigonometricTokens(dimensionality=1, freq_from_data=2)
    with pytest.warns(EPDEUsageWarning, match='only 1'):
        grid = tokens.resolve_from_data([traj])
    assert sorted(grid) == [0, 1]
    assert np.allclose(grid[0], (3.0,), atol=1e-2) and np.allclose(grid[1], (5.0,), atol=1e-2)
    one_axis = TrigonometricTokens(dimensionality=0, freq_from_data=2)
    with pytest.warns(EPDEUsageWarning, match='only 1'):
        assert np.allclose(one_axis.resolve_from_data([traj]), (3.0,), atol=1e-2)


def test_per_axis_peaks_stay_on_their_axis():
    x = np.linspace(0, 12, 601)
    tt, xx = np.meshgrid(T[:801], x, indexing='ij')
    traj = _Traj([tt, xx], {'u': np.sin(3 * tt) * np.sin(5 * xx) + 0.5 * np.sin(7 * tt)})
    with pytest.warns(EPDEUsageWarning, match='along axis 1; the data has only 1'):
        found = dominant_frequencies([traj], axes=(0, 1), n=2, unit='angular', per_axis=True)
    assert np.allclose(found[0], (3.0, 7.0), atol=1e-2) and np.allclose(found[1], (5.0,), atol=1e-2)
    # Pooled, as before, when asked for.
    assert np.allclose(dominant_frequencies([traj], axes=(0, 1), n=2, unit='angular'), (3.0, 5.0), atol=1e-2)


def test_an_axis_without_a_peak_gets_no_frequency():
    x = np.linspace(0, 1, 3)                                     # too few points for a spectrum
    tt, xx = np.meshgrid(T[:801], x, indexing='ij')
    traj = _Traj([tt, xx], {'u': np.sin(3 * tt) * (1 + xx)})
    with pytest.warns(EPDEUsageWarning, match='no spectral peak along axis 1'):
        found = dominant_frequencies([traj], axes=(0, 1), n=1, unit='angular', per_axis=True)
    assert list(found) == [0] and np.allclose(found[0], (3.0,), atol=1e-2)


PER_AXIS = {0: (3.0,), 1: (5.0, 6.0)}


def _per_axis_family():
    return _family(freq_from_data=2, dimensionality=1)


def test_a_factor_draws_its_axis_then_one_of_its_frequencies():
    fam = _per_axis_family()
    assert fam.install_param_grid('freq', PER_AXIS) == PER_AXIS
    assert fam.token_params['freq'] == (3.0, 6.0) and fam.equality_ranges['freq'] == 0.5
    np.random.seed(3)
    seen = set()
    for _ in range(200):
        _, factor = fam.create(label='sin', all_vars=['u'])
        axis = int(factor.params[2])
        assert _freq(factor) in PER_AXIS[axis]
        assert factor.params_description[1]['grid'] == PER_AXIS[axis]
        seen.add((axis, _freq(factor)))
    assert seen == {(0, 3.0), (1, 5.0), (1, 6.0)}


def test_only_the_axes_with_a_grid_are_drawn():
    fam = _per_axis_family()
    fam.install_param_grid('freq', {1: (5.0,)})
    np.random.seed(4)
    assert {int(fam.create(label='cos', all_vars=['u'])[1].params[2]) for _ in range(50)} == {1}


def test_a_given_frequency_snaps_to_the_grid_of_its_axis():
    fam = _per_axis_family()
    fam.install_param_grid('freq', PER_AXIS)
    _, factor = fam.create(label='sin', all_vars=['u'], power=1, freq=6.0, dim=1)
    assert _freq(factor) == 6.0
    with pytest.warns(EPDEUsageWarning, match='not one of its grid'):
        _, factor = fam.create(label='sin', all_vars=['u'], power=1, freq=3.0, dim=1)
    assert _freq(factor) == 5.0
    fam.install_param_grid('freq', {1: (5.0,)})
    with pytest.raises(ValueError, match='no freq along axis 0'):
        fam.create(label='sin', all_vars=['u'], power=1, freq=3.0, dim=0)


def test_a_per_axis_grid_is_validated():
    fam = _per_axis_family()
    with pytest.raises(ValueError, match='outside dim'):
        fam.install_param_grid('freq', {2: (1.0,)})
    with pytest.raises(ValueError, match='empty on some axis'):
        fam.install_param_grid('freq', {0: ()})


def test_bad_data_is_refused():
    with pytest.raises(ValueError, match='uniform spacing'):
        bent = T.copy()
        bent[500] += 1e-3
        dominant_frequencies([_Traj([bent], {'u': np.sin(3 * T)})], axes=(0,), n=1, unit='angular')
    with pytest.raises(ValueError, match='non-finite'):
        field = np.sin(3 * T)
        field[10] = np.nan
        dominant_frequencies([_Traj([T], {'u': field})], axes=(0,), n=1, unit='angular')
    with pytest.raises(ValueError, match='has 1 axis'):
        dominant_frequencies([_Traj([T], {'u': np.sin(3 * T)})], axes=(0, 1), n=1, unit='angular')
    with pytest.raises(ValueError, match='constant'):
        dominant_frequencies([_Traj([T], {'u': np.ones_like(T)})], axes=(0,), n=1, unit='angular')


def _merge_peaks_quadratic(peaks):
    """The pairwise merge ``_merge_peaks`` replaced, kept as its oracle."""
    peaks = sorted(peaks, key=lambda p: (-p[0], p[1]))
    merged, clusters = [False] * len(peaks), []
    for i, (energy, freq, width) in enumerate(peaks):
        if merged[i]:
            continue
        merged[i] = True
        members = [peaks[i]]
        for j in range(i + 1, len(peaks)):
            if not merged[j] and abs(peaks[j][1] - freq) < max(width, peaks[j][2]):
                merged[j] = True
                members.append(peaks[j])
        finest = min(members, key=lambda p: (p[2], -p[0]))
        clusters.append((sum(p[0] for p in members), finest[1]))
    return clusters


def _peak_set(seed):
    """Peaks from a few spectra of different resolutions, with exact
    boundary distances, energy ties, repeated frequencies and dense clumps."""
    rng = np.random.default_rng(seed)
    peaks = []
    for width in rng.choice([0.01, 0.0125, 0.05, 0.1, 0.4], size=rng.integers(1, 5), replace=False):
        width = float(width)
        k = rng.integers(2, 400, size=rng.integers(5, 120))
        freqs = (k + rng.uniform(-0.5, 0.5, size=k.size)) * width
        freqs[::7] = np.round(freqs[::7] / width) * width            # on-bin, so exact width gaps occur
        energy = rng.choice([1e-3, 2e-3, 5e-3], size=k.size) if seed % 2 else rng.exponential(1e-3, size=k.size)
        peaks += [(float(e), float(f), width) for e, f in zip(energy, freqs)]
    clump = float(rng.uniform(1, 3))
    peaks += [(float(e), clump + float(d), 0.05) for e, d in zip(rng.exponential(1e-3, 30), rng.normal(0, 0.03, 30))]
    peaks += peaks[:3]                                                # bitwise-equal duplicates
    return peaks


@pytest.mark.parametrize('seed', range(40))
def test_the_peak_merge_is_the_pairwise_merge(seed):
    from epde.interface.token_frequencies import _merge_peaks
    peaks = _peak_set(seed)
    assert _merge_peaks(peaks) == _merge_peaks_quadratic(peaks)


def test_the_peak_merge_scales_to_a_long_record():
    """A lorenz-sized pool (~20k peaks of one width) merges in well under a
    second; the pairwise merge took minutes."""
    import time
    from epde.interface.token_frequencies import _merge_peaks
    rng = np.random.default_rng(0)
    width = 1e-3
    k = np.sort(rng.choice(np.arange(2, 120_000), size=20_000, replace=False))
    peaks = [(float(e), float(f), width) for e, f in zip(rng.exponential(1e-5, k.size),
                                                         (k + rng.uniform(-0.4, 0.4, k.size)) * width)]
    start = time.perf_counter()
    clusters = _merge_peaks(peaks)
    assert time.perf_counter() - start < 1.0
    assert sum(e for e, _ in clusters) == pytest.approx(sum(p[0] for p in peaks), rel=1e-12)


# --- through EpdeSearch -----------------------------------------------------

def _search():
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _trajectory(search, cache_id, omega):
    grid = np.linspace(0, 40, 1600)
    domain = search.createDomain(grid, boundary_width=10, ID=cache_id)[1]
    return search.createTrajectory({'u': np.sin(omega * grid)}, domain, cache_id=cache_id)[1]


def test_create_pool_resolves_the_frequencies_from_the_data():
    search = _search()
    tokens = TrigonometricTokens(dimensionality=0, freq_from_data=1)
    search.create_pool(data=[_trajectory(search, 0, 3.0)], additional_tokens=[tokens],
                       max_deriv_order=(1,))
    grid = tokens._token_family.param_grids['freq']
    assert len(grid) == 1 and grid[0] == pytest.approx(3.0, abs=1e-3)
    assert tokens._token_family.deferred_params == set()


def test_a_failed_resolution_leaves_the_pool_as_it_was():
    search = _search()
    trajectory = _trajectory(search, 0, 3.0)
    search.create_pool(data=[trajectory], additional_tokens=[_family_tokens()], max_deriv_order=(1,))
    pool, key = search.pool, search.pool_params
    good = TrigonometricTokens(dimensionality=0, freq_from_data=1)
    unreachable = PhasedSine1DTokens(freq=(100.0, 200.0), freq_from_data=1)
    with pytest.raises(ValueError, match='no spectral peak'):
        search.create_pool(data=[trajectory], additional_tokens=[good, unreachable], max_deriv_order=(1,))
    assert search.pool is pool and search.pool_params is key          # identity: not even reassigned
    assert good._token_family.deferred_params == {'freq'}              # nothing installed on the good one


def test_the_phased_sine_finds_cyclic_frequencies_on_its_first_axis():
    search = _search()
    tokens = PhasedSine1DTokens(freq_from_data=1)
    search.create_pool(data=[_trajectory(search, 0, 3.0)], additional_tokens=[tokens], max_deriv_order=(1,))
    grid = tokens._token_family.param_grids['freq']
    assert len(grid) == 1 and grid[0] == pytest.approx(3.0 / (2 * np.pi), abs=1e-3)


def test_a_config_declared_family_is_resolved_through_fit(monkeypatch):
    """Resolved through fit, the config family sits in the pool once, resolved."""
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False},
                             tokens=[{'family': 'trigonometric', 'freq_from_data': 1, 'dimensionality': 0}])
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    monkeypatch.setattr(search, '_run_optimization', lambda *args, **kwargs: None)
    search.fit(data=[_trajectory(search, 0, 3.0)], max_deriv_order=(1,))
    trig = [family for family in search.pool.families if family.ftype == 'trigonometric']
    assert len(trig) == 1
    for family in trig:
        assert family.deferred_params == set()
        assert family.param_grids['freq'][0] == pytest.approx(3.0, abs=1e-3)


def _family_tokens():
    return TrigonometricTokens(freq=(1.0, 2.0), dimensionality=0)


def test_a_bare_deferred_family_is_refused_at_pool_build():
    search = _search()
    family = TrigonometricTokens(freq=(1.0, 2.0), dimensionality=0)._token_family
    family.deferred_params = {'freq'}
    with pytest.raises(ValueError, match='still wait for values from the data'):
        search.create_pool(data=[_trajectory(search, 0, 3.0)], additional_tokens=[family],
                           max_deriv_order=(1,))


def _count_pool_builds(search, monkeypatch):
    calls = []
    original = search._build_pool

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(search, '_build_pool', counted)
    monkeypatch.setattr(search, '_run_optimization', lambda *args, **kwargs: None)
    return calls


def test_every_fit_resolves_its_own_data(monkeypatch):
    search = _search()
    calls = _count_pool_builds(search, monkeypatch)
    tokens = TrigonometricTokens(dimensionality=0, freq_from_data=1)
    search.fit(data=[_trajectory(search, 0, 3.0)], additional_tokens=[tokens], max_deriv_order=(1,))
    assert tokens._token_family.param_grids['freq'][0] == pytest.approx(3.0, abs=1e-3)
    search.fit(data=[_trajectory(search, 1, 5.0)], additional_tokens=[tokens], max_deriv_order=(1,))
    assert tokens._token_family.param_grids['freq'][0] == pytest.approx(5.0, abs=1e-3)
    assert len(calls) == 2


def test_a_continuous_family_does_not_force_a_rebuild(monkeypatch):
    search = _search()
    calls = _count_pool_builds(search, monkeypatch)
    trajectory = _trajectory(search, 0, 3.0)
    tokens = _family_tokens()
    search.fit(data=[trajectory], additional_tokens=[tokens], max_deriv_order=(1,))
    search.fit(data=[trajectory], additional_tokens=[tokens], max_deriv_order=(1,))
    assert len(calls) == 1
