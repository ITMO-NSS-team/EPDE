"""``search_space.system_rps_params['target_axes']`` of plain mode: the grid axes along which a variable's
plain target derivative may be taken.

Pinned here:

* the configuration -- the accepted shapes, the refused ones, the refusal without ``plain_target``, the
  default, and ``search_config.target_axes_of``;
* the one target test (``plain_fit.is_target_term``) where plain mode picks a target -- the target
  candidates, the target repair and its roll-back, the pool check and its cache;
* the weak rows' window score, which reads only the allowed derivatives;
* the fit-time refusal of a variable the search does not have;
* end to end on a travelling wave, strong and weak rows: every target is a time derivative;
* a seeded fit with ``target_axes=None`` equals the same fit without the key.
"""
import warnings
from contextlib import contextmanager

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.search_config import (active_config, collect_overrides, load_search_config,
                                          set_active_config, target_axes_of)
from epde.operators.common import fill_prune as FP
from epde.operators.common import plain_fit as PF
from epde.operators.common import product_rule as PR
from epde.operators.common import system_redundancy as SR
from epde.preprocessing import weak_form as WF
from epde.structure.weak_rows import build_weak_rows

DT, DX, DXX = 'du/dx0{power: 1.0}', 'du/dx1{power: 1.0}', 'd^2u/dx1^2{power: 1.0}'
#: The factors setting the translated equations get (what a live search draws with).
LIVE_FACTORS = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
UNIVERSAL = {'system_rps_mode': 'plain', 'engine': 'nsga2'}
TIME_ONLY = {'target_axes': {'u': 0}}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _config(**overrides):
    return load_search_config(overrides=collect_overrides(**overrides))


@contextmanager
def _active(config):
    """Run the block under ``config``, putting back the configuration active before."""
    previous = active_config()
    set_active_config(config)
    try:
        yield config
    finally:
        set_active_config(previous)


def _wave():
    """u = sin(x - t) on a 2-D grid (t, x): u_t = -u_x."""
    t = np.linspace(0, 2 * np.pi, 40)
    x = np.linspace(0, 2 * np.pi, 48)
    T, X = np.meshgrid(t, x, indexing='ij')
    return T, X, np.sin(X - T)


def _pool_search():
    search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _create_pool(search, max_deriv_order=(1, 2)):
    T, X, u = _wave()
    domain = search.createDomain([T, X], ID=0)[1]
    trajectory = search.createTrajectory({'u': u}, domain, cache_id=0)[1]
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.create_pool(data=[trajectory], max_deriv_order=max_deriv_order, data_fun_pow=1)
    return search.pool


def _equation(pool, text, weights):
    """``text`` (an equation of u) translated on ``pool``, pinned to a fitted state: target last."""
    from epde.interface.equation_translator import translate_equation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        system = translate_equation({'u': text}, pool, all_vars=['u'])
    eq = system.vals['u']
    eq.main_var_to_explain = 'u'
    eq.metaparameters['max_factors_in_term']['value'] = dict(LIVE_FACTORS)
    for term in eq.structure:
        term.max_factors_in_term = dict(LIVE_FACTORS)
    w = np.array(weights, dtype=float)
    eq.weights_internal = w
    eq.weights_final = w.copy()
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    eq.target_idx = len(eq.structure) - 1
    return eq


def _along(term, axes):
    """A plain derivative of u whose deriv_code axes all lie in ``axes``."""
    return PF.is_plain_term(term, 'u') and set(term.structure[0].deriv_code) <= set(axes)


@pytest.fixture(autouse=True)
def _plain_state(monkeypatch):
    """Every cache plain mode keeps between calls starts empty and is put back after the test."""
    for module, name, empty in ((PF, '_SIBLINGS', {}), (PF, '_PLAIN_TOKENS', {}), (FP, '_LIBRARY', {}),
                                (FP, '_STORES', {}), (FP, '_WARNED', set()), (PR, '_GROUPS', {}),
                                (SR, '_REPORTED', set())):
        monkeypatch.setattr(module, name, type(empty)())
    monkeypatch.setitem(PF._ORDER, 'rng', None)
    yield


@pytest.fixture(scope='module')
def pool():
    """A 2-D pool of u(t, x): u, du/dx0, du/dx1, d^2u/dx1^2. Only its tokens are read."""
    previous = active_config()
    try:
        return _create_pool(_pool_search())
    finally:
        global_var.samples_manager.reset()
        set_active_config(previous)


# ---------------------------------------------------------------------------
# 1. The configuration.
# ---------------------------------------------------------------------------

class TestConfig:

    def test_the_default_is_none(self):
        assert _config(**UNIVERSAL).search_space.system_rps_params['target_axes'] is None
        assert _config(system_rps_mode='plain').search_space.system_rps_params['target_axes'] is None

    @pytest.mark.parametrize('axes', [None, 0, 1, (0,), (0, 1), [0, 2], {'u': 0}, {'u': (0, 1), 'v': None},
                                      {'u': [1]}, {}])
    def test_every_accepted_shape_loads(self, axes):
        config = _config(**UNIVERSAL, system_rps_params={'target_axes': axes})
        assert config.search_space.system_rps_params['target_axes'] == axes

    @pytest.mark.parametrize('axes', [True, -1, (), [], (0, -1), (0, True), 'x', 1.0, {0: 0}, {'u': 'x'},
                                      {'u': ()}, {'u': True}])
    def test_bad_shapes_are_refused(self, axes):
        with pytest.raises(ValueError, match="target_axes"):
            _config(**UNIVERSAL, system_rps_params={'target_axes': axes})

    def test_it_needs_plain_target(self):
        with pytest.raises(ValueError, match="target_axes.*plain_target"):
            _config(system_rps_mode='plain', system_rps_params={'target_axes': 0})
        with pytest.raises(ValueError, match="target_axes.*plain_target"):
            _config(**UNIVERSAL, system_rps_params={'target_axes': {'u': 0}, 'plain_target': False})
        _config(**UNIVERSAL, system_rps_params={'target_axes': {'u': 0}})
        _config(system_rps_mode='plain', system_rps_params={'target_axes': 0, 'plain_target': True})

    def test_target_axes_of(self):
        assert target_axes_of({'target_axes': None}, 'u') is None
        assert target_axes_of({}, 'u') is None
        assert target_axes_of({'target_axes': 1}, 'u') == frozenset({1})
        assert target_axes_of({'target_axes': (0, 2)}, 'v') == frozenset({0, 2})
        assert target_axes_of({'target_axes': [1]}, 'v') == frozenset({1})
        per_variable = {'target_axes': {'u': 0, 'v': (0, 1), 'w': None}}
        assert target_axes_of(per_variable, 'u') == frozenset({0})
        assert target_axes_of(per_variable, 'v') == frozenset({0, 1})
        assert target_axes_of(per_variable, 'w') is None
        assert target_axes_of(per_variable, 'z') is None
        assert isinstance(target_axes_of(per_variable, 'u'), frozenset)


# ---------------------------------------------------------------------------
# 2-4. The target test on a real 2-D pool.
# ---------------------------------------------------------------------------

class TestTargetTest:

    def test_only_time_derivatives_are_candidates(self, pool):
        eq = _equation(pool, f'0.5 * {DX} + 0.3 * {DXX} + 0.0 = {DT}', [0.5, 0.3, 0.0])
        dx, dxx, dt = eq.structure
        assert [t.name for t in (dx, dxx, dt)] == [DX, DXX, DT]
        with _active(_config(**UNIVERSAL, system_rps_params=TIME_ONLY)):
            assert PF.own_derivative_terms(eq) == [dt]
            assert PF.is_target_term(dt, 'u', True, frozenset({0}))
            assert not PF.is_target_term(dx, 'u', True, frozenset({0}))
            assert not PF.is_target_term(dxx, 'u', True, frozenset({0}))
        with _active(_config(**UNIVERSAL)):
            assert PF.own_derivative_terms(eq) == [dx, dxx, dt]
        with _active(_config(**UNIVERSAL, system_rps_params={'target_axes': {'u': 1}})):
            assert PF.own_derivative_terms(eq) == [dx, dxx]

    def _spatial_only(self, pool):
        eq = _equation(pool, f'0.5 * {DX} + 0.0 = {DXX}', [0.5, 0.0])
        assert all(PF.is_plain_term(term, 'u') and not _along(term, (0,)) for term in eq.structure)
        return eq

    def test_the_repair_brings_a_time_derivative(self, pool):
        eq = self._spatial_only(pool)
        with _active(_config(**UNIVERSAL, system_rps_params=TIME_ONLY)):
            np.random.seed(0)
            assert PF._ensure_target(eq, plain=True)
            assert any(_along(term, (0,)) for term in eq.structure)
            assert PF.own_derivative_terms(eq) and all(_along(t, (0,)) for t in PF.own_derivative_terms(eq))

    def test_without_axes_a_spatial_plain_term_is_enough(self, pool):
        eq = self._spatial_only(pool)
        before = list(eq.structure)
        with _active(_config(**UNIVERSAL)):
            assert PF._ensure_target(eq, plain=True)
        assert all(a is b for a, b in zip(eq.structure, before)) and len(eq.structure) == len(before)

    def test_a_failed_repair_rolls_the_structure_back(self, pool, monkeypatch):
        eq = self._spatial_only(pool)
        calls = []

        def no_time_term(self, deriv=False, mandatory_family=False, t_derivative=False):
            calls.append(deriv)
            self.structure = [self.structure[0]]          # a change that brings no time derivative
        monkeypatch.setattr(type(eq), 'restore_property', no_time_term)
        before = list(eq.structure)
        with _active(_config(**UNIVERSAL, system_rps_params=TIME_ONLY)):
            assert PF._ensure_target(eq, plain=True) is False
        assert len(calls) == PF.ENSURE_MAX_ITER and all(calls)
        assert len(eq.structure) == len(before)
        assert all(a is b for a, b in zip(eq.structure, before))

    def test_the_pool_check_names_target_axes(self, pool):
        eq = _equation(pool, f'0.5 * {DX} + 0.0 = {DT}', [0.5, 0.0])
        with pytest.raises(ValueError, match=r"target_axes.*'u'.*\[5\]"):
            PF._check_plain_tokens(eq, frozenset({5}))

    def test_the_verdict_is_kept_per_axes(self, pool):
        eq = _equation(pool, f'0.5 * {DX} + 0.0 = {DT}', [0.5, 0.0])
        for _ in range(2):
            with pytest.raises(ValueError, match='target_axes'):
                PF._check_plain_tokens(eq, frozenset({5}))
            PF._check_plain_tokens(eq, frozenset({0}))
            PF._check_plain_tokens(eq, frozenset({1}))
            PF._check_plain_tokens(eq)
        verdicts = {key[2]: entry[1] for key, entry in PF._PLAIN_TOKENS.items()}
        assert verdicts == {frozenset({5}): False, frozenset({0}): True, frozenset({1}): True, None: True}

    def test_no_time_derivative_in_the_pool_is_refused(self):
        previous = active_config()
        try:
            no_time = _create_pool(_pool_search(), max_deriv_order=(0, 2))
        finally:
            global_var.samples_manager.reset()
            set_active_config(previous)
        eq = _equation(no_time, f'0.5 * {DX} + 0.0 = {DXX}', [0.5, 0.0])
        with pytest.raises(ValueError, match=r"target_axes.*\[0\]"):
            PF._check_plain_tokens(eq, frozenset({0}))
        PF._check_plain_tokens(eq, frozenset({1}))
        with _active(_config(**UNIVERSAL, system_rps_params=TIME_ONLY)):
            with pytest.raises(ValueError, match='target_axes'):
                PF._ensure_target(eq, plain=True)


# ---------------------------------------------------------------------------
# 5. The weak rows' window score.
# ---------------------------------------------------------------------------

class _Stop(Exception):
    pass


class TestWeakRows:

    @pytest.fixture
    def live_pool(self):
        previous = active_config()
        try:
            yield _create_pool(_pool_search())
        finally:
            global_var.samples_manager.reset()
            set_active_config(previous)

    def _scored(self, pool, monkeypatch, target_axes):
        """The derivative orders of the candidate targets the window score is given (the window search
        itself is not run)."""
        seen = []

        def capture(fields, library, targets, rng, rows_ok=None, gate=None):
            seen.append(sorted(library[k].data[0][1] for k in targets))
            raise _Stop()
        monkeypatch.setattr(WF, 'choose_window', capture)
        with pytest.raises(_Stop):
            build_weak_rows(pool, global_var.samples_manager, lambda shape: True, target_axes=target_axes)
        return seen[0]

    def test_only_the_allowed_derivatives_are_scored(self, live_pool, monkeypatch):
        every = self._scored(live_pool, monkeypatch, None)
        assert every == [(0, 1), (0, 2), (1, 0)]
        assert self._scored(live_pool, monkeypatch, {'u': None}) == every
        assert self._scored(live_pool, monkeypatch, {'u': frozenset({0})}) == [(1, 0)]
        assert self._scored(live_pool, monkeypatch, {'u': frozenset({1})}) == [(0, 1), (0, 2)]

    def test_a_slot_without_a_candidate_is_refused(self, live_pool, monkeypatch):
        monkeypatch.setattr(WF, 'choose_window', lambda *args, **kwargs: (_ for _ in ()).throw(_Stop()))
        with pytest.raises(ValueError, match=r"target_axes.*'u'.*\[3\]"):
            build_weak_rows(live_pool, global_var.samples_manager, lambda shape: True,
                            target_axes={'u': frozenset({3})})


# ---------------------------------------------------------------------------
# 6-8. Fits.
# ---------------------------------------------------------------------------

def _search(form='strong', **extra):
    """A small universal search on one variable: three terms, population 8, two epochs."""
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, engine='nsga2', system_rps_mode='plain',
                             discrepancy_metric='wape', instability_metric='dyad', evaluation_form=form,
                             equation_terms_max_number=3, equation_factors_max_number=1,
                             data_fun_pow=1, population_size=8, training_epochs=2, **extra)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _fit_wave(search, seed):
    T, X, u = _wave()
    np.random.seed(seed)
    domain = search.createDomain([T, X], ID=0)[1]
    trajectory = search.createTrajectory({'u': u}, domain, cache_id=0)[1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.fit(data=[trajectory], max_deriv_order=(1, 2))


def _oscillator_fit(search, seed):
    """u = sin t, v = cos t; returns the reported front as text and objective values."""
    t = np.linspace(0, 4 * np.pi, 300)
    np.random.seed(seed)
    domain = search.createDomain(t, ID=0)[1]
    trajectory = search.createTrajectory({'u': np.sin(t), 'v': np.cos(t)}, domain, cache_id=0)[1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.fit(data=[trajectory], max_deriv_order=(1,))
    front = [(system.text_form, [float(v) for v in system.obj_fun]) for system in search.optimizer.reported_front()]
    history = [[(entry['text_form'], [float(v) for v in entry['obj_fun']]) for entry in epoch]
               for epoch in search.pareto_history]
    return front, history


def test_an_unknown_variable_is_refused_at_fit():
    previous = active_config()
    search = _search(system_rps_params={'target_axes': {'u': 0, 'w': 0}})
    try:
        with pytest.raises(ValueError, match=r"target_axes.*\['w'\].*\['u', 'v'\]"):
            _oscillator_fit(search, seed=0)
    finally:
        global_var.samples_manager.reset()
        PF.release()
        set_active_config(previous)


def _wave_targets(form, params):
    """(the reported front, the targets of the fitted equations of the front and the final population) of a
    seeded wave fit."""
    from epde.operators.common.objectives import LOSS_NAN_VAL
    previous = active_config()
    search = _search(form, system_rps_params=params)
    try:
        _fit_wave(search, seed=0)
        front = list(search.optimizer.reported_front())
        population = list(search.optimizer.pareto_levels.population)
        targets = []
        for system in front + population:
            eq = system.vals['u']
            if PF._fitted(eq) and eq.fitness_value is not None and eq.fitness_value < LOSS_NAN_VAL:
                targets.append(eq.target)
        if form == 'weak':
            assert global_var.samples_manager.weak.plans[(0, 'u')] is not None
        return front, targets
    finally:
        global_var.samples_manager.reset()
        PF.release()
        set_active_config(previous)


@pytest.mark.parametrize('form', ['strong', 'weak'])
def test_every_target_of_a_wave_fit_is_a_time_derivative(form):
    front, targets = _wave_targets(form, TIME_ONLY)
    assert front
    assert len(targets) >= len(front)
    assert all(_along(target, (0,)) for target in targets), [target.name for target in targets]
    # The same seeded fit without the setting reports spatial targets: the restriction is what keeps them out.
    _, free = _wave_targets(form, {})
    assert any(not _along(target, (0,)) for target in free)


def test_target_axes_none_is_the_default_path():
    previous = active_config()
    results = []
    try:
        for extra in ({}, {'system_rps_params': {'target_axes': None}}):
            search = _search(**extra)
            try:
                results.append(_oscillator_fit(search, seed=4))
            finally:
                global_var.samples_manager.reset()
                PF.release()
    finally:
        set_active_config(previous)
    assert results[0][0] and len(results[0][1]) == 2
    assert results[0] == results[1]
