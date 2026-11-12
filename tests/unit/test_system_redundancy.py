#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A system whose equations restate one relation (``system_redundancy``).

Pinned here, on Lotka-Volterra-shaped systems (a = b = c = d = 20, so u' + v' = 20 (u - v) is an identity):

* the sum relation beside its own time derivative: the derivative follows, and is the one found;
* the sum relation beside an identity that adds the difference of the laws (u'' = -v'' + 400 u + 400 v -
  800 u v): the two still pin down both laws, so nothing follows;
* the true pair, and an equation that is another multiplied by a field;
* inexact (fitted) coefficients on both sides of the tolerance;
* the test draws from its own generator; plain mode re-rolls a following equation and drops a following
  sibling, testing it before copying it, on the equations the sibling holds; assignment mode declines a
  following pick;
* the samples manager's axis ranges are memoised, and recomputed for a new trajectory, a reset, a changed
  boundary or a rebuilt grid.
"""
import numpy as np
import pytest

import epde.operators.common.plain_fit as plain_fit
import epde.operators.common.system_redundancy as redundancy
from epde.operators.common.objectives import LOSS_NAN_VAL
from epde.interface.search_config import REDUNDANCY_RTOL
from epde.operators.common.system_redundancy import redundant_equations

from test_duplicate_equation_guard import _system, uv_pool  # noqa: F401

RANGES = [(0.0, 4 * np.pi)]
U, V, UV = 'u{power: 1.0}', 'v{power: 1.0}', 'u{power: 1.0} * v{power: 1.0}'
DU, DV = 'du/dx0{power: 1.0}', 'dv/dx0{power: 1.0}'
DDU, DDV = 'd^2u/dx0^2{power: 1.0}', 'd^2v/dx0^2{power: 1.0}'


@pytest.fixture(scope='module')
def lv_pool():
    import epde
    search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False})
    t = np.linspace(0.0, 4 * np.pi, 200)
    _, domain = search.createDomain(t, boundary_width=10, ID=0)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory({'u': np.sin(t) + 2.0, 'v': np.cos(t) + 0.3 * np.sin(2 * t) + 2.0},
                                            domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(2,), data_fun_pow=2)
    return search.pool


def _equations(pool, u_text, u_weights, v_text, v_weights):
    system = _system(pool, {'u': u_text, 'v': v_text}, {'u': u_weights, 'v': v_weights})
    return [system.vals['u'], system.vals['v']]


SUM = (f'-1.0 * {DU} + 20.0 * {U} + -20.0 * {V} + 0.0 = {DV}', [-1.0, 20.0, -20.0, 0.0])
SUM_DERIVATIVE = (f'-1.0 * {DDV} + 20.0 * {DU} + -20.0 * {DV} + 0.0 = {DDU}', [-1.0, 20.0, -20.0, 0.0])
# D(law_u + law_v) + 20 (law_u - law_v): with the sum relation it gives both laws.
SUM_AND_DIFFERENCE = (f'-1.0 * {DDV} + 400.0 * {U} + 400.0 * {V} + -800.0 * {UV} + 0.0 = {DDU}',
                      [-1.0, 400.0, 400.0, -800.0, 0.0])
LAW_U = (f'20.0 * {U} + -20.0 * {UV} + 0.0 = {DU}', [20.0, -20.0, 0.0])
LAW_V = (f'-20.0 * {V} + 20.0 * {UV} + 0.0 = {DV}', [-20.0, 20.0, 0.0])


class TestTheTest:

    def test_a_relation_beside_its_derivative_restates_it(self, lv_pool):
        equations = _equations(lv_pool, *SUM_DERIVATIVE, *SUM)
        assert redundant_equations(equations, RANGES) == [0]
        # The order of the equations does not change which one follows.
        assert redundant_equations(equations[::-1], RANGES) == [1]

    def test_identities_that_pin_down_both_laws_are_kept(self, lv_pool):
        assert redundant_equations(_equations(lv_pool, *SUM_AND_DIFFERENCE, *SUM), RANGES) == []

    def test_the_true_pair_is_kept(self, lv_pool):
        assert redundant_equations(_equations(lv_pool, *LAW_U, *LAW_V), RANGES) == []

    def test_an_equation_multiplied_by_a_field_follows(self, lv_pool):
        # v * (sum relation), written for the target u' v.
        text = (f'-1.0 * {DV} * {V} + 20.0 * {UV} + -20.0 * v{{power: 2.0}} + 0.0 = {DU} * {V}',
                [-1.0, 20.0, -20.0, 0.0])
        assert redundant_equations(_equations(lv_pool, *text, *SUM), RANGES) == [0]

    def test_fitted_coefficients_are_judged_with_a_tolerance(self, lv_pool):
        # About 1e-3 off, as fitted: the derivative still follows, the difference law still does not.
        rough = [-0.999, 19.987, -20.004, 0.0]
        assert redundant_equations(_equations(lv_pool, SUM_DERIVATIVE[0], rough, *SUM), RANGES) == [0]
        rough = [-0.998, 396.7, 394.5, -792.2, 0.0]
        assert redundant_equations(_equations(lv_pool, SUM_AND_DIFFERENCE[0], rough, *SUM), RANGES) == []
        assert REDUNDANCY_RTOL == 0.1

    def test_one_equation_and_the_random_stream(self, lv_pool):
        equations = _equations(lv_pool, *SUM_DERIVATIVE, *SUM)
        assert redundant_equations(equations[:1], RANGES) == []
        state = np.random.get_state()[1].copy()
        redundant_equations(equations, RANGES)
        assert np.array_equal(np.random.get_state()[1], state)

    def test_the_data_ranges_are_read_when_not_given(self, lv_pool):
        assert redundant_equations(_equations(lv_pool, *SUM_DERIVATIVE, *SUM)) == [0]

    def test_the_data_ranges_are_the_inner_grids(self, lv_pool):
        import epde.globals as global_var
        manager = global_var.samples_manager
        manager._axis_ranges_memo = None
        ranges = redundancy._axis_ranges()
        grids = manager.grids(mode='solver')
        assert ranges == [(float(np.min(axes[0])), float(np.max(axes[0]))) for axes in grids.values()]
        assert redundancy._axis_ranges() == ranges


class _GridDomain(object):
    """A domain as ``TrajectoriesManager.axis_ranges`` reads it: one cached grid per axis, the
    boundary widths and the evaluation stride."""

    def __init__(self, axes, boundary=1, stride=None, ID=0):
        self._ID, self.axes = ID, list(axes)
        self.boundary_width_per_axis = [boundary] * len(self.axes)
        self._eval_stride = stride
        self._grid_cache = self

    def get_all(self, subcache_ID=None):
        return [None] * len(self.axes), list(self.axes)


class _GridTrajectory(object):
    def __init__(self, domain, ID=0):
        self.ID, self._domain, self.reads = ID, domain, 0

    def grids(self, mode='full'):
        self.reads += 1
        if self._domain._eval_stride is not None:
            raise ValueError('no solver grid on a lattice')
        return [axis[b:axis.size - b] for axis, b in zip(self._domain.axes, self._domain.boundary_width_per_axis)]


def _trajectory(lo, hi, n=11, ID=0, **kwargs):
    return _GridTrajectory(_GridDomain([np.linspace(lo, hi, n)], ID=ID, **kwargs), ID=ID)


def _inner(lo, hi, n=11, boundary=1):
    axis = np.linspace(lo, hi, n)
    return float(axis[boundary]), float(axis[n - 1 - boundary])


class TestTheAxisRanges:

    def _manager(self, *trajectories):
        from epde.structure.domain import TrajectoriesManager
        manager = TrajectoriesManager()
        for trajectory in trajectories:
            manager.addTrajectory(trajectory, trajectory._domain)
        return manager

    def test_a_repeat_is_the_memo(self):
        trajectory = _trajectory(0.0, 1.0)
        manager = self._manager(trajectory)
        first = manager.axis_ranges()
        assert first == [_inner(0.0, 1.0)] and trajectory.reads == 1
        first.append('scribble')
        assert manager.axis_ranges() == [_inner(0.0, 1.0)] and trajectory.reads == 1

    def test_a_new_trajectory_or_a_reset_recomputes(self):
        manager = self._manager(_trajectory(0.0, 1.0))
        manager.axis_ranges()
        second = _trajectory(-1.0, 0.5, n=16, ID=1)
        manager.addTrajectory(second, second._domain)
        assert manager.axis_ranges() == [(_inner(-1.0, 0.5, n=16)[0], _inner(0.0, 1.0)[1])]
        manager.reset()
        assert manager.axis_ranges() is None
        # Another fit: its domain under the same ID, over another range.
        third = _trajectory(2.0, 3.0)
        manager.addTrajectory(third, third._domain)
        assert manager.axis_ranges() == [_inner(2.0, 3.0)]

    def test_a_changed_domain_recomputes(self):
        trajectory = _trajectory(0.0, 1.0)
        manager = self._manager(trajectory)
        manager.axis_ranges()
        trajectory._domain.boundary_width_per_axis[0] = 2
        assert manager.axis_ranges() == [_inner(0.0, 1.0, boundary=2)] and trajectory.reads == 2
        trajectory._domain.axes = [np.linspace(0.0, 1.0, 11)]       # the same values, rebuilt
        manager.axis_ranges()
        assert trajectory.reads == 3

    def test_a_lattice_has_no_ranges_and_is_memoised(self):
        trajectory = _trajectory(0.0, 1.0, stride=(2,))
        manager = self._manager(trajectory)
        assert manager.axis_ranges() is None and manager.axis_ranges() is None
        assert trajectory.reads == 1


class TestTheHooks:

    def test_plain_mode_re_rolls_a_following_equation(self, uv_pool, monkeypatch):
        from test_plain_mode import _fresh, _run
        monkeypatch.setattr(plain_fit, 'redundant_equations', lambda equations, order=None: [0])
        system = _run(_fresh(uv_pool))
        assert system.vals['u'].fitness_value >= LOSS_NAN_VAL
        assert system.vals['v'].fitness_value < LOSS_NAN_VAL
        assert plain_fit.take_siblings(system) == []

    def _sibling_run(self, uv_pool, monkeypatch, verdict):
        """A plain-mode fit with every version kept (three sibling combos), ``verdict`` the
        redundancy test's answer: the lists it was asked about, and the systems deep-copied."""
        from test_plain_mode import _fresh, _run
        from epde.structure.main_structures import SoEq
        monkeypatch.setattr(plain_fit, '_dominates', lambda a, b, attrs: False)
        asked, copied = [], []
        monkeypatch.setattr(plain_fit, 'redundant_equations',
                            lambda equations, order=None: asked.append(list(equations)) or verdict)
        real_deepcopy = plain_fit.deepcopy

        def counting_deepcopy(obj, *args):
            if isinstance(obj, SoEq):
                copied.append(obj)
            return real_deepcopy(obj, *args)
        monkeypatch.setattr(plain_fit, 'deepcopy', counting_deepcopy)
        system = _run(_fresh(uv_pool))
        return system, asked, copied

    def test_a_refused_sibling_is_never_copied(self, uv_pool, monkeypatch):
        system, asked, copied = self._sibling_run(uv_pool, monkeypatch, [0])
        assert len(asked) == 4, 'three combos, then the system itself'
        assert copied == [] and plain_fit.take_siblings(system) == []

    def test_a_sibling_holds_the_equations_it_was_tested_on(self, uv_pool, monkeypatch):
        system, asked, copied = self._sibling_run(uv_pool, monkeypatch, [])
        siblings = plain_fit.take_siblings(system)
        assert len(asked) == len(siblings) + 1 == len(copied) + 1 == 4
        assert [eq.text_form for eq in asked[-1]] == [eq.text_form for eq in system]
        for tested, sibling in zip(asked, siblings):
            held = list(sibling)
            assert [eq.main_var_to_explain for eq in tested] == list(system.vars_to_describe)
            assert [eq.text_form for eq in tested] == [eq.text_form for eq in held]
            for a, b in zip(tested, held):
                assert np.asarray(a.weights_final).tobytes() == np.asarray(b.weights_final).tobytes()
                assert a.target_idx == b.target_idx

    def test_assignment_mode_declines_a_following_pick(self, uv_pool, monkeypatch):
        from test_assignment_rps import _fresh, _run
        monkeypatch.setattr(redundancy, 'redundant_equations', lambda equations: [0])
        system = _run(_fresh(uv_pool))
        assert all(system.vals[var].fitness_value >= LOSS_NAN_VAL for var in ('u', 'v'))
