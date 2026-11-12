"""``domain.weak_window_gate``: the half-power gate of the weak-form window chooser.

Pinned here:

* ``weak_form.kept_shares`` -- on a noise-free sinusoid, against a direct evaluation of its formula, on a
  constant field, and with the chooser's caches;
* ``weak_form.choose_window`` with a gate -- ``gate=None`` is the ungated chooser, the gated pick is the best
  admissible window, the fallback when no window is admissible, the gate with products, and the smoothness
  extension that follows the ungated best window;
* the configuration -- the default, the menu, the refusal, the ``EpdeSearch`` keyword and the config dict;
* ``build_weak_rows`` -- ``'off'`` and ``'half_power'`` against the chooser, ``None`` read from the active
  configuration, the refusal and the fallback warning;
* end to end -- a seeded weak fit under the default and under ``'off'``.
"""
import warnings

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.search_config import (WEAK_WINDOW_GATES, active_config, load_search_config,
                                          set_active_config)
from epde.operators.common import fill_prune as FP
from epde.operators.common import plain_fit as PF
from epde.operators.common import product_rule as PR
from epde.operators.common import system_redundancy as SR
from epde.preprocessing import weak_form as WF
from epde.structure import weak_rows as WR

WT = WF.WeakTerm
U, UT, UX, UXX = ('u', (0, 0)), ('u', (1, 0)), ('u', (0, 1)), ('u', (0, 2))
#: A one-factor selection library of u(t, x) and its target (u_t).
LIBRARY = [WT([U], (0, 0)), WT([UT], (0, 0)), WT([UX], (0, 0)), WT([UXX], (0, 0))]
TARGETS = [1]
PAIRS = [WT([a, b], (0, 0)) for i, a in enumerate((U, UT, UX, UXX)) for b in (U, UT, UX, UXX)[i:]]
#: A one-factor selection library of u(t) and its targets (u_t, u_tt).
LIBRARY_1D = [WT([('u', (0,))], (0,)), WT([('u', (1,))], (0,)), WT([('u', (2,))], (0,))]
TARGETS_1D = [1, 2]
SEED = 7


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _plain_state(monkeypatch):
    """Every cache plain mode keeps between calls starts empty and is put back after the test; the active
    configuration is put back too."""
    for module, name, empty in ((PF, '_SIBLINGS', {}), (PF, '_PLAIN_TOKENS', {}), (FP, '_LIBRARY', {}),
                                (FP, '_STORES', {}), (FP, '_WARNED', set()), (PR, '_GROUPS', {}),
                                (SR, '_REPORTED', set())):
        monkeypatch.setattr(module, name, type(empty)())
    monkeypatch.setitem(PF._ORDER, 'rng', None)
    previous = active_config()
    yield
    set_active_config(previous)


def _wave_fields(noise, seed=0, extra=None):
    """A noisy 2-D field u(t, x) (and ``extra`` fields on the same grid)."""
    t = np.linspace(0, 2 * np.pi, 33)
    x = np.linspace(0, 2 * np.pi, 41)
    T, X = np.meshgrid(t, x, indexing='ij')
    u = np.sin(X - T) + 0.3 * np.cos(2 * X + T)
    rng = np.random.default_rng(seed)
    values = {'u': u + noise * u.std() * rng.standard_normal(u.shape)}
    values.update(extra or {})
    return WF.Fields(values, [t, x])


def _sine_fields(noise, periods=8.0, n=400, seed=2):
    """A noisy 1-D field u = sin t over ``periods`` periods."""
    t = np.linspace(0, periods * 2 * np.pi, n)
    rng = np.random.default_rng(seed)
    return WF.Fields({'u': np.sin(t) + noise * rng.standard_normal(n)}, [t])


def _independent_draw(fields, seed=11, scale=0.05):
    rng = np.random.default_rng(seed)
    return fields.with_values({k: v + scale * rng.standard_normal(v.shape) for k, v in fields.values.items()})


def _choose(fields, library=LIBRARY, targets=TARGETS, **kwargs):
    return WF.choose_window(fields, library, targets, np.random.default_rng(SEED), **kwargs)


def _best(rows):
    return min(rows, key=lambda x: (x[4], x[0]))


def _admissible(plan, level):
    return [row for row in plan.table if np.isfinite(row[4]) and plan.shares[(row[0], row[1])] >= level]


def _counting(monkeypatch):
    calls = []
    real = WF.kept_shares

    def wrapper(*args, **kwargs):
        calls.append(tuple(args[3]))
        return real(*args, **kwargs)
    monkeypatch.setattr(WF, 'kept_shares', wrapper)
    return calls


def _pool(values, t, max_deriv_order):
    search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    domain = search.createDomain(t, ID=0)[1]
    trajectory = search.createTrajectory(values, domain, cache_id=0)[1]
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.create_pool(data=[trajectory], max_deriv_order=max_deriv_order, data_fun_pow=1)
    return search.pool


@pytest.fixture
def gated_pool():
    """A 1-D pool of noisy u = sin t over 16 periods (u, du/dx0, d^2u/dx0^2), whose ungated window keeps less
    than half of u's variance."""
    t = np.linspace(0, 16 * 2 * np.pi, 600)
    rng = np.random.default_rng(2)
    try:
        yield _pool({'u': np.sin(t) + 0.05 * rng.standard_normal(t.size)}, t, (2,))
    finally:
        global_var.samples_manager.reset()


@pytest.fixture
def two_variable_pool():
    """A 1-D pool of noisy u = sin t, v = cos t: u, du/dx0, v, dv/dx0."""
    t = np.linspace(0, 4 * np.pi, 300)
    rng = np.random.default_rng(2)
    values = {'u': np.sin(t) + 0.02 * rng.standard_normal(t.size),
              'v': np.cos(t) + 0.02 * rng.standard_normal(t.size)}
    try:
        yield _pool(values, t, (1,))
    finally:
        global_var.samples_manager.reset()


def _tables(rows):
    return {key: (repr(plan), repr(plan.table), plan.kept, plan.admissible, plan.shares)
            for key, plan in rows.plans.items()}


def _chooser_plans(pool, rows, gate):
    """The plans ``choose_window`` gives on the fields of ``rows`` for every slot of ``pool``."""
    out = {}
    for idx, key in enumerate(rows.keys):
        fields = rows.fields[key]
        library = WR.selection_library(pool, fields.D)
        for family in pool.families_demand_equation:
            slot = family.variable
            targets = [k for k, t in enumerate(library) if len(t.data) == 1 and not any(t.coords)
                       and t.data[0][0] == slot and sum(t.data[0][1]) > 0]
            out[(key, slot)] = WF.choose_window(fields, library, targets, np.random.default_rng([0, idx]),
                                                gate=gate)
    return {key: (repr(plan), repr(plan.table), plan.kept, plan.admissible, plan.shares)
            for key, plan in out.items()}


# ---------------------------------------------------------------------------
# 1. kept_shares.
# ---------------------------------------------------------------------------

class TestKeptShares:

    def test_a_clean_sinusoid_loses_its_variance_as_the_window_grows(self):
        fields = _sine_fields(0.0, periods=16.0, n=1001)          # about 62.5 samples per period
        clean = fields.with_values(dict(fields.values))
        shares = [WF.kept_shares(fields, clean, (0,), (m,), (2,))[0] for m in (2, 4, 8, 16, 32, 64, 128)]
        assert shares[0] > 0.99
        assert all(a > b for a, b in zip(shares[:4], shares[1:5]))
        assert shares[4] < 0.5
        assert all(abs(s) < 0.05 for s in shares[5:])              # windows of two periods and more

    def test_it_equals_a_direct_evaluation(self):
        fields = _sine_fields(0.1)
        draw = _independent_draw(fields)
        cut, q = (4,), (3,)
        for m in (4, 8, 16):
            a = fields.values['u'][cut[0]:-cut[0]]
            b = draw.values['u'][cut[0]:-cut[0]]
            w = WF.window_kernels((m,), fields.steps, q, (0,))[0][0]
            rows = (a.size - 2 * m - 1) // m + 1
            x = np.array([w @ a[i * m:i * m + 2 * m + 1] for i in range(rows)])
            e = np.array([w @ b[i * m:i * m + 2 * m + 1] for i in range(rows)]) - x
            centres = m + m * np.arange(rows)
            expected = (np.var(x) - np.var(e)) / (np.var(a[centres]) - np.var((b - a)[centres]))
            assert 0.0 < expected < 1.0
            assert WF.kept_shares(fields, draw, cut, (m,), q) == [pytest.approx(expected, rel=1e-9, abs=0)]

    def test_two_fields_on_a_2d_grid_against_their_weak_columns(self):
        t, x = np.linspace(0, 2 * np.pi, 33), np.linspace(0, 2 * np.pi, 41)
        T, X = np.meshgrid(t, x, indexing='ij')
        fields = WF.Fields({'u': np.sin(X - T), 'v': np.cos(2 * X + T)}, [t, x])
        draw = _independent_draw(fields)
        cut, ms, q = (2, 2), (4, 4), (3, 3)
        out = WF.kept_shares(fields, draw, cut, ms, q)
        assert len(out) == 2
        for k, var in enumerate(('u', 'v')):
            term = [WT([(var, (0, 0))], (0, 0)).monomials()]
            col, shape = WF.weak_matrix(term, [()], fields, cut, ms, ms, q)
            cold, _ = WF.weak_matrix(term, [()], draw, cut, ms, ms, q)
            col, err = col[:, 0], cold[:, 0] - col[:, 0]
            a = fields.values[var][2:-2, 2:-2]
            b = draw.values[var][2:-2, 2:-2]
            c = np.ix_(*[m + m * np.arange(n) for m, n in zip(ms, shape)])
            expected = (np.var(col) - np.var(err)) / (np.var(a[c]) - np.var((b - a)[c]))
            assert out[k] == pytest.approx(expected, rel=1e-9, abs=0)

    def test_the_caches_do_not_change_the_value(self):
        fields = _wave_fields(0.05, extra={'v': np.cos(_wave_fields(0.0).values['u'])})
        draw = _independent_draw(fields)
        caches = ({}, {})
        plain = WF.kept_shares(fields, draw, (0, 0), (4, 8), (2, 2))
        cached = WF.kept_shares(fields, draw, (0, 0), (4, 8), (2, 2), caches)
        assert caches[0] and caches[1]
        again = WF.kept_shares(fields, draw, (0, 0), (4, 8), (2, 2), caches)
        assert plain == cached == again

    def test_a_constant_field_keeps_nothing(self):
        shape = _wave_fields(0.02).shape
        fields = _wave_fields(0.02, extra={'c': np.full(shape, 3.0)})
        for draw in (_independent_draw(fields), fields.with_values(dict(fields.values))):
            shares = WF.kept_shares(fields, draw, (0, 0), (4, 4), (2, 2))
            assert shares[1] == 0.0
            assert shares[0] > 0.5


# ---------------------------------------------------------------------------
# 2. choose_window with a gate.
# ---------------------------------------------------------------------------

class TestChooseWindow:

    def test_no_gate_is_the_ungated_chooser(self, monkeypatch):
        calls = _counting(monkeypatch)
        fields = _wave_fields(0.02)
        base = _choose(fields)
        none = _choose(fields, gate=None)
        assert repr(none) == repr(base) and repr(none.table) == repr(base.table)
        assert (none.kept, none.admissible, none.shares) == (None, None, {})
        assert 'kept' not in repr(none)
        assert calls == []

    @pytest.mark.parametrize('case', ['wave', 'sine'])
    def test_the_pick_is_the_best_admissible_window(self, case):
        if case == 'wave':
            fields, library, targets = _wave_fields(0.02), LIBRARY, TARGETS
        else:
            fields, library, targets = _sine_fields(0.3), LIBRARY_1D, TARGETS_1D
        ungated = _choose(fields, library, targets)
        plan = _choose(fields, library, targets, gate=WF.HALF_POWER)
        assert repr(plan.table) == repr(ungated.table)
        assert set(plan.shares) == {(row[0], row[1]) for row in plan.table if np.isfinite(row[4])}
        assert plan.shares[(ungated.j, ungated.ms)] < WF.HALF_POWER
        inside = _admissible(plan, WF.HALF_POWER)
        assert plan.admissible == len(inside) > 0
        best = _best(inside)
        assert (plan.j, plan.ms, plan.ss, plan.shape, plan.score) == best
        assert plan.kept == plan.shares[(plan.j, plan.ms)] >= WF.HALF_POWER
        assert (plan.j, plan.ms) != (ungated.j, ungated.ms)
        assert repr(plan).endswith(f', kept={plan.kept:.2f})')

    def test_the_shares_are_those_of_kept_shares(self):
        fields = _sine_fields(0.3)
        plan = _choose(fields, LIBRARY_1D, TARGETS_1D, gate=WF.HALF_POWER)
        rng = np.random.default_rng(SEED)
        sig, rhos, _ = WF.noise_model(fields.values['u'])
        draw = fields.with_values({'u': fields.values['u'] + WF.ar_noise(rng, fields.shape, sig, rhos)})
        for (j, ms), share in plan.shares.items():
            q = [r + j for r in plan.r]
            assert share == min(WF.kept_shares(fields, draw, plan.cut, ms, q))

    def test_an_impossible_level_falls_back_to_the_ungated_pick(self):
        fields = _wave_fields(0.02)
        ungated = _choose(fields)
        plan = _choose(fields, gate=2.0)
        assert plan.admissible == 0
        assert (plan.j, plan.ms, plan.ss, plan.shape, plan.score) == \
            (ungated.j, ungated.ms, ungated.ss, ungated.shape, ungated.score)
        assert plan.kept == plan.shares[(ungated.j, ungated.ms)]
        assert repr(plan.table) == repr(ungated.table)

    def test_the_gate_with_products(self):
        fields = _wave_fields(0.02)
        products_only = _choose(fields, products=PAIRS)
        gated = _choose(fields, gate=WF.HALF_POWER)
        both = _choose(fields, products=PAIRS, gate=WF.HALF_POWER)
        assert repr(both.table) == repr(products_only.table)
        assert both.shares == gated.shares
        inside = _admissible(both, WF.HALF_POWER)
        assert both.admissible == len(inside) > 0
        assert (both.j, both.ms, both.ss, both.shape, both.score) == _best(inside)
        assert both.kept >= WF.HALF_POWER

    def test_the_extension_follows_the_ungated_best(self):
        # the ungated best sits at the largest smoothness tried: the table is extended, gate or not
        fields = _wave_fields(0.2)
        ungated = _choose(fields)
        plan = _choose(fields, gate=WF.HALF_POWER)
        assert max(row[0] for row in ungated.table) > max(WF.J_START)
        assert ungated.j == max(row[0] for row in ungated.table) - 1
        assert repr(plan.table) == repr(ungated.table)
        assert set(plan.shares) == {(row[0], row[1]) for row in plan.table if np.isfinite(row[4])}
        # the gated pick sits at the largest smoothness tried, the ungated best below it: no extension
        fields = _sine_fields(0.3)
        ungated = _choose(fields, LIBRARY_1D, TARGETS_1D)
        plan = _choose(fields, LIBRARY_1D, TARGETS_1D, gate=WF.HALF_POWER)
        top = max(row[0] for row in plan.table)
        assert top == max(WF.J_START) == plan.j > ungated.j
        assert repr(plan.table) == repr(ungated.table)


# ---------------------------------------------------------------------------
# 3. The configuration.
# ---------------------------------------------------------------------------

class TestConfig:

    def test_the_default_is_half_power(self):
        assert load_search_config().domain.weak_window_gate == 'half_power'
        assert epde.EpdeSearch().config.domain.weak_window_gate == 'half_power'

    def test_the_menu(self):
        assert WEAK_WINDOW_GATES == ('half_power', 'off')

    def test_a_value_outside_the_menu_is_refused(self):
        with pytest.raises(ValueError, match='weak_window_gate must be one of'):
            load_search_config(None, {'evaluation_form': 'weak', 'weak_window_gate': 'quarter_power'})
        with pytest.raises(ValueError, match='weak_window_gate must be one of'):
            epde.EpdeSearch(weak_window_gate=None)

    def test_the_search_keyword_reaches_the_config(self):
        search = epde.EpdeSearch(evaluation_form='weak', weak_window_gate='off')
        assert search.config.domain.weak_window_gate == 'off'

    def test_a_config_dict(self):
        cfg = load_search_config({'domain': {'evaluation_form': 'weak', 'weak_window_gate': 'off'}})
        assert cfg.domain.weak_window_gate == 'off'
        search = epde.EpdeSearch({'domain': {'evaluation_form': 'weak', 'weak_window_gate': 'off'}})
        assert search.config.domain.weak_window_gate == 'off'

    @pytest.mark.parametrize('gate', WEAK_WINDOW_GATES)
    def test_strong_rows_accept_every_value(self, gate):
        cfg = load_search_config(None, {'weak_window_gate': gate})
        assert cfg.domain.evaluation_form == 'strong'
        assert cfg.domain.weak_window_gate == gate


# ---------------------------------------------------------------------------
# 4. build_weak_rows.
# ---------------------------------------------------------------------------

class TestBuildWeakRows:

    def test_off_and_half_power_are_the_chooser_without_and_with_the_gate(self, gated_pool):
        manager = global_var.samples_manager
        off = WR.build_weak_rows(gated_pool, manager, lambda shape: True, window_gate='off')
        on = WR.build_weak_rows(gated_pool, manager, lambda shape: True, window_gate='half_power')
        assert _tables(off) == _chooser_plans(gated_pool, off, None)
        assert _tables(on) == _chooser_plans(gated_pool, on, WF.HALF_POWER)
        plan_off, plan_on = off.plans[(0, 'u')], on.plans[(0, 'u')]
        assert (plan_on.j, plan_on.ms) != (plan_off.j, plan_off.ms)
        assert plan_on.shares[(plan_off.j, plan_off.ms)] < WF.HALF_POWER <= plan_on.kept

    def test_none_reads_the_active_config(self, gated_pool):
        manager = global_var.samples_manager
        expected = {gate: _tables(WR.build_weak_rows(gated_pool, manager, lambda shape: True, window_gate=gate))
                    for gate in WEAK_WINDOW_GATES}
        assert expected['off'] != expected['half_power']
        for gate in WEAK_WINDOW_GATES:
            set_active_config(load_search_config(None, {'evaluation_form': 'weak', 'weak_window_gate': gate}))
            assert _tables(WR.build_weak_rows(gated_pool, manager, lambda shape: True)) == expected[gate]

    def test_an_unknown_value_is_refused(self, gated_pool):
        with pytest.raises(ValueError, match='window_gate must be one of'):
            WR.build_weak_rows(gated_pool, global_var.samples_manager, lambda shape: True,
                               window_gate='quarter_power')

    def test_the_fallback_warns_once_per_slot(self, two_variable_pool, monkeypatch):
        manager = global_var.samples_manager
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            WR.build_weak_rows(two_variable_pool, manager, lambda shape: True, window_gate='half_power')
        assert not [w for w in caught if issubclass(w.category, global_var.EPDEUsageWarning)]
        monkeypatch.setattr(WF, 'HALF_POWER', 2.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rows = WR.build_weak_rows(two_variable_pool, manager, lambda shape: True, window_gate='half_power')
        usage = [str(w.message) for w in caught if issubclass(w.category, global_var.EPDEUsageWarning)]
        assert len(usage) == 2
        off = WR.build_weak_rows(two_variable_pool, manager, lambda shape: True, window_gate='off')
        for slot in ('u', 'v'):
            plan = rows.plans[(0, slot)]
            assert plan.admissible == 0
            assert (plan.j, plan.ms) == (off.plans[(0, slot)].j, off.plans[(0, slot)].ms)
            text = [m for m in usage if f'equation slot {slot!r}' in m]
            assert len(text) == 1
            assert 'trajectory 0' in text[0] and f'{plan.kept:.2f}' in text[0]


# ---------------------------------------------------------------------------
# 5. End to end.
# ---------------------------------------------------------------------------

def _oscillator_fit(seed, **extra):
    """A seeded weak plain-mode NSGA-II fit of u = sin t, v = cos t: the reported front, the history and the
    window plans."""
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, engine='nsga2', system_rps_mode='plain',
                             discrepancy_metric='wape', instability_metric='dyad', evaluation_form='weak',
                             equation_terms_max_number=3, equation_factors_max_number=2,
                             data_fun_pow=1, population_size=8, training_epochs=2, **extra)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    t = np.linspace(0, 4 * np.pi, 300)
    np.random.seed(seed)
    try:
        domain = search.createDomain(t, ID=0)[1]
        trajectory = search.createTrajectory({'u': np.sin(t), 'v': np.cos(t)}, domain, cache_id=0)[1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            search.fit(data=[trajectory], max_deriv_order=(1,))
        front = [(system.text_form, [float(v) for v in system.obj_fun])
                 for system in search.optimizer.reported_front()]
        history = [[(entry['text_form'], [float(v) for v in entry['obj_fun']]) for entry in epoch]
                   for epoch in search.pareto_history]
        plans = dict(global_var.samples_manager.weak.plans)
        return front, history, plans
    finally:
        global_var.samples_manager.reset()
        PF.release()


def test_a_fit_under_the_default_and_under_off():
    gated = _oscillator_fit(4)
    off = _oscillator_fit(4, weak_window_gate='off')
    for front, history, plans in (gated, off):
        assert front and len(history) == 2
        assert set(plans) == {(0, 'u'), (0, 'v')}
    for key, plan in gated[2].items():
        assert plan.admissible > 0 and plan.kept >= WF.HALF_POWER
        assert off[2][key].kept is None and off[2][key].admissible is None
        # the oscillator's ungated windows keep more than half: the gate picks the same windows
        assert (plan.j, plan.ms, plan.ss) == (off[2][key].j, off[2][key].ms, off[2][key].ss)
    assert gated[:2] == off[:2]
