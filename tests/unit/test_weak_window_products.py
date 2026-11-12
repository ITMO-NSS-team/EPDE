"""``domain.weak_window_products``: the opt-in product part of the weak-form window score.

Pinned here:

* the configuration -- the default, the menu, the refusal without ``evaluation_form='weak'`` and the
  ``EpdeSearch`` keyword;
* ``weak_rows.product_library`` -- the pairwise products of the selection library's data and derivative tokens;
* ``weak_form.product_noise`` -- against a direct computation, on clean data, on a constant column, and
  independent of the product grouping;
* ``weak_form.choose_window`` with products -- the summed score, the unchanged path without products, the
  shared cache;
* ``build_weak_rows`` -- one cache per trajectory, and ``'off'`` equal to the argument absent;
* end to end -- a seeded weak fit with ``'pairs'``, and ``'off'`` equal to the key absent.
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.evaluators import simple_function_evaluator
from epde.interface.search_config import active_config, load_search_config, set_active_config
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


def _wave_values(n=(33, 41)):
    t = np.linspace(0, 2 * np.pi, n[0])
    x = np.linspace(0, 2 * np.pi, n[1])
    T, X = np.meshgrid(t, x, indexing='ij')
    return t, x, T, X, np.sin(X - T) + 0.3 * np.cos(2 * X + T)


def _noisy_fields(noise=0.02, seed=0, extra=None):
    """A noisy 2-D field u(t, x) (and ``extra`` fields on the same grid)."""
    t, x, _, _, u = _wave_values()
    rng = np.random.default_rng(seed)
    values = {'u': u + noise * u.std() * rng.standard_normal(u.shape)}
    values.update(extra or {})
    return WF.Fields(values, [t, x])


def _independent_draw(fields, seed=11, scale=0.05):
    rng = np.random.default_rng(seed)
    return fields.with_values({k: v + scale * rng.standard_normal(v.shape) for k, v in fields.values.items()})


def _choose_window_draw(fields, seed):
    """The noise draw ``choose_window`` makes with ``np.random.default_rng(seed)`` on every field."""
    rng = np.random.default_rng(seed)
    drawn = dict(fields.values)
    for k in fields.values:
        sig, rhos, _ = WF.noise_model(fields.values[k])
        drawn[k] = fields.values[k] + WF.ar_noise(rng, fields.values[k].shape, sig, rhos)
    return fields.with_values(drawn)


def _direct(fields, draw, products, cut, ms, q):
    """The product part computed term by term, one ``weak_matrix`` call per column."""
    total = []
    for term in products:
        monos = term.monomials(psi_cap=q)
        x = WF.weak_matrix([monos], [()], fields, cut, ms, ms, q)[0][:, 0]
        xd = WF.weak_matrix([monos], [()], draw, cut, ms, ms, q)[0][:, 0]
        energy = np.var(x) * x.size
        noise = np.var(xd - x) * x.size
        a = noise / energy if energy > 0 else np.inf
        total.append(min(a, WF.NOISE_DOMINATED) ** 2 if np.isfinite(a) else WF.NOISE_DOMINATED ** 2)
    return float(np.sum(total))


def _counting(monkeypatch):
    calls = []
    real = WF.product_noise

    def wrapper(fields, draw, products, cut, ms, q):
        calls.append((tuple(q), tuple(ms)))
        return real(fields, draw, products, cut, ms, q)
    monkeypatch.setattr(WF, 'product_noise', wrapper)
    return calls


def _pool_search():
    search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _wave_pool():
    """A 2-D pool of u(t, x): u, du/dx0, du/dx1, d^2u/dx1^2; the samples manager holds its trajectory."""
    search = _pool_search()
    _, _, T, X, u = _wave_values()
    domain = search.createDomain([T, X], ID=0)[1]
    trajectory = search.createTrajectory({'u': u}, domain, cache_id=0)[1]
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.create_pool(data=[trajectory], max_deriv_order=(1, 2), data_fun_pow=1)
    return search.pool


def _two_variable_pool():
    """A 1-D pool of noisy u = sin t, v = cos t: u, du/dx0, v, dv/dx0."""
    search = _pool_search()
    t = np.linspace(0, 4 * np.pi, 300)
    rng = np.random.default_rng(2)
    values = {'u': np.sin(t) + 0.02 * rng.standard_normal(t.size),
              'v': np.cos(t) + 0.02 * rng.standard_normal(t.size)}
    domain = search.createDomain(t, ID=0)[1]
    trajectory = search.createTrajectory(values, domain, cache_id=0)[1]
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.create_pool(data=[trajectory], max_deriv_order=(1,), data_fun_pow=1)
    return search.pool


@pytest.fixture
def wave_pool():
    try:
        yield _wave_pool()
    finally:
        global_var.samples_manager.reset()


@pytest.fixture
def two_variable_pool():
    try:
        yield _two_variable_pool()
    finally:
        global_var.samples_manager.reset()


def _tables(rows):
    return {key: (repr(plan), repr(plan.table)) for key, plan in rows.plans.items()}


# ---------------------------------------------------------------------------
# 1. The configuration.
# ---------------------------------------------------------------------------

class TestConfig:

    def test_the_default_is_off(self):
        assert load_search_config().domain.weak_window_products == 'off'

    def test_a_value_outside_the_menu_is_refused(self):
        with pytest.raises(ValueError, match='weak_window_products must be one of'):
            load_search_config(None, {'evaluation_form': 'weak', 'weak_window_products': 'all'})

    def test_pairs_needs_weak_rows(self):
        with pytest.raises(ValueError, match="weak_window_products.*evaluation_form='weak'"):
            load_search_config(None, {'weak_window_products': 'pairs'})

    def test_pairs_with_weak_rows_loads(self):
        cfg = load_search_config(None, {'evaluation_form': 'weak', 'weak_window_products': 'pairs'})
        assert cfg.domain.weak_window_products == 'pairs'

    def test_the_search_keyword_reaches_the_config(self):
        search = epde.EpdeSearch(evaluation_form='weak', weak_window_products='pairs')
        assert search.config.domain.weak_window_products == 'pairs'
        assert epde.EpdeSearch().config.domain.weak_window_products == 'off'


# ---------------------------------------------------------------------------
# 2. The product library.
# ---------------------------------------------------------------------------

def _fake_pool(*variables):
    families = []
    for v in variables:
        labels = [v, f'd{v}/dx0', f'd{v}/dx1']
        families.append(SimpleNamespace(ftype=f'deriv of {v}', variable=v, _evaluator=simple_function_evaluator,
                                        tokens=labels, derivs_ords={labels[0]: [None], labels[1]: [0],
                                                                    labels[2]: [1]}))
    families.append(SimpleNamespace(ftype='grids', variable='grids', _evaluator=None, tokens=['x_0', 'x_1'],
                                    token_fixed_params={'x_0': {'dim': 0}, 'x_1': {'dim': 1}}))
    return SimpleNamespace(families=families)


class TestProductLibrary:

    def test_one_variable_on_a_2d_grid(self, wave_pool):
        library = WR.selection_library(wave_pool, 2)
        tokens = [t.data[0] for t in library if len(t.data) == 1 and not any(t.coords)]
        assert sorted(tokens) == sorted([U, UT, UX, UXX])
        products = WR.product_library(wave_pool, 2)
        n = len(tokens)
        assert len(products) == n * (n + 1) // 2 == 10
        assert all(len(p.data) == 2 and p.coords == (0, 0) and p.opaque == () for p in products)
        expected = [WT([tokens[i], tokens[j]], (0, 0)) for i in range(n) for j in range(i, n)]
        assert products == expected
        assert all(WT([tok, tok], (0, 0)) in products for tok in tokens)

    def test_grid_tokens_are_not_factors(self):
        products = WR.product_library(_fake_pool('u'), 2)
        assert len(WR.selection_library(_fake_pool('u'), 2)) == 5      # three data tokens, two coordinates
        assert products == [WT([a, b], (0, 0)) for i, a in enumerate((U, UT, UX)) for b in (U, UT, UX)[i:]]

    def test_two_variables_give_cross_products(self, two_variable_pool):
        products = WR.product_library(two_variable_pool, 1)
        assert len(products) == 10
        datas = {p.data for p in products}
        for a in (('u', (0,)), ('u', (1,))):
            for b in (('v', (0,)), ('v', (1,))):
                assert tuple(sorted([a, b])) in datas
        assert (('u', (0,)), ('u', (0,))) in datas and (('v', (1,)), ('v', (1,))) in datas


# ---------------------------------------------------------------------------
# 3. product_noise.
# ---------------------------------------------------------------------------

class TestProductNoise:

    CUT, MS, Q = (0, 0), (4, 4), (3, 3)

    def test_it_equals_a_direct_computation(self):
        fields = _noisy_fields()
        draw = _independent_draw(fields)
        pointwise = [WT([UX, UX], (0, 0)), WT([UT, UXX], (0, 0))]
        divergence = [WT([U, U], (0, 0)), WT([U, UX], (0, 0))]
        for term in pointwise:
            assert any(WF.leftover_orders(term.monomials(psi_cap=self.Q), 2))
        for term in divergence:
            assert not any(WF.leftover_orders(term.monomials(psi_cap=self.Q), 2))
        for products in (pointwise, divergence, pointwise + divergence):
            value = WF.product_noise(fields, draw, products, self.CUT, self.MS, self.Q)
            assert 0.0 < value <= len(products) * WF.NOISE_DOMINATED ** 2
            assert value == pytest.approx(_direct(fields, draw, products, self.CUT, self.MS, self.Q),
                                          rel=1e-10, abs=0)

    def test_clean_data_gives_zero(self):
        fields = _noisy_fields()
        clean = fields.with_values(dict(fields.values))
        assert WF.product_noise(fields, clean, PAIRS, self.CUT, self.MS, self.Q) == 0.0

    def test_a_column_constant_over_the_rows_counts_the_cap(self):
        shape = _noisy_fields().shape
        fields = _noisy_fields(extra={'z': np.zeros(shape), 'c': np.ones(shape)})
        draw = _independent_draw(fields)
        for var in ('z', 'c'):
            term = WT([(var, (0, 0)), (var, (0, 0))], (0, 0))
            assert WF.product_noise(fields, draw, [term], self.CUT, self.MS, self.Q) == WF.NOISE_DOMINATED ** 2
        clean = fields.with_values(dict(fields.values))
        assert WF.product_noise(fields, clean, [WT([('z', (0, 0))] * 2, (0, 0))], self.CUT, self.MS,
                                self.Q) == WF.NOISE_DOMINATED ** 2

    def test_the_grouping_does_not_change_the_value(self, monkeypatch):
        fields = _noisy_fields()
        draw = _independent_draw(fields)
        calls = []
        real = WF.weak_matrix

        def counting(*args, **kwargs):
            calls.append(len(args[0]))
            return real(*args, **kwargs)
        monkeypatch.setattr(WF, 'weak_matrix', counting)
        whole = WF.product_noise(fields, draw, PAIRS, self.CUT, self.MS, self.Q)
        assert calls == [len(PAIRS)] * 2
        calls.clear()
        monkeypatch.setattr(WF, 'PRODUCT_GROUP_BYTES', 1)
        single = WF.product_noise(fields, draw, PAIRS, self.CUT, self.MS, self.Q)
        assert calls == [1] * (2 * len(PAIRS))
        assert single == pytest.approx(whole, rel=1e-12, abs=0)


# ---------------------------------------------------------------------------
# 4. choose_window with products.
# ---------------------------------------------------------------------------

class TestChooseWindow:

    SEED = 7

    def _plan(self, fields, **kwargs):
        return WF.choose_window(fields, LIBRARY, TARGETS, np.random.default_rng(self.SEED), **kwargs)

    def test_the_score_is_the_window_score_plus_the_product_part(self):
        fields = _noisy_fields()
        base = self._plan(fields)
        summed = self._plan(fields, products=PAIRS)
        draw = _choose_window_draw(fields, self.SEED)
        own = {(row[0], row[1]): row for row in base.table}
        compared = 0
        for j, ms, ss, shape, score in summed.table:
            if (j, ms) not in own:
                continue
            assert own[(j, ms)][2:4] == (ss, shape)
            if not np.isfinite(own[(j, ms)][4]):
                assert score == own[(j, ms)][4]
                continue
            q = [r + j for r in base.r]
            part = WF.product_noise(fields, draw, PAIRS, base.cut, ms, q)
            assert part > 0
            assert score == pytest.approx(own[(j, ms)][4] + part, rel=1e-12, abs=0)
            compared += 1
        assert compared >= 10
        assert summed.score == min(row[4] for row in summed.table)

    @pytest.mark.parametrize('products', [None, []])
    def test_no_products_is_the_plain_path(self, products, monkeypatch):
        calls = _counting(monkeypatch)
        fields = _noisy_fields()
        base = self._plan(fields)
        other = self._plan(fields, products=products)
        assert repr(other) == repr(base) and repr(other.table) == repr(base.table)
        assert calls == []

    def test_a_shared_cache_is_reused(self, monkeypatch):
        calls = _counting(monkeypatch)
        fields = _noisy_fields()
        cache = {}
        first = self._plan(fields, products=PAIRS, product_cache=cache)
        assert calls and len(calls) == len(set(calls)) == len(cache)
        assert set(calls) == {(tuple(r + row[0] for r in first.r), row[1]) for row in first.table
                              if np.isfinite(row[4])}
        calls.clear()
        second = self._plan(fields, products=PAIRS, product_cache=cache)
        assert calls == []
        assert repr(second) == repr(first) and repr(second.table) == repr(first.table)


# ---------------------------------------------------------------------------
# 5. build_weak_rows.
# ---------------------------------------------------------------------------

class TestBuildWeakRows:

    def test_the_slots_of_a_trajectory_share_one_cache(self, two_variable_pool, monkeypatch):
        calls = _counting(monkeypatch)
        rows = WR.build_weak_rows(two_variable_pool, global_var.samples_manager, lambda shape: True,
                                  window_products='pairs')
        assert set(rows.plans) == {(0, 'u'), (0, 'v')}
        scored = [(tuple(r + row[0] for r in plan.r), row[1]) for plan in rows.plans.values()
                  for row in plan.table if np.isfinite(row[4])]
        assert len(calls) == len(set(calls)) == len(set(scored)) < len(scored)

    def test_off_is_the_argument_absent(self, two_variable_pool, monkeypatch):
        calls = _counting(monkeypatch)
        manager = global_var.samples_manager
        absent = WR.build_weak_rows(two_variable_pool, manager, lambda shape: True)
        off = WR.build_weak_rows(two_variable_pool, manager, lambda shape: True, window_products='off')
        assert _tables(absent) == _tables(off)
        assert calls == []

    def test_an_unknown_value_is_refused(self, two_variable_pool):
        with pytest.raises(ValueError, match='window_products must be one of'):
            WR.build_weak_rows(two_variable_pool, global_var.samples_manager, lambda shape: True,
                               window_products='all')


# ---------------------------------------------------------------------------
# 6. End to end.
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
        plans = _tables(global_var.samples_manager.weak)
        return front, history, plans
    finally:
        global_var.samples_manager.reset()
        PF.release()


def test_a_fit_with_pairs_finishes(monkeypatch):
    calls = _counting(monkeypatch)
    front, history, plans = _oscillator_fit(4, weak_window_products='pairs')
    assert front and len(history) == 2
    assert set(plans) == {(0, 'u'), (0, 'v')}
    assert calls


def test_off_is_the_default_path():
    absent = _oscillator_fit(4)
    off = _oscillator_fit(4, weak_window_products='off')
    assert absent[0] and len(absent[1]) == 2
    assert absent == off
