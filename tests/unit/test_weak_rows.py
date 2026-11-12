"""Weak-form rows in the search (``domain.evaluation_form='weak'``): the setting and its refusals, the term
mapping, the fail-loud strong accessors, and small fits that recover known equations on weak rows."""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.evaluators import CustomEvaluator, simple_function_evaluator
from epde.interface.search_config import check_evaluation_form, load_search_config
from epde.preprocessing import weak_form as WF
from epde.structure import weak_rows as WR


# ------------------------------------------------------------------------------------------------- setting
class TestEvaluationFormSetting:

    def test_the_default_is_strong(self):
        assert load_search_config().domain.evaluation_form == 'strong'

    def test_weak_resolves(self):
        assert load_search_config(None, {'evaluation_form': 'weak'}).domain.evaluation_form == 'weak'

    @pytest.mark.parametrize('overrides, match', [
        ({'evaluation_form': 'integral'}, 'must be one of'),
        ({'evaluation_form': 'weak', 'evaluation_stride': 2}, 'evaluation_stride'),
        ({'evaluation_form': 'weak', 'instability_metric': 'vcoef'}, 'vcoef'),
    ])
    def test_refusals(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            load_search_config(None, overrides)

    def test_the_solver_is_refused(self):
        with pytest.raises(ValueError, match='use_solver'):
            check_evaluation_form('weak', None, True, 'dyad')

    def test_strong_refuses_nothing_new(self):
        check_evaluation_form('strong', 2, True, 'vcoef')


# --------------------------------------------------------------------------------------------- term mapping
def _factor(ftype, variable, power, evaluator, deriv_code=None, dim=None, label=None):
    names = ['power'] + (['dim'] if dim is not None else [])
    params = [float(power)] + ([float(dim)] if dim is not None else [])
    label = label or ftype
    return SimpleNamespace(ftype=ftype, variable=variable, evaluator=evaluator, deriv_code=deriv_code,
                           is_deriv=deriv_code is not None, label=label, structural_label=(label, tuple(params)),
                           params_description={i: {'name': n} for i, n in enumerate(names)},
                           params=np.array(params))


class TestWeakTerm:

    def test_data_derivative_and_grid_factors(self):
        term = SimpleNamespace(structure=[
            _factor('poly of u', 'u', 2, simple_function_evaluator, deriv_code=[None]),
            _factor('deriv of u', 'u', 1, simple_function_evaluator, deriv_code=[1, 1]),
            _factor('grids', 'grids', 2, None, dim=0, label='x_0')])
        wt, opaque = WR.weak_term(term, 2)
        assert wt.data == (('u', (0, 0)), ('u', (0, 0)), ('u', (0, 2)))
        assert wt.coords == (2, 0) and wt.opaque == () and opaque == {}

    def test_a_coordinate_function_is_opaque(self):
        cos = CustomEvaluator(lambda t, power: np.cos(t) ** power, ['power'], native_vectorized=True)
        trig = _factor('trigonometric', 'trigonometric', 1, cos, label='cos')
        wt, opaque = WR.weak_term(SimpleNamespace(structure=[trig]), 1)
        assert wt.opaque == (trig.structural_label,) and opaque == {trig.structural_label: trig}

    def test_a_data_token_without_full_grid_values_is_refused(self):
        stored = _factor('stored', 'stored', 1, simple_function_evaluator, label='w')
        with pytest.raises(ValueError, match='cannot evaluate token'):
            WR.weak_term(SimpleNamespace(structure=[stored]), 1)

    def test_a_stored_token_with_full_grid_values_is_opaque(self):
        f = _factor('stored', 'stored', 1, simple_function_evaluator, label='w')
        wt, factors = WR.weak_term(SimpleNamespace(structure=[f]), 1, stored={'w'})
        assert wt.opaque == (f.structural_label,) and factors == {f.structural_label: f}


def _stored_pool(labels=('w',)):
    family = SimpleNamespace(ftype='stored', variable='stored', _evaluator=simple_function_evaluator,
                             family_of_derivs=False, tokens=list(labels))
    return SimpleNamespace(families=[family])


class TestCheckFamilies:

    def test_a_stored_family_needs_its_tensors(self):
        with pytest.raises(ValueError, match=r"full-grid tensors.*\['w'\]"):
            WR.check_families(_stored_pool())
        with pytest.raises(ValueError, match='cannot evaluate the token family'):
            WR.check_families(_stored_pool(('w', 'v')), stored_labels={'w'})
        WR.check_families(_stored_pool(), stored_labels={'w'})


class TestColumns:

    def _stored_setup(self, monkeypatch, tensor):
        t = np.linspace(0, 20, 801)
        monkeypatch.setattr(global_var, 'samples_manager',
                            SimpleNamespace(grids=lambda: {0: [t]}, trajecatoryIDs=[0]), raising=False)
        fields = WF.Fields({'u': np.sin(t)}, [t])
        plan = WF.WindowPlan(1, (1,), (32,), (16,), (4,), (WF.lattice_shape([793], [32], [16])), 0.0)
        return fields, WR.WeakRows({0: fields}, {(0, 'u'): plan}, [0], tokens={0: {'w': tensor}})

    def test_a_stored_token_column_matches_the_coordinate_function(self, monkeypatch):
        t = np.linspace(0, 20, 801)
        fields, rows = self._stored_setup(monkeypatch, np.cos(t))
        cos = CustomEvaluator(lambda x, power: np.cos(x) ** power, ['power'], native_vectorized=True)
        for power in (1, 2):
            stored = SimpleNamespace(structure=[_factor('stored', 'stored', power, simple_function_evaluator,
                                                        label='w')], factors_labels=frozenset({f'w{power}'}))
            trig = SimpleNamespace(structure=[_factor('trigonometric', 'trigonometric', power, cos, label='cos')],
                                   factors_labels=frozenset({f'cos{power}'}))
            a, b = rows.columns(stored, 'u')[0], rows.columns(trig, 'u')[0]
            assert np.abs(b).max() > 0
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * np.abs(b).max())
        assert fields.opaque == {}

    def test_a_stored_tensor_of_the_wrong_shape_is_refused(self, monkeypatch):
        t = np.linspace(0, 20, 801)
        fields, rows = self._stored_setup(monkeypatch, np.cos(t[:-1]))
        stored = SimpleNamespace(structure=[_factor('stored', 'stored', 1, simple_function_evaluator, label='w')],
                                 factors_labels=frozenset({'w'}))
        with pytest.raises(ValueError, match="shape.*'w'|'w'.*shape"):
            rows.columns(stored, 'u')
        assert fields.opaque == {}

    def test_the_opaque_path_agrees_with_integration_by_parts(self, monkeypatch):
        """u = sin(t): the IBP column of u' (-u psi') and the pointwise column of cos(t) are one column."""
        t = np.linspace(0, 20, 801)
        monkeypatch.setattr(global_var, 'samples_manager',
                            SimpleNamespace(grids=lambda: {0: [t]}, trajecatoryIDs=[0]), raising=False)
        cos = CustomEvaluator(lambda x, power: np.cos(x) ** power, ['power'], native_vectorized=True)
        du = SimpleNamespace(structure=[_factor('deriv of u', 'u', 1, simple_function_evaluator, deriv_code=[0])],
                             factors_labels=frozenset({'du'}))
        trig = SimpleNamespace(structure=[_factor('trigonometric', 'trigonometric', 1, cos, label='cos')],
                               factors_labels=frozenset({'cos'}))
        fields = WF.Fields({'u': np.sin(t)}, [t])
        plan = WF.WindowPlan(1, (1,), (32,), (16,), (4,), (WF.lattice_shape([793], [32], [16])), 0.0)
        rows = WR.WeakRows({0: fields}, {(0, 'u'): plan}, [0])
        a, b = rows.columns(du, 'u')[0], rows.columns(trig, 'u')[0]
        assert a.shape == b.shape == (plan.rows,)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6 * np.abs(b).max())
        assert fields.opaque == {}                     # opaque values do not outlive the column
        assert rows.columns(du, 'u') is rows.columns(du, 'u')


# ------------------------------------------------------------------------------------------------------ fits
def _search(form, epochs=2, terms=3):
    np.random.seed(0)
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, population_size=6,
                             training_epochs=epochs, equation_terms_max_number=terms,
                             equation_factors_max_number=1, evaluation_form=form)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _oscillator(search, n=600, cache_id=0):
    t = np.linspace(0, 12, n)
    domain = search.createDomain(t, ID=cache_id)[1]
    return search.createTrajectory({'u': np.sin(2 * t) + 0.5 * np.cos(2 * t)}, domain, cache_id=cache_id)[1]


def _fit(search, data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.fit(data=data, **kwargs)


def _equations(search):
    return [eq for sol in search.optimizer.pareto_levels.population for eq in sol.vals]


@pytest.fixture
def weak_oscillator():
    search = _search('weak')
    try:
        _fit(search, [_oscillator(search)], max_deriv_order=(2,))
        yield search
    finally:
        global_var.samples_manager.reset()


class TestWeakFit:

    def test_recovers_the_oscillator(self, weak_oscillator):
        found = []
        for eq in _equations(weak_oscillator):
            names = {term.name for term in eq.structure}
            if names == {'d^2u/dx0^2{power: 1.0}', 'u{power: 1.0}'} and eq.weights_final[-1] == 0:
                tgt = eq.target_idx
                coef = eq.weights_final[eq.weight_index(1 - tgt, tgt)]
                found.append(coef if eq.structure[tgt].name.startswith('d^2u') else 1.0 / coef)
        assert found and all(abs(c + 4.0) < 1e-6 for c in found)

    def test_rows_are_the_slot_window(self, weak_oscillator):
        plan = global_var.samples_manager.weak.plans[(0, 'u')]
        eq = _equations(weak_oscillator)[0]
        target, features = eq.evaluate()
        assert target[0].shape == (plan.rows,)
        assert features is None or features[0].shape[0] == plan.rows
        np.testing.assert_array_equal(global_var.samples_manager.row_weights('u', 'dm')[0], np.ones(plan.rows))
        np.testing.assert_array_equal(global_var.samples_manager.row_shapes('u')[0], plan.shape)

    def test_strong_accessors_fail_loud(self, weak_oscillator):
        manager = global_var.samples_manager
        term = _equations(weak_oscillator)[0].structure[0]
        with pytest.raises(RuntimeError, match='without an equation slot'):
            term.evaluate()
        for read in (lambda: manager.gFunc('dm'), lambda: manager.gFunc('dmf'), lambda: manager.inner_shapes):
            with pytest.raises(RuntimeError, match='weak rows'):
                read()
        assert set(manager.gFunc('m')) == {0}          # the strong pool's masks stay readable

    def test_a_strong_fit_clears_the_weak_rows(self, weak_oscillator):
        search = _search('strong', epochs=0)
        _fit(search, [_oscillator(search)], max_deriv_order=(2,))
        assert global_var.samples_manager.weak is None


def test_every_trajectory_gets_its_own_window():
    search = _search('weak', epochs=0)
    try:
        _fit(search, [_oscillator(search, 600, 0), _oscillator(search, 300, 1)], max_deriv_order=(2,))
        plans = global_var.samples_manager.weak.plans
        assert set(plans) == {(0, 'u'), (1, 'u')}
        target, _ = _equations(search)[0].evaluate()
        assert {key: len(col) for key, col in target.items()} == {0: plans[(0, 'u')].rows,
                                                                 1: plans[(1, 'u')].rows}
    finally:
        global_var.samples_manager.reset()


def test_weak_heat_recovers_the_diffusivity():
    """u_t = 0.3 u_xx from two decaying modes: the weak fit's coefficient is exact to quadrature error."""
    search = _search('weak', epochs=2)
    t, x = np.linspace(0, 1, 64), np.linspace(0, 2 * np.pi, 96)
    T, X = np.meshgrid(t, x, indexing='ij')
    u = np.exp(-0.3 * T) * np.sin(X) + 0.5 * np.exp(-1.2 * T) * np.sin(2 * X)
    try:
        domain = search.createDomain([T, X], ID=0)[1]
        trajectory = search.createTrajectory({'u': u}, domain, cache_id=0)[1]
        _fit(search, [trajectory], max_deriv_order=(1, 2))
        coefs = []
        for eq in _equations(search):
            names = {term.name for term in eq.structure}
            if names == {'du/dx0{power: 1.0}', 'd^2u/dx1^2{power: 1.0}'} and eq.weights_final[-1] == 0:
                tgt = eq.target_idx
                c = eq.weights_final[eq.weight_index(1 - tgt, tgt)]
                coefs.append(c if eq.structure[tgt].name.startswith('du/dx0') else 1.0 / c)
        assert coefs and all(abs(c - 0.3) < 1e-3 for c in coefs)
    finally:
        global_var.samples_manager.reset()


def test_weak_fit_with_a_stored_token_family():
    """u' = 3 g with g supplied as a full-grid tensor: the stored token is an opaque pointwise factor of the
    weak rows, and the fit recovers the coefficient. The trajectory keeps the tensor by reference."""
    t = np.linspace(0, 12, 600)
    g = np.cos(2 * t) + 0.3 * np.sin(5 * t)
    u = 1.5 * np.sin(2 * t) - 0.18 * np.cos(5 * t)
    search = _search('weak', epochs=3)
    try:
        domain = search.createDomain(t, ID=0)[1]
        trajectory = search.createTrajectory({'u': u}, domain, cache_id=0)[1]
        family = epde.CacheStoredTokens(token_type='stored', token_labels=['g'], token_tensors={'g': g},
                                        params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                                        meaningful=True)
        _fit(search, [trajectory], max_deriv_order=(1,), additional_tokens=[family])
        coefs = []
        for eq in _equations(search):
            names = {term.name for term in eq.structure}
            if names == {'du/dx0{power: 1.0}', 'g{power: 1.0}'} and eq.weights_final[-1] == 0:
                tgt = eq.target_idx
                c = eq.weights_final[eq.weight_index(1 - tgt, tgt)]
                coefs.append(c if eq.structure[tgt].name.startswith('du/dx0') else 1.0 / c)
        assert coefs and all(abs(c - 3.0) < 1e-3 for c in coefs)

        stored = global_var.samples_manager[0]
        assert np.shares_memory(stored.raw_token('g'), g) and set(stored.raw_tokens()) == {'g'}
        with pytest.raises(KeyError, match="'missing'"):
            stored.raw_token('missing')
    finally:
        global_var.samples_manager.reset()
