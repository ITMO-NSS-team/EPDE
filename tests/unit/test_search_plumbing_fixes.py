#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search-plumbing correctness fixes: the RPS pre-check, the published
configuration, and the keep-rule's sweep order.

* **Stale pre-check (F1).** ``SolverFreeFitness.apply`` asked
  ``primary.is_degenerate`` BEFORE computing the candidate's discrepancy, and
  that test also compares ``fitness_value`` against the degeneracy threshold.
  Inside the RPS term-sweep ``fitness_value`` still held the PREVIOUS target's
  discrepancy, so a target that fits exactly was declined on its predecessor's
  WAPE -- and with no target left the equation was rerolled.
* **Fit-time config (D9).** ``EpdeSearch`` published its configuration only in
  the constructor, so ``fit(equation_terms_max_number=N)`` reached the
  population but not ``active_config()``, which is where the tile/het
  partition width is read (``survival._reference_width``).
* **Tie order (B15).** ``PhysicsInformedLasso.fit`` visited columns in
  ``np.argsort(scores)[::-1]`` order. ``argsort``'s default kind is not
  stable, so exact score ties (het's unresolved sentinel, het_raw's overflow)
  were visited in an implementation-defined order, and when killing one tied
  column rescues the other the support depended on the library's column
  order.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.search_config import active_config


# --------------------------------------------------------------------------- #
#  F1 -- the RPS pre-check must not read another target's discrepancy         #
# --------------------------------------------------------------------------- #
U_T = 'du/dx0{power: 1.0}'
U_TT = 'd^2u/dx0^2{power: 1.0}'
U = 'u{power: 1.0}'


@pytest.fixture(scope='module')
def sine_pool():
    """``u = sin t``: ``u_tt = -u`` holds to finite-difference accuracy, while
    ``u_t = cos t`` is orthogonal to ``u``."""
    search = epde.EpdeSearch(use_solver=False,
                             verbose_params={'show_iter_idx': False})
    t = np.linspace(0.0, 4 * np.pi, 200)
    _, domain = search.createDomain(t, boundary_width=5, ID=0)
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    _, trajectory = search.createTrajectory({'u': np.sin(t)}, domain,
                                            cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(2,),
                       data_fun_pow=1)
    return search.pool


def _sweep_equation(pool):
    """Structure ``[u_t, u, u_tt]``: the sweep visits ``u_t`` first, then
    ``u_tt`` (``u`` carries no derivative and is skipped)."""
    from epde.interface.equation_translator import translate_equation
    eq = translate_equation(f'0.0 * {U_T} + -1.0 * {U} + 0.0 = {U_TT}', pool,
                            all_vars=['u']).vals['u']
    eq.main_var_to_explain = 'u'
    return eq


#: The support decision per candidate target, by feature name. ``u_t`` gets a
#: POOR fit (``u_t ~ u``: WAPE ~ sqrt(2), over the threshold of 1); ``u_tt``
#: gets its exact law ``u_tt = -u``.
_SUPPORT = {U_T: {U: 1.0}, U_TT: {U: -1.0}}


def _fixed_support_operator():
    """A sparsity step that installs ``_SUPPORT`` for whatever target is
    current, zeros for anything else -- the decline ``SolverFreeFitness``
    makes for an all-zero support. It declares ``fits_physical_scale`` so
    ``LinRegBasedCoeffsEquation`` promotes the vector unchanged."""
    from epde.operators.utils.template import CompoundOperator

    class FixedSupport(CompoundOperator):
        fits_physical_scale = True

        def apply(self, objective, arguments):
            coefs = _SUPPORT.get(objective.target.name, {})
            objective.weights_internal = np.array(
                [coefs.get(term.name, 0.0)
                 for idx, term in enumerate(objective.structure)
                 if idx != objective.target_idx] + [0.0])
            objective.weights_internal_evald = True

        def use_default_tags(self):
            self._tags = {'sparsity', 'gene level', 'no suboperators', 'inplace'}

    return FixedSupport()


def _selector():
    from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
    from epde.operators.common.fitness import SolverFreeFitness
    from epde.operators.common.objectives import Discrepancy, Instability
    from epde.operators.common.right_part_selection import EqRightPartSelector

    discrepancy = Discrepancy()
    host = SolverFreeFitness(['penalty_coeff'],
                             objectives=[discrepancy, Instability()],
                             primary=discrepancy)
    host.params = {'penalty_coeff': 0.2}
    selector = EqRightPartSelector()
    selector.set_suboperators({'fitness_calculation': host,
                               'sparsity': _fixed_support_operator(),
                               'coeff_calc': LinRegBasedCoeffsEquation()})
    return selector


class TestRPSPrecheckIsNotStale:

    def test_an_exact_target_after_a_poor_one_is_accepted(self, sine_pool):
        """The sweep body, step by step: score ``u_t`` (declined on its own,
        freshly computed WAPE), clear the fit flags exactly as the sweep does,
        then score ``u_tt``. ``u_tt = -u`` is exact, so it must come back
        finite -- the old pre-check returned None on ``u_t``'s WAPE."""
        eq = _sweep_equation(sine_pool)
        selector = _selector()
        _, subop_args = selector.parse_suboperator_args(arguments={})

        eq.target_idx = 0
        assert eq.target.name == U_T
        assert selector._fit_and_score(eq, subop_args) is None
        assert eq.fitness_value > 1.0          # declined on its OWN value
        eq.weights_internal_evald = eq.weights_final_evald = False

        eq.target_idx = 2
        assert eq.target.name == U_TT
        fit = selector._fit_and_score(eq, subop_args)
        assert fit is not None
        assert fit < 1e-2
        assert eq.fitness_value == fit

    def test_the_sweep_selects_the_exact_target(self, sine_pool):
        """End to end: with the stale decline the sweep found no target,
        rerolled the equation and never returned to ``u_tt = -u``."""
        eq = _sweep_equation(sine_pool)
        _selector().apply(eq, arguments={})
        assert eq.target.name == U_TT
        assert [term.name for term in eq.structure] == [U, U_TT]
        assert eq.weights_final[0] == pytest.approx(-1.0)

    def test_an_all_zero_support_is_still_declined_before_scoring(
            self, sine_pool, monkeypatch):
        """The pre-check keeps its structural half: an all-zero support is
        declined without computing a discrepancy for it."""
        from epde.operators.common.objectives import Discrepancy

        eq = _sweep_equation(sine_pool)
        selector = _selector()
        _, subop_args = selector.parse_suboperator_args(arguments={})
        calls = []
        original = Discrepancy.compute
        monkeypatch.setattr(Discrepancy, 'compute',
                            lambda self, *a: calls.append(1) or original(self, *a))
        monkeypatch.setitem(_SUPPORT, U_TT, {})
        eq.target_idx = 2
        assert selector._fit_and_score(eq, subop_args) is None
        assert calls == []

    def test_the_precheck_reads_no_fitness_value(self):
        """Source pin: the only ``is_degenerate`` consults in the
        out-of-place branch come after ``primary.compute``."""
        import inspect
        from epde.operators.common.fitness import SolverFreeFitness

        source = inspect.getsource(SolverFreeFitness.apply)
        code = '\n'.join(line.split('#')[0] for line in source.splitlines())
        first_compute = code.index('primary.compute(')
        assert 'is_degenerate' not in code[:first_compute]


# --------------------------------------------------------------------------- #
#  D9 -- fit publishes the configuration it runs under                        #
# --------------------------------------------------------------------------- #
def _search(**kwargs):
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, **kwargs)
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    return search


def _trajectory(search, cache_id):
    grid = np.linspace(0, 4 * np.pi, 60)
    domain = search.createDomain(grid, boundary_width=5, ID=cache_id)[1]
    return search.createTrajectory({'u': np.sin(grid)}, domain,
                                   cache_id=cache_id)[1]


def _fit_and_capture(monkeypatch, search, **fit_kwargs):
    """Run ``fit`` up to the optimizer and report what the operators would
    read from the active configuration at that point."""
    from epde.operators.common.survival import _reference_width
    captured = {}

    def recorder(self, terms, factors, interval, optimizer=None,
                 population=None):
        space = active_config().search_space
        captured.update(population_terms=terms, population_factors=factors,
                        terms=space.equation_terms_max_number,
                        factors=space.equation_factors_max_number,
                        reference_width=_reference_width(2))

    monkeypatch.setattr(type(search), '_run_optimization', recorder)
    search.fit(data=[_trajectory(search, 71)], max_deriv_order=(2,),
               **fit_kwargs)
    return captured


class TestFitPublishesItsConfig:

    def test_the_fit_time_term_cap_reaches_the_estimators(self, monkeypatch):
        search = _search()
        assert search.config.search_space.equation_terms_max_number == 6
        seen = _fit_and_capture(monkeypatch, search,
                                equation_terms_max_number=9)
        assert seen['population_terms'] == 9
        assert seen['terms'] == 9
        assert seen['reference_width'] == 9
        # ... and it stays the active value once fit has returned.
        assert active_config().search_space.equation_terms_max_number == 9

    def test_the_other_search_space_values_follow(self, monkeypatch):
        """Every config-backed fit argument is published, not just the one a
        consumer happens to read today."""
        search = _search()
        seen = _fit_and_capture(monkeypatch, search,
                                equation_factors_max_number=2,
                                data_fun_pow=2)
        assert seen['population_factors'] == 2
        assert seen['factors'] == 2
        cfg = active_config()
        assert cfg.search_space.data_fun_pow == 2
        assert cfg.preprocessing.max_deriv_order == (2,)

    def test_the_search_keeps_its_own_config(self, monkeypatch):
        """A fit argument is for that fit: the instance config is untouched,
        so the next fit without it is back on the constructor's value."""
        search = _search()
        _fit_and_capture(monkeypatch, search, equation_terms_max_number=9)
        assert search.config.search_space.equation_terms_max_number == 6
        seen = _fit_and_capture(monkeypatch, search)
        assert seen['terms'] == 6
        assert seen['reference_width'] == 6

    def test_fit_republishes_over_a_later_constructed_search(self, monkeypatch):
        """The constructor-only publish made the LAST search constructed win,
        even for a fit on an earlier one."""
        first = _search(equation_terms_max_number=9)
        _search()
        assert active_config().search_space.equation_terms_max_number == 6
        seen = _fit_and_capture(monkeypatch, first)
        assert seen['terms'] == 9


# --------------------------------------------------------------------------- #
#  B15 -- exact keep-rule ties are visited in a deterministic order           #
# --------------------------------------------------------------------------- #
_N = 200
_TIE = 1.0       # het's unresolved sentinel


def _tie_library():
    """``y = 2u + a`` over ``[u, a, b, w]``, ``a``/``b`` a near-exact
    duplicate pair: under ``_tie_scores`` both members score ``_TIE`` while
    the pair is present, and whichever is visited first is killed -- after
    which the survivor is no longer a duplicate and keeps its place."""
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 1.0, _N)
    m = np.sin(6 * np.pi * t) * (0.5 + t)
    e1, e2 = rng.normal(size=(2, _N))
    columns = {'u': np.cos(2 * np.pi * t) + 0.2 * t,
               'a': m + 1e-5 * e1,
               'b': m + 1e-5 * e2,
               'w': np.sin(10 * np.pi * t)}
    target = 2.0 * columns['u'] + columns['a']
    return columns, target


def _tie_scores(metric, X, y, sw, grid_shape, active_mask, n_features, **kw):
    """A content-equivariant keep-rule score: a column with a near-exact
    duplicate among the active ones is 'unresolved' (``_TIE``), everything
    else -- the intercept included -- scores 1e-3. Permuting the columns
    permutes the scores, so any change of support under a permutation comes
    from the sweep order alone."""
    cols = np.where(active_mask)[0]
    feats = [c for c in cols if c < n_features]
    out = np.full(cols.size, 1e-3)
    for i, c in enumerate(cols):
        if c < n_features and any(
                d != c and abs(np.corrcoef(X[:, c], X[:, d])[0, 1]) > 0.999
                for d in feats):
            out[i] = _TIE
    return out


class _Term:
    def __init__(self, name):
        self.name = name
        self.factors_labels = frozenset({(name, (1.0,))})


def _vwsr_support(order, monkeypatch):
    """Run ``VWSRSparsity.apply`` on an equation whose structure lists the
    library in ``order`` (the target last) and return the surviving feature
    names. The super-Gram slot carries the evaluated columns, as the RPS
    sweep would leave it."""
    import epde.operators.common.sparsity as sparsity
    columns, target = _tie_library()
    monkeypatch.setattr(sparsity, 'instability_scores', _tie_scores)
    monkeypatch.setattr(global_var, 'samples_manager', SimpleNamespace(
        gFunc=lambda mode: {0: np.ones(_N)}, inner_shapes={0: (_N,)}),
        raising=False)
    terms = [_Term(name) for name in order] + [_Term('y')]
    Z = np.column_stack([columns[name] for name in order] + [target])
    eq = SimpleNamespace(structure=terms, target_idx=len(order),
                         main_var_to_explain='u',
                         _gram_super={'mode': None, 'Z': {0: Z}})
    sparsity.VWSRSparsity([]).apply(eq, {})
    return frozenset(name for name, w in zip(order, eq.weights_internal[:-1])
                     if w != 0)


class TestKeepRuleTieOrder:

    def test_the_support_is_free_of_the_library_column_order(self, monkeypatch):
        """All 24 orders of the library give ONE support. The old unstable
        argsort kept ``a`` in some orders and ``b`` in others."""
        supports = {_vwsr_support(order, monkeypatch)
                    for order in itertools.permutations('uabw')}
        assert len(supports) == 1, supports

    def test_ties_are_broken_by_term_label(self, monkeypatch):
        """``a`` sorts before ``b``, so it is visited first and is the member
        killed; the label decides, not the column position."""
        for order in itertools.permutations('uabw'):
            assert _vwsr_support(order, monkeypatch) == {'u', 'b'}, order

    def test_without_labels_ties_go_by_column_position(self, monkeypatch):
        """The bare estimator has no labels; ties are then visited in column
        order, so the EARLIER member of the pair is the one killed."""
        import epde.operators.common.sparsity as sparsity
        columns, target = _tie_library()
        monkeypatch.setattr(sparsity, 'instability_scores', _tie_scores)
        for order in itertools.permutations('uabw'):
            X = np.column_stack([columns[name] for name in order])
            estimator = sparsity.PhysicsInformedLasso().fit(
                X, target, grid_shape=(_N,), sample_weights=np.ones(_N))
            kept = {name for name, w in zip(order, estimator.coef_) if w != 0}
            later = max('ab', key=order.index)
            assert kept == {'u', later}, order

    def test_labels_must_match_the_columns(self):
        from epde.operators.common.sparsity import PhysicsInformedLasso
        columns, target = _tie_library()
        X = np.column_stack([columns[name] for name in 'uabw'])
        with pytest.raises(ValueError, match='feature_labels'):
            PhysicsInformedLasso().fit(X, target, grid_shape=(_N,),
                                       feature_labels=['u', 'a'])
