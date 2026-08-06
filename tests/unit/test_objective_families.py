#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The three objective-family classes (``Discrepancy`` / ``Instability`` /
``Complexity``) and their wiring.

The ``Discrepancy`` grid is the fold-is-verbatim proof: each option's value
is asserted against the pre-fold formula written out explicitly on a fixed
synthetic equation mock. Mock conventions follow test_eq_mo_objectives /
test_instability_filler (SimpleNamespace, no pool).
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import epde.globals as global_var
from epde.operators.common.objectives import (Complexity, Discrepancy,
                                              FitContext, Instability)
from epde.eq_mo_objectives import (_complexity_of_equation, _terms_of_equation,
                                   complexity_deriv, equation_complexity,
                                   equation_complexity_by_factors,
                                   equation_complexity_by_terms)


def _ctx(for_rps=False):
    return FitContext(g_fun_vals=None, data_shape=(40,), penalty_coeff=0.2,
                      for_rps=for_rps)


# Fixed synthetic: y deliberately NOT an exact fit so every residual metric
# is nonzero. VWSR convention: weights_internal carries a trailing intercept
# PRESENCE flag; weights_final = [coefs..., intercept] when the flag is set.
_N = 40
_x = np.linspace(0.0, 2 * np.pi, _N)
_X = np.column_stack([np.sin(_x), np.cos(2 * _x)])
_COEFS = np.array([2.0, -1.0])
_INTERCEPT = 0.5
_Y = _X @ _COEFS + _INTERCEPT + 0.1 * np.sin(5 * _x)


def _vwsr_eq():
    return SimpleNamespace(
        evaluate=lambda normalize, return_val: (None, _Y, _X),
        weights_internal=np.array([2.0, -1.0, 1.0]),   # trailing intercept flag
        weights_final=np.array([2.0, -1.0, _INTERCEPT]),
        weights_final_evald=True)


class TestDiscrepancyOptions:

    def test_wape_matches_prefold_formula(self):
        val = Discrepancy('wape').compute(_vwsr_eq(), _ctx())
        resid = _Y - (_X @ _COEFS + _INTERCEPT)
        assert val == pytest.approx(np.sum(np.abs(resid)) / np.sum(np.abs(_Y)))

    def test_l2_relative_matches_prefold_formula(self):
        val = Discrepancy('l2_relative').compute(_vwsr_eq(), _ctx())
        resid = _Y - (_X @ _COEFS + _INTERCEPT)
        assert val == pytest.approx(np.linalg.norm(resid) / np.linalg.norm(_Y))

    def test_scale_invariant_matches_prefold_formula(self):
        val = Discrepancy('scale_invariant').compute(_vwsr_eq(), _ctx())
        contribs = _X * _COEFS[None, :]
        resid = _Y - contribs.sum(axis=1) - _INTERCEPT
        mass = np.abs(_Y) + np.abs(contribs).sum(axis=1) + abs(_INTERCEPT)
        assert val == pytest.approx(np.mean(np.abs(resid) / mass))

    def test_l2_matches_prefold_formula(self):
        # 'l2' is the LASSO-paired legacy core: weights_internal is
        # position-indexed with NO intercept slot; weights_final[-1] is read
        # as the intercept unconditionally.
        eq = SimpleNamespace(
            evaluate=lambda normalize, return_val: (None, _Y, _X),
            weights_internal=np.array([2.0, -1.0]),
            weights_final=np.array([2.0, -1.0, _INTERCEPT]),
            weights_final_evald=True)
        val = Discrepancy('l2').compute(eq, _ctx())
        resid = _X @ np.array([2.0, -1.0]) + _INTERCEPT - _Y
        assert val == pytest.approx(np.linalg.norm(resid, ord=2))

    def test_aliases_canonicalize_and_agree(self):
        assert Discrepancy('l2_scaled').metric == 'l2_relative'
        assert Discrepancy('l2_rel').metric == 'l2_relative'
        assert Discrepancy('residual').metric == 'l2_relative'
        assert Discrepancy('scale_inv').metric == 'scale_invariant'
        assert Discrepancy('sinv').metric == 'scale_invariant'
        assert Discrepancy('cancellation').metric == 'scale_invariant'
        a = Discrepancy('l2_scaled').compute(_vwsr_eq(), _ctx())
        b = Discrepancy('l2_relative').compute(_vwsr_eq(), _ctx())
        assert a == pytest.approx(b)

    def test_unknown_metric_raises(self):
        # The regression pin for the killed silent typo-to-WAPE catch-all.
        with pytest.raises(ValueError):
            Discrepancy('wapee')

    def test_is_degenerate_branches_per_option(self):
        # 'l2' keeps the legacy all-zero-weights-only test (NO threshold);
        # the other options apply the post-fit degenerate_threshold.
        eq = SimpleNamespace(weights_internal=np.array([1.0, 0.0, 0.0]),
                             fitness_value=5.0)   # way above threshold 1.0
        assert Discrepancy('l2').is_degenerate(eq) is False
        assert Discrepancy('wape').is_degenerate(eq) is True
        eq_zero = SimpleNamespace(weights_internal=np.zeros(3),
                                  fitness_value=0.1)
        assert Discrepancy('l2').is_degenerate(eq_zero) is True
        assert Discrepancy('wape').is_degenerate(eq_zero) is True


class TestSolverDiscrepancyOptions:
    """The solver-based options of the SAME Discrepancy family
    ('solver_l2' / 'pic' / 'deepxde'), computed through the
    compute(eq, eq_idx, sctx) protocol. Formula pins mirror the pre-fold
    SolverL2Discrepancy / PICError / DeepXDEError bodies."""

    @staticmethod
    def _tensor_cache(ref):
        import epde.globals as gv
        had = hasattr(gv, 'tensor_cache')
        prev = getattr(gv, 'tensor_cache', None)
        gv.tensor_cache = SimpleNamespace(get=lambda key: ref)
        return had, prev

    @staticmethod
    def _restore(had, prev):
        import epde.globals as gv
        if had:
            gv.tensor_cache = prev
        else:
            del gv.tensor_cache

    def test_solver_l2_matches_prefold_formula(self):
        from epde.operators.common.objectives import SolverContext
        rng = np.random.default_rng(3)
        n = 30
        sol = rng.normal(size=(n, 1))
        ref = rng.normal(size=n)
        g = rng.random(size=n)
        sctx = SolverContext(solution=sol, loss_add=0.5, g_fun_vals=g,
                             penalty_coeff=0.2, pinn_loss_mult=2.0)
        eq = SimpleNamespace(main_var_to_explain='u',
                             weights_final=np.array([1.0]))
        had, prev = self._tensor_cache(ref)
        try:
            val = Discrepancy('solver_l2').compute(eq, 0, sctx)
        finally:
            self._restore(had, prev)
        expected = np.linalg.norm((sol[..., 0] - ref) * g, ord=2) + 2.0 * 0.5
        assert val == pytest.approx(expected)

    def test_solver_l2_penalty_division_on_zero_weights(self):
        from epde.operators.common.objectives import SolverContext
        rng = np.random.default_rng(4)
        n = 30
        sol = rng.normal(size=(n, 1)); ref = rng.normal(size=n)
        g = np.ones(n)
        sctx = SolverContext(solution=sol, loss_add=0.0, g_fun_vals=g,
                             penalty_coeff=0.2, pinn_loss_mult=1.0)
        eq = SimpleNamespace(main_var_to_explain='u',
                             weights_final=np.zeros(2))
        had, prev = self._tensor_cache(ref)
        try:
            val = Discrepancy('solver_l2').compute(eq, 0, sctx)
        finally:
            self._restore(had, prev)
        assert val == pytest.approx(np.linalg.norm(sol[..., 0] - ref) / 0.2)

    def test_pic_matches_prefold_formula(self):
        from epde.operators.common.objectives import SolverContext
        rng = np.random.default_rng(5)
        n = 30
        sol = rng.normal(size=(n, 1)); ref = rng.normal(size=n)
        g = rng.random(size=n)
        sctx = SolverContext(solution=sol, loss_add=0.25, g_fun_vals=g,
                             penalty_coeff=0.2, pinn_loss_mult=3.0)
        eq = SimpleNamespace(main_var_to_explain='u',
                             weights_final=np.array([1.0]))
        had, prev = self._tensor_cache(ref)
        try:
            val = Discrepancy('pic').compute(eq, 0, sctx)
        finally:
            self._restore(had, prev)
        expected = np.mean(((sol[..., 0] - ref) * g) ** 2) + 3.0 * 0.25
        assert val == pytest.approx(expected)

    def test_deepxde_inner_metrics(self):
        from epde.operators.common.objectives import SolverContext
        rng = np.random.default_rng(6)
        masked_sol = [rng.normal(size=25)]
        masked_data = [rng.normal(size=25)]
        sctx = SolverContext(solution=masked_sol, loss_add=0.0,
                             g_fun_vals=masked_data,
                             penalty_coeff=0.2, pinn_loss_mult=0.0)
        eq = SimpleNamespace(weights_final=np.array([1.0]))
        d = masked_sol[0] - masked_data[0]
        f = Discrepancy('deepxde')                       # rmse default
        assert f.compute(eq, 0, sctx) == pytest.approx(np.sqrt(np.mean(d ** 2)))
        f.error_metric = 'l2'
        assert f.compute(eq, 0, sctx) == pytest.approx(np.linalg.norm(d, ord=2))
        f.error_metric = 'mae'
        assert f.compute(eq, 0, sctx) == pytest.approx(np.mean(np.abs(d)))

    def test_protocol_mismatch_fails_loudly(self):
        from epde.operators.common.objectives import SolverContext
        sctx = SolverContext(solution=np.zeros((5, 1)), loss_add=0.0,
                             g_fun_vals=np.ones(5), penalty_coeff=0.2,
                             pinn_loss_mult=0.0)
        # solver-free option driven through the solver protocol:
        with pytest.raises(TypeError):
            Discrepancy('wape').compute(_vwsr_eq(), 0, sctx)
        # solver option driven through the solver-free protocol:
        with pytest.raises(TypeError):
            Discrepancy('solver_l2').compute(_vwsr_eq(), _ctx())

    def test_solver_l2_distinct_from_solverfree_l2(self):
        assert Discrepancy('l2').metric == 'l2'
        assert Discrepancy('solver_l2').metric == 'solver_l2'


# ---- complexity family ---------------------------------------------------- #

def _fake_factor(deriv_code, power=1.0):
    return SimpleNamespace(deriv_code=deriv_code,
                           param=lambda name, _p=power: _p)


def _fake_term(*factors):
    return SimpleNamespace(structure=list(factors))


def _factors_eq():
    # 3 terms, target_idx=1; non-target weights [1., 0.] -> active: term0 +
    # target term; term2 inactive.
    t0 = _fake_term(_fake_factor([0]))                    # deriv order 1
    t1 = _fake_term(_fake_factor([None]))                 # target, non-deriv
    t2 = _fake_term(_fake_factor([0, 0]))                 # deriv order 2
    return SimpleNamespace(structure=[t0, t1, t2], target_idx=1,
                           weights_internal=np.array([1.0, 0.0]))


class TestComplexityFiller:

    def teardown_method(self):
        global_var.set_complexity_metric(None)

    def test_factors_matches_legacy_reader(self):
        eq = _factors_eq()
        system = SimpleNamespace(vals={'u': eq}, vars_to_describe=['u'])
        filler_val = Complexity('factors').compute(eq, _ctx())
        assert filler_val == pytest.approx(_complexity_of_equation(eq))
        assert filler_val == pytest.approx(
            equation_complexity_by_factors(system, 'u'))

    def test_terms_vwsr_counts_intercept_slot(self):
        # VWSR: weights_internal has the trailing intercept slot (len m+1).
        eq = SimpleNamespace(structure=[None] * 4,                # m = 3
                             weights_internal=np.array([1.0, 0.0, 2.0, 0.5]),
                             weights_final_evald=True,
                             weights_final=np.array([1.0, 2.0, 0.5]))
        assert _terms_of_equation(eq) == 3        # 2 active + intercept
        eq.weights_internal = np.array([1.0, 0.0, 2.0, 0.0])
        assert _terms_of_equation(eq) == 2        # zero intercept not counted

    def test_terms_lasso_reads_weights_final_intercept(self):
        # LASSO: no internal intercept slot (len m); intercept trails
        # weights_final.
        eq = SimpleNamespace(structure=[None] * 4,                # m = 3
                             weights_internal=np.array([1.0, 0.0, 2.0]),
                             weights_final_evald=True,
                             weights_final=np.array([1.0, 2.0, 0.7]))
        assert _terms_of_equation(eq) == 3
        eq.weights_final = np.array([1.0, 2.0, 0.0])
        assert _terms_of_equation(eq) == 2

    def test_terms_uniform_across_conventions(self):
        # The user-locked invariant: identical active support + intercept
        # state must count identically under both sparsity pairings.
        vwsr = SimpleNamespace(structure=[None] * 4,
                               weights_internal=np.array([1.0, 0.0, 2.0, 0.5]),
                               weights_final_evald=True,
                               weights_final=np.array([1.0, 2.0, 0.5]))
        lasso = SimpleNamespace(structure=[None] * 4,
                                weights_internal=np.array([1.0, 0.0, 2.0]),
                                weights_final_evald=True,
                                weights_final=np.array([1.0, 2.0, 0.7]))
        assert _terms_of_equation(vwsr) == _terms_of_equation(lasso) == 3

    def test_terms_degenerate_stamped_lasso_no_raise(self):
        # weights_final_evald False (the degenerate-stamped path): the
        # intercept contributes 0 instead of raising.
        eq = SimpleNamespace(structure=[None] * 4,
                             weights_internal=np.array([1.0, 0.0, 2.0]),
                             weights_final_evald=False)
        assert _terms_of_equation(eq) == 2

    def test_filler_default_follows_global(self):
        eq = SimpleNamespace(structure=[None] * 4,
                             weights_internal=np.array([1.0, 0.0, 2.0, 0.5]),
                             weights_final_evald=True,
                             weights_final=np.array([1.0, 2.0, 0.5]))
        global_var.set_complexity_metric('terms')
        assert Complexity().compute(eq, _ctx()) == pytest.approx(3.0)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            Complexity('bogus')

    def test_by_terms_reader_tuple_scalar_parity(self):
        eq = SimpleNamespace(structure=[None] * 4,
                             weights_internal=np.array([1.0, 0.0, 2.0, 0.5]),
                             weights_final_evald=True,
                             weights_final=np.array([1.0, 2.0, 0.5]))
        system = SimpleNamespace(vals={'u': eq}, vars_to_describe=['u'])
        assert equation_complexity_by_terms(system, 'u') == 3
        assert equation_complexity_by_terms(system) == (3,)


class TestStampContract:

    def test_stamped_on_failure_attributes(self):
        # The host's failure stamps (unfitted + degenerate-verdict loops in
        # SolverFreeFitness.apply) consult this flag; Complexity opts out of
        # both so its structure-derived value stays real and the reader's
        # lazy fallback reproduces the pre-filler legacy semantics.
        assert Discrepancy('wape').stamped_on_failure is True
        assert Instability().stamped_on_failure is True
        assert Complexity().stamped_on_failure is False

    @staticmethod
    def _host():
        # A SolverFreeFitness with the suboperator machinery stubbed out so
        # apply() reaches the stamp loops (the review found the attribute
        # pins alone let a guard inversion ship silently).
        from epde.operators.common.fitness import SolverFreeFitness
        disc = Discrepancy('wape')
        host = SolverFreeFitness(['penalty_coeff'],
                                 objectives=[disc, Complexity('terms')],
                                 primary=disc)
        host.params = {'penalty_coeff': 0.2}
        noop = SimpleNamespace(apply=lambda *a, **k: None)
        host._suboperators = {'sparsity': noop, 'coeff_calc': noop}
        host.parse_suboperator_args = (
            lambda arguments: ({}, {'sparsity': {}, 'coeff_calc': {}}))
        return host

    def test_unfitted_stamp_skips_complexity(self):
        # RPS-exhausted equation (weights_final never produced): fitness is
        # stamped LOSS_NAN_VAL, but the complexity flag STAYS DOWN so the
        # equation_complexity reader takes its lazy fallback -- the exact
        # pre-filler legacy semantics.
        from epde.operators.common.objectives import LOSS_NAN_VAL
        eq = SimpleNamespace(weights_internal_evald=True,
                             weights_final_evald=False,
                             remove_zero_terms=lambda: None,
                             complexity_calculated=False,
                             complexity_value=None)
        self._host().apply(eq, {})
        assert eq.fitness_value == LOSS_NAN_VAL
        assert eq.fitness_calculated is True
        assert eq.complexity_calculated is False      # NOT stamped
        assert eq.complexity_value is None            # NOT stamped

    def test_degenerate_stamp_keeps_complexity_real(self):
        # Degenerate-but-fitted equation: the verdict stamps fitness to
        # LOSS_NAN_VAL but the structure-derived complexity keeps its real
        # computed value (stamping it would change the legacy Pareto
        # geometry for degenerate forms).
        from epde.operators.common.objectives import LOSS_NAN_VAL
        eq = SimpleNamespace(
            evaluate=lambda normalize, return_val: (None, _Y, None),
            structure=[None] * 3,                      # m = 2
            weights_internal=np.array([0.0, 0.0, 1.0]),  # degenerate + flag
            weights_final=np.array([0.7]),
            weights_internal_evald=True,
            weights_final_evald=True,
            remove_zero_terms=lambda: None)
        self._host().apply(eq, {})
        assert eq.fitness_value == LOSS_NAN_VAL       # stamped
        assert eq.complexity_calculated is True
        assert eq.complexity_value == pytest.approx(1.0)   # real: intercept only


class TestStrategyConstruction:
    """Director-level construction over the (use_pic x second_objective)
    grid -- the exact gap that let the Complexity NameError ship: the filler
    tests imported Complexity directly and never assembled the strategy."""

    def teardown_method(self):
        global_var.set_second_objective(None)

    def test_moeadd_strategy_constructs_for_all_cells(self):
        from epde.optimizers.builder import StrategyBuilder
        from epde.optimizers.moeadd.strategy import MOEADDDirector
        from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser
        for second in (None, 'instability', 'complexity'):
            for use_pic in (True, False):
                global_var.set_second_objective(second)
                director = MOEADDDirector()
                director.builder = StrategyBuilder(MOEADDSectorProcesser)
                # Must not raise -- the unset/use_pic=False cell is the
                # default legacy pipeline (4 of 8 thesis labels).
                director.use_baseline(use_solver=False, use_pic=use_pic)


class TestEquationComplexityReader:

    def teardown_method(self):
        global_var.set_complexity_metric(None)

    def test_attr_path_wins_when_flag_set(self):
        eq = SimpleNamespace(complexity_calculated=True, complexity_value=7.5)
        system = SimpleNamespace(vals={'u': eq}, vars_to_describe=['u'])
        assert equation_complexity(system, 'u') == 7.5
        assert equation_complexity(system) == (7.5,)

    def test_lazy_fallback_routes_on_global(self):
        eq = _factors_eq()
        eq.complexity_calculated = False
        eq.weights_final_evald = True
        eq.weights_final = np.array([1.0, 0.0])
        system = SimpleNamespace(vals={'u': eq}, vars_to_describe=['u'])
        assert equation_complexity(system, 'u') == pytest.approx(
            _complexity_of_equation(eq))
        global_var.set_complexity_metric('terms')
        assert equation_complexity(system, 'u') == pytest.approx(
            _terms_of_equation(eq))


class TestComplexityDerivQuirk:

    def test_last_factor_power_multiplies_whole_term(self):
        # PINNED legacy quirk: ``factor`` leaks from the loop, so the summed
        # total is multiplied by the LAST factor's power only. Non-deriv
        # (0.5, power 2) + deriv order 1 (1.0, power 3) -> (0.5+1.0)*3 = 4.5,
        # NOT 0.5*2 + 1.0*3 = 4.0. An accidental "fix" must trip this test
        # and force the deliberate bit-compat conversation.
        factors = [_fake_factor([None], power=2.0), _fake_factor([0], power=3.0)]
        assert complexity_deriv(factors) == pytest.approx(4.5)


class TestRoutingAndSignatures:

    def teardown_method(self):
        global_var.set_second_objective(None)

    def test_use_default_multiobjective_function_routing(self):
        from epde.structure.main_structures import SoEq
        calls = []
        rec = SimpleNamespace(
            use_new_multiobjective_function=lambda: calls.append('new'),
            use_legacy_multiobjective_function=lambda: calls.append('legacy'))
        for second, use_pic, expect in [
                (None, True, 'new'), (None, False, 'legacy'),
                ('instability', False, 'new'), ('complexity', True, 'legacy')]:
            global_var.set_second_objective(second)
            calls.clear()
            SoEq.use_default_multiobjective_function(rec, use_pic=use_pic)
            assert calls == [expect], (second, use_pic)

    def test_baseline_director_signature_config_driven(self):
        # sparsity_cls is honored as a named param; the discrepancy metric
        # deliberately is NOT a director parameter -- objectives build from
        # the search configuration (globals.discrepancy_metric, written by
        # EpdeSearch.__init__), never from threaded metric strings.
        from epde.optimizers.single_criterion.strategy import BaselineDirector
        params = inspect.signature(BaselineDirector.use_baseline).parameters
        assert 'sparsity_cls' in params
        assert 'discrepancy_metric' not in params


class TestConfigDrivenDiscrepancy:
    """``Discrepancy()`` builds bare and resolves its option from the
    search-level configuration at compute time."""

    def teardown_method(self):
        global_var.set_discrepancy_metric(None)

    def test_bare_instance_defaults_to_wape(self):
        global_var.set_discrepancy_metric(None)
        bare = Discrepancy().compute(_vwsr_eq(), _ctx())
        assert bare == pytest.approx(Discrepancy('wape').compute(_vwsr_eq(), _ctx()))

    def test_bare_instance_follows_configuration(self):
        global_var.set_discrepancy_metric('l2_scaled')   # alias -> l2_relative
        bare = Discrepancy().compute(_vwsr_eq(), _ctx())
        assert bare == pytest.approx(
            Discrepancy('l2_relative').compute(_vwsr_eq(), _ctx()))

    def test_setter_normalizes_aliases_and_validates(self):
        global_var.set_discrepancy_metric('scale_inv')
        assert global_var.discrepancy_metric == 'scale_invariant'
        assert global_var.resolve_discrepancy_metric() == 'scale_invariant'
        with pytest.raises(ValueError):
            global_var.set_discrepancy_metric('wapee')
        # solver options are host-internal, not search configuration:
        with pytest.raises(ValueError):
            global_var.set_discrepancy_metric('solver_l2')

    def test_explicit_internal_wiring_beats_configuration(self):
        # Fixed-role hosts (RPS lightweight 'l2_relative') keep their metric
        # regardless of the search-level setting.
        global_var.set_discrepancy_metric('l2')
        wired = Discrepancy('l2_relative').compute(_vwsr_eq(), _ctx())
        assert wired == pytest.approx(
            Discrepancy('l2_relative').compute(_vwsr_eq(), _ctx()))
        assert Discrepancy('l2_relative')._resolved_metric() == 'l2_relative'


class TestIdealPoint:

    def teardown_method(self):
        global_var.set_second_objective(None)
        global_var.set_complexity_metric(None)

    def test_ideal_point_grid(self):
        # instability -> [0,0]; complexity+'factors' -> [0,1] (bit-compat);
        # complexity+'terms' -> [0,0] (a terms count can reach 0, and a
        # utopia above an achievable value breaks the PBI assumption).
        from epde.interface.interface import EpdeSearch
        rip = EpdeSearch._resolve_ideal_point
        assert rip(True) == [0., 0.]              # unset -> instability
        assert rip(False) == [0., 1.]             # unset -> factors complexity
        global_var.set_second_objective('instability')
        assert rip(False) == [0., 0.]
        global_var.set_second_objective('complexity')
        assert rip(True) == [0., 1.]
        global_var.set_complexity_metric('terms')
        assert rip(True) == [0., 0.]


class TestSolverBranchSecondObjective:
    """use_pic ONLY selects the second objective (instability in place of the
    baseline complexity). The solver backend -- and the backend-implied
    primary discrepancy option -- is chosen by ``solver_backend``, fully
    decoupled from use_pic. The second-objective filler rides the uniform
    ``objectives`` list of the host (no separate stability channel)."""

    def teardown_method(self):
        global_var.set_second_objective(None)

    def _built_fitness(self, monkeypatch, use_pic, backend):
        import epde.optimizers.moeadd.strategy as strategy_mod
        from epde.optimizers.builder import StrategyBuilder
        from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser
        from epde.operators.common.fitness import SolverBasedFitness
        created = []

        class _Recorder(SolverBasedFitness):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

        monkeypatch.setattr(strategy_mod, 'SolverBasedFitness', _Recorder)
        director = strategy_mod.MOEADDDirector()
        director.builder = StrategyBuilder(MOEADDSectorProcesser)
        director.use_baseline(use_solver=True, use_pic=use_pic,
                              solver_backend=backend)
        assert len(created) == 1
        return created[0]

    def test_backend_and_second_axis_fully_decoupled(self, monkeypatch):
        expected_metric = {'autograd': 'solver_l2', 'deepxde': 'deepxde'}
        for backend in ('autograd', 'deepxde'):
            for second in (None, 'instability', 'complexity'):
                for use_pic in (True, False):
                    global_var.set_second_objective(second)
                    fit = self._built_fitness(monkeypatch, use_pic, backend)
                    cell = (backend, second, use_pic)
                    # Backend + primary metric never depend on use_pic.
                    assert fit.backend == backend, cell
                    assert fit.primary.metric == expected_metric[backend], cell
                    resolved = second if second is not None else (
                        'instability' if use_pic else 'complexity')
                    non_primary = [f for f in fit.objectives
                                   if f is not fit.primary]
                    if resolved == 'instability':
                        assert len(non_primary) == 1, cell
                        assert isinstance(non_primary[0], Instability), cell
                    else:
                        # complexity axis: lazy reader, no filler
                        assert non_primary == [], cell

    def test_unknown_backend_raises(self):
        from epde.optimizers.builder import StrategyBuilder
        from epde.optimizers.moeadd.strategy import MOEADDDirector
        from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser
        director = MOEADDDirector()
        director.builder = StrategyBuilder(MOEADDSectorProcesser)
        with pytest.raises(ValueError, match='solver_backend'):
            director.use_baseline(use_solver=True, solver_backend='foo')

    def test_instability_kwarg_folds_into_objectives(self):
        # The legacy ``instability=`` convenience channel appends the filler
        # to the uniform objectives list exactly once.
        from epde.operators.common.fitness import SolverBasedFitness
        disc = Discrepancy('solver_l2')
        inst = Instability()
        host = SolverBasedFitness(['penalty_coeff', 'pinn_loss_mult'],
                                  objectives=[disc], primary=disc,
                                  instability=inst, backend='autograd')
        assert host.objectives == [disc, inst]
        host2 = SolverBasedFitness(['penalty_coeff', 'pinn_loss_mult'],
                                   objectives=[disc, inst], primary=disc,
                                   instability=inst, backend='autograd')
        assert host2.objectives == [disc, inst]        # no duplicate


class TestDirectorParamsWiring:
    """The bundled ``params=`` dict (EpdeSearch's director_params) must reach
    the same operator constructors as the individually named ``*_params`` --
    it used to land in ``**kwargs['params']`` and was silently inert."""

    def test_bundle_matches_named_kwargs_and_wins(self, monkeypatch):
        import epde.optimizers.moeadd.strategy as strategy_mod
        from epde.optimizers.builder import StrategyBuilder
        from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser
        seen = []
        real = strategy_mod.get_basic_variation
        monkeypatch.setattr(strategy_mod, 'get_basic_variation',
                            lambda variation_params={}:
                                (seen.append(variation_params), real({}))[1])

        def build(**kw):
            director = strategy_mod.MOEADDDirector()
            director.builder = StrategyBuilder(MOEADDSectorProcesser)
            director.use_baseline(use_solver=False, use_pic=False, **kw)

        marker = {'marker': 'bundled'}
        build(params={'variation_params': marker})
        build(variation_params=marker)
        build(params={'variation_params': marker},
              variation_params={'marker': 'named'})
        assert seen == [marker, marker, marker]


class TestInterfaceGuards:

    def teardown_method(self):
        from epde.operators.utils.default_parameter_loader import EvolutionaryParams
        EvolutionaryParams.reset()

    def test_single_objective_use_solver_raises(self):
        from epde.interface.interface import EpdeSearch
        with pytest.raises(ValueError, match='solver'):
            EpdeSearch(multiobjective_mode=False, use_solver=True,
                       define_domain=False)

    def test_unknown_solver_backend_raises(self):
        from epde.interface.interface import EpdeSearch
        with pytest.raises(ValueError, match='solver_backend'):
            EpdeSearch(define_domain=False, solver_backend='foo')

    def test_single_objective_constructs_without_solver(self):
        from epde.interface.interface import EpdeSearch
        obj = EpdeSearch(multiobjective_mode=False, define_domain=False)
        assert obj.multiobjective_mode is False


class TestMultisampleUsePicForwarding:
    """EpdeMultisample.fit used to drop use_pic on both channels
    (population_instruct + _create_optimizer), so a PIC multisample search
    aimed at the complexity utopia [0., 1.] while its fillers computed
    instability. Both channels must carry the flag."""

    def test_fit_forwards_use_pic_to_both_channels(self):
        from epde.interface.interface import EpdeMultisample
        captured = {}

        def fake_create(mo_mode, init_params, director, population=None,
                        use_pic=False):
            captured['use_pic'] = use_pic
            captured['instruct'] = init_params['population_instruct']
            return SimpleNamespace(optimize=lambda **kw: None)

        fake = SimpleNamespace(_use_pic=True, multiobjective_mode=True,
                               director='DIR', optimizer_init_params={},
                               optimizer_exec_params={},
                               _create_optimizer=fake_create)
        EpdeMultisample.fit(fake, samples=None, pool='POOL')
        assert captured['use_pic'] is True
        assert captured['instruct']['use_pic'] is True
        assert fake.search_conducted is True


class TestSolverPoolWarning:
    """use_solver=True without a pretrained data_nn used to silently skip the
    data-representation ANN setup (the training branch is commented out)."""

    def teardown_method(self):
        from epde.operators.utils.default_parameter_loader import EvolutionaryParams
        EvolutionaryParams.reset()

    def test_create_pool_warns_without_data_nn(self):
        from epde.interface.interface import EpdeSearch
        t = np.linspace(0., 4 * np.pi, 60)
        u = np.sin(t)
        obj = EpdeSearch(use_solver=True, coordinate_tensors=[t], boundary=0,
                         verbose_params={'show_iter_idx': False})
        obj.set_preprocessor()
        with pytest.warns(UserWarning, match='data_nn'):
            obj.create_pool(data=u, variable_names=['u'], max_deriv_order=2)


class TestAmplifiedFitCondemnation:
    """In-place fitness backstop: a fit whose amplification ratio
    A = sum|c|*||col|| / ||target|| exceeds rps_amplification_cap is stamped
    LOSS_NAN_VAL even though its discrepancy is excellent -- the guard-clean
    RPS sweep can be bypassed by the exit-guarantee fallback, so the
    condemnation must not depend on how the weights arose."""

    def teardown_method(self):
        global_var.set_rps_amplification_cap(100.)

    @staticmethod
    def _host():
        from epde.operators.common.fitness import SolverFreeFitness
        disc = Discrepancy('wape')
        host = SolverFreeFitness(['penalty_coeff'], objectives=[disc],
                                 primary=disc)
        host.params = {'penalty_coeff': 0.2}
        noop = SimpleNamespace(apply=lambda *a, **k: None)
        host._suboperators = {'sparsity': noop, 'coeff_calc': noop}
        host.parse_suboperator_args = (
            lambda arguments: ({}, {'sparsity': {}, 'coeff_calc': {}}))
        return host

    @staticmethod
    def _term(col):
        return SimpleNamespace(evaluate=lambda normalize, grids=None, _c=col: _c)

    @classmethod
    def _amplified_eq(cls):
        # Near-null combination: colA - colB = 1e-6*cos(x); coefficients 1e6
        # reproduce the target EXACTLY (WAPE ~ 0, so the plain degeneracy
        # threshold cannot catch it) while A ~ 2e6 >> cap.
        base = np.sin(_x)
        eps = 1e-6 * np.cos(_x)
        col_a, col_b = base + eps, base
        t = 1e6 * eps
        X = np.column_stack([col_a, col_b])
        # VWSR reader convention: weights_internal trailing slot is the
        # intercept-PRESENCE flag (0 -> weights_final carries no intercept);
        # _amplification_ratio reads the same trailing 0.0 as a zero
        # intercept coefficient -- both consistent here.
        return SimpleNamespace(
            evaluate=lambda normalize, return_val: (None, t, X),
            structure=[cls._term(col_a), cls._term(col_b), cls._term(t)],
            target_idx=2,
            weights_internal=np.array([1e6, -1e6, 0.0]),
            weights_final=np.array([1e6, -1e6]),
            weights_internal_evald=True, weights_final_evald=True,
            remove_zero_terms=lambda: None)

    @classmethod
    def _modest_eq(cls):
        col_a, col_b = np.sin(_x), np.cos(_x)
        t = 2.0 * col_a - 1.0 * col_b
        X = np.column_stack([col_a, col_b])
        return SimpleNamespace(
            evaluate=lambda normalize, return_val: (None, t, X),
            structure=[cls._term(col_a), cls._term(col_b), cls._term(t)],
            target_idx=2,
            weights_internal=np.array([2.0, -1.0, 0.0]),
            weights_final=np.array([2.0, -1.0]),
            weights_internal_evald=True, weights_final_evald=True,
            remove_zero_terms=lambda: None)

    def test_amplified_fit_is_condemned(self):
        from epde.operators.common.objectives import LOSS_NAN_VAL
        eq = self._amplified_eq()
        self._host().apply(eq, {})
        assert eq.fitness_value == LOSS_NAN_VAL

    def test_amplified_sweep_fit_returns_none(self):
        # Out-of-place (RPS term-sweep) unified decline semantics: an
        # amplified fit returns None exactly like the all-features-zeroed
        # sparsity case, so the sweep never considers the candidate at all.
        eq = self._amplified_eq()
        assert self._host().apply(eq, {}, force_out_of_place=True) is None

    def test_modest_sweep_fit_returns_value(self):
        eq = self._modest_eq()
        val = self._host().apply(eq, {}, force_out_of_place=True)
        assert val is not None and val < 1.0

    def test_modest_fit_is_kept(self):
        from epde.operators.common.objectives import LOSS_NAN_VAL
        eq = self._modest_eq()
        self._host().apply(eq, {})
        assert eq.fitness_calculated is True
        assert eq.fitness_value < 1.0 and eq.fitness_value != LOSS_NAN_VAL

    def test_cap_none_disables_backstop(self):
        from epde.operators.common.objectives import LOSS_NAN_VAL
        global_var.set_rps_amplification_cap(None)
        eq = self._amplified_eq()
        self._host().apply(eq, {})
        assert eq.fitness_value != LOSS_NAN_VAL
