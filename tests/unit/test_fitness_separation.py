#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The fitness hosts score; ``EqRightPartSelector`` fits and prunes.

The hosts used to own ``sparsity`` and ``coeff_calc`` and to end their
in-place pass with ``objective.remove_zero_terms()``. Two consequences:

* the in-place pass could silently re-sparsify an already-fitted equation
  (the ``needs_sparsity`` fallback, ``for_rps or not weights_internal_evald``)
  and then prune it, so an equation could come out of "scoring" with a
  different structure than it went in with;
* because of that, the structural label MOEA/D dedups on was not settled
  until after the fit -- which is why ``InitialParetoLevelSorting`` scored
  every candidate before it read the duplicate verdict. On a solver-based AC
  run that meant 109 solves for 32 distinct systems, 77 of them (71%) handed
  to the solver for a system already in ``objective.history``.

The prune was a leftover from the retired ``nnz+1`` coefficient layout. Under
the unified layout (``Equation._validate_weight_layout``) both weight vectors
are structure-aligned with zeros retained, so nothing downstream needs the
structure compacted -- pinned below by scoring a zero-padded equation against
its pruned twin.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import epde.globals as global_var
from epde.operators.common.fitness import SolverBasedFitness, SolverFreeFitness
from epde.operators.common.objectives import EquationObjective, LOSS_NAN_VAL
from epde.operators.common.right_part_selection import EqRightPartSelector
from epde.operators.multiobjective.moeadd_specific import _rps_fitness_regenerate


# --------------------------------------------------------------------------- #
#  The hosts own nothing                                                       #
# --------------------------------------------------------------------------- #
class TestHostsAreScorersOnly:

    @pytest.mark.parametrize('host', [SolverFreeFitness, SolverBasedFitness])
    def test_apply_reaches_for_no_suboperator(self, host):
        """A host that names ``sparsity`` / ``coeff_calc`` in its own body is
        fitting, whatever the wiring says."""
        source = inspect.getsource(host.apply)
        code = [line.split('#')[0] for line in source.splitlines()]
        # ``self.suboperators[`` is the access; the bare word also occurs
        # inside the guard's error message, which is prose about the wiring.
        assert not [line for line in code if 'self.suboperators[' in line], source

    def test_the_in_place_pass_does_not_prune(self):
        """``remove_zero_terms`` belongs to RPS. A prune here is what made the
        post-RPS structural label unreliable for the history dedup."""
        source = inspect.getsource(SolverFreeFitness.apply)
        code = [line.split('#')[0] for line in source.splitlines()]
        code = [line for line in code if '``' not in line]
        assert not [line for line in code if 'remove_zero_terms' in line], source

    def test_the_sparsity_hook_is_gone_from_the_filler_protocol(self):
        """``needs_sparsity`` was the hook the host asked before fitting; with
        the host out of the fitting business it has no caller and no meaning."""
        assert not hasattr(EquationObjective, 'needs_sparsity')


# --------------------------------------------------------------------------- #
#  ... so an unfitted equation must fail loudly, not get fitted on the sly     #
# --------------------------------------------------------------------------- #
class _BareEquation:
    """Enough of an Equation for the guard, which runs before anything else.

    Both flags, because both are the precondition: the fillers read
    ``weights_final`` (through ``Equation.residual``), so a support decision
    alone is not enough to score on. ``final`` defaults to ``support`` so the
    common "fitted / not fitted" case stays a single argument.
    """

    def __init__(self, evald, final=None):
        self.weights_internal_evald = evald
        self.weights_final_evald = evald if final is None else final
        self.main_var_to_explain = 'u'


@pytest.fixture
def unstrided_samples(monkeypatch):
    """The solver host's first check, ``require_unstrided``, reads the global
    trajectories manager. A stand-in with nothing strided keeps a test from
    depending on whichever earlier test built a real one."""
    monkeypatch.setattr(global_var, 'samples_manager',
                        SimpleNamespace(require_unstrided=lambda where: None),
                        raising=False)


class TestUnfittedEquationRaises:

    def test_solver_free_host_refuses_to_score_it(self):
        host = SolverFreeFitness(['penalty_coeff'])
        host.params = {'penalty_coeff': 0.2}
        with pytest.raises(RuntimeError, match='not fitted'):
            host.apply(_BareEquation(False), {})

    def test_a_support_decision_alone_is_not_enough(self):
        """The host reads ``weights_final``, so sparsity-without-coefficients
        is exactly as unscoreable as nothing at all -- and used to slip through
        a check that only asked about the support."""
        host = SolverFreeFitness(['penalty_coeff'])
        host.params = {'penalty_coeff': 0.2}
        with pytest.raises(RuntimeError, match='not fitted'):
            host.apply(_BareEquation(True, final=False), {})

    def test_the_message_names_the_variable_and_the_owner(self):
        host = SolverFreeFitness(['penalty_coeff'])
        host.params = {'penalty_coeff': 0.2}
        with pytest.raises(RuntimeError) as excinfo:
            host.apply(_BareEquation(False), {})
        assert 'EqRightPartSelector' in str(excinfo.value)
        assert "'u'" in str(excinfo.value)

    def test_the_solver_host_refuses_the_whole_system(self, unstrided_samples):
        host = SolverBasedFitness(['penalty_coeff', 'pinn_loss_mult'])
        system = type('S', (), {'vals': [_BareEquation(True),
                                         _BareEquation(False)]})()
        with pytest.raises(RuntimeError, match='not fitted'):
            host.apply(system, {})


# --------------------------------------------------------------------------- #
#  ... and a declined equation keeps its stamp                                 #
# --------------------------------------------------------------------------- #
class _FittedEquation(_BareEquation):
    """A fitted equation as the solver host meets it; ``declined`` gives it
    the stamp right-part selection leaves on one it cannot fit
    (``_mark_unassignable``: ``fitness_value`` at ``LOSS_NAN_VAL`` with
    ``fitness_calculated`` raised)."""

    def __init__(self, declined=False):
        super().__init__(True)
        self.fitness_calculated = declined
        self.fitness_value = LOSS_NAN_VAL if declined else None

    def assert_state_invariants(self, where):
        pass

    def release_eval_cache(self):
        """The host drops each equation's evaluate() memo once it has scored."""
        self.released = True


class TestADeclinedEquationKeepsItsStamp:
    """The real wiring, on real fits, is driven in test_rps_repair_contract.py
    (``TestADeclinedCandidate``); this pins the host's own check, for every
    backend, ahead of the backend dispatch."""

    @staticmethod
    def _host(backend, monkeypatch):
        solved = []
        for name in ('_apply_autograd', '_apply_deepxde'):
            monkeypatch.setattr(SolverBasedFitness, name,
                                lambda self, objective, force, _n=name: solved.append(_n))
        host = SolverBasedFitness(['penalty_coeff', 'pinn_loss_mult'], backend=backend)
        return host, solved

    @pytest.mark.parametrize('backend', ['autograd', 'deepxde', 'basis'])
    def test_the_solver_host_does_not_solve_a_system_holding_one(
            self, backend, monkeypatch, unstrided_samples):
        """Solving would overwrite the stamp with a finite error for every
        equation, and the declined one would be placed. The host leaves the
        system as it came: the stamp stands, the other equation unscored."""
        host, solved = self._host(backend, monkeypatch)
        declined, healthy = _FittedEquation(declined=True), _FittedEquation()
        system = SimpleNamespace(vals=[healthy, declined])

        assert host.apply(system, {}) is None
        assert host.apply(system, {}, force_out_of_place=True) == LOSS_NAN_VAL
        assert solved == []
        assert declined.fitness_value == LOSS_NAN_VAL and declined.fitness_calculated
        assert healthy.fitness_value is None and not healthy.fitness_calculated

    @pytest.mark.parametrize('backend', ['autograd', 'deepxde', 'basis'])
    def test_a_system_without_one_is_solved(self, backend, monkeypatch,
                                            unstrided_samples):
        """The control: the same system with both equations fitted reaches the
        backend."""
        host, solved = self._host(backend, monkeypatch)
        system = SimpleNamespace(vals=[_FittedEquation(), _FittedEquation()])
        host.apply(system, {})
        assert solved == ['_apply_autograd' if backend == 'autograd' else '_apply_deepxde']
        # Solved and scored: every equation's evaluate() memo is released.
        assert all(getattr(eq, 'released', False) for eq in system.vals)


# --------------------------------------------------------------------------- #
#  RPS owns the fit, and is the one place that prunes                          #
# --------------------------------------------------------------------------- #
class TestRPSOwnsTheFit:

    def test_every_out_of_place_site_shares_one_helper(self):
        """The term sweep is the one site that fits a candidate, through the
        helper -- no site may call the host directly. The other two sites, the
        exit-guarantee probe and the exit refit for the target the probe
        installed, served the rerolls and the cap break; both now decline the
        candidate instead (test_rps_repair_contract.py)."""
        source = inspect.getsource(EqRightPartSelector.apply)
        assert source.count('_fit_and_score(') == 1
        code = [line.split('#')[0] for line in source.splitlines()]
        assert not [line for line in code if 'force_out_of_place' in line]

    def test_the_equation_leaves_rps_fitted(self):
        """The one way out of the loop that reaches the prune is convergence:
        the winner's fit installed with BOTH flags up -- the scorer reads
        ``weights_final``, so a support decision alone is not a fit -- and
        ``simplify_equation`` changing nothing after it. Every other exit
        declines the candidate: it leaves stamped fit-degenerate with zero
        weights, and the chromosome fitness keeps the stamp
        (``TestADeclinedEquationKeepsItsStamp`` below and
        test_rps_repair_contract.py). (The cap break used to leave the loop
        with the flags down, which the removed exit refit had to repair.)
        """
        source = inspect.getsource(EqRightPartSelector.apply)
        body = source.split('"""', 2)[2]                 # past the docstring
        code = '\n'.join(line.split('#')[0] for line in body.splitlines())
        loop = code[:code.index('objective.remove_zero_terms()')]
        assert loop.count('break') == 1
        simplify = loop.rindex('self.simplify_equation(objective)', 0, loop.index('break'))
        assert 'objective.weights_internal_evald = True' in loop[:simplify]
        assert 'objective.weights_final_evald = True' in loop[:simplify]
        returns = [line.strip() for line in loop.splitlines()
                   if line.strip().startswith('return')]
        assert returns and all(r.startswith('return self._decline(') for r in returns)

    def test_the_helper_fits_before_it_scores(self):
        """sparsity -> coeff_calc -> fitness, in that order."""
        source = inspect.getsource(EqRightPartSelector._fit_and_score)
        order = [name for name in ('sparsity', 'coeff_calc', 'fitness_calculation')
                 if "'%s'" % name in source]
        assert order == ['sparsity', 'coeff_calc', 'fitness_calculation']
        assert source.index("'sparsity'") < source.index("'coeff_calc'")
        assert source.index("'coeff_calc'") < source.index("'fitness_calculation'")

    def test_rps_still_prunes_exactly_once(self):
        source = inspect.getsource(EqRightPartSelector.apply)
        code = [line.split('#')[0] for line in source.splitlines()]
        assert len([l for l in code if 'remove_zero_terms' in l]) == 1


# --------------------------------------------------------------------------- #
#  Leaving zero-weight terms in the structure changes no score                 #
# --------------------------------------------------------------------------- #
def _scored_pair(metric):
    """(padded, pruned) discrepancies for the same fitted law.

    ``padded`` carries two extra terms whose ``weights_internal`` slot is 0 --
    exactly what an un-pruned equation looks like. ``pruned`` is the same law
    with those terms physically removed. The two must score identically, or
    dropping the in-place prune would have moved the objective.
    """
    from epde.operators.common.objectives import Discrepancy, FitContext

    rng = np.random.default_rng(0)
    n = 64
    cols = rng.normal(size=(n, 4))
    coefs = np.array([2.0, -0.5, 0.0, 0.0])
    target = cols @ coefs + 0.3

    def make(active_only_cols):
        eq = type('E', (), {})()
        idx = list(range(4)) if active_only_cols is None else active_only_cols
        eq_coefs = np.append(coefs[idx], 0.3)
        eq.weights_final = eq_coefs
        eq.weights_internal = eq_coefs
        eq.weights_final_evald = eq.weights_internal_evald = True
        eq.active_mask = np.asarray(coefs[idx]) != 0
        eq.evaluate = lambda *, active_only=False, _i=idx: (
            {0: target},
            {0: cols[:, [j for j in _i if coefs[j] != 0]] if active_only
                else cols[:, _i]})
        return eq

    ctx = FitContext(g_fun_vals={0: np.ones(n)}, data_shape={0: (n,)},
                     penalty_coeff=0.2, for_rps=False)
    filler = Discrepancy(metric)
    return (filler.compute(make(None), ctx),
            filler.compute(make([0, 1]), ctx))


class TestZeroPaddingIsInert:

    @pytest.mark.parametrize('metric', ['wape', 'l2', 'l2_relative',
                                        'scale_invariant'])
    def test_padded_and_pruned_score_the_same(self, metric):
        padded, pruned = _scored_pair(metric)
        assert padded == pytest.approx(pruned, rel=1e-12, abs=1e-15)


# --------------------------------------------------------------------------- #
#  The initial population reads the duplicate verdict before it pays for it    #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  Stamp first: a condemned form's stamped fillers are not computed            #
# --------------------------------------------------------------------------- #
class _SpyFiller(EquationObjective):
    """A filler counting its computes; the primary's verdict is set from
    outside (``degenerate``)."""

    def __init__(self, name, value_attr, flag_attr, value, stamped=True, calls=None):
        self.name, self.value_attr, self.flag_attr = name, value_attr, flag_attr
        self.value, self.stamped_on_failure = value, stamped
        self.degenerate = False
        self.calls = calls if calls is not None else []

    def compute(self, equation, ctx):
        self.calls.append(self.name)
        return self.value

    def is_degenerate(self, equation):
        self.calls.append(self.name + '.is_degenerate')
        return self.degenerate

    def reads_wide_evaluate(self):
        return True


class _ScoredEquation(_FittedEquation):
    def __init__(self):
        super().__init__()
        self.weights_internal = np.array([1.0, 0.0])


def _stamp_host(degenerate, amplified, monkeypatch, stamp_first=True):
    import epde.operators.common.fitness as fitness
    calls = []
    primary = _SpyFiller('discrepancy', 'fitness_value', 'fitness_calculated', 0.25, calls=calls)
    primary.degenerate = degenerate
    instability = _SpyFiller('instability', 'coefficients_stability', 'stability_calculated', 3.0, calls=calls)
    complexity = _SpyFiller('complexity', 'complexity_value', 'complexity_calculated', 2.0, stamped=False,
                            calls=calls)
    monkeypatch.setattr(fitness, '_amplification_trips',
                        lambda objective: calls.append('amplification') or amplified)
    monkeypatch.setattr(global_var, 'samples_manager',
                        SimpleNamespace(gFunc=lambda key: None, inner_shapes=None,
                                        row_weights=lambda slot, mode='dm': None, row_shapes=lambda slot: None),
                        raising=False)
    host = SolverFreeFitness(['penalty_coeff'], objectives=[primary, instability, complexity])
    host.params = {'penalty_coeff': 0.2}
    if not stamp_first:
        host._stamp_first = lambda: False
    return host, calls


_SCORED_ATTRS = ('fitness_value', 'fitness_calculated', 'coefficients_stability', 'stability_calculated',
                 'complexity_value', 'complexity_calculated', 'aic', 'aic_calculated', 'released')


class TestStampFirst:

    @pytest.mark.parametrize('degenerate, amplified', [(True, False), (False, True), (True, True), (False, False)])
    def test_the_scored_state_is_the_compute_everything_one(self, degenerate, amplified, monkeypatch):
        states = []
        for stamp_first in (True, False):
            host, _ = _stamp_host(degenerate, amplified, monkeypatch, stamp_first)
            equation = _ScoredEquation()
            host.apply(equation, {})
            states.append({attr: getattr(equation, attr, None) for attr in _SCORED_ATTRS})
        assert states[0] == states[1]
        condemned = degenerate or amplified
        assert (states[0]['fitness_value'] == LOSS_NAN_VAL) == condemned
        assert (states[0]['coefficients_stability'] == LOSS_NAN_VAL) == condemned
        assert states[0]['complexity_value'] == 2.0 and states[0]['released']

    @pytest.mark.parametrize('degenerate, amplified', [(True, False), (False, True)])
    def test_a_condemned_form_computes_no_stamped_filler(self, degenerate, amplified, monkeypatch):
        host, calls = _stamp_host(degenerate, amplified, monkeypatch)
        host.apply(_ScoredEquation(), {})
        # The verdict in its usual order (the amplification ratio only for a
        # form the discrepancy did not condemn), right after the primary.
        expected = ['discrepancy', 'discrepancy.is_degenerate'] + ([] if degenerate else ['amplification'])
        assert calls == expected + ['complexity']

    def test_a_healthy_form_computes_every_filler_once(self, monkeypatch):
        host, calls = _stamp_host(False, False, monkeypatch)
        host.apply(_ScoredEquation(), {})
        assert calls == ['discrepancy', 'discrepancy.is_degenerate', 'amplification', 'instability', 'complexity']

    def test_only_when_the_primary_comes_first_and_owns_its_attributes(self, monkeypatch):
        host, _ = _stamp_host(False, False, monkeypatch)
        assert host._stamp_first()
        primary, instability, complexity = host.objectives
        host.objectives = [instability, primary, complexity]
        assert not host._stamp_first()
        host.objectives = [primary, instability, complexity]
        instability.value_attr = 'fitness_value'
        assert not host._stamp_first()
        instability.value_attr = 'coefficients_stability'
        primary.value_attr = 'other_value'
        assert not host._stamp_first()
        primary.value_attr = 'fitness_value'
        primary.reads_wide_evaluate = lambda: False
        assert not host._stamp_first()
        primary.reads_wide_evaluate = None
        assert not host._stamp_first(), 'a primary that does not say so is not taken early'

    @pytest.mark.parametrize('metric, wide', [('wape', True), ('l2_relative', True), ('scale_invariant', True),
                                              ('l2', False)])
    def test_the_l2_discrepancy_is_not_taken_early(self, metric, wide):
        from epde.operators.common.objectives import Discrepancy
        assert Discrepancy(metric).reads_wide_evaluate() is wide
        assert set(Discrepancy.WIDE_EVALUATE) | {'l2'} == set(Discrepancy.OPTIONS) - set(Discrepancy.SOLVER_OPTIONS)


class _Candidate:
    def __init__(self, labels):
        self.equations_labels = labels


class TestInitialPopulationGate:

    def test_signature_accepts_the_predicate(self):
        params = inspect.signature(_rps_fitness_regenerate).parameters
        assert 'skip_fitness_if' in params
        assert params['skip_fitness_if'].default is None

    def test_a_known_duplicate_is_never_scored(self):
        calls = {'rps': 0, 'fitness': 0}
        rps = type('R', (), {'apply': lambda s, objective, arguments: calls
                             .__setitem__('rps', calls['rps'] + 1)})()
        fitness = type('F', (), {'apply': lambda s, objective, arguments: calls
                                 .__setitem__('fitness', calls['fitness'] + 1)})()
        history = {('a',)}

        scored = _rps_fitness_regenerate(
            _Candidate(('a',)), rps, fitness, {}, {},
            skip_fitness_if=lambda c: c.equations_labels in history)

        assert scored is False
        # RPS still ran -- it is what produces the label the gate reads.
        assert calls == {'rps': 1, 'fitness': 0}

    def test_a_novel_candidate_is_scored(self):
        calls = {'rps': 0, 'fitness': 0}
        rps = type('R', (), {'apply': lambda s, objective, arguments: calls
                             .__setitem__('rps', calls['rps'] + 1)})()
        fitness = type('F', (), {'apply': lambda s, objective, arguments: calls
                                 .__setitem__('fitness', calls['fitness'] + 1)})()
        candidate = _Candidate(('b',))
        candidate.vals = []          # no equations -> no degenerate re-roll

        scored = _rps_fitness_regenerate(
            candidate, rps, fitness, {}, {},
            skip_fitness_if=lambda c: c.equations_labels in {('a',)})

        assert scored is True
        assert calls == {'rps': 1, 'fitness': 1}

    def test_no_predicate_always_scores(self):
        calls = {'fitness': 0}
        rps = type('R', (), {'apply': lambda s, objective, arguments: None})()
        fitness = type('F', (), {'apply': lambda s, objective, arguments: calls
                                 .__setitem__('fitness', calls['fitness'] + 1)})()
        candidate = _Candidate(('a',))
        candidate.vals = []
        assert _rps_fitness_regenerate(candidate, rps, fitness, {}, {}) is True
        assert calls['fitness'] == 1
