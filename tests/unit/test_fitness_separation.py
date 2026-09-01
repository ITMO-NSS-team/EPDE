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

import numpy as np
import pytest

from epde.operators.common.fitness import SolverBasedFitness, SolverFreeFitness
from epde.operators.common.objectives import EquationObjective
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
    """Enough of an Equation for the guard, which runs before anything else."""

    def __init__(self, evald):
        self.weights_internal_evald = evald
        self.main_var_to_explain = 'u'


class TestUnfittedEquationRaises:

    def test_solver_free_host_refuses_to_score_it(self):
        host = SolverFreeFitness(['penalty_coeff'])
        host.params = {'penalty_coeff': 0.2}
        with pytest.raises(RuntimeError, match='no support decision'):
            host.apply(_BareEquation(False), {})

    def test_the_message_names_the_variable_and_the_owner(self):
        host = SolverFreeFitness(['penalty_coeff'])
        host.params = {'penalty_coeff': 0.2}
        with pytest.raises(RuntimeError) as excinfo:
            host.apply(_BareEquation(False), {})
        assert 'EqRightPartSelector' in str(excinfo.value)
        assert "'u'" in str(excinfo.value)

    def test_the_solver_host_refuses_the_whole_system(self):
        host = SolverBasedFitness(['penalty_coeff', 'pinn_loss_mult'])
        system = type('S', (), {'vals': [_BareEquation(True),
                                         _BareEquation(False)]})()
        with pytest.raises(RuntimeError, match='no support decision'):
            host.apply(system, {})


# --------------------------------------------------------------------------- #
#  RPS owns the fit, and is the one place that prunes                          #
# --------------------------------------------------------------------------- #
class TestRPSOwnsTheFit:

    def test_every_out_of_place_site_shares_one_helper(self):
        """Three sites fit a candidate: the term-sweep, the exit-guarantee
        probe, and the exit contract that refits for the target the probe
        finally installed. They must fit identically; a second copy of the
        sequence is how they drift apart. No site may call the host directly."""
        source = inspect.getsource(EqRightPartSelector.apply)
        assert source.count('_fit_and_score(') == 3
        code = [line.split('#')[0] for line in source.splitlines()]
        assert not [line for line in code if 'force_out_of_place' in line]

    def test_the_equation_leaves_rps_fitted(self):
        """The probe clears the weight flags after every candidate it tries.
        Without a refit for the installed target the scorer is handed an
        equation with no support decision -- which used to be papered over by
        the host's ``needs_sparsity`` fallback and now raises. Reproduced live
        by the legacy-LASSO pipeline."""
        source = inspect.getsource(EqRightPartSelector.apply)
        guard = "if not getattr(objective, 'weights_internal_evald', False):"
        assert guard in source
        assert source.index(guard) < source.index('objective.remove_zero_terms()')

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
