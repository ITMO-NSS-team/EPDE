#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The solver discrepancy axis, and the scales it is compared on.

Four regressions, all measured on an Allen-Cahn run with ``use_solver=True``
(30 candidates, popsize 16):

* the objective was ``rl_error + 1e4 * pinn_loss``, and the PINN term was
  **99.4-100%** of it -- the Pareto axis ranked candidates by how easy they
  were for the solver, not by how well they explained the data. The best
  composite of the run belonged to a form with ``wape`` 1.001;
* ``rl_error`` was a bare Euclidean norm over every grid point, so it grew
  like ``sqrt(n_points)`` and was comparable neither with the solver-free
  options nor with the PINN term it was added to. Across the 30 candidates it
  spanned only 26-62, truth and junk alike;
* MOEA/D assigned solutions to weight vectors by the acute angle of the RAW
  objective while selecting within a sector on the NORMALIZED one. With the
  axes orders of magnitude apart the whole population occupied 4.5e-4 rad
  (1.54 rad normalized), so the sectors carried almost no information;
* the gene-swap branch of ``ChromosomeCrossover`` cleared
  ``right_part_selected`` but not ``fitness_calculated``. RPS then re-ran and
  its term-sweep overwrote ``fitness_value`` with its own (solver-free)
  metric, while the solver host declined to re-score -- putting two different
  metrics on one Pareto axis. That per-branch flag juggling is gone: the
  offspring's whole fitted state now dies once, at the deepcopy site in
  ``ParetoLevelsCrossover``, which is where a candidate stops being its parent.
"""

import numpy as np
import pytest

import epde.globals as global_var
from epde.interface.search_config import SolverConfig
from epde.operators.common.objectives import (_relative_mean_square,
                                              _relative_norm)
from epde.operators.multiobjective.variation import ParetoLevelsCrossover
from epde.operators.utils.template import CompoundOperator
from epde.optimizers.moeadd.moeadd import (ObjFunNormalizer,
                                           marriageSolutionAssignment)


class TestDiscrepancyIsRelative:
    """The data term is a ratio, so it does not move with the grid."""

    def test_norm_is_grid_independent(self):
        # The same physical residual-to-reference ratio, sampled twice as
        # densely. The bare 2-norm grows like sqrt(n); the ratio must not.
        coarse_ref = np.ones(100)
        fine_ref = np.ones(400)
        coarse = _relative_norm(0.1 * coarse_ref, coarse_ref)
        fine = _relative_norm(0.1 * fine_ref, fine_ref)
        assert coarse == pytest.approx(fine)
        assert coarse == pytest.approx(0.1)
        # ... and the raw norms it replaced really do differ by sqrt(4).
        assert (np.linalg.norm(0.1 * fine_ref) /
                np.linalg.norm(0.1 * coarse_ref)) == pytest.approx(2.0)

    def test_norm_is_scale_independent(self):
        reference = np.linspace(1.0, 5.0, 64)
        residual = 0.25 * reference
        assert _relative_norm(residual, reference) == pytest.approx(
            _relative_norm(1000 * residual, 1000 * reference))

    def test_zero_reference_falls_back_to_the_bare_norm(self):
        """No scale to divide by; 0/0 must not reach the Pareto front."""
        residual = np.array([3.0, 4.0])
        assert _relative_norm(residual, np.zeros(2)) == pytest.approx(5.0)

    def test_mean_square_option_is_the_matching_ratio(self):
        reference = np.array([2.0, 2.0, 2.0, 2.0])
        residual = np.array([1.0, 1.0, 1.0, 1.0])
        # mean(1)/mean(4)
        assert _relative_mean_square(residual, reference) == pytest.approx(0.25)

    def test_the_two_solver_options_agree_on_a_common_case(self):
        """'solver_l2' and 'pic' differed by more than their reduction: one
        was a bare 2-norm, the other a bare mean-square. As ratios, a uniform
        residual gives the same fraction under both (squared, for 'pic')."""
        reference = np.full(50, 3.0)
        residual = 0.2 * reference
        assert _relative_norm(residual, reference) == pytest.approx(0.2)
        assert _relative_mean_square(residual, reference) == pytest.approx(0.04)


class TestPinnLossIsNotFused:

    def test_multiplier_defaults_to_zero(self):
        """At 1e4 this term WAS the objective. The data term now stands
        alone; raising it fuses the two deliberately."""
        assert SolverConfig().pinn_loss_mult == 0.0

    def test_it_is_still_configurable(self):
        assert SolverConfig(pinn_loss_mult=1e4).pinn_loss_mult == 1e4

    def test_it_is_declared_exactly_once(self):
        """It used to be written in three files, and the dataclass default --
        the one that reads like the definition -- was the one that lost.

        The two JSONs are gone; the operator block references the setting
        rather than restating it, so there is nothing left to disagree.
        """
        from epde.interface.search_config import (MULTI_OBJECTIVE_OPERATORS,
                                                  FromConfig)
        for operator in ('SolverBasedFitness', 'PIC'):
            declared = MULTI_OBJECTIVE_OPERATORS[operator]['pinn_loss_mult']
            assert isinstance(declared, FromConfig)
            assert (declared.group, declared.key) == ('solver', 'pinn_loss_mult')

    def test_the_resolved_config_is_what_the_dataclass_says(self):
        """The end-to-end check the old suite did not make: a value written on
        the dataclass is the value a search actually runs with."""
        from epde.interface.search_config import load_search_config
        cfg = load_search_config()
        assert cfg.solver.pinn_loss_mult == SolverConfig().pinn_loss_mult
        assert (cfg.evolution.operators['SolverBasedFitness']['pinn_loss_mult']
                == SolverConfig().pinn_loss_mult)


class _FakeSolution:
    """The surface ``marriageSolutionAssignment`` touches."""

    def __init__(self, objectives):
        self.obj_fun = np.asarray(objectives, dtype=float)
        self.vals = [None]           # one equation -> weight_full == weight
        self.domain = None

    def set_domain(self, idx):
        self.domain = idx


@pytest.fixture
def quiet_verbose(monkeypatch):
    """``marriageSolutionAssignment`` ends by reading
    ``global_var.verbose.show_iter_idx``.

    The attribute is set directly rather than through ``init_verbose``: that
    helper also installs a process-wide ``filterwarnings('ignore')``, which
    would leak out of this module and silence warnings other tests assert on.
    """
    class _Verbose:
        show_iter_idx = False

    monkeypatch.setattr(global_var, 'verbose', _Verbose(), raising=False)


class TestSectorAssignmentUsesTheSameScaleAsPBI:

    @staticmethod
    def _assign(normalizer):
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])
        # Raw, both point along axis 0 -- indistinguishable directions.
        # Normalized by the worst values they become (0.5, 0.02) and (1, 1),
        # i.e. one hugs the discrepancy axis and one sits at 45 degrees.
        solutions = [_FakeSolution([1e5, 0.1]), _FakeSolution([2e5, 5.0])]
        marriageSolutionAssignment(weights, solutions, normalizer)
        return solutions

    def test_normalized_assignment_separates_the_two_trade_offs(self, quiet_verbose):
        normalizer = ObjFunNormalizer(np.array([2e5, 5.0]))
        first, second = self._assign(normalizer)
        # The discrepancy-weighted sector takes the solution that is nearly
        # pure discrepancy; the other sector takes the balanced one.
        assert first.domain == 0
        assert second.domain == 1

    def test_raw_objectives_are_nearly_collinear(self):
        """Why the normalizer matters: the raw directions differ by ~1e-5 rad,
        so the assignment above cannot be read off them reliably."""
        raw = [np.arctan2(0.1, 1e5), np.arctan2(5.0, 2e5)]
        assert abs(raw[1] - raw[0]) < 1e-4
        normed = [np.arctan2(0.1 / 5.0, 1e5 / 2e5), np.arctan2(1.0, 1.0)]
        assert abs(normed[1] - normed[0]) > 0.7

    def test_assignment_still_works_without_a_normalizer(self, quiet_verbose):
        """The parameter is optional -- callers that have no scale yet
        (weights associated before any population is placed) still work."""
        solutions = self._assign(None)
        assert sorted(sol.domain for sol in solutions) == [0, 1]


class _FakeGene:
    def __init__(self):
        self.right_part_selected = True
        self.fitness_calculated = True
        self.stability_calculated = True
        self.complexity_calculated = True
        self.weights_internal_evald = True
        self.weights_final_evald = True
        # ParetoLevelsCrossover re-asserts the label/structure alignment on
        # both parents and both offspring.
        self.structure = ['term']
        self.terms_labels = frozenset({'term'})

    def reset_state(self, reset_right_part=True):
        self.right_part_selected = False
        self.fitness_calculated = False
        self.stability_calculated = False
        self.complexity_calculated = False
        self.weights_internal_evald = False
        self.weights_final_evald = False


class _FakeGenes:
    def __init__(self, keys):
        self.equation_keys = list(keys)
        self._genes = {key: _FakeGene() for key in keys}

    def same_encoding(self, other):
        return self.equation_keys == other.equation_keys

    def replace_gene(self, gene_key, value):
        self._genes[gene_key] = value

    def __getitem__(self, key):
        return self._genes[key]


class _FakeChromosome:
    def __init__(self, keys):
        self.vals = _FakeGenes(keys)
        self._times = 1

    def crossover_times(self):
        return self._times

    def reset_counter(self):
        self._times = 0

    def reset_state(self, reset_right_part=True):
        for key in self.vals.equation_keys:
            self.vals[key].reset_state(reset_right_part)


class _PassThroughCrossover(CompoundOperator):
    """Stands in for ``chromosome_crossover``: returns the pair untouched, so
    the only thing under test is what the copy site guarantees."""

    key = 'PassThroughCrossover'

    def apply(self, objective, arguments):
        return objective

    def use_default_tags(self):
        self._tags = {'crossover', 'chromosome level', 'no suboperators'}


class _FakeParetoLevels:
    def __init__(self, population):
        self.population = population
        self.unplaced_candidates = None


class TestOffspringOwnNoFit:
    """An offspring owns no fit, no scores and no right-part verdict.

    ``Equation.__deepcopy__`` carries both weight vectors and both ``*_evald``
    flags verbatim, so the deepcopy site in ``ParetoLevelsCrossover`` is the
    boundary where a candidate stops being its parent -- and where the fitted
    state has to die. Historically nothing cleared it there (the reset was
    commented out), and the gene-swap branch of ``ChromosomeCrossover`` cleared
    four flags by hand while deliberately KEEPING ``simplified`` /
    ``is_correct_right_part``. Those two were the outer-loop condition of
    ``EqRightPartSelector.apply``, so keeping them made RPS skip its body --
    including the ``reset_state`` on its first line -- and the gene rode out
    still carrying its parent's coefficients. It then reached the front with an
    ``l2_relative`` value beside neighbours carrying solver values.
    """

    @staticmethod
    def _offspring():
        operator = ParetoLevelsCrossover([])
        operator.set_suboperators({'chromosome_crossover': _PassThroughCrossover([])})
        levels = _FakeParetoLevels([_FakeChromosome(['u', 'v']),
                                    _FakeChromosome(['u', 'v'])])
        operator.apply(objective=levels, arguments={})
        return levels.unplaced_candidates

    def test_the_support_decision_is_cleared(self):
        for chromosome in self._offspring():
            for key in chromosome.vals.equation_keys:
                assert chromosome.vals[key].weights_internal_evald is False

    def test_the_fitted_magnitudes_are_cleared(self):
        for chromosome in self._offspring():
            for key in chromosome.vals.equation_keys:
                assert chromosome.vals[key].weights_final_evald is False

    def test_right_part_selection_is_cleared(self):
        """So ``rps_cond`` fires and the selector actually re-runs."""
        for chromosome in self._offspring():
            for key in chromosome.vals.equation_keys:
                assert chromosome.vals[key].right_part_selected is False

    def test_every_objective_flag_goes_too(self):
        """Both Pareto axes are recomputed from the re-scored equation."""
        for chromosome in self._offspring():
            for key in chromosome.vals.equation_keys:
                gene = chromosome.vals[key]
                assert gene.fitness_calculated is False
                assert gene.stability_calculated is False
                assert gene.complexity_calculated is False

    def test_the_parents_keep_their_own_fit(self):
        """The reset lands on the copies, not on the population members."""
        operator = ParetoLevelsCrossover([])
        operator.set_suboperators({'chromosome_crossover': _PassThroughCrossover([])})
        parents = [_FakeChromosome(['u', 'v']), _FakeChromosome(['u', 'v'])]
        operator.apply(objective=_FakeParetoLevels(parents), arguments={})
        for parent in parents:
            for key in parent.vals.equation_keys:
                assert parent.vals[key].weights_final_evald is True
