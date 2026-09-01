"""``Equation.evaluate``'s arguments name what they do.

The method used to be ``evaluate(normalize=True, return_val=False,
grids=None)``, and not one of the three described its own behaviour:

* ``normalize`` normalised nothing. It chose which non-target terms became
  feature columns -- all of them, or only the ones the sparsity step left with
  a non-zero ``weights_internal`` slot. The normalisation it was named after
  had already been deleted from ``Term.evaluate`` (it survives there only as a
  commented-out block), yet callers went on reasoning about column widths
  through a word for scaling. ``objectives._extract_coefs_intercept`` had to
  document the real contract on its behalf.
* ``return_val`` selected no evaluation option: it swapped the operation for a
  different one, a residual, and left every other caller opening with
  ``_, targets, features = ...``.
* ``grids`` could not be passed at all -- ``Term.evaluate`` raises
  ``NotImplementedError`` for any non-``None`` value.

Now: ``evaluate(*, active_only=False) -> (target, features)`` and a separate
``residual(*, active_only=False)``. These tests pin the new contract, the
column selection behind ``active_only``, the memoization policy, and the four
arithmetic defects the old residual branch carried.
"""

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import epde
from epde.structure.main_structures import Equation

EPDE_ROOT = Path(epde.__file__).parent

#: Deliberately unequal to any column count used below, so a features/points
#: axis mix-up cannot broadcast by luck.
N_POINTS = 7


# --------------------------------------------------------------------------- #
#  A bare Equation: __new__ plus the slots evaluate/residual touch. Building a  #
#  real one needs a token pool, a domain and cached derivatives; the fake Terms #
#  just hand back their per-trajectory dict, which is all evaluate asks of      #
#  them. Same construction trick as test_weight_layout._equation.               #
# --------------------------------------------------------------------------- #
def _term_values(seed, keys, n=N_POINTS):
    rng = np.random.default_rng(seed)
    return {key: rng.normal(size=n) for key in keys}


def _equation(term_coefs, target_pos=0, intercept=0.0, final=None, keys=(0,)):
    """``len(term_coefs) + 1`` terms; ``term_coefs`` are the NON-target ones,
    in ``weight_index`` order."""
    eq = Equation.__new__(Equation)
    eq._history = ''
    eq.structure = []
    for i in range(len(term_coefs) + 1):
        values = _term_values(100 + i, keys)
        eq.structure.append(SimpleNamespace(name='t{0}'.format(i),
                                            values=values,
                                            evaluate=(lambda v=values: v)))
    eq._target_term = eq.structure[target_pos]
    internal = np.asarray(list(term_coefs) + [intercept], dtype=float)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    eq.weights_internal = internal
    eq.weights_final = internal.copy() if final is None else np.asarray(final, dtype=float)
    eq._eval_cache = {}
    return eq


def _columns(eq, key=0):
    """The fake Terms' values in structure order, target excluded."""
    tgt = eq.target_idx
    return [eq.structure[i].values[key]
            for i in range(len(eq.structure)) if i != tgt]


class TestTheSignatureNamesWhatItDoes:

    def test_active_only_is_the_only_parameter(self):
        params = inspect.signature(Equation.evaluate).parameters
        assert list(params) == ['self', 'active_only']
        assert params['active_only'].default is False

    def test_it_is_keyword_only(self):
        """``logger.py`` called ``evaluate(False, True)`` positionally, which
        is how a renamed flag could still have been passed silently. Keyword-
        only makes that spelling impossible rather than merely wrong."""
        params = inspect.signature(Equation.evaluate).parameters
        assert params['active_only'].kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize('name', ['normalize', 'return_val', 'grids'])
    def test_the_old_parameters_are_gone(self, name):
        assert name not in inspect.signature(Equation.evaluate).parameters

    def test_the_old_positional_call_no_longer_parses(self):
        eq = _equation([2.0, -1.0])
        with pytest.raises(TypeError):
            eq.evaluate(False, True)

    def test_it_returns_a_pair(self):
        targets, features = _equation([2.0, -1.0]).evaluate()
        assert isinstance(targets, dict)
        assert isinstance(features, dict)

    def test_residual_mirrors_the_flag(self):
        params = inspect.signature(Equation.residual).parameters
        assert list(params) == ['self', 'active_only']
        assert params['active_only'].kind is inspect.Parameter.KEYWORD_ONLY


class TestColumnSelection:
    """``active_only`` is the only thing that changes between the two widths --
    no scaling is applied on either path."""

    def test_the_default_is_every_non_target_term(self):
        eq = _equation([2.0, 0.0, -1.0])          # one zero weight
        _, features = eq.evaluate()
        assert features[0].shape == (N_POINTS, len(eq.structure) - 1)

    def test_active_only_keeps_the_non_zero_weight_terms(self):
        eq = _equation([2.0, 0.0, -1.0])
        _, features = eq.evaluate(active_only=True)
        assert features[0].shape == (N_POINTS, int(eq.active_mask.sum()))
        assert features[0].shape[1] == 2

    def test_the_surviving_columns_are_the_right_terms(self):
        eq = _equation([2.0, 0.0, -1.0])
        _, features = eq.evaluate(active_only=True)
        kept = _columns(eq)
        np.testing.assert_allclose(features[0][:, 0], kept[0])
        np.testing.assert_allclose(features[0][:, 1], kept[2])

    def test_the_wide_columns_are_in_structure_order(self):
        eq = _equation([2.0, 0.0, -1.0], target_pos=2)
        _, features = eq.evaluate()
        for position, column in enumerate(_columns(eq)):
            np.testing.assert_allclose(features[0][:, position], column)

    def test_neither_path_rescales_anything(self):
        """The name it carried for years promised otherwise."""
        eq = _equation([2.0, -1.0])
        _, wide = eq.evaluate()
        _, narrow = eq.evaluate(active_only=True)
        for position, column in enumerate(_columns(eq)):
            np.testing.assert_allclose(wide[0][:, position], column)
            np.testing.assert_allclose(narrow[0][:, position], column)

    def test_features_is_none_when_no_term_qualifies(self):
        eq = _equation([0.0, 0.0])
        assert eq.evaluate(active_only=True)[1] is None

    def test_a_single_term_equation_has_no_features(self):
        eq = _equation([])
        assert eq.evaluate()[1] is None

    def test_the_target_is_never_a_column(self):
        eq = _equation([2.0, -1.0], target_pos=1)
        targets, features = eq.evaluate()
        np.testing.assert_allclose(targets[0], eq.structure[1].values[0])
        assert features[0].shape[1] == 2

    def test_every_trajectory_is_carried_through(self):
        eq = _equation([2.0, -1.0], keys=(0, 1, 2))
        targets, features = eq.evaluate()
        assert set(targets) == set(features) == {0, 1, 2}
        assert all(features[key].shape == (N_POINTS, 2) for key in features)


class TestTheMemoizationPolicy:
    """Only the wide result is cached -- the narrow one reads the weights, and
    callers move those between calls."""

    def test_the_wide_result_is_memoized(self):
        eq = _equation([2.0, -1.0])
        assert eq.evaluate() is eq.evaluate()

    def test_the_key_is_the_target_index(self):
        eq = _equation([2.0, -1.0], target_pos=1)
        eq.evaluate()
        assert set(eq._eval_cache) == {eq.target_idx} == {1}

    def test_the_narrow_branch_is_never_cached(self):
        eq = _equation([2.0, -1.0])
        eq.evaluate(active_only=True)
        assert eq._eval_cache == {}

    def test_the_narrow_branch_tracks_a_weight_change(self):
        """The reason it must not be cached."""
        eq = _equation([2.0, -1.0])
        assert eq.evaluate(active_only=True)[1][0].shape[1] == 2
        eq.weights_internal = np.array([2.0, 0.0, 0.0])
        assert eq.evaluate(active_only=True)[1][0].shape[1] == 1

    def test_a_second_target_gets_its_own_entry(self):
        eq = _equation([2.0, -1.0])
        eq.evaluate()
        eq.target_idx = 2
        eq.evaluate()
        assert set(eq._eval_cache) == {0, 2}

    def test_a_structural_mutation_drops_it(self):
        eq = _equation([2.0, -1.0])
        eq.evaluate()
        eq._invalidate_label_cache()
        assert eq._eval_cache == {}

    def test_the_cache_survives_a_deepcopy_as_a_fresh_dict(self):
        """Cited by ``_EQ_CACHE_AVOID_COPY``'s comment as
        ``test_eval_cache_after_deepcopy_is_fresh_dict``, which is not in this
        tree."""
        eq = _equation([2.0, -1.0])
        eq.evaluate()
        assert eq._eval_cache
        clone = copy.deepcopy(eq)
        assert clone._eval_cache == {}
        assert clone._eval_cache is not eq._eval_cache


class TestResidual:
    """The old ``return_val`` branch carried four defects at once; every one of
    them was invisible because its only caller (``Logger.add_log``) is itself
    unreachable."""

    @staticmethod
    def _expected(eq, key=0, active_only=False):
        tgt = eq.target_idx
        columns, coefs = [], []
        for i in range(len(eq.structure)):
            if i == tgt:
                continue
            slot = eq.weight_index(i, tgt)
            if active_only and eq.weights_internal[slot] == 0:
                continue
            columns.append(eq.structure[i].values[key])
            coefs.append(eq.weights_final[slot])
        prediction = (sum(c * col for c, col in zip(coefs, columns))
                      if columns else 0.0)
        return eq.structure[tgt].values[key] - (prediction + eq.weights_final[-1])

    def test_the_shape_matches_the_target(self):
        """DEFECT 1: the intercept column was appended with ``np.vstack`` --
        a ROW onto an ``(n_points, n_features)`` matrix -- so the weighted sum
        came out ``n_features`` long and the subtraction could only broadcast
        when features and points happened to be equinumerous."""
        eq = _equation([2.0, -1.0])
        assert eq.residual()[0].shape == (N_POINTS,)

    def test_the_intercept_is_included(self):
        """DEFECT 2: the ones-column was built and then never read -- the sum
        ran over the feature indexes alone."""
        without = _equation([2.0, -1.0], intercept=0.0)
        with_intercept = _equation([2.0, -1.0], intercept=0.0,
                                   final=[2.0, -1.0, 5.0])
        np.testing.assert_allclose(with_intercept.residual()[0],
                                   without.residual()[0] - 5.0)

    def test_it_equals_target_minus_prediction(self):
        eq = _equation([2.0, -1.0, 0.5], target_pos=2, intercept=3.0)
        np.testing.assert_allclose(eq.residual()[0], self._expected(eq))

    def test_every_trajectory_keeps_its_own_residual(self):
        """DEFECT 3: ``targets[idx]`` leaked the feature-index loop's ``idx``
        where the trajectory key belonged."""
        eq = _equation([2.0, -1.0], keys=(0, 1, 2), intercept=1.5)
        residual = eq.residual()
        assert set(residual) == {0, 1, 2}
        for key in residual:
            np.testing.assert_allclose(residual[key], self._expected(eq, key))

    @pytest.mark.parametrize('target_pos', [0, 1, 2, 3])
    def test_the_coefficients_are_read_at_the_right_slots(self, target_pos):
        """DEFECT 4: both weight vectors skip the target, but the old code
        indexed them by full-STRUCTURE position, so every term past the target
        read its neighbour's coefficient."""
        eq = _equation([2.0, -1.0, 0.5], target_pos=target_pos, intercept=0.25)
        np.testing.assert_allclose(eq.residual()[0], self._expected(eq))

    def test_it_uses_the_fitted_magnitudes(self):
        """The wide path used to read ``weights_internal`` -- the sparsity
        step's SUPPORT decision -- in place of ``weights_final``."""
        eq = _equation([1.0, 1.0], final=[2.0, -3.0, 0.0])
        np.testing.assert_allclose(eq.residual()[0], self._expected(eq))
        # and it genuinely differs from what weights_internal would have given
        columns = _columns(eq)
        internal_prediction = columns[0] + columns[1]
        assert not np.allclose(eq.residual()[0],
                               eq.structure[0].values[0] - internal_prediction)

    def test_the_two_column_sets_agree_when_every_weight_is_non_zero(self):
        eq = _equation([2.0, -1.0], intercept=0.75)
        np.testing.assert_allclose(eq.residual()[0],
                                   eq.residual(active_only=True)[0])

    def test_the_narrow_path_drops_the_inactive_terms(self):
        eq = _equation([2.0, 0.0, -1.0], intercept=0.5)
        np.testing.assert_allclose(eq.residual(active_only=True)[0],
                                   self._expected(eq, active_only=True))

    def test_a_featureless_equation_is_target_minus_intercept(self):
        eq = _equation([0.0, 0.0], final=[0.0, 0.0, 4.0])
        np.testing.assert_allclose(eq.residual(active_only=True)[0],
                                   eq.structure[0].values[0] - 4.0)

    def test_it_is_not_memoized(self):
        eq = _equation([2.0, -1.0])
        first = eq.residual()[0].copy()
        eq.weights_final = np.array([2.0, -1.0, 10.0])
        np.testing.assert_allclose(eq.residual()[0], first - 10.0)


class TestNoCallerStillStatesTheOldContract:
    """A source scan, the same instrument
    ``test_globals_runtime_only.TestNothingReadsTheOldNames`` uses: a stale
    keyword is a ``TypeError`` at runtime, but only on the branch that reaches
    it, and several of these live in solver paths the fast suite never runs."""

    def _scan(self, predicate):
        offenders = []
        for path in EPDE_ROOT.rglob('*.py'):
            for lineno, line in enumerate(
                    path.read_text(encoding='utf-8', errors='ignore')
                        .splitlines(), 1):
                code = line.split('#')[0]
                if '.evaluate(' in code and predicate(code):
                    offenders.append('%s:%s %s' % (path.name, lineno,
                                                   code.strip()))
        return offenders

    def test_no_evaluate_call_passes_the_retired_keywords(self):
        # The keyword spellings, not the bare word: the vendored solver has an
        # unrelated ``Solution.evaluate`` whose callers write
        # ``loss, loss_normalized = ...``.
        assert not self._scan(
            lambda code: any(spelling in code
                             for spelling in ('normalize=', 'normalize =',
                                              'return_val=', 'return_val =')))

    def test_no_caller_still_unpacks_three_values(self):
        assert not self._scan(lambda code: code.strip().startswith('_,'))
