"""Check pooled coefficient fits and amplification at extreme scales."""
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import using_config
from epde.operators.common.multisample import fit_row_weights, joint_subset_table
from epde.operators.common.right_part_selection import amplification_ratio


@pytest.mark.parametrize('amplitude', [1.0, 1e155, 1e-170])
@pytest.mark.parametrize('weighting, mass', [('samples', 1.0), ('samples', 1e307),
                                           ('points', 1e307)])
def test_finite_units_keep_identifiable_slope_and_amplification(amplitude, weighting, mass):
    x = amplitude * np.linspace(1.0, 2.0, 40)
    blocks = [(key, x[:, None], 2 * x, np.full(x.size, mass)) for key in (7, 19)]
    discrepancy, amplification, _, coefficients, full_rank = joint_subset_table(
        blocks, discrepancy='wape', subsets=[1], weighting=weighting)
    assert full_rank.tolist() == [True]
    np.testing.assert_allclose(coefficients[0], [2.0, 0.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(amplification, [1.0], rtol=1e-12)
    assert discrepancy[0] < 1e-12


@pytest.mark.parametrize('mass', [1.0, 1e307, 1e-300])
def test_sample_mass_normalisation_preserves_relative_weights(mass):
    raw = np.tile([0.0, 1.0, 3.0], 20) * mass
    expected = np.tile([0.0, 1.0 / 80, 3.0 / 80], 20)
    np.testing.assert_allclose(fit_row_weights(raw, 'samples'), expected, rtol=1e-14)
    np.testing.assert_array_equal(fit_row_weights(raw, 'points'), raw)


class _EvaluatedTerm:
    """An already evaluated term: only the guard's column protocol is needed."""
    def __init__(self, columns):
        self.columns = columns

    def evaluate(self, **kwargs):
        return self.columns


@pytest.mark.parametrize('amplitude', [1.0, 1e155, 1e-170])
def test_materialised_joint_guard_agrees_with_the_physical_slope(amplitude):
    fields = {7: amplitude * np.linspace(1.0, 2.0, 40),
              19: amplitude * np.linspace(2.0, 3.0, 60)}
    equation = SimpleNamespace(
        structure=[_EvaluatedTerm(fields),
                   _EvaluatedTerm({key: 2 * x for key, x in fields.items()})],
        target_idx=1, main_var_to_explain='u', weights_internal=np.array([2.0, 0.0]))
    with using_config(multisample_mode='joint', system_rps_mode='plain'):
        assert amplification_ratio(equation) == pytest.approx(1.0, rel=1e-12)


def test_zero_target_still_declines_instead_of_dropping_an_experiment():
    fields = {7: np.arange(1.0, 41.0), 19: np.arange(1.0, 61.0)}
    equation = SimpleNamespace(
        structure=[_EvaluatedTerm(fields),
                   _EvaluatedTerm({key: np.zeros_like(x) for key, x in fields.items()})],
        target_idx=1, main_var_to_explain='u', weights_internal=np.array([2.0, 0.0]))
    with using_config(multisample_mode='joint', system_rps_mode='plain'):
        assert amplification_ratio(equation) == np.inf


def test_unused_large_column_cannot_poison_a_finite_selected_support():
    x = np.arange(1.0, 5.0)
    features = np.column_stack([x, np.full(x.size, 1e308)])
    blocks = [(key, features, 2 * x, np.ones(x.size)) for key in (7, 19)]
    discrepancy, amplification, _, coefficients, full_rank = joint_subset_table(
        blocks, discrepancy='wape', subsets=[1], weighting='points')
    assert full_rank.tolist() == [True]
    np.testing.assert_allclose(coefficients[0], [2.0, 0.0, 0.0], rtol=1e-12)
    assert discrepancy[0] < 1e-12
    assert amplification[0] == pytest.approx(1.0, rel=1e-12)
