"""Numerical scoring regressions, independent of EPDE search."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epde_bench.metrics import (canonical_tokens, coefficient_error_best,
                               select_compromise, score_run)

U = 'u{power: 1.0}'
V = 'v{power: 1.0}'
DT = 'du/dx0{power: 1.0}'


class CoefficientParsingTests(unittest.TestCase):
    def test_system_decoration_does_not_change_coefficients(self):
        truth = f'-20.0 * {U} * {V} + 20.0 * {U} = {DT}'
        for prefix in ('/ ', '| ', '\\ '):
            with self.subTest(prefix=prefix):
                self.assertEqual(coefficient_error_best([prefix + truth], [[truth]]), 0.)

    def test_scientific_notation_plus_is_not_a_term_separator(self):
        truth = f'100000 * {U} + -20 * {V} = {DT}'
        discovered = f'1e+05 * {U} + -2E+01 * {V} = {DT}'
        self.assertEqual(coefficient_error_best([discovered], [[truth]]), 0.)

    def test_zero_scientific_coefficient_does_not_create_structural_term(self):
        self.assertEqual(canonical_tokens([f'0e+05 * {V} + {U} = {DT}']),
                         canonical_tokens([f'{U} = {DT}']))

    def test_target_flip_and_equation_permutation_keep_zero_error(self):
        truth = [f'-2 * {U} = {DT}', f'3 * {U} = {V}']
        discovered = [f'0.3333333333333333 * {V} = {U}', f'-.5 * {DT} = {U}']
        self.assertAlmostEqual(coefficient_error_best(discovered, [truth]), 0.)


class CompromiseTests(unittest.TestCase):
    def test_nonfinite_candidates_never_beat_valid_candidates(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(bad=bad):
                self.assertEqual(select_compromise([[bad, 0], [1, 1], [2, 2]]), 1)

    def test_missing_vector_is_excluded_without_changing_original_index(self):
        self.assertEqual(select_compromise([None, [2, 2], [1, 1]]), 2)

    def test_all_invalid_has_no_selection(self):
        for objectives in ([], [None], [[], []], [[float('nan'), 1], [1, float('inf')]]):
            with self.subTest(objectives=objectives):
                self.assertIsNone(select_compromise(objectives))

    def test_inconsistent_dimensions_have_no_defined_compromise(self):
        self.assertIsNone(select_compromise([[1, 2], [1]]))

    def test_malformed_vectors_are_excluded(self):
        self.assertEqual(select_compromise([3, ['invalid', 2], [1, 1]]), 2)

    def test_finite_extreme_values_still_choose_smallest_objective(self):
        self.assertEqual(select_compromise([[1e308], [-1e308], [0.]]), 1)

    def test_scoring_all_invalid_keeps_front_metrics_without_selected_metrics(self):
        truth = [f'-2 * {U} = {DT}']
        result = score_run([truth], [[float('nan'), 1]], [truth])
        self.assertIsNone(result['selected_index'])
        self.assertTrue(result['success_front'])
        self.assertFalse(result['success_selected'])
        self.assertIsNone(result['hamming_selected'])

    def test_normalization_and_ties_preserve_original_order(self):
        self.assertEqual(select_compromise([[1, 10], [2, 5], [3, 1]]), 1)
        self.assertEqual(select_compromise([[1, 1], [1, 1]]), 0)


if __name__ == '__main__':
    unittest.main()
