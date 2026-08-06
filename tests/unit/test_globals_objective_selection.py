#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validators and resolvers for the objective-family globals
(``complexity_metric``, ``second_objective``) and the ``'chi'`` alias of
``set_instability_metric``. Template: test_amplification_guard's
``test_cap_validator``."""

import pytest

import epde.globals as global_var


class TestComplexityMetricGlobal:

    def teardown_method(self):
        global_var.set_complexity_metric(None)

    def test_accepts_menu(self):
        for value in (None, 'factors', 'terms'):
            global_var.set_complexity_metric(value)
            assert global_var.complexity_metric == value

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            global_var.set_complexity_metric('bogus')

    def test_resolver_default_is_factors(self):
        global_var.set_complexity_metric(None)
        assert global_var.resolve_complexity_metric() == 'factors'

    def test_resolver_follows_override(self):
        global_var.set_complexity_metric('terms')
        assert global_var.resolve_complexity_metric() == 'terms'


class TestSecondObjectiveGlobal:

    def teardown_method(self):
        global_var.set_second_objective(None)

    def test_accepts_menu(self):
        for value in (None, 'instability', 'complexity'):
            global_var.set_second_objective(value)
            assert global_var.second_objective == value

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            global_var.set_second_objective('bogus')

    def test_unset_derives_from_use_pic(self):
        global_var.set_second_objective(None)
        assert global_var.resolve_second_objective(True) == 'instability'
        assert global_var.resolve_second_objective(False) == 'complexity'

    def test_override_wins_for_both_use_pic_values(self):
        # The lockstep contract: an explicit selection beats use_pic
        # regardless of what use_pic says.
        global_var.set_second_objective('complexity')
        assert global_var.resolve_second_objective(True) == 'complexity'
        assert global_var.resolve_second_objective(False) == 'complexity'
        global_var.set_second_objective('instability')
        assert global_var.resolve_second_objective(True) == 'instability'
        assert global_var.resolve_second_objective(False) == 'instability'


class TestChiAlias:

    def teardown_method(self):
        global_var.set_instability_metric(None)

    def test_chi_normalized_to_chi2_at_set_time(self):
        # Un-normalized 'chi' would silently fall through to the 'cv'
        # branch of Instability.compute -- the alias must canonicalize HERE.
        global_var.set_instability_metric('chi')
        assert global_var.instability_metric == 'chi2'
        assert global_var.resolve_instability_metric() == 'chi2'

    def test_chi2_still_accepted(self):
        global_var.set_instability_metric('chi2')
        assert global_var.instability_metric == 'chi2'

    def test_invalid_still_rejected(self):
        with pytest.raises(ValueError):
            global_var.set_instability_metric('bogus')
