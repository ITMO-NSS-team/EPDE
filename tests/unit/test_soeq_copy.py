#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A system's copy shares the population-wide token families (``SoEq._SHARED_ON_COPY``).

Pinned here:

* a copy holds its original's ``tokens_for_eq`` / ``tokens_supp``, and its own equations;
* ``create`` on a copy draws what it drew over the deep-copied families, from the same random stream;
* the families stay as the pool built them through copies, fresh systems and a plain-mode fit
  (status, parameter descriptions, equality ranges, derivative orders, tokens, evaluators).
"""
from copy import deepcopy

import numpy as np
import pytest

from epde.structure.main_structures import SoEq

from test_duplicate_equation_guard import _system, uv_pool  # noqa: F401
from test_plain_mode import LEAKY, _fresh, _run


def _families(system):
    return list(system.tokens_for_eq.families) + list(system.tokens_supp.families)


def _family_state(family):
    return (dict(family.status), deepcopy(family.token_params), deepcopy(family.equality_ranges),
            deepcopy(getattr(family, 'derivs_ords', None)), list(family.tokens), family._evaluator)


def _same_state(a, b):
    status_a, params_a, ranges_a, ords_a, tokens_a, evaluator_a = a
    status_b, params_b, ranges_b, ords_b, tokens_b, evaluator_b = b
    return (status_a == status_b and repr(params_a) == repr(params_b) and repr(ranges_a) == repr(ranges_b)
            and repr(ords_a) == repr(ords_b) and tokens_a == tokens_b and evaluator_a is evaluator_b)


class TestTheCopy:

    def test_a_copy_shares_the_families_and_owns_its_equations(self, uv_pool):
        system = _system(uv_pool, *LEAKY)
        copied = deepcopy(system)
        assert copied.tokens_for_eq is system.tokens_for_eq
        assert copied.tokens_supp is system.tokens_supp
        for key in system.vars_to_describe:
            assert copied.vals[key] is not system.vals[key]
            assert copied.vals[key].text_form == system.vals[key].text_form

    def test_create_on_a_copy_draws_as_over_copied_families(self, uv_pool, monkeypatch):
        system = _system(uv_pool, *LEAKY)

        def draw():
            copied = deepcopy(system)
            np.random.seed(11)
            copied.create()
            return [eq.text_form for eq in copied], np.random.get_state()[1].copy(), copied

        shared_forms, shared_state, shared = draw()
        monkeypatch.setattr(SoEq, '_SHARED_ON_COPY', ())
        copied_forms, copied_state, copied = draw()
        assert copied.tokens_for_eq is not system.tokens_for_eq, 'the old copy, for the comparison'
        assert shared_forms == copied_forms
        assert np.array_equal(shared_state, copied_state)
        families = {id(family) for family in _families(system)}
        for eq in shared:
            for term in eq.structure:
                for factor in term.structure:
                    assert any(factor._evaluator is family._evaluator for family in _families(system))
        assert all(id(family) in families for family in _families(shared))

    def test_the_families_stay_as_built(self, uv_pool):
        system = _fresh(uv_pool)
        before = [_family_state(family) for family in _families(system)]
        copies = [deepcopy(system) for _ in range(3)]
        np.random.seed(3)
        for copied in copies[:2]:
            copied.create()
        _run(copies[2])
        _run(system)
        after = [_family_state(family) for family in _families(system)]
        assert len(before) == len(after)
        assert all(_same_state(a, b) for a, b in zip(before, after))
