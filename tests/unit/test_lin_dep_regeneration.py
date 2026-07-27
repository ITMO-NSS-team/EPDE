"""Unit tests for the exact linear-dependence scrub in RPS simplify
(`_flag_dependent_columns` + the `extra_ok` extension of the
regenerate-n-then-drop policy).

The detector must flag ONLY machine-exact dependence -- the
"infinite solutions" rank deficiency observed on lv / lorenz -- and must
never fire on near-collinear physics (no-collinearity-machinery
principle: valid identities are themselves near-degenerate and must not
be penalized).
"""
import numpy as np
import pytest

from epde.operators.common.right_part_selection import (
    _LIN_DEP_RTOL, _flag_dependent_columns,
)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


def test_exact_scaled_duplicate_flagged(rng):
    a = rng.normal(size=200)
    flagged, basis = _flag_dependent_columns([a, 2.0 * a])
    assert flagged == [1]
    # basis: intercept + the first (kept) column
    assert len(basis) == 2


def test_exact_combination_flagged(rng):
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    c = 3.0 * a - 0.5 * b  # exactly in span{a, b}
    flagged, _ = _flag_dependent_columns([a, b, c])
    assert flagged == [2]


def test_constant_column_flagged_against_intercept(rng):
    a = rng.normal(size=200)
    const = np.full(200, 7.3)  # redundant with the fitted intercept
    flagged, _ = _flag_dependent_columns([const, a])
    assert flagged == [0]


def test_zero_column_flagged(rng):
    a = rng.normal(size=200)
    flagged, _ = _flag_dependent_columns([a, np.zeros(200)])
    assert flagged == [1]


def test_near_collinear_not_flagged(rng):
    # r ~ 0.9999995 -- far beyond any physical collinearity, still far
    # above machine exactness: must NOT be flagged.
    a = rng.normal(size=200)
    b = a + 1e-3 * rng.normal(size=200)
    flagged, _ = _flag_dependent_columns([a, b])
    assert flagged == []


def test_full_rank_clean(rng):
    cols = [rng.normal(size=200) for _ in range(5)]
    flagged, basis = _flag_dependent_columns(cols)
    assert flagged == []
    assert len(basis) == 6  # intercept + all five


def test_nonfinite_returns_none(rng):
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    b[13] = np.nan
    assert _flag_dependent_columns([a, b]) is None


def test_scale_invariance(rng):
    # Column rescaling must not change the verdict (unit normalization).
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    c = a + b
    f1, _ = _flag_dependent_columns([a, b, c])
    f2, _ = _flag_dependent_columns([1e8 * a, 1e-8 * b, 1e5 * c])
    assert f1 == f2 == [2]


def test_flagged_order_keeps_first_seen(rng):
    # Greedy scan keeps the earlier column of a dependent pair -- the
    # later duplicate is the regeneration candidate.
    a = rng.normal(size=200)
    flagged, _ = _flag_dependent_columns([a, -a, a * 0.5])
    assert flagged == [1, 2]


def test_extra_ok_extends_acceptability():
    """`_regen_or_drop_term(extra_ok=...)`: a label-unique term failing
    ``extra_ok`` must NOT short-circuit to 'ok' (the pre-extension
    behavior would)."""
    from epde.operators.common.right_part_selection import _regen_or_drop_term

    class _FakeTerm:
        def __init__(self, label):
            self.structure = [object()]
            self._label = label

        @property
        def factors_labels(self):
            return frozenset({self._label})

        def contains_meaningful(self):
            return True

        def randomize(self):
            pass

        def reset_saved_state(self):
            pass

    class _FakeEq:
        def __init__(self, terms):
            self.structure = terms
            self.target = None

        def _invalidate_label_cache(self):
            pass

    t1, t2, t3 = _FakeTerm('a'), _FakeTerm('b'), _FakeTerm('c')
    eq = _FakeEq([t1, t2, t3])

    # extra_ok permanently False -> regeneration exhausts -> term dropped
    status = _regen_or_drop_term(eq, t3, max_iter=3,
                                 extra_ok=lambda term: False)
    assert status == 'dropped'
    assert t3 not in eq.structure

    # extra_ok True -> unchanged 'ok' fast path
    status = _regen_or_drop_term(eq, t2, max_iter=3,
                                 extra_ok=lambda term: True)
    assert status == 'ok'
    assert t2 in eq.structure
