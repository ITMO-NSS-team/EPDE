"""The search must not depend on Python's per-process string-hash salt.

``simplify_equation`` walks the factors common to every term of an equation
while mutating those terms, so the order it walks them in decides the
simplified form. Taking that order straight from a ``frozenset`` made it
depend on the hash salt, and with it the whole trajectory of the search: the
same explicit seed on the same data reached different equations in different
processes.
"""
import os
import subprocess
import sys
import textwrap

from epde.operators.common.right_part_selection import _common_factors

LABELS = [('u', (1.0,)),
          ('du/dx0', (1.0,)),
          ('d^2u/dx1^2', (1.0,)),
          ('x_0', (1.0,)),
          ('cos(t)sin(x)', (1.0,))]

#: Enough salts to make a coincidental agreement of the unordered form
#: implausible: 12 salts give 11 distinct orders for the labels above.
HASH_SEEDS = [str(seed) for seed in range(1, 13)]

#: Importing epde in a subprocess costs seconds, so the salts that need it are
#: kept few; the salts that only need the interpreter are cheap and plentiful.
HASH_SEEDS_WITH_IMPORT = HASH_SEEDS[:4]


def _run_under_hash_seed(seed: str, snippet: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    completed = subprocess.run([sys.executable, '-c', textwrap.dedent(snippet)],
                               env=env, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def test_common_factors_returns_the_intersection():
    first = frozenset(LABELS)
    second = frozenset(LABELS[:3])
    assert _common_factors([first, second]) == sorted(LABELS[:3])


def test_common_factors_order_does_not_depend_on_argument_order():
    first = frozenset(LABELS)
    second = frozenset(LABELS[:3])
    assert _common_factors([first, second]) == _common_factors([second, first])


def test_raw_set_order_does_vary_with_the_hash_seed():
    """Pins the hazard itself, so the test below cannot pass vacuously."""
    orders = {_run_under_hash_seed(seed, f'''
        labels = {LABELS!r}
        print(list(frozenset(labels)))
    ''') for seed in HASH_SEEDS}
    assert len(orders) > 1, 'expected the hash salt to reshuffle set iteration'


def test_common_factors_order_is_stable_across_hash_seeds():
    orders = {_run_under_hash_seed(seed, f'''
        from epde.operators.common.right_part_selection import _common_factors
        labels = {LABELS!r}
        print(_common_factors([frozenset(labels), frozenset(labels)]))
    ''') for seed in HASH_SEEDS_WITH_IMPORT}
    assert len(orders) == 1, f'order varies with PYTHONHASHSEED: {orders}'
