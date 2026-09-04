"""The trig frequency equality tolerance, and the quantisation it drives.

``TrigonometricTokens`` declares ``freq_equality_fraction = 0.05`` and
documents it as "fraction of allowed frequency interval, that is considered as
the same". The code divided by it instead of multiplying, making the tolerance
TWENTY TIMES the whole declared interval. Since
``Factor._quantized_params`` buckets a continuous parameter as
``int((v - bounds[0]) / tol)`` and the numerator can never reach a denominator
twenty times its own range, every admissible frequency quantised to bucket 0.

The consequences were not cosmetic. All sin/cos factors shared one
``structural_label``, hence one ``Term.factors_labels``, hence ONE entry in the
term tensor cache -- so the first frequency evaluated was silently reused for
every later one, and the frequency printed in a discovered equation was not the
frequency that had been scored. The same tolerance drives ``Factor.__eq__``,
so the whole trig family also compared mutually equal.
"""
import numpy as np
import pytest

from epde.interface.prepared_tokens import TrigonometricTokens
from epde.structure.factor import Factor

INTERVALS = [
    pytest.param((np.pi / 2.0, 2 * np.pi), id='library-default'),
    pytest.param((0.999, 1.001), id='ode-narrow'),
    pytest.param((2 - 1e-8, 2 + 1e-8), id='pinned-freq-2'),
]


def _family(freq):
    return TrigonometricTokens(freq=freq, dimensionality=0)._token_family


def _factor(family, freq, bounds, label='sin'):
    f = Factor(label, status=family.status, family_type='trigonometric',
               latex_constructor=family.latex_constructor,
               variable='u', all_vars=['u'])
    f.set_parameters({'power': (1, 1), 'freq': bounds, 'dim': (0, 0)},
                     family.equality_ranges, random=False,
                     power=1, freq=freq, dim=0)
    return f


@pytest.mark.parametrize('freq', INTERVALS)
def test_tolerance_is_a_fraction_of_the_interval(freq):
    """A MULTIPLICATION. The divide made it 20x the interval instead of 1/20."""
    width = freq[1] - freq[0]
    tol = _family(freq).equality_ranges['freq']
    assert tol == pytest.approx(width * 0.05, rel=1e-12)
    assert tol < width, 'tolerance must be smaller than the range it partitions'


@pytest.mark.parametrize('freq', INTERVALS)
def test_the_interval_is_not_one_bucket(freq):
    """The regression proper: the endpoints of the declared range must be
    distinguishable. Under the divide they both quantised to 0."""
    fam = _family(freq)
    lo = _factor(fam, freq[0], freq)
    hi = _factor(fam, freq[1] * (1 - 1e-15) if freq[1] else freq[1], freq)
    assert lo.structural_label != hi.structural_label
    assert not (lo == hi)


@pytest.mark.parametrize('freq', INTERVALS)
def test_frequencies_within_tolerance_still_compare_equal(freq):
    """The tolerance must not become so tight that genuinely-identical
    frequencies separate -- the fix is a rescale, not a removal."""
    fam = _family(freq)
    tol = fam.equality_ranges['freq']
    base = freq[0] + 0.5 * (freq[1] - freq[0])
    assert _factor(fam, base, freq) == _factor(fam, base + 0.01 * tol, freq)


@pytest.mark.parametrize('freq', INTERVALS)
def test_a_shared_label_implies_the_frequencies_are_within_tolerance(freq):
    """THE load-bearing direction, because ``structural_label`` keys the term
    tensor cache: two factors may share a cached column only if they really are
    the same frequency.

    The converse is deliberately not asserted. Bucketing makes an approximate
    relation transitive and hashable, at the cost that two values closer than
    ``tol`` can straddle a bin edge and label differently. That direction is
    harmless -- it makes dedup conservative. The direction asserted here is the
    dangerous one: if it failed, two different frequencies would silently share
    one evaluated tensor, which is exactly the bug this module pins.
    """
    fam = _family(freq)
    tol = fam.equality_ranges['freq']
    rng = np.random.default_rng(0)
    draws = rng.uniform(freq[0], freq[1], size=200)
    factors = [_factor(fam, float(v), freq) for v in draws]
    for i, fi in enumerate(factors):
        for j in range(i + 1, len(factors)):
            if fi.structural_label == factors[j].structural_label:
                assert abs(draws[i] - draws[j]) <= tol, (
                    f'{draws[i]} and {draws[j]} share a label but differ by '
                    f'more than the tolerance {tol}')


def test_distinct_frequencies_span_many_buckets():
    """Sanity: the partition is actually used. 5% of the interval means the
    declared range resolves into ~20 distinguishable frequencies."""
    freq = (0.999, 1.001)
    fam = _family(freq)
    labels = {_factor(fam, float(v), freq).structural_label
              for v in np.linspace(freq[0], freq[1], 200)}
    assert len(labels) >= 19, f'only {len(labels)} distinct frequency buckets'
