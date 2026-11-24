"""Check coefficient scaling, noise polarity and rank deficiency against exact reference fits."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parents[3]), str(ROOT.parent/'complementary_windows_2026-11-24')]
from cpu_limit import cap_worker

if __name__ == '__main__':
    cap_worker()

import numpy as np
from uncertainty import improves, regressions, sensitivity


def check():
    rng = np.random.default_rng(71)
    x, z = np.linalg.qr(rng.normal(size=(48, 2)), mode='reduced')[0].T
    a = np.column_stack((2*x - z, x, z))
    noise = np.zeros_like(a)
    noise[:, 0] = .3*x + .1*z
    scale, basis = regressions(a, noise, [0])
    expected = .3 / np.sqrt(5.)
    np.testing.assert_allclose(sensitivity(a, noise, scale, basis), expected)
    np.testing.assert_allclose(sensitivity(a, -noise, scale, basis), expected)

    # Frozen reduced coordinates handle exact collinearity and a zero column.
    duplicated = np.column_stack((a, x, np.zeros_like(x)))
    perturbation = np.column_stack((noise, np.zeros((len(x), 2))))
    units = np.array([-.02, 130., -7., .8, 4.])
    for scaled, draw in ((duplicated, perturbation), (duplicated*units, perturbation*units)):
        scale, fits = regressions(scaled, draw, [0])
        assert fits[0][2].shape == (3, 2)
        np.testing.assert_allclose(sensitivity(scaled, draw, scale, fits), expected/2)
    assert not improves(0., 0.) and not improves(0., np.inf)
    assert not np.isfinite(sensitivity(a, noise, scale, []))
    print('Exact signed refit, rescaling, rank deficiency and empty-target checks passed.')


if __name__ == '__main__':
    check()
