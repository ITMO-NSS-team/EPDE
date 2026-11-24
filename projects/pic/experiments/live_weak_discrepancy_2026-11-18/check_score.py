"""Check the live columns and fitted score against the earlier matrix implementation."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import live_score as L
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core15_implicit_correction_2026-11-18'))
from evaluate import corrected_fit
from train import TrainingMoments


def checks():
    rng = np.random.default_rng(53)
    t = np.linspace(0, 1, 61)
    window = L.Window([t], (2,), 7)
    u = np.cos(2*t)
    weak, op = L.R.symbols(SimpleNamespace(shape=u.shape, data={'u': u}, grids=[t]), [
        (), (L.R.atom('u'),), (L.R.atom('du/dx0'),),
        (('u', (('power', 2.),)), L.R.atom('du/dx0')),
        (L.R.atom('cos', freq=2., dim=0.), L.R.atom('du/dx0'))])
    scalar = L.Measurements(window, {'u': u})
    cov = L.O.Covariance([t], np.eye(len(t)), {'u': (.2, [0.], [])})
    held = L.Measurements(window, {'u': u}, cov)
    bank = SimpleNamespace(terms=weak, monos=[term.monomials(window.q) for term in weak],
                           shape=window.shape, axes=[t], kernels=window.kernels, opaque=op)
    expected = window.basis.convert(L.O.matrix(bank, {'u': u}, cov))
    actual = np.column_stack([held.column(term, op) for term in weak])
    np.testing.assert_allclose(actual, expected, rtol=1.e-10, atol=1.e-11)
    batched = L.Measurements(window, {'u': np.stack([u, u])})
    for term in weak:
        a = scalar.column(term, op)
        np.testing.assert_allclose(batched.column(term, op), np.stack([a, a]), atol=1.e-12)
    fourth = L.R.WF.WeakTerm([('u',(0,))]*4, [0])
    fourth_column = window.real(window.integrate(u**4 - 6*.2**2*u**2 + 3*.2**4, (0,)))
    np.testing.assert_allclose(held.column(fourth, {}), fourth_column, atol=1.e-12)

    units = np.array([.01, -2., 300.])
    labels = tuple(range(len(units)))
    train = rng.normal(size=(3, L.DRAWS+1, 9, len(units)))
    raw = rng.normal(size=(3, 9, len(units)))
    white = rng.normal(size=(3, 9, 9))
    def live(mult):
        scorer = L.LiveScore.__new__(L.LiveScore)
        scorer.columns = {j: [(train[g,...,j]*mult[j], raw[g,:,j]*mult[j]) for g in range(3)] for j in labels}
        scorer.folds = [(None, None, w) for w in white]
        return scorer
    observed = live(np.ones(3)).support_score(labels)
    scale_changed = live(units).support_score(labels[::-1])
    ref = []
    for g in range(3):
        stats = TrainingMoments(train[g,0])
        for sample in train[g,1:]:
            stats.update(sample)
        vector, _, status = corrected_fit(stats.corrected(), np.linalg.norm(train[g,0],axis=0))
        assert status == 'ok'
        left,right = [white[g] @ raw[h] for h in range(3) if g != h]
        ref.append(L.score(left, right, np.linalg.norm(white[g] @ train[g,0],axis=0), vector))
    np.testing.assert_allclose([observed,scale_changed], np.mean(ref), rtol=1.e-10, atol=1.e-12)
    print(json.dumps(dict(column_error=float(np.max(abs(actual-expected))), fourth_power_checked=True,
                          score_error=float(abs(observed-np.mean(ref))), rescaling_error=float(abs(observed-scale_changed)))))


if __name__ == '__main__':
    checks()
