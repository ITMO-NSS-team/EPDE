"""Conditional joint residual-covariance IRLS; no differential-equation solver."""
import os, sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'variance_reduction_2026-11-16'))
import moments as M
np, R, Q, O, J, F = M.np, M.R, M.Q, M.O, M.J, M.F
RCOND = R.SV._DYAD_RCOND


def least_squares(x, y):
    scale = np.linalg.norm(x, axis=0)
    if np.any(scale == 0):
        return None, dict(status='rank_deficient_design')
    normalized = x / scale
    coef, _, rank, singular = np.linalg.lstsq(normalized, y, rcond=np.sqrt(RCOND))
    ratio = float((singular[-1] / singular[0])**2) if singular[0] else 0.
    if rank != x.shape[1]:
        return None, dict(status='rank_deficient_design', rank=int(rank), eigen_ratio=ratio)
    return coef / scale, dict(status='ok', rank=int(rank), eigen_ratio=ratio)


def ordinary(point, candidate):
    return least_squares(point[:, candidate['xi']], point[:, candidate['yi']])


def irls(point, draws, candidate, tolerance=1e-8, iterations=100):
    xi, yi = candidate['xi'], candidate['yi']
    x, y = point[:, xi], point[:, yi]
    coefficient, initial = least_squares(x, y)
    if coefficient is None:
        return None, dict(initial, iterations=0, trace=[])
    centered = draws - draws.mean(axis=0)
    ex, ey = centered[:, :, xi], centered[:, :, yi]
    trace = []
    for iteration in range(1, iterations + 1):
        error = ey - np.einsum('brj,j->br', ex, coefficient)
        covariance = error.T @ error / (len(draws) - 1)
        eigen, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
        rank = int(np.sum(eigen > RCOND * max(float(eigen[-1]), 0.)))
        entry = dict(iteration=iteration, covariance_rank=rank, rows=len(y),
                     covariance_eigen_ratio=float(eigen[0]/eigen[-1]) if eigen[-1] > 0 else 0.)
        if rank != len(y):
            trace.append(entry)
            return None, dict(status='singular_residual_covariance', iterations=iteration,
                              trace=trace, last_coefficients=coefficient)
        white = (vectors / np.sqrt(eigen)).T
        updated, info = least_squares(white @ x, white @ y)
        if updated is None:
            trace.append(entry)
            return None, dict(status='rank_deficient_weighted_design', iterations=iteration,
                              trace=trace, last_coefficients=coefficient)
        change = float(np.linalg.norm(x @ (updated-coefficient)) /
                       max(np.linalg.norm(y)+np.linalg.norm(x @ coefficient), np.finfo(float).tiny))
        residual = white @ (y-x @ updated)
        entry.update(prediction_change=change, weighted_loss=float(residual @ residual),
                     design_eigen_ratio=info['eigen_ratio'], coefficients=updated)
        trace.append(entry)
        coefficient = updated
        if change <= tolerance:
            return coefficient, dict(status='ok', iterations=iteration, trace=trace)
    return None, dict(status='nonconverged', iterations=iterations, trace=trace,
                      last_coefficients=coefficient)


def checks():
    rng = np.random.default_rng(71523)
    x = np.column_stack([np.ones(11), np.linspace(-1, 1, 11)])
    y = 3 + 2*x[:, 1] + rng.normal(size=11)*.04
    point = np.column_stack([x, y]); candidate = dict(xi=[0, 1], yi=2)
    draws = np.broadcast_to(point, (512, *point.shape)).copy()
    draws[:, :, 2] += rng.normal(size=(512, 11))*np.linspace(.1, .5, 11)
    centered = draws[:, :, 2] - draws[:, :, 2].mean(axis=0)
    covariance = centered.T @ centered / 511
    expected = np.linalg.solve(x.T @ np.linalg.solve(covariance, x), x.T @ np.linalg.solve(covariance, y))
    actual, info = irls(point, draws, candidate)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert info['status'] == 'ok' and info['iterations'] <= 3
    joint = draws.copy(); joint[:, :, 1] += rng.normal(size=(512, 11))*.03
    units = np.array([.01, 100., .2])
    base, bm = irls(point, joint, candidate)
    scaled, sm = irls(point*units, joint*units, candidate)
    assert bm['status'] == sm['status'] == 'ok'
    np.testing.assert_allclose(scaled*units[:2]/units[2], base, rtol=2e-10, atol=2e-10)
    exact = point.copy(); exact[:, 2] = 3 + 2*x[:, 1]
    kept, km = irls(exact, draws, candidate)
    np.testing.assert_allclose(kept, [3., 2.], atol=2e-12)
    zero, zm = irls(point, np.zeros_like(draws), candidate)
    assert zero is None and zm['status'] == 'singular_residual_covariance'
    # Include target/predictor cross terms: C = Cyy - Cyx*c - c*Cxy + c*Cxx*c.
    e = joint - joint.mean(axis=0); coefficient = np.array([1.5, -.7])
    ex = e[:, :, :2] @ coefficient; ey = e[:, :, 2]
    direct = (ey-ex).T @ (ey-ex) / 511
    expanded = (ey.T@ey - ey.T@ex - ex.T@ey + ex.T@ex) / 511
    np.testing.assert_allclose(direct, expanded, atol=2e-15)
    R.dump(ROOT/'checks.json', dict(known_covariance_gls=True, gls_coefficients=actual,
           joint_feature_units_invariant=True, constant_preserved=kept[0],
           zero_covariance_explicitly_infeasible=True, joint_covariance_algebra=True))


if __name__ == '__main__':
    checks()
