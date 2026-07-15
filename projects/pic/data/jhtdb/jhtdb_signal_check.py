"""Gate check for the JHTDB pilot plane: does the NS momentum balance
close on the sampled data?

  LHS = du/dt + (u.grad)u        (du/dt frame-differenced, rest DNS-exact)
  RHS = -grad p + nu * lap u

Reports per-component OLS R^2 of LHS on [-grad p, nu*lap u] (coefficients
should be ~[1, 1]) plus the fixed-coefficient residual ratio. R^2 ~ 1
passes the gate -> filtered-closure experiment + adapter wiring proceed.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'jhtdb_pilot_plane.npz')


def main():
    d = np.load(NPZ)
    u = d['velocity_field'].astype(np.float64)        # (nt, ny, nx, 3)
    gu = d['velocity_gradient'].astype(np.float64)    # (nt, ny, nx, 9)
    gp = d['pressure_gradient'].astype(np.float64)    # (nt, ny, nx, 3)
    lu = d['velocity_laplacian'].astype(np.float64)   # (nt, ny, nx, 3)
    t, nu = d['t'], float(d['nu'])
    dt = float(t[1] - t[0])

    step = np.abs(np.diff(u, axis=0)).mean()
    print(f'frames: {u.shape[0]}, mean |du| between frames = {step:.4e} '
          f'(zero would mean identical stored frames)')

    # central time derivative on interior frames
    dudt = (u[2:] - u[:-2]) / (2 * dt)
    sl = slice(1, -1)
    u_i, gu_i, gp_i, lu_i = u[sl], gu[sl], gp[sl], lu[sl]

    # advective term: adv_i = u_j * d u_i / d x_j, gradient layout 3*i + j
    adv = np.empty_like(dudt)
    for i in range(3):
        adv[..., i] = np.einsum('...j,...j->...',
                                u_i, gu_i[..., 3 * i:3 * i + 3])

    lhs = dudt + adv
    for i, comp in enumerate('uvw'):
        y = lhs[..., i].ravel()
        F = np.column_stack([-gp_i[..., i].ravel(), nu * lu_i[..., i].ravel()])
        th, *_ = np.linalg.lstsq(np.column_stack([F, np.ones_like(y)]), y,
                                 rcond=None)
        fit = np.column_stack([F, np.ones_like(y)]) @ th
        r2 = 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2)
        resid_fixed = y - F.sum(axis=1)
        ratio = np.linalg.norm(resid_fixed) / np.linalg.norm(y)
        print(f'{comp}-momentum: R2 = {r2:.5f}   coefs [-gradp, nu*lap] = '
              f'[{th[0]:.4f}, {th[1]:.4f}]   fixed-coef |resid|/|lhs| = {ratio:.4f}')


if __name__ == '__main__':
    main()
