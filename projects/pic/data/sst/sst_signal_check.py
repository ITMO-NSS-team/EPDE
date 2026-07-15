"""Is there ANY diffusion/advection signal in real SST's dT/dt, or did VWSR sparsity
correctly delete the terms? Plain OLS (no sparsity, no EPDE) of dT/dt on the spatial
derivatives over the real field -- if even unregularized OLS can't find a coherent
diffusion coefficient with decent R^2, the terms genuinely carry no signal.

Derivatives via np.gradient (FD); boundaries trimmed. Reports per-term correlation with
dT/dt, the diffusion-only fit, the full advection-diffusion fit, and term magnitudes.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from sst import load_sst, REGION, COARSEN, N_TIME, TIME_STRIDE

DATA = os.path.join(os.path.dirname(__file__), 'sst_l4.nc')


def d1(F, ax, h):
    return np.gradient(F, h, axis=ax)


def main():
    (t, y, x), field = load_sst(DATA, REGION, COARSEN, N_TIME, TIME_STRIDE)
    dt = float(np.mean(np.diff(t)))
    dy = float(np.mean(np.diff(y)))
    dx = float(np.mean(np.diff(x)))

    T = field
    Tt = d1(T, 0, dt)
    Ty = d1(T, 1, dy)
    Tx = d1(T, 2, dx)
    Tyy = d1(Ty, 1, dy)
    Txx = d1(Tx, 2, dx)
    lap = Txx + Tyy

    s = (slice(2, -2), slice(3, -3), slice(3, -3))   # trim FD edges (t,y,x)
    def flat(A):
        return A[s].reshape(-1)
    Tt_, Txx_, Tyy_, Tx_, Ty_, lap_ = map(flat, (Tt, Txx, Tyy, Tx, Ty, lap))

    print(f"\n[signal-check] real SST, N={Tt_.size:,} interior points")
    print(f"  dt={dt:.2f} day, dx~{dx:.0f} km, dy~{dy:.0f} km")
    print(f"  magnitudes (std):  dT/dt={Tt_.std():.3e}   Laplacian(T_xx+T_yy)={lap_.std():.3e}"
          f"   T_x={Tx_.std():.3e}  T_y={Ty_.std():.3e}")

    print("\n  Pearson corr with dT/dt:")
    for nm, v in [('T_xx', Txx_), ('T_yy', Tyy_), ('T_x', Tx_), ('T_y', Ty_),
                  ('Laplacian', lap_)]:
        r = np.corrcoef(Tt_, v)[0, 1]
        print(f"    corr(dT/dt, {nm:<10}) = {r:+.4f}")

    def ols(cols, names):
        X = np.column_stack([c for c in cols] + [np.ones_like(Tt_)])
        beta, *_ = np.linalg.lstsq(X, Tt_, rcond=None)
        pred = X @ beta
        ss_res = np.sum((Tt_ - pred) ** 2)
        ss_tot = np.sum((Tt_ - Tt_.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        terms = "  ".join(f"{b:+.4e}*{n}" for b, n in zip(beta[:-1], names))
        print(f"    R^2={r2:6.4f}   dT/dt = {terms}  {beta[-1]:+.4e}")
        return r2

    print("\n  OLS fits (no sparsity):")
    print("   diffusion only (kappa*Laplacian):")
    ols([lap_], ['Lap'])
    print("   isotropic diffusion (separate T_xx, T_yy):")
    ols([Txx_, Tyy_], ['T_xx', 'T_yy'])
    print("   advection-diffusion (T_xx,T_yy,T_x,T_y):")
    ols([Txx_, Tyy_, Tx_, Ty_], ['T_xx', 'T_yy', 'T_x', 'T_y'])
    print("   constant only (drift baseline):")
    r2c = 1 - np.sum((Tt_ - Tt_.mean()) ** 2) / np.sum((Tt_ - Tt_.mean()) ** 2)
    print(f"    R^2={r2c:6.4f}   dT/dt = {Tt_.mean():+.4e}   (any model must beat 0)")


if __name__ == "__main__":
    main()
