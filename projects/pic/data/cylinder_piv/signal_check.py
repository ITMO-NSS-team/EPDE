"""OLS vorticity-transport signal-check for the real TR-PIV cylinder-wake (von Karman
Institute / MODULO Ex_5). Real-instrument counterpart to the synthetic ns/ Navier-Stokes
test. BEFORE any EPDE run: can the 2D vorticity-transport PDE explain the measured field?
  w_t + u w_x + v w_y = nu (w_xx + w_yy),   w = v_x - u_y
Regress w_t on {-u w_x, -v w_y, (w_xx+w_yy)} over the developed shedding region.
Tells: the two advection coeffs ~ EQUAL (and ~ -1 if dt is right); R^2 is dt-robust
(scaling w_t does not change R^2). Viscous coeff small-positive; on a coarse 71x30 grid
the 2nd-derivative (diffusion) term is expected to be the marginal one.
"""
import os
import zipfile
from io import StringIO
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(__file__)
ZIP = os.path.join(HERE, 'Ex_5_TR_PIV_Cylinder.zip')
DT = 1.0 / 3000.0       # TR-PIV ~3 kHz (R^2 is robust to this; coeff magnitude is not)
I0, I1 = 300, 800       # contiguous snapshot block in developed shedding
# Light Gaussian denoise on (u,v) over (t,y,x) BEFORE differentiating. Raw PIV vorticity
# (curl of a noisy field, then w_t / Laplacian) is noise-dominated (advection R^2~0.10);
# sigma~1 peaks the recoverable advection structure at R^2~0.60. Heavier smoothing
# (sigma>=2) blurs the shedding vortices and collapses R^2 again -- a narrow sweet spot.
SMOOTH = (1.0, 1.0, 1.0)


def load_dat(z, name):
    return np.loadtxt(StringIO(z.read(name).decode('latin-1')), skiprows=1)


def ols(y, cols, names):
    X = np.column_stack(cols + [np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1.0 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, dict(zip(names + ['const'], beta))


def main():
    z = zipfile.ZipFile(ZIP)
    mesh = load_dat(z, 'MESH.dat')
    x_mm, y_mm = mesh[:, 0], mesh[:, 1]
    N = mesh.shape[0]
    nx = int(np.where(np.diff(y_mm) != 0)[0][0]) + 1   # x is the fast axis
    ny = N // nx
    X = (x_mm.reshape(ny, nx)) * 1e-3                   # meters
    Y = (y_mm.reshape(ny, nx)) * 1e-3
    dx = float(np.median(np.diff(X[0, :])))
    dy = float(np.median(np.diff(Y[:, 0])))
    print(f"[piv] grid {ny} x {nx} = {N} pts, dx={dx*1e3:.3f}mm dy={dy*1e3:.3f}mm, "
          f"snapshots {I0}..{I1} @ dt={DT*1e3:.3f}ms")

    U, V = [], []
    for i in range(I0, I1):
        a = load_dat(z, f'Res{i:05d}.dat')
        U.append(a[:, 0].reshape(ny, nx))
        V.append(a[:, 1].reshape(ny, nx))
    U, V = np.array(U), np.array(V)
    print(f"  u in [{np.nanmin(U):.2f},{np.nanmax(U):.2f}] m/s, NaNs: {np.isnan(U).sum()+np.isnan(V).sum()}")
    U = np.nan_to_num(U); V = np.nan_to_num(V)
    U = gaussian_filter(U, SMOOTH); V = gaussian_filter(V, SMOOTH)   # denoise before deriv
    print(f"  Gaussian denoise sigma(t,y,x)={SMOOTH}")

    ddx = lambda f: np.gradient(f, dx, axis=2)
    ddy = lambda f: np.gradient(f, dy, axis=1)
    w = ddx(V) - ddy(U)
    w_x, w_y = ddx(w), ddy(w)
    lap = ddx(w_x) + ddy(w_y)
    w_t = np.gradient(w, DT, axis=0)

    sl = (slice(1, -1), slice(3, -3), slice(3, -3))   # drop time + spatial edges
    yt = w_t[sl].ravel()
    aU = (U[sl] * w_x[sl]).ravel()
    aV = (V[sl] * w_y[sl]).ravel()
    lp = lap[sl].ravel()

    r_adv, b_adv = ols(yt, [aU, aV], ['u*w_x', 'v*w_y'])
    r_full, b_full = ols(yt, [aU, aV, lp], ['u*w_x', 'v*w_y', 'lap_w'])
    print(f"\n  advection-only {{u w_x, v w_y}}        R^2={r_adv:.4f}"
          f"   coeffs {b_adv['u*w_x']:+.3f}/{b_adv['v*w_y']:+.3f} (expect ~equal, ~-1)")
    print(f"  + diffusion    {{..., w_xx+w_yy}}      R^2={r_full:.4f}"
          f"   adv {b_full['u*w_x']:+.3f}/{b_full['v*w_y']:+.3f}  nu={b_full['lap_w']:+.5f} (expect small +)")
    verdict = 'RECOVERABLE' if r_full > 0.9 else ('marginal' if r_full > 0.6 else 'WEAK')
    print(f"\n  verdict: advection structure {'PRESENT' if r_adv > 0.6 else 'weak'} | full R^2={r_full:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
