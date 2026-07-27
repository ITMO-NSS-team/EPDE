"""
Convert `cylinder_nektar_wake.mat` (Raissi 2D cylinder wake) into a clean
`ns_data.npz` for the PINN harness.

The .mat is the standard NS-PINN benchmark:
    incompressible flow past a 2D cylinder, Re ~ 100
    -> true (lambda_1, lambda_2) = (1.0, 0.01)

PDE residual form (used by all downstream scripts):
    f_u = u_t + lambda_1 (u u_x + v u_y) + p_x - lambda_2 (u_xx + u_yy)
    f_v = v_t + lambda_1 (u v_x + v v_y) + p_y - lambda_2 (v_xx + v_yy)
    f_div = u_x + v_y                                (incompressibility)

The .mat data is on a regular 100 x 50 grid in (x, y) for 200 timesteps.
We slice to the first `T_TRAIN` timesteps to keep PINN training tractable.
"""

import numpy as np
import scipy.io as scio


MAT_PATH = "cylinder_nektar_wake.mat"
OUT_PATH = "ns_data.npz"

T_TRAIN  = 50           # number of timesteps to keep (matches the existing ns.py)
LAMBDA_1 = 1.0          # ground truth advection coefficient
LAMBDA_2 = 0.01         # ground truth viscosity (= 1/Re, Re=100)


def main():
    data = scio.loadmat(MAT_PATH)
    U_star = data["U_star"]   # (N, 2, T)
    P_star = data["p_star"]   # (N, T)
    t_star = data["t"]        # (T, 1)
    X_star = data["X_star"]   # (N, 2)

    x = np.unique(X_star[:, 0])
    y = np.unique(X_star[:, 1])
    t = t_star.flatten()[:T_TRAIN]

    Nx, Ny, Nt = len(x), len(y), len(t)

    # Reshape: each spatial column of X_star corresponds to a (y, x) cell.
    # The .mat stores points in (x outer, y inner) raveled order — that's
    # what `np.unique(X_star[:,0])` recovers as the x axis.
    u = U_star[:, 0, :T_TRAIN].T.reshape(Nt, Ny, Nx).astype(np.float32)
    v = U_star[:, 1, :T_TRAIN].T.reshape(Nt, Ny, Nx).astype(np.float32)
    p = P_star[:,    :T_TRAIN].T.reshape(Nt, Ny, Nx).astype(np.float32)

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    t = t.astype(np.float32)

    np.savez(
        OUT_PATH,
        t=t, x=x, y=y, u=u, v=v, p=p,
        lambda_1=np.float32(LAMBDA_1),
        lambda_2=np.float32(LAMBDA_2),
    )

    print(f"Generated {OUT_PATH}")
    print(f"  shapes: t={t.shape}, x={x.shape}, y={y.shape}, u/v/p={u.shape}")
    print(f"  t in [{t[0]:.3f}, {t[-1]:.3f}]   (Nt={Nt})")
    print(f"  x in [{x[0]:.3f}, {x[-1]:.3f}]   (Nx={Nx})")
    print(f"  y in [{y[0]:.3f}, {y[-1]:.3f}]   (Ny={Ny})")
    print(f"  u range = [{u.min():.4f}, {u.max():.4f}]")
    print(f"  v range = [{v.min():.4f}, {v.max():.4f}]")
    print(f"  p range = [{p.min():.4f}, {p.max():.4f}]")
    print(f"  (lambda_1, lambda_2) = ({LAMBDA_1}, {LAMBDA_2})")


if __name__ == "__main__":
    main()
