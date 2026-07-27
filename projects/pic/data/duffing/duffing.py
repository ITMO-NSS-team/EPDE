"""
Forced Duffing oscillator data generator.

    x_tt + delta * x_t + alpha * x + beta * x^3 = gamma * cos(omega * t)

We use a SINGLE-WELL forced regime (alpha=1) so the response is regular
periodic / quasi-periodic -- the PINN has a fair chance to fit it. The
classic double-well chaotic set (alpha=-1, beta=1, delta=0.3, gamma=0.5,
omega=1.2) is too brutal for a stock PINN on long horizons and would
muddy the CV-vs-baseline comparison.

Saves `duffing.npz` with `t`, `x`, plus the ground-truth coefficients.
"""

import numpy as np
from scipy.integrate import solve_ivp


# Single-well forced Duffing: alpha > 0 -> stiffening spring, no double-well.
ALPHA  = 1.0     # linear stiffness
BETA   = 1.0     # cubic stiffness
DELTA  = 0.2     # damping
GAMMA  = 0.3     # forcing amplitude
OMEGA  = 1.0     # forcing frequency

T_MIN, T_MAX = 0.0, 20.0
N_GRID = 1001    # ~50 samples per forcing period (T = 2*pi/omega ~ 6.28)
X0, V0 = 0.0, 0.0


def rhs(t, y):
    x, v = y
    return [v, -ALPHA * x - BETA * x**3 - DELTA * v + GAMMA * np.cos(OMEGA * t)]


def main():
    t_eval = np.linspace(T_MIN, T_MAX, N_GRID)
    sol = solve_ivp(
        rhs, (T_MIN, T_MAX), [X0, V0],
        t_eval=t_eval, method="RK45",
        rtol=1e-10, atol=1e-12,
    )
    assert sol.success, sol.message

    t = sol.t.astype(np.float32)
    x = sol.y[0].astype(np.float32)
    v = sol.y[1].astype(np.float32)

    np.savez(
        "duffing.npz",
        t=t, x=x, v=v,
        alpha=np.float32(ALPHA), beta=np.float32(BETA),
        delta=np.float32(DELTA), gamma=np.float32(GAMMA),
        omega=np.float32(OMEGA),
        x0=np.float32(X0), v0=np.float32(V0),
    )

    print(f"Generated duffing.npz")
    print(f"  N points        = {len(t)}")
    print(f"  t range         = [{t[0]:.2f}, {t[-1]:.2f}]")
    print(f"  x range         = [{x.min():.4f}, {x.max():.4f}]")
    print(f"  v range         = [{v.min():.4f}, {v.max():.4f}]")
    print(f"  (alpha, beta, delta, gamma, omega) = "
          f"({ALPHA}, {BETA}, {DELTA}, {GAMMA}, {OMEGA})")


if __name__ == "__main__":
    main()
