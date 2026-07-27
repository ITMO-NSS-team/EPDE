"""
Double pendulum data generator.

Equal masses and lengths (m1=m2=1, L1=L2=1) under gravity g. The Lagrangian
gives two coupled ODEs in (theta1, theta2):

    2*theta1_tt + theta2_tt*cos(theta1-theta2)
                + theta2_t^2 * sin(theta1-theta2)
                + 2*g*sin(theta1) = 0

    theta2_tt + theta1_tt*cos(theta1-theta2)
              - theta1_t^2 * sin(theta1-theta2)
              + g*sin(theta2) = 0

We integrate the explicit form (linear-solve in the (theta1_tt, theta2_tt)
pair at each step) with RK45 high-precision.

PINN-friendly initial conditions: (theta1, theta2) = (1.0, -0.5) and zero
angular velocities. This is mildly nonlinear (the double pendulum is
chaotic at larger amplitudes -- (2.0, 0.5) etc.) but stays bounded and
non-chaotic over the chosen horizon, so a stock PINN has a fair chance to
fit it. The CV-vs-baseline comparison should focus on coefficient recovery,
not on the network's ability to track chaos.

Saves `dp.npz` with t, theta1, theta2, omega1, omega2, plus ground-truth
constants.
"""

import numpy as np
from scipy.integrate import solve_ivp


G   = 9.81    # gravity
L1  = 1.0
L2  = 1.0
M1  = 1.0
M2  = 1.0

T_MIN, T_MAX = 0.0, 10.0
N_GRID = 1001

# PINN-friendly mid-amplitude IC; non-chaotic over [0, 10].
THETA1_0 = 1.0
THETA2_0 = -0.5
OMEGA1_0 = 0.0
OMEGA2_0 = 0.0


def rhs(t, y):
    """Equal-mass, equal-length double pendulum -- analytic explicit form."""
    th1, th2, w1, w2 = y
    d = th1 - th2
    c, s = np.cos(d), np.sin(d)
    # M @ [th1_tt, th2_tt]^T = b
    #   M = [[2, c], [c, 1]],  b = [-w2^2*s - 2*g*sin(th1),
    #                                w1^2*s - g*sin(th2)]
    denom = 2.0 - c * c
    b1 = -w2 * w2 * s - 2.0 * G * np.sin(th1)
    b2 =  w1 * w1 * s -        G * np.sin(th2)
    th1_tt = (b1 - c * b2) / denom
    th2_tt = (2.0 * b2 - c * b1) / denom
    return [w1, w2, th1_tt, th2_tt]


def main():
    t_eval = np.linspace(T_MIN, T_MAX, N_GRID)
    sol = solve_ivp(
        rhs, (T_MIN, T_MAX),
        [THETA1_0, THETA2_0, OMEGA1_0, OMEGA2_0],
        t_eval=t_eval, method="RK45",
        rtol=1e-10, atol=1e-12,
    )
    assert sol.success, sol.message

    t  = sol.t.astype(np.float32)
    th1 = sol.y[0].astype(np.float32)
    th2 = sol.y[1].astype(np.float32)
    w1  = sol.y[2].astype(np.float32)
    w2  = sol.y[3].astype(np.float32)

    np.savez(
        "dp.npz",
        t=t,
        theta1=th1, theta2=th2,
        omega1=w1,  omega2=w2,
        g=np.float32(G),
        L1=np.float32(L1), L2=np.float32(L2),
        m1=np.float32(M1), m2=np.float32(M2),
        theta1_0=np.float32(THETA1_0), theta2_0=np.float32(THETA2_0),
        omega1_0=np.float32(OMEGA1_0), omega2_0=np.float32(OMEGA2_0),
    )

    print(f"Generated dp.npz")
    print(f"  N points        = {len(t)}")
    print(f"  t range         = [{t[0]:.2f}, {t[-1]:.2f}]")
    print(f"  theta1 range    = [{th1.min():.4f}, {th1.max():.4f}]")
    print(f"  theta2 range    = [{th2.min():.4f}, {th2.max():.4f}]")
    print(f"  omega1 range    = [{w1.min():.4f}, {w1.max():.4f}]")
    print(f"  omega2 range    = [{w2.min():.4f}, {w2.max():.4f}]")
    print(f"  (g, L1, L2, m1, m2) = ({G}, {L1}, {L2}, {M1}, {M2})")
    print(f"  IC (th1, th2, w1, w2) = "
          f"({THETA1_0}, {THETA2_0}, {OMEGA1_0}, {OMEGA2_0})")


if __name__ == "__main__":
    main()
