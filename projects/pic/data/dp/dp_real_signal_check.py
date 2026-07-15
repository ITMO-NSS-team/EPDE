"""OLS equation-of-motion signal-check for the low-cost (HardwareX) REAL double
pendulum, Trial 1, video-tracked link angles. Mirrors the SST signal-check: BEFORE any
EPDE run, test whether the classic double-pendulum EOM is recoverable from the measured
theta1, theta2 -- can the EOM library terms explain the measured theta1'', theta2''?

Data: DPmean_data_RB0 = [t, phi1 (upper link)], DPmean_data_RB1 = [t, phi2 (lower)],
500 Hz, radians (unwrapped over chaotic rotations), tracking std ~0.18 rad.

Classic double-pendulum EOM (absolute angles from vertical, Delta = th1 - th2):
  th1'' + A*cos(D)*th2'' + A*sin(D)*th2'^2 + (g/l1)*sin(th1) = 0     A = m2 l2/((m1+m2) l1)
  th2'' + B*cos(D)*th1'' - B*sin(D)*th1'^2 + (g/l2)*sin(th2) = 0     B = l1/l2
So regress th1'' on {cos(D)th2'', sin(D)th2'^2, sin th1} and th2'' on {cos(D)th1'',
sin(D)th1'^2, sin th2}. High R^2 => the EOM structure is present in the real data.
A consistency tell: in eq1 the cos- and sin-term coefficients should be EQUAL; in eq2
they should be NEGATIVES. Derivatives via Savitzky-Golay (noise-robust).
"""
import os
import numpy as np
from scipy.signal import savgol_filter

BASE = os.path.join(os.path.dirname(__file__), 'Video_Tracking_Data', 'Video_Tracking_Data', 'Trial1')


def load(trial_dir):
    rb0 = np.load(os.path.join(trial_dir, 'DPmean_data_RB0.npy'))  # [t, phi1]
    rb1 = np.load(os.path.join(trial_dir, 'DPmean_data_RB1.npy'))  # [t, phi2]
    t, th1, th2 = rb0[0], rb0[1], rb1[1]
    ok = np.isfinite(t) & np.isfinite(th1) & np.isfinite(th2)
    # Tracked angles are in DEGREES (smooth ~140 units/s median => 2.5 rad/s if deg,
    # vs an impossible 140 rad/s if rad; per-sample noise ~0.19 deg). Convert so the
    # trig terms sin(th), cos(th1-th2) get the right argument.
    return t[ok], np.deg2rad(th1[ok]), np.deg2rad(th2[ok])


def ols(y, cols, names):
    X = np.column_stack(cols + [np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, dict(zip(names + ['const'], beta))


def main():
    t, th1, th2 = load(BASE)
    dt = float(np.median(np.diff(t)))
    print(f"[dp signal-check] Trial1 video angles: N={t.size}, dt={dt:.4f}s (fs~{1/dt:.0f}Hz)")
    print(f"  theta1 in [{th1.min():.1f},{th1.max():.1f}] rad   theta2 in [{th2.min():.1f},{th2.max():.1f}] rad")
    print(f"  EOM library: eq1 target th1''  {{cos(D)th2'', sin(D)th2'^2, sin th1}};"
          f"  eq2 target th2''  {{cos(D)th1'', sin(D)th1'^2, sin th2}}\n")
    print(f"  {'SG win':>7} {'win(s)':>7} | {'eq1 R^2':>8} {'cos=sin?':>18} | {'eq2 R^2':>8} {'cos=-sin?':>18}")
    poly = 4
    for win in (21, 51, 101, 201, 401):
        if win >= t.size:
            continue
        d0 = lambda x: savgol_filter(x, win, poly, deriv=0, delta=dt, mode='interp')
        d1 = lambda x: savgol_filter(x, win, poly, deriv=1, delta=dt, mode='interp')
        d2 = lambda x: savgol_filter(x, win, poly, deriv=2, delta=dt, mode='interp')
        T1, T2 = d0(th1), d0(th2)
        w1, a1 = d1(th1), d2(th1)
        w2, a2 = d1(th2), d2(th2)
        D = T1 - T2
        s = slice(win, -win)
        T1, T2, w1, a1, w2, a2, D = (v[s] for v in (T1, T2, w1, a1, w2, a2, D))

        r1, b1 = ols(a1, [np.cos(D) * a2, np.sin(D) * w2 ** 2, np.sin(T1)],
                     ['cosD*a2', 'sinD*w2^2', 'sin_th1'])
        r2, b2 = ols(a2, [np.cos(D) * a1, np.sin(D) * w1 ** 2, np.sin(T2)],
                     ['cosD*a1', 'sinD*w1^2', 'sin_th2'])
        eq1_tell = f"{b1['cosD*a2']:+.3f}/{b1['sinD*w2^2']:+.3f}"
        eq2_tell = f"{b2['cosD*a1']:+.3f}/{b2['sinD*w1^2']:+.3f}"
        print(f"  {win:>7} {win*dt:>7.3f} | {r1:>8.4f} {eq1_tell:>18} | {r2:>8.4f} {eq2_tell:>18}")

    # baselines at a mid window: how much do the NONLINEAR coupling terms add over
    # a gravity-only (sin theta) model?
    win = 101
    d0 = lambda x: savgol_filter(x, win, poly, deriv=0, delta=dt, mode='interp')
    d1 = lambda x: savgol_filter(x, win, poly, deriv=1, delta=dt, mode='interp')
    d2 = lambda x: savgol_filter(x, win, poly, deriv=2, delta=dt, mode='interp')
    T1, T2 = d0(th1), d0(th2)
    w1, a1, w2, a2 = d1(th1), d2(th1), d1(th2), d2(th2)
    D = T1 - T2
    s = slice(win, -win)
    T1, T2, w1, a1, w2, a2, D = (v[s] for v in (T1, T2, w1, a1, w2, a2, D))
    print(f"\n  --- eq1 (th1'') term-ablation @ win={win} ---")
    print(f"    gravity only  {{sin th1}}                      R^2={ols(a1,[np.sin(T1)],['s1'])[0]:.4f}")
    print(f"    + coupling    {{cosD a2, sinD w2^2, sin th1}}  R^2={ols(a1,[np.cos(D)*a2,np.sin(D)*w2**2,np.sin(T1)],['c','s','g'])[0]:.4f}")
    r1f, b1f = ols(a1, [np.cos(D)*a2, np.sin(D)*w2**2, np.sin(T1)], ['cosD*a2','sinD*w2^2','sin_th1'])
    print(f"    full coeffs: {', '.join(f'{k}={v:+.4f}' for k,v in b1f.items())}")
    print(f"  --- eq2 (th2'') term-ablation @ win={win} ---")
    print(f"    gravity only  {{sin th2}}                      R^2={ols(a2,[np.sin(T2)],['s2'])[0]:.4f}")
    print(f"    + coupling    {{cosD a1, sinD w1^2, sin th2}}  R^2={ols(a2,[np.cos(D)*a1,np.sin(D)*w1**2,np.sin(T2)],['c','s','g'])[0]:.4f}")
    r2f, b2f = ols(a2, [np.cos(D)*a1, np.sin(D)*w1**2, np.sin(T2)], ['cosD*a1','sinD*w1^2','sin_th2'])
    print(f"    full coeffs: {', '.join(f'{k}={v:+.4f}' for k,v in b2f.items())}")


if __name__ == "__main__":
    main()
