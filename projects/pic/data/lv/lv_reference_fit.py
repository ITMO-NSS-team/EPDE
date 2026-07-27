"""Reference Lotka-Volterra least-squares fit to the real Leigh-1968 hare-lynx
data, to benchmark what EPDE discovery should recover.

Conventions match harelynx.py: hare (prey) -> u, lynx (predator) -> v, both in
thousands; t in years (unit step). LV form:
    du/dt =  a*u - b*u*v
    dv/dt =  c*u*v - d*v

Three fits:
  (A) collocation LS on central finite-difference derivatives (closed-form),
  (B) collocation LS on Savitsky-Golay derivatives (mirrors the 'poly' preprocessor),
  (C) trajectory-matching nonlinear LS (integrate the ODE, fit the time series).
"""
import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import odeint
from scipy.optimize import least_squares


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot


def _rel_l2(y, yhat):
    return np.linalg.norm(y - yhat) / np.linalg.norm(y)


def fit_collocation(u, v, du, dv):
    """Linear LS for the LV form (no intercept). Returns (a, b, c, d) and diagnostics."""
    Xu = np.column_stack([u, u * v])          # du = a*u + (-b)*(u v)
    cu, *_ = np.linalg.lstsq(Xu, du, rcond=None)
    Xv = np.column_stack([u * v, v])          # dv = c*(u v) + (-d)*v
    cv, *_ = np.linalg.lstsq(Xv, dv, rcond=None)
    a, b = cu[0], -cu[1]
    c, d = cv[0], -cv[1]
    diag = {
        'r2_u': _r2(du, Xu @ cu), 'r2_v': _r2(dv, Xv @ cv),
        'relL2_u': _rel_l2(du, Xu @ cu), 'relL2_v': _rel_l2(dv, Xv @ cv),
    }
    return (a, b, c, d), diag


def lv_rhs(y, t, a, b, c, d):
    U, V = y
    return [a * U - b * U * V, c * U * V - d * V]


def fit_trajectory(t, u, v, seeds):
    """Multi-start trajectory-matching NLS; returns the best (lowest-cost) fit.

    LV trajectory matching is strongly non-convex, so a single start can collapse
    to a degenerate near-flat minimum (b->0). Try several seeds and keep the best.
    """
    def resid(p):
        sol = odeint(lv_rhs, [u[0], v[0]], t, args=tuple(p))
        return np.concatenate([sol[:, 0] - u, sol[:, 1] - v])
    best = None
    for p0 in seeds:
        try:
            res = least_squares(resid, np.clip(p0, 1e-4, None),
                                bounds=(0.0, np.inf), method='trf', max_nfev=4000)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    a, b, c, d = best.x
    sol = odeint(lv_rhs, [u[0], v[0]], t, args=(a, b, c, d))
    wape_u = np.sum(np.abs(sol[:, 0] - u)) / np.sum(np.abs(u))
    wape_v = np.sum(np.abs(sol[:, 1] - v)) / np.sum(np.abs(v))
    period = 2 * np.pi / np.sqrt(a * d) if a * d > 0 else float('nan')
    return (a, b, c, d), {'wape_u': wape_u, 'wape_v': wape_v,
                          'cost': best.cost, 'LV_period_yr': period}


def dominant_period(t, x):
    """Dominant oscillation period (yr) of a series via FFT of the detrended signal."""
    xd = x - np.polyval(np.polyfit(t, x, 1), t)
    freqs = np.fft.rfftfreq(len(t), d=t[1] - t[0])
    amp = np.abs(np.fft.rfft(xd))
    amp[0] = 0.0
    return 1.0 / freqs[np.argmax(amp)]


def _show(tag, p, diag):
    a, b, c, d = p
    print(f"[{tag}]")
    print(f"    du/dt = {a:+.5f}*u {(-b):+.5f}*u*v")
    print(f"    dv/dt = {c:+.5f}*u*v {(-d):+.5f}*v")
    print(f"    a={a:.5f}  b={b:.5f}  c={c:.5f}  d={d:.5f}")
    print("    " + "  ".join(f"{k}={v:.4f}" for k, v in diag.items()))


def main():
    csv = os.path.join(os.path.dirname(__file__), 'Leigh1968_harelynx.csv')
    raw = np.genfromtxt(csv, delimiter=',', skip_header=1)
    t = raw[:, 0] - raw[0, 0]
    u = raw[:, 1] / 1000.0
    v = raw[:, 2] / 1000.0
    print(f"N={len(t)} points, t in [{t[0]:.0f},{t[-1]:.0f}] yr, "
          f"u in [{u.min():.0f},{u.max():.0f}]k, v in [{v.min():.0f},{v.max():.0f}]k\n")

    # (A) finite-difference collocation
    du_fd, dv_fd = np.gradient(u, t), np.gradient(v, t)
    pA, dA = fit_collocation(u, v, du_fd, dv_fd)
    _show("A  FD collocation", pA, dA)

    # (B) Savitsky-Golay collocation (window/order ~ the 'poly' preprocessor)
    w, po = 9, 3
    u_s, v_s = savgol_filter(u, w, po), savgol_filter(v, w, po)
    du_sg = savgol_filter(u, w, po, deriv=1, delta=1.0)
    dv_sg = savgol_filter(v, w, po, deriv=1, delta=1.0)
    pB, dB = fit_collocation(u_s, v_s, du_sg, dv_sg)
    _show("B  SavGol collocation (matches 'poly')", pB, dB)

    # (C) trajectory-matching NLS, multi-start (collocation fits + literature + variants)
    seeds = [np.array(pB), np.array(pA),
             np.array([0.55, 0.028, 0.026, 0.84]),   # canonical hare-lynx LV
             np.array([0.5, 0.02, 0.02, 0.8]),
             np.array([1.0, 0.05, 0.05, 1.0]),
             np.array([0.3, 0.01, 0.01, 0.5])]
    pC, dC = fit_trajectory(t, u, v, seeds)
    _show("C  trajectory NLS (gold reference, multi-start)", pC, dC)

    print(f"\nObserved dominant period: hare={dominant_period(t, u):.1f} yr, "
          f"lynx={dominant_period(t, v):.1f} yr")

    # SavGol-window sweep: the 'poly' preprocessor defaults to window=9, which
    # spans ~a full 10yr cycle. Show how the fitted rates / R2 react to width.
    print("\nSavGol-window sweep (collocation):")
    print("  win   a       b        c        d        R2_u    R2_v")
    for w in (5, 7, 9, 11, 15):
        us, vs = savgol_filter(u, w, 3), savgol_filter(v, w, 3)
        dus = savgol_filter(u, w, 3, deriv=1, delta=1.0)
        dvs = savgol_filter(v, w, 3, deriv=1, delta=1.0)
        (a, b, c, d), dg = fit_collocation(us, vs, dus, dvs)
        print(f"  {w:>3}  {a:6.3f}  {b:7.4f}  {c:7.4f}  {d:7.4f}  "
              f"{dg['r2_u']:6.3f}  {dg['r2_v']:6.3f}")


if __name__ == "__main__":
    main()
