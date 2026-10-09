r"""Two-timescale velocity memory, fitted through the MSD (window-bias free).

For a stationary velocity with per-component autocovariance C(s), the per-component MSD is
    MSD(t) = 2 int_0^t (t - s) C(s) ds  (+ 2 sn2, the tracking-noise floor)
and every model below has it in closed form (C = A Re e^{-lambda s}:  2 A Re[t/lambda - (1 - e^{-lambda t})/lambda^2]).

Models (per recording, MSD pooled over robots and over x, y; lags 1 frame .. LAG_MAX):
  1exp     C = A e^{-s/tau}                                    (OU, the single-timescale picture)
  2exp     C = A1 e^{-s/tau1} + A2 e^{-s/tau2}                 (fast + slow memory)
  exp+spin C = A1 e^{-s/tau1} + A2 <cos(w_i s)>_i e^{-s/tau2}  (fast memory + propulsion along a
           heading that spins at each robot's own measured rate w_i, losing coherence over tau2)
Fitted in log-MSD; AIC compares them; robot-resampling bootstrap gives the spread.

Validation. The earlier velocity-level regression slope at window w (robot_friction, friction only)
is fixed by the MSD at w and 2w:  Var V = MSD(w)/w^2, Cov(V_next, V) = (MSD(2w) - 2 MSD(w)) / (2 w^2),
b = (Cov/Var - 1)/w. Each model's b(w) is compared with the measured slopes (W = 2, 3, 4, 6 frames).

python robot_vmemory.py
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from robot_discovery import load_track                                      # noqa: E402
from swarm_data import swarm_files, swarm_meta, load_swarm, spin_rates, FPS  # noqa: E402

DT = 1 / FPS
LAG_MAX_S = 5.0    # long enough to see the spin-locked return (MSD dips at ~1-2 s) and caging
LAGS = np.unique(np.round(np.logspace(0, np.log10(LAG_MAX_S * FPS), 32)).astype(int))
W_CHECK = (2, 3, 4, 6)
NBOOT = 40
rng = np.random.default_rng(0)


def per_robot_msd(x, y, lags):
    """(lags, robots) per-component MSD, x and y averaged."""
    out = np.empty((len(lags), x.shape[1]))
    for i, l in enumerate(lags):
        out[i] = 0.5 * (np.mean((x[l:] - x[:-l]) ** 2, axis=0) + np.mean((y[l:] - y[:-l]) ** 2, axis=0))
    return out


def msd_term(t, A, lam):
    """2 A Re[t/lam - (1 - exp(-lam t))/lam^2] for complex or real lam (per-component MSD)."""
    lam = np.asarray(lam, dtype=complex)
    return 2 * A * np.real(t / lam - (1 - np.exp(-lam * t)) / lam ** 2)


def model_msd(name, p, t, omegas):
    sn2 = p[-1]
    if name == "1exp":
        A, tau = p[0], p[1]
        m = msd_term(t, A, 1 / tau)
    elif name == "2exp":                         # A2 free in sign: slow persistence OR cage rebound
        A1, tau1, A2, tau2 = p[:4]
        m = msd_term(t, A1, 1 / tau1) + msd_term(t, A2, 1 / tau2)
    else:                                        # exp+spin(+cage): robot-averaged rotating propulsion
        A1, tau1, A2, tau2 = p[:4]
        rot = np.mean([msd_term(t, A2, 1 / tau2 - 1j * w) for w in omegas], axis=0)
        m = msd_term(t, A1, 1 / tau1) + rot
        if name == "exp+spin+cage":
            m = m + msd_term(t, p[4], 1 / p[5])
    return m + 2 * sn2


MODELS = ("1exp", "2exp", "exp+spin", "exp+spin+cage")


def fit(name, t, msd, omegas, v2):
    """Least squares in log MSD, best of a few starts (v2 = short-lag velocity variance)."""
    if name == "1exp":
        starts, lo, hi = [[v2, 0.1, 0.4]], [1e-3, 1e-3, 0.0], [1e6, 50.0, 25.0]
    elif name == "2exp":
        starts = [[0.8 * v2, 0.1, -0.2 * v2, 1.0, 0.4], [0.7 * v2, 0.05, 0.3 * v2, 0.5, 0.4]]
        lo, hi = [1e-3, 1e-3, -1e6, 1e-3, 0.0], [1e6, 5.0, 1e6, 200.0, 25.0]
    elif name == "exp+spin":
        starts = [[0.7 * v2, 0.1, 0.3 * v2, 5.0, 0.4], [0.9 * v2, 0.05, 0.1 * v2, 1.0, 0.4]]
        lo, hi = [1e-3, 1e-3, 1e-3, 1e-3, 0.0], [1e6, 5.0, 1e6, 200.0, 25.0]
    else:
        starts = [[0.7 * v2, 0.1, 0.3 * v2, 5.0, -0.1 * v2, 1.0, 0.4],
                  [0.9 * v2, 0.05, 0.1 * v2, 2.0, -0.05 * v2, 3.0, 0.4]]
        lo = [1e-3, 1e-3, 1e-3, 1e-3, -1e6, 1e-2, 0.0]
        hi = [1e6, 5.0, 1e6, 200.0, 0.0, 200.0, 25.0]
        starts = [s[:6] + [s[6]] for s in starts]

    def resid(p):
        return np.log(np.maximum(model_msd(name, p, t, omegas), 1e-9)) - np.log(msd)

    best = None
    for x0 in starts:
        x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=20000)
        if best is None or res.cost < best.cost:
            best = res
    p = best.x.copy()
    if name == "2exp" and p[1] > p[3]:           # keep tau1 the fast one
        p = np.array([p[2], p[3], p[0], p[1], p[4]])
    rss = float(np.sum(best.fun ** 2))
    aic = len(t) * np.log(rss / len(t)) + 2 * len(p)
    return p, rss, aic


def slope_from_msd(m_w, m_2w, w):
    var = m_w / w ** 2
    cov = (m_2w - 2 * m_w) / (2 * w ** 2)
    return (cov / var - 1) / w


def analyse(label, x, y, omegas, measured_b=None):
    t = LAGS * DT
    R = per_robot_msd(x, y, LAGS)
    msd = R.mean(axis=1)
    v2 = msd[0] / t[0] ** 2
    out = {"label": label}
    print(f"\n=== {label}: {x.shape[1]} robot(s), |omega| median {np.median(np.abs(omegas)):.2f} rad/s")
    fits = {}
    for name in MODELS:
        p, rss, aic = fit(name, t, msd, omegas, v2)
        boot = []
        for _ in range(NBOOT if x.shape[1] > 1 else 0):
            idx = rng.integers(0, x.shape[1], x.shape[1])
            pb, _, _ = fit(name, t, R[:, idx].mean(axis=1), omegas[idx], v2)
            boot.append(pb)
        sd = np.std(boot, axis=0) if boot else np.full(len(p), np.nan)
        fits[name] = dict(p=p.tolist(), sd=sd.tolist(), rss=rss, aic=aic)
        if name == "1exp":
            desc = f"A {p[0]:.0f}  tau {p[1]:.3f}+-{sd[1]:.3f}s"
        else:
            desc = (f"A1 {p[0]:.0f} tau1 {p[1]:.3f}+-{sd[1]:.3f}s | A2 {p[2]:+.0f} tau2 {p[3]:.3f}+-{sd[3]:.3f}s")
            if name == "exp+spin+cage":
                desc += f" | A3 {p[4]:+.0f} tau3 {p[5]:.3f}+-{sd[5]:.3f}s"
            if name.startswith("exp+spin"):
                desc += f" | circle radius sqrt(A2)/|w| ~ {np.sqrt(p[2]) / np.median(np.abs(omegas)):.1f}px"
        print(f"  {name:8s} {desc} | noise {np.sqrt(p[-1]):.2f}px | log-RSS {rss:.2e} AIC {aic:.1f}")
    best = min(fits, key=lambda k: fits[k]["aic"])
    print(f"  best by AIC: {best}")
    out["fits"] = fits
    # validation against the measured window slopes
    rows = []
    for W in W_CHECK:
        w = W * DT
        pred = {name: slope_from_msd(model_msd(name, np.array(f["p"]), np.array([w]), omegas)[0],
                                     model_msd(name, np.array(f["p"]), np.array([2 * w]), omegas)[0], w)
                for name, f in fits.items()}
        meas = measured_b.get(W) if measured_b else None
        rows.append(dict(W=W, measured=meas, **pred))
        print(f"  W={W} ({w:.2f}s): measured slope {meas if meas is None else round(meas, 3)} | predicted "
              + ", ".join(f"{k} {v:+.3f}" for k, v in pred.items()))
    out["slopes"] = rows
    return out


def measured_slopes():
    path = os.path.join(HERE, "results", "wide_friction.json")
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    out = {}
    for tag, rec in d.items():
        by_w = {}
        for r in rec["calibration"]:
            by_w.setdefault(r["W"], []).append(r["b"])
        out[tag] = {W: float(np.mean(v)) for W, v in by_w.items()}
    return out


def main():
    meas = measured_slopes()
    res = []
    t, x, y, phi = load_track()
    res.append(analyse("single", x[:, None], y[:, None], np.array([np.polyfit(t, phi, 1)[0]]),
                       meas.get("single")))
    for f in swarm_files():
        tag = swarm_meta(f)["tag"]
        t, x, y, phi, ids = load_swarm(f)
        res.append(analyse(tag, x, y, spin_rates(t, phi), meas.get(tag)))
    json.dump(res, open(os.path.join(HERE, "results", "vmemory.json"), "w"), indent=1, default=float)
    print("\nsaved results/vmemory.json")


if __name__ == "__main__":
    main()
