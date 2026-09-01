"""OLS equation-of-motion signal-check for the MultiArm-Pendulum (Brunton/HardwareX,
optical-encoder) free-swing data -- the higher-SNR analogue of dp/dp_real_signal_check.py
(video markers). Encoders log angle AND angular velocity directly, so the acceleration
needs only ONE derivative of the MEASURED velocity (not a double-difference of position).

Gate (same as the SST/video-DP gate): regress each link's angular acceleration onto the
classic Lagrangian EOM library; a high R^2 jump from gravity-only to +coupling means the
EOM structure is genuinely present. Single: th'' ~ {sin th, th'} (gravity + damping).
Double (Delta=th1-th2):
  th1'' ~ {cos(D) th2'', sin(D) th2'^2, sin th1}   (cos- and sin-coeffs ~ equal)
  th2'' ~ {cos(D) th1'', sin(D) th1'^2, sin th2}   (cos- and sin-coeffs ~ negatives)
Triple: per-link with both neighbour couplings.
"""
import os
import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter

HERE = os.path.dirname(__file__)


def ols(y, cols, names):
    X = np.column_stack(cols + [np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, dict(zip(names + ['const'], beta))


def _sg(x, win, poly, deriv, dt):
    return savgol_filter(x, win, poly, deriv=deriv, delta=dt, mode='interp')


def _rng(name, a):
    return f"{name} in [{a.min():.2f},{a.max():.2f}]"


# --------------------------------------------------------------------------- #
def check_single(path, win=201, poly=4):
    m = sio.loadmat(path)
    th = m['Theta1']['signals'][0, 0]['values'][0, 0].ravel()
    tm = m['Theta1']['time'][0, 0].ravel()
    dt = float(np.median(np.diff(tm)))
    print(f"\n### SINGLE  N={th.size}  dt={dt:.5f}s (fs~{1/dt:.0f}Hz)  {_rng('th', th)} rad")
    s = slice(win, -win)
    T = _sg(th, win, poly, 0, dt)[s]
    w = _sg(th, win, poly, 1, dt)[s]
    a = _sg(th, win, poly, 2, dt)[s]
    r_g, _ = ols(a, [np.sin(T)], ['sin'])
    r_f, b = ols(a, [np.sin(T), w], ['sin_th', 'damp'])
    print(f"  gravity-only {{sin th}}      R^2={r_g:.4f}")
    print(f"  + damping    {{sin th, th'}} R^2={r_f:.4f}   g/l={-b['sin_th']:+.3f}  c={b['damp']:+.4f}")
    return ('single', r_g, r_f)


def check_double(path, win=151, poly=4):
    m = sio.loadmat(path)
    th1, th2 = m['Theta1'].ravel(), m['Theta2'].ravel()
    w1m, w2m = m['dTheta1'].ravel(), m['dTheta2'].ravel()
    tm = m['Time'].ravel()
    dt = float(np.median(np.diff(tm)))
    print(f"\n### DOUBLE  N={th1.size}  dt={dt:.5f}s (fs~{1/dt:.0f}Hz)  {_rng('th1',th1)} {_rng('th2',th2)} rad")
    s = slice(win, -win)
    T1 = _sg(th1, win, poly, 0, dt)[s]
    T2 = _sg(th2, win, poly, 0, dt)[s]
    w1 = _sg(w1m, win, poly, 0, dt)[s]      # smooth measured velocity
    w2 = _sg(w2m, win, poly, 0, dt)[s]
    a1 = _sg(w1m, win, poly, 1, dt)[s]      # accel = 1st deriv of measured velocity
    a2 = _sg(w2m, win, poly, 1, dt)[s]
    D = T1 - T2
    rg1, _ = ols(a1, [np.sin(T1)], ['g'])
    rf1, b1 = ols(a1, [np.cos(D)*a2, np.sin(D)*w2**2, np.sin(T1)], ['cosD*a2', 'sinD*w2^2', 'sin_th1'])
    rg2, _ = ols(a2, [np.sin(T2)], ['g'])
    rf2, b2 = ols(a2, [np.cos(D)*a1, np.sin(D)*w1**2, np.sin(T2)], ['cosD*a1', 'sinD*w1^2', 'sin_th2'])
    print(f"  eq1 th1'': gravity-only R^2={rg1:.4f}  -> +coupling R^2={rf1:.4f}"
          f"   tell cos/sin={b1['cosD*a2']:+.3f}/{b1['sinD*w2^2']:+.3f} (expect ~equal)")
    print(f"  eq2 th2'': gravity-only R^2={rg2:.4f}  -> +coupling R^2={rf2:.4f}"
          f"   tell cos/sin={b2['cosD*a1']:+.3f}/{b2['sinD*w1^2']:+.3f} (expect ~negatives)")
    return ('double', min(rg1, rg2), min(rf1, rf2))


def check_triple(path, win=151, poly=4):
    m = sio.loadmat(path)
    th = [m[f'Theta{i}'].ravel() for i in (1, 2, 3)]
    wm = [m[f'dTheta{i}'].ravel() for i in (1, 2, 3)]
    tm = m['Time'].ravel()
    dt = float(np.median(np.diff(tm)))
    print(f"\n### TRIPLE  N={th[0].size}  dt={dt:.5f}s (fs~{1/dt:.0f}Hz)")
    s = slice(win, -win)
    T = [_sg(x, win, poly, 0, dt)[s] for x in th]
    w = [_sg(x, win, poly, 0, dt)[s] for x in wm]
    a = [_sg(x, win, poly, 1, dt)[s] for x in wm]
    rgs, rfs = [], []
    for i, j, k in [(0, 1, 2), (1, 0, 2), (2, 0, 1)]:
        Dij, Dik = T[i]-T[j], T[i]-T[k]
        rg, _ = ols(a[i], [np.sin(T[i])], ['g'])
        rf, b = ols(a[i], [np.cos(Dij)*a[j], np.cos(Dik)*a[k],
                           np.sin(Dij)*w[j]**2, np.sin(Dik)*w[k]**2, np.sin(T[i])],
                    ['cij*aj', 'cik*ak', 'sij*wj2', 'sik*wk2', 'sin'])
        print(f"  eq{i+1} th{i+1}'': gravity-only R^2={rg:.4f}  -> +coupling R^2={rf:.4f}")
        rgs.append(rg); rfs.append(rf)
    return ('triple', min(rgs), min(rfs))


if __name__ == "__main__":
    results = []
    results.append(check_single(os.path.join(HERE, 'Single_FreeSwing_1.mat')))
    results.append(check_double(os.path.join(HERE, 'Double_FreeSwing_1.mat')))
    results.append(check_triple(os.path.join(HERE, 'Triple_FreeSwing_1.mat')))
    print("\n" + "=" * 60)
    print(f"  {'system':8} {'gravity-only R^2':>18} {'+coupling R^2':>16}   verdict")
    for name, rg, rf in results:
        verdict = 'RECOVERABLE' if rf > 0.9 else ('marginal' if rf > 0.6 else 'WEAK')
        print(f"  {name:8} {rg:>18.4f} {rf:>16.4f}   {verdict}")
