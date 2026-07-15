"""OLS Lotka-Volterra signal-check for the REAL Hudson's Bay Company lynx-hare pelt
records (1900-1920, 21 annual points) -- the real-data counterpart to the synthetic lv/.
Gate: can the LV kinetics explain the measured population rates?
  dH/dt = a*H - b*H*V       (prey/hare: +H, -H*V)
  dV/dt = -c*V + d*H*V       (predator/lynx: -V, +H*V)
21 noisy annual points => derivatives via central finite differences; report R^2 and,
crucially, whether the FOUR coefficient SIGNS come out LV-canonical. Signs matter more
than R^2 here: a tiny noisy series can hit modest R^2 yet still confirm the structure if
the signs and relative magnitudes are physical.
"""
import os
import numpy as np

HERE = os.path.dirname(__file__)
# CSV: Year, Lynx, Hare  (thousands of pelts). H=hare (prey), V=lynx (predator).
CSV = os.path.join(HERE, 'hudson-bay-lynx-hare.csv')


def ols(y, cols, names):
    X = np.column_stack(cols + [np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, dict(zip(names + ['const'], beta))


def main():
    rows = []
    with open(CSV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line[0].isalpha():
                continue
            rows.append([float(x) for x in line.split(',')])
    arr = np.array(rows)
    year, lynx, hare = arr[:, 0], arr[:, 1], arr[:, 2]
    H, V = hare, lynx
    t = year - year[0]
    print(f"[lynx-hare LV] N={H.size} annual points {int(year[0])}-{int(year[-1])}")
    print(f"  hare(H) in [{H.min():.1f},{H.max():.1f}]  lynx(V) in [{V.min():.1f},{V.max():.1f}]")
    # central differences for interior points
    dH = np.gradient(H, t)
    dV = np.gradient(V, t)
    # drop endpoints (one-sided, noisier)
    sl = slice(1, -1)
    H, V, dH, dV = H[sl], V[sl], dH[sl], dV[sl]

    rH, bH = ols(dH, [H, H * V], ['H', 'H*V'])
    rV, bV = ols(dV, [V, H * V], ['V', 'H*V'])
    print(f"\n  dH/dt ~ a*H + b*H*V    R^2={rH:.3f}   a={bH['H']:+.4f} (want +)   b={bH['H*V']:+.5f} (want -)")
    print(f"  dV/dt ~ c*V + d*H*V    R^2={rV:.3f}   c={bV['V']:+.4f} (want -)   d={bV['H*V']:+.5f} (want +)")
    signs_ok = (bH['H'] > 0 and bH['H*V'] < 0 and bV['V'] < 0 and bV['H*V'] > 0)
    print(f"\n  LV-canonical signs all correct? {signs_ok}")
    print(f"  verdict: {'STRUCTURE PRESENT (signs LV-canonical)' if signs_ok else 'signs inconsistent with LV'}"
          f"  | fit quality R^2 = ({rH:.2f}, {rV:.2f})")


if __name__ == "__main__":
    main()
