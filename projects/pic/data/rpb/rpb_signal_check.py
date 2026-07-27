"""OLS vorticity-transport signal-check for RealPDEBench `cylinder` REAL (PIV) data.

RealPDEBench (AI4Science-WestlakeU, ICLR 2026) ships paired real-PIV / numerical-CFD
trajectories. This is the real-instrument counterpart to the synthetic ns/ test and a
direct sibling of cylinder_piv/signal_check.py -- but reading the HF Arrow shards on
D:\\RPB instead of the MODULO .dat zip.

BEFORE any EPDE run: can the 2D vorticity-transport PDE explain the measured field?
  w_t + u w_x + v w_y = nu (w_xx + w_yy),   w = v_x - u_y
Regress w_t on {-u w_x, -v w_y, (w_xx+w_yy)}. Tells: the two advection coeffs ~ EQUAL
(and ~ -1 if dt/dx are physical); R^2 is dt/dx-robust. The signal is concentrated in
the DEVELOPED SHEDDING region, so we restrict the regression to a data-driven wake
mask (cells whose vorticity varies strongly in time) and sweep the pre-derivative
Gaussian denoise -- exactly the cylinder_piv playbook.

cylinder-real Arrow schema (dataset_info.json), float32 fields + float64 coords:
  u, v, vo : (shape_t, shape_h, shape_w)   -- vo is the dataset's own vorticity (f32)
  x, y     : (x_shape_h, x_shape_w)        -- physical coordinate grids (f64)
  t        : (t_shape,)                    -- physical timestamps (f64)
"""
import os
import argparse
import numpy as np
import pyarrow as pa
from scipy.ndimage import gaussian_filter

DATA_ROOT = r'D:\RPB\cylinder\real'
DEFAULT_SHARD = 'data-00072-of-00073.arrow'


def open_arrow(path):
    """HF `datasets` writes Arrow IPC *stream* (.arrow); try file then stream."""
    src = pa.memory_map(path, 'r')
    try:
        return pa.ipc.open_file(src).read_all()
    except pa.lib.ArrowInvalid:
        src.seek(0)
        return pa.ipc.open_stream(src).read_all()


def decode(row, table, key, shape_keys):
    """Fields (u,v,vo) are float32 blobs; coords (x,y,t) are float64. Infer dtype
    from buffer bytes vs the element count implied by the shape. Returns None if
    the column is absent/null (numerical split stores p but not vo)."""
    if key not in table.column_names:
        return None
    raw = table.column(key)[row].as_py()
    if raw is None:
        return None
    shape = tuple(int(table.column(k)[row].as_py()) for k in shape_keys)
    n = int(np.prod(shape))
    if len(raw) == 4 * n:
        dtype = np.float32
    elif len(raw) == 8 * n:
        dtype = np.float64
    else:
        raise ValueError(f"{key}: {len(raw)} bytes != 4x/8x the {n} elements of {shape}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float64)


def ols(y, cols, names):
    X = np.column_stack(cols + [np.ones_like(y)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1.0 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, dict(zip(names + ['const'], beta))


def transport_ols(U, V, W, dx, dy, dt, mask, label):
    """Vorticity-transport regression of W_t on {U W_x, V W_y, lap W} over the
    interior-time x spatial-`mask` points. W may be the computed curl OR the
    dataset's provided vorticity `vo`."""
    ddx = lambda f: np.gradient(f, dx, axis=2)
    ddy = lambda f: np.gradient(f, dy, axis=1)
    w_x, w_y = ddx(W), ddy(W)
    lap = ddx(w_x) + ddy(w_y)
    w_t = np.gradient(W, dt, axis=0)

    tsl = slice(1, -1)                       # drop first/last frame (central diff)
    m3 = np.broadcast_to(mask[None], W[tsl].shape)
    yt = w_t[tsl][m3].ravel()
    aU = (U[tsl] * w_x[tsl])[m3].ravel()
    aV = (V[tsl] * w_y[tsl])[m3].ravel()
    lp = lap[tsl][m3].ravel()

    r_adv, b_adv = ols(yt, [aU, aV], ['u*w_x', 'v*w_y'])
    r_full, b_full = ols(yt, [aU, aV, lp], ['u*w_x', 'v*w_y', 'lap_w'])
    print(f"  [{label}] adv-only  R^2={r_adv:.4f}  coeffs {b_adv['u*w_x']:+.3f}/{b_adv['v*w_y']:+.3f}"
          f"   | +diff  R^2={r_full:.4f}  adv {b_full['u*w_x']:+.3f}/{b_full['v*w_y']:+.3f}"
          f"  nu={b_full['lap_w']:+.6f}")
    return r_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', default=os.path.join(DATA_ROOT, DEFAULT_SHARD))
    ap.add_argument('--traj', type=int, default=0)
    ap.add_argument('--mask-frac', type=float, default=0.15,
                    help='wake mask: keep cells with std_t(vo) > frac*max(std_t(vo))')
    args = ap.parse_args()

    table = open_arrow(args.shard)
    print(f"[rpb] shard {os.path.basename(args.shard)}: {table.num_rows} trajectories")
    r = args.traj
    U = decode(r, table, 'u', ['shape_t', 'shape_h', 'shape_w'])
    V = decode(r, table, 'v', ['shape_t', 'shape_h', 'shape_w'])
    VO = decode(r, table, 'vo', ['shape_t', 'shape_h', 'shape_w'])
    X = decode(r, table, 'x', ['x_shape_h', 'x_shape_w'])
    Y = decode(r, table, 'y', ['y_shape_h', 'y_shape_w'])
    T = decode(r, table, 't', ['t_shape'])
    U = np.nan_to_num(U); V = np.nan_to_num(V)

    dx = float(np.median(np.diff(X[0, :])))
    dy = float(np.median(np.diff(Y[:, 0])))
    dt = float(np.median(np.diff(T)))
    print(f"[rpb] sim_id={table.column('sim_id')[r].as_py()}  field {U.shape} (T,H,W)")
    print(f"      x in [{X.min():.4f},{X.max():.4f}]  y in [{Y.min():.4f},{Y.max():.4f}]  "
          f"dx={dx:.5g} dy={dy:.5g} dt={dt:.5g}")

    # Vorticity for the wake mask + cross-check. The numerical split has no `vo`
    # column -> fall back to the computed curl for the mask.
    w_raw = np.gradient(V, dx, axis=2) - np.gradient(U, dy, axis=1)
    if VO is not None:
        VO = np.nan_to_num(VO)
        print(f"      |u|max={np.abs(U).max():.3f} |v|max={np.abs(V).max():.3f} "
              f"|vo|max={np.abs(VO).max():.2f}")
        print(f"[rpb] corr(provided vo, computed v_x-u_y) = "
              f"{np.corrcoef(VO.ravel(), w_raw.ravel())[0,1]:.4f}")
        activity = VO.std(axis=0)
    else:
        print(f"      |u|max={np.abs(U).max():.3f} |v|max={np.abs(V).max():.3f} "
              f"(no vo column -> mask from computed curl)")
        activity = w_raw.std(axis=0)
    mask = activity > args.mask_frac * activity.max()
    print(f"[rpb] wake mask: {mask.sum()}/{mask.size} cells ({mask.mean():.1%}) "
          f"with std_t(vo) > {args.mask_frac:g}*max\n")

    print("== vorticity-transport OLS over wake mask, smoothing sweep ==")
    print("-- W = computed curl (v_x - u_y) --")
    best = 0.0
    for sig in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        if sig > 0:
            Us, Vs = gaussian_filter(U, (sig,)*3), gaussian_filter(V, (sig,)*3)
        else:
            Us, Vs = U, V
        W = np.gradient(Vs, dx, axis=2) - np.gradient(Us, dy, axis=1)
        best = max(best, transport_ols(Us, Vs, W, dx, dy, dt, mask, f'curl sig={sig:g}'))

    if VO is not None:
        print("-- W = dataset-provided vo (smooth vo + u,v together) --")
        for sig in (0.0, 1.0):
            if sig > 0:
                Us, Vs, Ws = (gaussian_filter(A, (sig,)*3) for A in (U, V, VO))
            else:
                Us, Vs, Ws = U, V, VO
            best = max(best, transport_ols(Us, Vs, Ws, dx, dy, dt, mask, f'vo   sig={sig:g}'))

    verdict = 'RECOVERABLE' if best > 0.9 else ('marginal' if best > 0.6 else 'WEAK')
    print(f"\n  verdict: best full-model R^2={best:.3f} -> {verdict}")


if __name__ == '__main__':
    main()
