"""
Compare the CV-augmented PINN and the torch baseline PINN on the 1D wave
problem. Pure aggregator: reads cached `.npz` files produced by the two
training scripts, computes the comparison metrics, and writes a stats
table + six plots under `plots/`.

Dependencies (must exist before running this script):
    wave_pinn_baseline.npz    -- from pinn_test_baseline.py
    wave_pinn_cv_ols.npz      -- from pinn_test_cv.py

Caveats surfaced in the output:
- The 60 c^2_i values per model come from sliding-window OLS over an
  identical eval pool (deterministic via seed). They are SPATIALLY
  CORRELATED (WIN_FRAC=0.5 -> windows overlap), so the histogram spread
  understates true sampling uncertainty.
- The legacy DeepXDE script `pinn_test.py` is kept on disk for reference
  but is NOT used here -- this comparison is torch-vs-torch only.

Usage:
    python compare_pinns.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cv_metric import cv_forms, error_stats


C2_TRUE = 0.04
HERE = Path(__file__).resolve().parent
PLOTS_DIR = HERE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================ load
def _load(name, hint):
    p = HERE / name
    if not p.exists():
        raise FileNotFoundError(
            f"{name} not found in {HERE}. Run `{hint}` first to produce it."
        )
    return np.load(p, allow_pickle=True)


baseline = _load("wave_pinn_baseline.npz", "python pinn_test_baseline.py")
cv_pinn = _load("wave_pinn_cv_ols.npz", "python pinn_test_cv.py")


# ============================================================ stats
def _row(name, b, c, fmt="{:.4e}"):
    return f"| {name:<22} | {fmt.format(b):>14} | {fmt.format(c):>14} |"


def _gather(npz):
    errs = error_stats(npz["u_pred"], npz["u_data"])
    forms = cv_forms(npz["c2_per_window"], c2_true=C2_TRUE)
    res = np.abs(npz["pde_residual_grid"])
    return {
        **errs,
        **forms,
        "mean_abs_residual": float(res.mean()),
        "max_abs_residual": float(res.max()),
        "training_time": float(npz["training_time"]),
    }


b_stats = _gather(baseline)
c_stats = _gather(cv_pinn)

header = (
    f"# PINN comparison: baseline (PDE residual) vs CV-augmented\n"
    f"# c^2_true = {C2_TRUE}\n\n"
    f"| {'metric':<22} | {'baseline':>14} | {'cv-pinn':>14} |\n"
    f"| {'-'*22} | {'-'*14:>14} | {'-'*14:>14} |"
)
rows = [
    _row("rel_l2 (vs data)",       b_stats["rel_l2"],            c_stats["rel_l2"]),
    _row("rmse",                   b_stats["rmse"],              c_stats["rmse"]),
    _row("max_abs_err",            b_stats["max_abs"],           c_stats["max_abs"]),
    _row("mean(c^2_i)",            b_stats["mean"],              c_stats["mean"],            fmt="{:.6f}"),
    _row("median(c^2_i)",          b_stats["median"],            c_stats["median"],          fmt="{:.6f}"),
    _row("std(c^2_i)",             b_stats["std"],               c_stats["std"]),
    _row("cv2 = var/mean^2",       b_stats["cv2"],               c_stats["cv2"]),
    _row("anchored_mse",           b_stats["anchored_mse"],      c_stats["anchored_mse"]),
    _row("rel_bias",               b_stats["rel_bias"],          c_stats["rel_bias"]),
    _row("c^2_i min",              b_stats["c2_min"],            c_stats["c2_min"],          fmt="{:.6f}"),
    _row("c^2_i max",              b_stats["c2_max"],            c_stats["c2_max"],          fmt="{:.6f}"),
    _row("mean|residual|",         b_stats["mean_abs_residual"], c_stats["mean_abs_residual"]),
    _row("max|residual|",          b_stats["max_abs_residual"],  c_stats["max_abs_residual"]),
    _row("training_time_s",        b_stats["training_time"],     c_stats["training_time"],   fmt="{:.1f}"),
]
table = header + "\n" + "\n".join(rows)
print(table)

# Training-time CV (CV PINN only) is captured at training and may differ from
# the post-hoc cv2 above (different eval pool semantics for autograd vs the
# fixed eval pool we use here).
if "cv2_final" in cv_pinn.files:
    print(f"\n# CV PINN training-time cv2_final (anchored form): "
          f"{float(cv_pinn['cv2_final']):.4e}")

(PLOTS_DIR / "comparison_stats.txt").write_text(table + "\n")


# ============================================================ plotting helpers
def _imshow(ax, field, x, t, **kw):
    """Heatmap with x on horizontal, t on vertical -- matches the data shape."""
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        aspect="auto",
        **kw,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    return im


x = baseline["x"]
t = baseline["t"]
u_data = baseline["u_data"]
u_pred_b = baseline["u_pred"]
u_pred_c = cv_pinn["u_pred"]
c2_b = baseline["c2_per_window"]
c2_c = cv_pinn["c2_per_window"]


# ----- Fig 1: u fields and errors
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
vmin = float(min(u_data.min(), u_pred_b.min(), u_pred_c.min()))
vmax = float(max(u_data.max(), u_pred_b.max(), u_pred_c.max()))
panels_row1 = [
    ("u_data",            u_data),
    ("u_pred baseline",   u_pred_b),
    ("u_pred cv-pinn",    u_pred_c),
]
for ax, (title, field) in zip(axes[0], panels_row1):
    im = _imshow(ax, field, x, t, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

err_b = np.abs(u_pred_b - u_data)
err_c = np.abs(u_pred_c - u_data)
diff = u_pred_b - u_pred_c
panels_row2 = [
    ("|err| baseline",   err_b,  "magma"),
    ("|err| cv-pinn",    err_c,  "magma"),
    ("baseline - cv",    diff,   "RdBu_r"),
]
for ax, (title, field, cmap) in zip(axes[1], panels_row2):
    if cmap == "RdBu_r":
        vlim = float(np.abs(field).max())
        im = _imshow(ax, field, x, t, vmin=-vlim, vmax=vlim, cmap=cmap)
    else:
        im = _imshow(ax, field, x, t, cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

fig.suptitle("Predicted u(x, t) and errors")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "fields.png", dpi=150)
plt.close(fig)


# ----- Fig 2: c^2_i histograms
fig, ax = plt.subplots(figsize=(8, 5))
bins = 30
ax.hist(c2_b, bins=bins, alpha=0.55, label="baseline", color="C0")
ax.hist(c2_c, bins=bins, alpha=0.55, label="cv-pinn",  color="C1")
ax.axvline(C2_TRUE, color="k", linestyle="--", label=f"c^2_true = {C2_TRUE}")
ax.set_xlabel("c^2 per window")
ax.set_ylabel("count (of 60 windows)")
ax.set_title("Distribution of per-window c^2 estimates")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "c2_hist.png", dpi=150)
plt.close(fig)


# ----- Fig 3: spatial scatter of c^2_i
# Window centers: same as make_windows. First 30 are t-windows (full x range),
# next 30 are x-windows (full t range). We plot them on a 2-row layout per model:
# row "t-windows" along x=0.5 horizontal slice, row "x-windows" along t=0.5.
def _window_centers(n=30, frac=0.5, lo=0.0, hi=1.0):
    w = (hi - lo) * frac
    h = w / 2
    return np.linspace(lo + h, hi - h, n, dtype=np.float32)

centers = _window_centers()
# axis along which each window slides:
#   first 30 entries -> moves in t (center_t varies; center_x = 0.5)
#   next 30 entries  -> moves in x (center_x varies; center_t = 0.5)
xs = np.concatenate([np.full(30, 0.5), centers])
ts = np.concatenate([centers,         np.full(30, 0.5)])

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
vmin = float(min(c2_b.min(), c2_c.min()))
vmax = float(max(c2_b.max(), c2_c.max()))
for ax, c2, label in [(axes[0], c2_b, "baseline"),
                      (axes[1], c2_c, "cv-pinn")]:
    sc = ax.scatter(xs, ts, c=c2, vmin=vmin, vmax=vmax, cmap="viridis", s=60)
    ax.set_title(f"{label}: window center vs c^2_i")
    ax.set_xlabel("x (window center)")
    ax.set_ylabel("t (window center)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
fig.colorbar(sc, ax=axes, shrink=0.85, label="c^2_i")
fig.savefig(PLOTS_DIR / "c2_spatial.png", dpi=150)
plt.close(fig)


# ----- Fig 4: c^2_i vs window index
fig, ax = plt.subplots(figsize=(10, 5))
idx = np.arange(len(c2_b))
ax.plot(idx, c2_b, "o-", label="baseline", color="C0", markersize=4)
ax.plot(idx, c2_c, "s-", label="cv-pinn",  color="C1", markersize=4)
ax.axhline(C2_TRUE, color="k", linestyle="--", label=f"c^2_true = {C2_TRUE}")
ax.axvline(29.5, color="0.6", linestyle=":", linewidth=1)
ax.text(14.5, ax.get_ylim()[1], "t-windows",
        ha="center", va="top", color="0.4", fontsize=9)
ax.text(44.5, ax.get_ylim()[1], "x-windows",
        ha="center", va="top", color="0.4", fontsize=9)
ax.set_xlabel("window index (0-29: t-windows, 30-59: x-windows)")
ax.set_ylabel("c^2_i")
ax.set_title("Per-window c^2 estimates")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "c2_per_window.png", dpi=150)
plt.close(fig)


# ----- Fig 5: derivative + residual heatmaps
uxx_b = baseline["u_xx_grid"]
uxx_c = cv_pinn["u_xx_grid"]
utt_b = baseline["u_tt_grid"]
utt_c = cv_pinn["u_tt_grid"]
res_b = baseline["pde_residual_grid"]
res_c = cv_pinn["pde_residual_grid"]

fig, axes = plt.subplots(3, 2, figsize=(11, 13))
row_data = [
    ("u_xx", uxx_b, uxx_c, "RdBu_r"),
    ("u_tt", utt_b, utt_c, "RdBu_r"),
    ("PDE residual = u_tt - 0.04 u_xx", res_b, res_c, "RdBu_r"),
]
for i, (label, fb, fc, cmap) in enumerate(row_data):
    vlim = float(max(np.abs(fb).max(), np.abs(fc).max()))
    for ax, field, title in [(axes[i, 0], fb, "baseline"),
                              (axes[i, 1], fc, "cv-pinn")]:
        im = _imshow(ax, field, x, t, vmin=-vlim, vmax=vlim, cmap=cmap)
        ax.set_title(f"{title}: {label}")
    fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.85)
fig.suptitle("Derivatives and PDE residual on the data grid")
fig.savefig(PLOTS_DIR / "derivatives.png", dpi=150)
plt.close(fig)


# ----- Fig 6: |residual| histograms
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(np.abs(res_b).ravel(), bins=60, alpha=0.55, label="baseline", color="C0")
ax.hist(np.abs(res_c).ravel(), bins=60, alpha=0.55, label="cv-pinn",  color="C1")
ax.set_yscale("log")
ax.set_xlabel("|u_tt - 0.04 u_xx|  (pointwise)")
ax.set_ylabel("count  (log scale)")
ax.set_title("Pointwise PDE residual distribution on the data grid")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS_DIR / "residual_hist.png", dpi=150)
plt.close(fig)


print(f"\nWrote 6 figures and comparison_stats.txt to {PLOTS_DIR}/")
