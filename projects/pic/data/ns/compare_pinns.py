"""
Compare NS PINN variants (baseline / cv-ols / cv-trainable) under the
three-candidate-equations generic form.

Models (each optional except baseline):
    baseline     -- ns_pinn_baseline.npz
    cv-ols       -- ns_pinn_cv_ols.npz
    cv-trainable -- ns_pinn_cv_trainable.npz

Stats and plots are split per equation (u_t, v_y, P_y) since each has its
own K and own true coefficient vector. The headline theta for each model:
- baseline: TRUE coefs broadcast across windows (uses truth in residual)
- cv-ols: post-hoc OLS recovery per window
- cv-trainable: trained nn.Parameter per window

Usage:
    python compare_pinns.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cv_metric import EQUATIONS, EQ_NAMES, cv_forms, error_stats


HERE = Path(__file__).resolve().parent
PLOTS_DIR = HERE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================ load
def _maybe_load(name, hint, required=False):
    p = HERE / name
    if not p.exists():
        if required:
            raise FileNotFoundError(
                f"{name} not found in {HERE}. Run `{hint}` first to produce it."
            )
        print(f"  [skip] {name} not found -- omitting (run `{hint}` to include)")
        return None
    return np.load(p, allow_pickle=True)


print("Loading model artifacts...")
baseline   = _maybe_load("ns_pinn_baseline.npz",
                         "python pinn_test_baseline.py", required=True)
cv_ols     = _maybe_load("ns_pinn_cv_ols.npz",
                         "python pinn_test_cv.py")
cv_trained = _maybe_load("ns_pinn_cv_trainable.npz",
                         "python pinn_test_cv_trainable_theta.py")

MODELS = [("baseline", baseline, "C0")]
if cv_ols is not None:
    MODELS.append(("cv-ols", cv_ols, "C1"))
if cv_trained is not None:
    MODELS.append(("cv-trainable", cv_trained, "C2"))

model_names = [name for (name, *_rest) in MODELS]
_NPZ_BY_NAME = {name: npz for (name, npz, _c) in MODELS}


def _model_field(name, key):
    return np.asarray(_NPZ_BY_NAME[name][key])


THETA_TRUE_BY_EQ = {eq["name"]: eq["theta_true"] for eq in EQUATIONS}


# ============================================================ stats
def _gather(npz):
    errs_u = error_stats(npz["u_pred"], npz["u_data"])
    errs_v = error_stats(npz["v_pred"], npz["v_data"])
    out = {
        "rel_l2_u": errs_u["rel_l2"], "rmse_u": errs_u["rmse"], "max_abs_u": errs_u["max_abs"],
        "rel_l2_v": errs_v["rel_l2"], "rmse_v": errs_v["rmse"], "max_abs_v": errs_v["max_abs"],
        "training_time": float(npz["training_time"]),
    }
    for eq in EQUATIONS:
        nm = eq["name"]
        tw = np.asarray(npz[f"theta_{nm}_per_window"])
        out[nm] = cv_forms(tw, THETA_TRUE_BY_EQ[nm])
        res = np.abs(npz[f"residual_{nm}_grid"])
        out[f"{nm}_mean_abs_res"] = float(res.mean())
        out[f"{nm}_max_abs_res"]  = float(res.max())
    return out


stats = {name: _gather(npz) for (name, npz, _c) in MODELS}


COL_WIDTH = 14


def _fmt(v, fmt):
    return fmt.format(v)


def _row(label, values, fmt="{:.4e}"):
    cells = " | ".join(f"{_fmt(v, fmt):>{COL_WIDTH}}" for v in values)
    return f"| {label:<32} | {cells} |"


header_cells = " | ".join(f"{n:>{COL_WIDTH}}" for n in model_names)
sep_cells = " | ".join("-" * COL_WIDTH for _ in model_names)
lines = [
    "# NS PINN comparison -- 3 candidate equations, generic form",
    "# true theta per eq:",
    *[f"#   {eq['name']:>4} ({len(eq['theta_true'])} terms): {eq['theta_true'].tolist()}"
      for eq in EQUATIONS],
    "",
    f"| {'metric':<32} | {header_cells} |",
    f"| {'-'*32} | {sep_cells} |",
    _row("rel_l2 (u)",          [stats[n]["rel_l2_u"]      for n in model_names]),
    _row("rel_l2 (v)",          [stats[n]["rel_l2_v"]      for n in model_names]),
    _row("rmse (u)",            [stats[n]["rmse_u"]        for n in model_names]),
    _row("rmse (v)",            [stats[n]["rmse_v"]        for n in model_names]),
    _row("training_time_s",     [stats[n]["training_time"] for n in model_names],
         fmt="{:.1f}"),
]

for eq in EQUATIONS:
    nm = eq["name"]
    lines.append("")
    lines.append(f"## Eq {nm} (K = {eq['k']}, true = {eq['theta_true'].tolist()})")
    lines.append(_row(f"  mean|residual|",  [stats[n][f"{nm}_mean_abs_res"] for n in model_names]))
    lines.append(_row(f"  max|residual|",   [stats[n][f"{nm}_max_abs_res"]  for n in model_names]))
    lines.append(_row(f"  cv2_sum",         [stats[n][nm]["cv2_sum"]        for n in model_names]))
    lines.append(_row(f"  anchored_mse_sum",[stats[n][nm]["anchored_mse_sum"] for n in model_names]))
    for j in range(eq["k"]):
        fname = eq["feature_names"][j]
        true_val = float(eq["theta_true"][j])
        lines.append(_row(
            f"  [{fname:<7}] mean  (true={true_val:.3f})",
            [stats[n][nm][j]["mean"] for n in model_names],
            fmt="{:.5f}"))
        lines.append(_row(
            f"  [{fname:<7}] std",
            [stats[n][nm][j]["std"] for n in model_names]))
        lines.append(_row(
            f"  [{fname:<7}] rel_bias",
            [stats[n][nm][j]["rel_bias"] for n in model_names]))

# cv-trainable: trained-theta vs post-hoc-OLS theta comparison
if cv_trained is not None:
    lines.append("")
    lines.append("## cv-trainable: trained theta vs post-hoc OLS theta")
    for eq in EQUATIONS:
        nm = eq["name"]
        trained = np.asarray(cv_trained[f"theta_{nm}_per_window"]).mean(axis=0)
        ols     = np.asarray(cv_trained[f"theta_{nm}_per_window_ols"]).mean(axis=0)
        for j in range(eq["k"]):
            fname = eq["feature_names"][j]
            lines.append(
                f"| {nm:>4} [{fname:<7}]          | "
                f"trained={trained[j]:>10.5f} | OLS={ols[j]:>10.5f} | "
                f"true={float(eq['theta_true'][j]):>10.5f} |"
            )

table = "\n".join(lines)
print(table)
(PLOTS_DIR / "comparison_stats.txt").write_text(table + "\n")


# ============================================================ plots
t = baseline["t"]
x = baseline["x"]
y = baseline["y"]
u_data = baseline["u_data"]
v_data = baseline["v_data"]
SNAPSHOT = u_data.shape[0] // 2


def _imshow_xy(ax, field2d, **kw):
    im = ax.imshow(
        field2d, origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto", **kw,
    )
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return im


# ----- Fig 1: u snapshot and errors
ncols = len(MODELS) + 1
fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7))
vmin = float(u_data[SNAPSHOT].min()); vmax = float(u_data[SNAPSHOT].max())
ax = axes[0, 0]
im = _imshow_xy(ax, u_data[SNAPSHOT], vmin=vmin, vmax=vmax, cmap="viridis")
ax.set_title(f"u_data  (t={t[SNAPSHOT]:.2f})"); fig.colorbar(im, ax=ax)
for k, (name, _npz, _c) in enumerate(MODELS):
    ax = axes[0, k + 1]
    im = _imshow_xy(ax, _model_field(name, "u_pred")[SNAPSHOT],
                    vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(f"u_pred {name}"); fig.colorbar(im, ax=ax)
axes[1, 0].axis("off")
for k, (name, _npz, _c) in enumerate(MODELS):
    ax = axes[1, k + 1]
    field = _model_field(name, "u_pred")[SNAPSHOT] - u_data[SNAPSHOT]
    vlim = float(np.abs(field).max() + 1e-12)
    im = _imshow_xy(ax, field, vmin=-vlim, vmax=vlim, cmap="RdBu_r")
    ax.set_title(f"u err ({name})"); fig.colorbar(im, ax=ax)
fig.suptitle(f"u snapshot at t = {t[SNAPSHOT]:.2f}")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "u_snapshot.png", dpi=150)
plt.close(fig)


# ----- Fig 2: v snapshot
fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7))
vmin = float(v_data[SNAPSHOT].min()); vmax = float(v_data[SNAPSHOT].max())
ax = axes[0, 0]
im = _imshow_xy(ax, v_data[SNAPSHOT], vmin=vmin, vmax=vmax, cmap="viridis")
ax.set_title(f"v_data  (t={t[SNAPSHOT]:.2f})"); fig.colorbar(im, ax=ax)
for k, (name, _npz, _c) in enumerate(MODELS):
    ax = axes[0, k + 1]
    im = _imshow_xy(ax, _model_field(name, "v_pred")[SNAPSHOT],
                    vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(f"v_pred {name}"); fig.colorbar(im, ax=ax)
axes[1, 0].axis("off")
for k, (name, _npz, _c) in enumerate(MODELS):
    ax = axes[1, k + 1]
    field = _model_field(name, "v_pred")[SNAPSHOT] - v_data[SNAPSHOT]
    vlim = float(np.abs(field).max() + 1e-12)
    im = _imshow_xy(ax, field, vmin=-vlim, vmax=vlim, cmap="RdBu_r")
    ax.set_title(f"v err ({name})"); fig.colorbar(im, ax=ax)
fig.suptitle(f"v snapshot at t = {t[SNAPSHOT]:.2f}")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "v_snapshot.png", dpi=150)
plt.close(fig)


# ----- Fig 3: per-equation residual snapshot per model
neq = len(EQUATIONS)
fig, axes = plt.subplots(neq, len(MODELS), figsize=(4 * len(MODELS), 3 * neq))
if len(MODELS) == 1:
    axes = axes.reshape(-1, 1)
if neq == 1:
    axes = axes.reshape(1, -1)
for r, eq in enumerate(EQUATIONS):
    nm = eq["name"]
    for c, (name, _npz, _color) in enumerate(MODELS):
        ax = axes[r, c]
        field = _model_field(name, f"residual_{nm}_grid")[SNAPSHOT]
        vlim = float(np.abs(field).max() + 1e-12)
        im = _imshow_xy(ax, field, vmin=-vlim, vmax=vlim, cmap="RdBu_r")
        ax.set_title(f"residual {nm}  ({name})"); fig.colorbar(im, ax=ax)
fig.suptitle(f"Per-equation residuals at t = {t[SNAPSHOT]:.2f}")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "residuals_snapshot.png", dpi=150)
plt.close(fig)


# ----- Fig 4: mean residual time series (one panel per equation)
fig, axes = plt.subplots(neq, 1, figsize=(12, 3 * neq), sharex=True)
if neq == 1:
    axes = [axes]
for r, eq in enumerate(EQUATIONS):
    nm = eq["name"]
    ax = axes[r]
    for name, _npz, color in MODELS:
        field = _model_field(name, f"residual_{nm}_grid")
        ax.plot(t, np.abs(field).mean(axis=(1, 2)),
                "-", color=color, label=name, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel(f"mean|residual {nm}|")
    ax.legend(); ax.grid(alpha=0.3)
axes[-1].set_xlabel("t")
fig.suptitle("Per-equation residual time series (mean over space, log scale)")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "residual_timeseries.png", dpi=150)
plt.close(fig)


# ----- Fig 5: per-window coefficient estimates (one figure per equation)
markers = {"baseline": "o", "cv-ols": "s", "cv-trainable": "D"}
for eq in EQUATIONS:
    nm = eq["name"]
    K = eq["k"]
    n_win = _model_field("baseline", f"theta_{nm}_per_window").shape[0]
    idx = np.arange(n_win)
    cols = min(3, K)
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows),
                              squeeze=False)
    for j in range(K):
        ax = axes[j // cols, j % cols]
        for name, _npz, color in MODELS:
            tw = _model_field(name, f"theta_{nm}_per_window")
            ax.plot(idx, tw[:, j], marker=markers[name], linestyle="-",
                    color=color, label=name, markersize=5)
        ax.axhline(float(eq["theta_true"][j]), color="k", linestyle="--",
                   label=f"true = {eq['theta_true'][j]:.3f}")
        ax.set_title(f"{nm}: {eq['feature_names'][j]}")
        ax.set_xlabel("window index"); ax.set_ylabel("coefficient")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for j in range(K, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(f"Per-window coefficients for equation {nm}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"coef_per_window_{nm}.png", dpi=150)
    plt.close(fig)


# ----- Fig 6: cv-trainable trained vs post-hoc OLS (one figure per equation)
if cv_trained is not None:
    for eq in EQUATIONS:
        nm = eq["name"]
        K = eq["k"]
        tw_trained = np.asarray(cv_trained[f"theta_{nm}_per_window"])
        tw_ols     = np.asarray(cv_trained[f"theta_{nm}_per_window_ols"])
        cols = min(3, K)
        rows = (K + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows),
                                  squeeze=False)
        idx_t = np.arange(tw_trained.shape[0])
        for j in range(K):
            ax = axes[j // cols, j % cols]
            ax.plot(idx_t, tw_trained[:, j], "D-", color="C2",
                    label="trained (nn.Parameter)", markersize=5)
            ax.plot(idx_t, tw_ols[:, j], "x--", color="C3",
                    label="post-hoc OLS", markersize=6)
            ax.axhline(float(eq["theta_true"][j]), color="k", linestyle=":",
                       label=f"true = {eq['theta_true'][j]:.3f}")
            ax.set_title(f"{nm}: {eq['feature_names'][j]}")
            ax.set_xlabel("window index"); ax.set_ylabel("coefficient")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        for j in range(K, rows * cols):
            axes[j // cols, j % cols].axis("off")
        fig.suptitle(f"cv-trainable: trained vs post-hoc OLS theta for {nm}")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"cv_trainable_theta_vs_ols_{nm}.png", dpi=150)
        plt.close(fig)


n_main = 4 + len(EQUATIONS)
n_extra = len(EQUATIONS) if cv_trained is not None else 0
print(f"\nWrote {n_main + n_extra} figures and comparison_stats.txt to {PLOTS_DIR}/")
