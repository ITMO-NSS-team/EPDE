"""
Compare PINN variants on the forced Duffing oscillator. Pure aggregator:
reads cached `.npz` files produced by the training scripts, computes a
stats table per coefficient, and writes plots under `plots/`.

Models (each is optional except baseline):
    baseline     -- duffing_pinn_baseline.npz     (pinn_test_baseline.py)
    cv-ols       -- duffing_pinn_cv_ols.npz       (pinn_test_cv.py)
    cv-trainable -- duffing_pinn_cv_trainable.npz (pinn_test_cv_trainable_theta.py)

For cv-trainable, the "headline" theta in the stats table is the
TRAINED `theta_per_window` (an nn.Parameter), not the post-hoc OLS
recovery. We also save a side-by-side plot of trained vs post-hoc OLS
theta for that model so you can see how close the two agree.

Usage:
    python compare_pinns.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cv_metric import cv_forms, error_stats, COEF_NAMES


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
baseline    = _maybe_load("duffing_pinn_baseline.npz",
                          "python pinn_test_baseline.py", required=True)
cv_ols      = _maybe_load("duffing_pinn_cv_ols.npz",
                          "python pinn_test_cv.py")
cv_trained  = _maybe_load("duffing_pinn_cv_trainable.npz",
                          "python pinn_test_cv_trainable_theta.py")

# Headline theta column per model: cv-trainable reports the TRAINED theta;
# the others have only the post-hoc OLS theta.
MODELS = [("baseline",     baseline,   "C0", "theta_per_window")]
if cv_ols is not None:
    MODELS.append(("cv-ols",       cv_ols,     "C1", "theta_per_window"))
if cv_trained is not None:
    MODELS.append(("cv-trainable", cv_trained, "C2", "theta_per_window"))


theta_true = np.array([
    float(baseline["alpha_true"]),
    float(baseline["beta_true"]),
    float(baseline["delta_true"]),
    float(baseline["gamma_true"]),
])
OMEGA = float(baseline["omega_true"])


# ============================================================ stats
def _gather(npz, theta_key):
    errs = error_stats(npz["x_pred"], npz["x_data"])
    forms = cv_forms(npz[theta_key], theta_true=theta_true)
    res = np.abs(npz["residual_grid"])
    return {
        **errs,
        **forms,
        "mean_abs_residual": float(res.mean()),
        "max_abs_residual": float(res.max()),
        "training_time": float(npz["training_time"]),
    }


stats = {name: _gather(npz, key) for (name, npz, _color, key) in MODELS}
model_names = [name for (name, *_rest) in MODELS]


def _fmt(v, fmt):
    return fmt.format(v)


COL_WIDTH = 14


def _row(label, values, fmt="{:.4e}"):
    cells = " | ".join(f"{_fmt(v, fmt):>{COL_WIDTH}}" for v in values)
    return f"| {label:<28} | {cells} |"


header_cells = " | ".join(f"{n:>{COL_WIDTH}}" for n in model_names)
sep_cells = " | ".join("-" * COL_WIDTH for _ in model_names)
lines = [
    f"# Duffing PINN comparison: baseline vs CV-OLS vs CV-trainable-theta",
    f"# true (alpha, beta, delta, gamma, omega) = "
    f"({theta_true[0]}, {theta_true[1]}, {theta_true[2]}, {theta_true[3]}, {OMEGA})",
    "",
    f"| {'metric':<28} | {header_cells} |",
    f"| {'-'*28} | {sep_cells} |",
    _row("rel_l2 (vs data)",       [stats[n]["rel_l2"]            for n in model_names]),
    _row("rmse",                   [stats[n]["rmse"]              for n in model_names]),
    _row("max_abs_err",            [stats[n]["max_abs"]           for n in model_names]),
    _row("mean|ODE residual|",     [stats[n]["mean_abs_residual"] for n in model_names]),
    _row("max|ODE residual|",      [stats[n]["max_abs_residual"]  for n in model_names]),
    _row("cv2_sum (4 coefs)",      [stats[n]["cv2_sum"]           for n in model_names]),
    _row("anchored_mse_sum",       [stats[n]["anchored_mse_sum"]  for n in model_names]),
    _row("training_time_s",        [stats[n]["training_time"]     for n in model_names],
         fmt="{:.1f}"),
]

# Per-coefficient breakdown.
for j, cname in enumerate(COEF_NAMES):
    true_val = theta_true[j]
    lines.append("")
    lines.append(f"## {cname} (true = {true_val})")
    for stat_key, stat_fmt in [
        ("mean",     "{:.6f}"),
        ("median",   "{:.6f}"),
        ("std",      "{:.4e}"),
        ("cv2",      "{:.4e}"),
        ("rel_bias", "{:.4e}"),
        ("min",      "{:.6f}"),
        ("max",      "{:.6f}"),
    ]:
        lines.append(_row(
            f"  {stat_key}",
            [stats[n][cname][stat_key] for n in model_names],
            fmt=stat_fmt,
        ))

# For cv-trainable, also show the post-hoc OLS comparison: did the trained
# theta converge to what an OLS-on-the-final-network would report?
if cv_trained is not None and "theta_per_window_ols" in cv_trained.files:
    ols_forms = cv_forms(cv_trained["theta_per_window_ols"], theta_true=theta_true)
    lines.append("")
    lines.append("## cv-trainable: trained theta vs post-hoc OLS theta")
    lines.append(f"| {'coef':<28} | {'trained mean':>14} | {'OLS mean':>14} | {'true':>14} |")
    lines.append(f"| {'-'*28} | {'-'*14:>14} | {'-'*14:>14} | {'-'*14:>14} |")
    for j, cname in enumerate(COEF_NAMES):
        trained_mean = stats["cv-trainable"][cname]["mean"]
        ols_mean = ols_forms[cname]["mean"]
        lines.append(
            f"| {'  ' + cname:<28} | {trained_mean:>14.6f} | "
            f"{ols_mean:>14.6f} | {theta_true[j]:>14.6f} |"
        )

table = "\n".join(lines)
print(table)
(PLOTS_DIR / "comparison_stats.txt").write_text(table + "\n")


# ============================================================ plots
t = baseline["t"]
x_data = baseline["x_data"]


_NPZ_BY_NAME = {name: npz for (name, npz, _c, _k) in MODELS}


def _model_field(name, key):
    return np.asarray(_NPZ_BY_NAME[name][key])


# ----- Fig 1: trajectory + error
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax = axes[0]
ax.plot(t, x_data, "k-", label="data", linewidth=1.5)
linestyles = {"baseline": "--", "cv-ols": ":", "cv-trainable": "-."}
for name, _npz, color, _ in MODELS:
    ax.plot(t, _model_field(name, "x_pred"), linestyle=linestyles[name],
            color=color, label=name, linewidth=1.1)
ax.set_ylabel("x(t)")
ax.set_title("Duffing trajectory: data vs PINN predictions")
ax.legend()
ax.grid(alpha=0.3)
ax = axes[1]
for name, _npz, color, _ in MODELS:
    ax.plot(t, _model_field(name, "x_pred") - x_data,
            "-", color=color, label=name, linewidth=0.8)
ax.axhline(0, color="k", linestyle=":", alpha=0.5)
ax.set_xlabel("t")
ax.set_ylabel("x_pred - x_data")
ax.set_title("Prediction error")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "trajectory.png", dpi=150)
plt.close(fig)


# ----- Fig 2: ODE residual time series
fig, ax = plt.subplots(figsize=(12, 4))
for name, _npz, color, _ in MODELS:
    ax.plot(t, _model_field(name, "residual_grid"),
            "-", color=color, label=name, linewidth=0.8)
ax.axhline(0, color="k", linestyle=":", alpha=0.5)
ax.set_xlabel("t")
ax.set_ylabel("x_tt + delta x_t + alpha x + beta x^3 - gamma cos(omega t)")
ax.set_title("ODE residual on the data grid (each model uses its own theta)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "residual_timeseries.png", dpi=150)
plt.close(fig)


# ----- Fig 3: per-window coefficient estimates (4 subplots)
markers = {"baseline": "o", "cv-ols": "s", "cv-trainable": "D"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
n_win = len(_model_field("baseline", "theta_per_window"))
idx = np.arange(n_win)
for j, (cname, ax) in enumerate(zip(COEF_NAMES, axes.flat)):
    for name, _npz, color, theta_key in MODELS:
        tw = _model_field(name, theta_key)
        ax.plot(idx, tw[:, j], marker=markers[name], linestyle="-",
                color=color, label=name, markersize=4)
    ax.axhline(theta_true[j], color="k", linestyle="--",
               label=f"true = {theta_true[j]}")
    ax.set_title(f"{cname} per window")
    ax.set_xlabel("window index")
    ax.set_ylabel(cname)
    ax.grid(alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8)
fig.suptitle("Per-window coefficient estimates "
             "(baseline/cv-ols: post-hoc OLS; cv-trainable: trained nn.Parameter)")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "coef_per_window.png", dpi=150)
plt.close(fig)


# ----- Fig 4: coefficient histograms (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for j, (cname, ax) in enumerate(zip(COEF_NAMES, axes.flat)):
    bins = 25
    for name, _npz, color, theta_key in MODELS:
        tw = _model_field(name, theta_key)
        ax.hist(tw[:, j], bins=bins, alpha=0.5, label=name, color=color)
    ax.axvline(theta_true[j], color="k", linestyle="--",
               label=f"true = {theta_true[j]}")
    ax.set_title(f"{cname} distribution across windows")
    ax.set_xlabel(cname)
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8)
fig.suptitle("Per-window coefficient distributions")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "coef_hist.png", dpi=150)
plt.close(fig)


# ----- Fig 5: phase portrait (x vs dx_dt)
dt = float(t[1] - t[0])
v_true = np.gradient(x_data, dt)

panels = [("data (FD dx/dt)", x_data, v_true, "k")]
for name, _npz, color, _ in MODELS:
    panels.append((name, _model_field(name, "x_pred"),
                   _model_field(name, "v_grid_pred"), color))

ncols = len(panels)
fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4),
                         sharex=True, sharey=True)
for ax, (title, xx, vv, color) in zip(axes, panels):
    ax.plot(xx, vv, linewidth=0.6, alpha=0.8, color=color)
    ax.set_xlabel("x")
    ax.set_ylabel("dx/dt")
    ax.set_title(title)
    ax.grid(alpha=0.3)
fig.suptitle("Phase portrait")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "phase_portrait.png", dpi=150)
plt.close(fig)


# ----- Fig 6 (cv-trainable only): trained theta vs post-hoc OLS theta
if cv_trained is not None and "theta_per_window_ols" in cv_trained.files:
    tw_trained = cv_trained["theta_per_window"]
    tw_ols = cv_trained["theta_per_window_ols"]
    n_win_t = tw_trained.shape[0]
    idx_t = np.arange(n_win_t)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, (cname, ax) in enumerate(zip(COEF_NAMES, axes.flat)):
        ax.plot(idx_t, tw_trained[:, j], "D-", color="C2",
                label="trained (nn.Parameter)", markersize=4)
        ax.plot(idx_t, tw_ols[:, j], "x--", color="C3",
                label="post-hoc OLS (diagnostic)", markersize=5)
        ax.axhline(theta_true[j], color="k", linestyle=":",
                   label=f"true = {theta_true[j]}")
        ax.set_title(f"{cname}: trained vs OLS")
        ax.set_xlabel("window index")
        ax.set_ylabel(cname)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("cv-trainable: how close is the trained theta to "
                 "an OLS recovery on the final network?")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cv_trainable_theta_vs_ols.png", dpi=150)
    plt.close(fig)


n_figs = 5 + (1 if cv_trained is not None and "theta_per_window_ols" in cv_trained.files else 0)
print(f"\nWrote {n_figs} figures and comparison_stats.txt to {PLOTS_DIR}/")
