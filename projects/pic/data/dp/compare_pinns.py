"""
Compare DP PINN variants (baseline / cv-ols / cv-trainable) under the
two-candidate-equations generic form.

Models (each optional except baseline):
    baseline     -- dp_pinn_baseline.npz
    cv-ols       -- dp_pinn_cv_ols.npz
    cv-trainable -- dp_pinn_cv_trainable.npz

Stats and plots are split per equation (th1, th2) since each has its own
K=3 and own true coefficient vector. The headline theta for each model:
- baseline: TRUE coefs broadcast across windows (uses truth in residual)
- cv-ols: post-hoc OLS recovery per window
- cv-trainable: trained nn.Parameter per window

Usage:
    python compare_pinns.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cv_metric import EQUATIONS, EQ_NAMES, cv_forms, error_stats, set_gravity


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
# Each algorithm has two variants: rectangular windowing and circular
# windowing (set CIRCULAR_WINDOWS=True in the relevant PINN script).
# Both are loaded as separate model entries if their .npz exists.
_CANDIDATES = [
    ("baseline",          "dp_pinn_baseline.npz",                 "C0",
     "python pinn_test_baseline.py"),
    ("baseline-circ",     "dp_pinn_baseline_circular.npz",        "C5",
     "python pinn_test_baseline.py with CIRCULAR_WINDOWS=True"),
    ("cv-ols",            "dp_pinn_cv_ols.npz",                   "C1",
     "python pinn_test_cv.py"),
    ("cv-ols-circ",       "dp_pinn_cv_ols_circular.npz",          "C6",
     "python pinn_test_cv.py with CIRCULAR_WINDOWS=True"),
    ("cv-trainable",      "dp_pinn_cv_trainable.npz",             "C2",
     "python pinn_test_cv_trainable_theta.py"),
    ("cv-trainable-circ", "dp_pinn_cv_trainable_circular.npz",    "C7",
     "python pinn_test_cv_trainable_theta.py with CIRCULAR_WINDOWS=True"),
    ("cv-pointwise",      "dp_pinn_cv_pointwise.npz",             "C4",
     "python pinn_test_cv_pointwise.py"),
    ("cv-pointwise-circ", "dp_pinn_cv_pointwise_circular.npz",    "C8",
     "python pinn_test_cv_pointwise.py with CIRCULAR_WINDOWS=True"),
    ("coef-match",        "dp_pinn_coef_match.npz",               "C9",
     "python pinn_test_coef_match.py"),
]

MODELS = []
for name, fname, color, hint in _CANDIDATES:
    npz = _maybe_load(fname, hint)
    if npz is not None:
        MODELS.append((name, npz, color))

if not MODELS:
    raise FileNotFoundError(
        "No model .npz artifacts found in this directory. "
        "Run at least one pinn_test_*.py script first."
    )

# First loaded model is the reference for shared data (t_grid, theta_data,
# g_true don't depend on which variant). Plot helpers also key off this.
REFERENCE = MODELS[0][1]

model_names = [name for (name, *_rest) in MODELS]
_NPZ_BY_NAME = {name: npz for (name, npz, _c) in MODELS}


def _model_field(name, key):
    return np.asarray(_NPZ_BY_NAME[name][key])


# Refresh truth vectors so EQUATIONS reflects the .npz's gravity.
G_TRUE = float(REFERENCE["g_true"])
set_gravity(G_TRUE)
THETA_TRUE_BY_EQ = {eq["name"]: eq["theta_true"] for eq in EQUATIONS}


# ============================================================ stats
def _gather(npz):
    errs_th1 = error_stats(npz["theta1_pred"], npz["theta1_data"])
    errs_th2 = error_stats(npz["theta2_pred"], npz["theta2_data"])
    out = {
        "rel_l2_th1": errs_th1["rel_l2"], "rmse_th1": errs_th1["rmse"],
        "max_abs_th1": errs_th1["max_abs"],
        "rel_l2_th2": errs_th2["rel_l2"], "rmse_th2": errs_th2["rmse"],
        "max_abs_th2": errs_th2["max_abs"],
        "training_time": float(npz["training_time"]),
        "rel_l2_th1_train": float(npz["rel_l2_th1_train"]),
        "rel_l2_th2_train": float(npz["rel_l2_th2_train"]),
        "rel_l2_th1_test":  float(npz["rel_l2_th1_test"]),
        "rel_l2_th2_test":  float(npz["rel_l2_th2_test"]),
        "train_frac": float(npz["train_frac"]),
        "t_split":    float(npz["t_split"]),
        "n_train":    int(npz["n_train"]),
    }
    for eq in EQUATIONS:
        nm = eq["name"]
        tw = np.asarray(npz[f"theta_{nm}_per_window"])
        out[nm] = cv_forms(tw, THETA_TRUE_BY_EQ[nm])
        res = np.abs(npz[f"residual_{nm}_grid"])
        out[f"{nm}_mean_abs_res"] = float(res.mean())
        out[f"{nm}_max_abs_res"]  = float(res.max())
        # Per-region residual breakdown.
        n_train = int(npz["n_train"])
        res_train = res[:n_train]
        res_test  = res[n_train:]
        out[f"{nm}_mean_abs_res_train"] = float(res_train.mean())
        out[f"{nm}_mean_abs_res_test"]  = float(res_test.mean()) if res_test.size else float("nan")
    return out


stats = {name: _gather(npz) for (name, npz, _c) in MODELS}


COL_WIDTH = 14


def _fmt(v, fmt):
    return fmt.format(v)


def _row(label, values, fmt="{:.4e}"):
    cells = " | ".join(f"{_fmt(v, fmt):>{COL_WIDTH}}" for v in values)
    return f"| {label:<34} | {cells} |"


header_cells = " | ".join(f"{n:>{COL_WIDTH}}" for n in model_names)
sep_cells = " | ".join("-" * COL_WIDTH for _ in model_names)
_split_summary = stats[model_names[0]]
TRAIN_FRAC = _split_summary["train_frac"]
T_SPLIT    = _split_summary["t_split"]
N_TRAIN    = _split_summary["n_train"]
lines = [
    "# Double pendulum PINN comparison -- 2 candidate equations, generic form",
    f"# g = {G_TRUE}",
    f"# temporal train/test split: TRAIN_FRAC = {TRAIN_FRAC:.2f}, "
    f"t_split = {T_SPLIT:.3f}, n_train = {N_TRAIN}",
    "# true theta per eq:",
    *[f"#   {eq['name']:>4} ({len(eq['theta_true'])} terms): {eq['theta_true'].tolist()}"
      for eq in EQUATIONS],
    "",
    f"| {'metric':<34} | {header_cells} |",
    f"| {'-'*34} | {sep_cells} |",
    _row("rel_l2 train (theta1)", [stats[n]["rel_l2_th1_train"] for n in model_names]),
    _row("rel_l2 test  (theta1)", [stats[n]["rel_l2_th1_test"]  for n in model_names]),
    _row("rel_l2 train (theta2)", [stats[n]["rel_l2_th2_train"] for n in model_names]),
    _row("rel_l2 test  (theta2)", [stats[n]["rel_l2_th2_test"]  for n in model_names]),
    _row("rel_l2 full  (theta1)", [stats[n]["rel_l2_th1"]       for n in model_names]),
    _row("rel_l2 full  (theta2)", [stats[n]["rel_l2_th2"]       for n in model_names]),
    _row("rmse  full  (theta1)", [stats[n]["rmse_th1"]      for n in model_names]),
    _row("rmse  full  (theta2)", [stats[n]["rmse_th2"]      for n in model_names]),
    _row("training_time_s",     [stats[n]["training_time"] for n in model_names],
         fmt="{:.1f}"),
]

for eq in EQUATIONS:
    nm = eq["name"]
    lines.append("")
    lines.append(f"## Eq {nm} (K = {eq['k']}, true = {eq['theta_true'].tolist()})")
    lines.append(_row(f"  mean|residual| train",[stats[n][f"{nm}_mean_abs_res_train"] for n in model_names]))
    lines.append(_row(f"  mean|residual| test", [stats[n][f"{nm}_mean_abs_res_test"]  for n in model_names]))
    lines.append(_row(f"  mean|residual|",      [stats[n][f"{nm}_mean_abs_res"]       for n in model_names]))
    lines.append(_row(f"  max|residual|",       [stats[n][f"{nm}_max_abs_res"]        for n in model_names]))
    lines.append(_row(f"  cv2_sum",          [stats[n][nm]["cv2_sum"]        for n in model_names]))
    lines.append(_row(f"  anchored_mse_sum", [stats[n][nm]["anchored_mse_sum"] for n in model_names]))
    for j in range(eq["k"]):
        fname = eq["feature_names"][j]
        true_val = float(eq["theta_true"][j])
        lines.append(_row(
            f"  [{fname:<18}] mean  (true={true_val:.3f})",
            [stats[n][nm][j]["mean"] for n in model_names],
            fmt="{:.5f}"))
        lines.append(_row(
            f"  [{fname:<18}] std",
            [stats[n][nm][j]["std"] for n in model_names]))
        lines.append(_row(
            f"  [{fname:<18}] rel_bias",
            [stats[n][nm][j]["rel_bias"] for n in model_names]))

# cv-trainable: trained-theta vs post-hoc-OLS theta comparison (one block
# per loaded cv-trainable* variant -- includes the circular variant if present).
_cv_trainable_models = [(name, npz) for (name, npz, _c) in MODELS
                        if name.startswith("cv-trainable")]
for trainable_name, trainable_npz in _cv_trainable_models:
    lines.append("")
    lines.append(f"## {trainable_name}: trained theta vs post-hoc OLS theta")
    for eq in EQUATIONS:
        nm = eq["name"]
        trained = np.asarray(trainable_npz[f"theta_{nm}_per_window"]).mean(axis=0)
        ols     = np.asarray(trainable_npz[f"theta_{nm}_per_window_ols"]).mean(axis=0)
        for j in range(eq["k"]):
            fname = eq["feature_names"][j]
            lines.append(
                f"| {nm:>4} [{fname:<18}] | "
                f"trained={trained[j]:>10.5f} | OLS={ols[j]:>10.5f} | "
                f"true={float(eq['theta_true'][j]):>10.5f} |"
            )

table = "\n".join(lines)
print(table)
(PLOTS_DIR / "comparison_stats.txt").write_text(table + "\n")


# ============================================================ plots
t = REFERENCE["t"]
th1_data = REFERENCE["theta1_data"]
th2_data = REFERENCE["theta2_data"]


# ----- Fig 1: trajectory (theta1, theta2) + error
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
# Linestyle keyed by algorithm family; circular variants share the same
# linestyle as their non-circular twin, color tells them apart.
linestyles = {
    "baseline":          "--", "baseline-circ":     "--",
    "cv-ols":            ":",  "cv-ols-circ":       ":",
    "cv-trainable":      "-.", "cv-trainable-circ": "-.",
    "cv-pointwise":      "-",  "cv-pointwise-circ": "-",
    "coef-match":        (0, (3, 1, 1, 1)),  # dash-dot-dot
}


def _mark_split(ax):
    ax.axvline(T_SPLIT, color="gray", linestyle="--", alpha=0.6,
               label=f"train/test split (t={T_SPLIT:.2f})")


ax = axes[0, 0]
ax.plot(t, th1_data, "k-", label="data", linewidth=1.5)
for name, _npz, color in MODELS:
    ax.plot(t, _model_field(name, "theta1_pred"), linestyle=linestyles[name],
            color=color, label=name, linewidth=1.1)
_mark_split(ax)
ax.set_ylabel(r"$\theta_1(t)$")
ax.set_title("theta1 trajectory")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t, th2_data, "k-", label="data", linewidth=1.5)
for name, _npz, color in MODELS:
    ax.plot(t, _model_field(name, "theta2_pred"), linestyle=linestyles[name],
            color=color, label=name, linewidth=1.1)
_mark_split(ax)
ax.set_ylabel(r"$\theta_2(t)$")
ax.set_title("theta2 trajectory")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
for name, _npz, color in MODELS:
    ax.plot(t, _model_field(name, "theta1_pred") - th1_data,
            "-", color=color, label=name, linewidth=0.8)
ax.axhline(0, color="k", linestyle=":", alpha=0.5)
_mark_split(ax)
ax.set_xlabel("t"); ax.set_ylabel(r"$\theta_1^{pred}-\theta_1^{data}$")
ax.set_title("theta1 prediction error")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
for name, _npz, color in MODELS:
    ax.plot(t, _model_field(name, "theta2_pred") - th2_data,
            "-", color=color, label=name, linewidth=0.8)
ax.axhline(0, color="k", linestyle=":", alpha=0.5)
_mark_split(ax)
ax.set_xlabel("t"); ax.set_ylabel(r"$\theta_2^{pred}-\theta_2^{data}$")
ax.set_title("theta2 prediction error")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(f"Double pendulum trajectories: data vs PINN predictions "
             f"(train: t < {T_SPLIT:.2f}, test: t ≥ {T_SPLIT:.2f})")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "trajectory.png", dpi=150)
plt.close(fig)


# ----- Fig 2: per-equation residual time series
neq = len(EQUATIONS)
fig, axes = plt.subplots(neq, 1, figsize=(12, 3 * neq), sharex=True)
if neq == 1:
    axes = [axes]
for r, eq in enumerate(EQUATIONS):
    nm = eq["name"]
    ax = axes[r]
    for name, _npz, color in MODELS:
        field = _model_field(name, f"residual_{nm}_grid")
        ax.plot(t, np.abs(field), "-", color=color, label=name, linewidth=0.9)
    ax.axvline(T_SPLIT, color="gray", linestyle="--", alpha=0.6,
               label=f"train/test split")
    ax.set_yscale("log")
    ax.set_ylabel(f"|residual {nm}|")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
axes[-1].set_xlabel("t")
fig.suptitle("Per-equation residual time series (log scale)")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "residual_timeseries.png", dpi=150)
plt.close(fig)


# ----- Fig 3: per-window coefficient estimates (one figure per equation)
# markers = {"baseline": "o", "cv-ols": "s", "cv-trainable": "D"}
# for eq in EQUATIONS:
#     nm = eq["name"]
#     K = eq["k"]
#     n_win = _model_field("baseline", f"theta_{nm}_per_window").shape[0]
#     idx = np.arange(n_win)
#     cols = min(3, K)
#     rows = (K + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows),
#                               squeeze=False)
#     for j in range(K):
#         ax = axes[j // cols, j % cols]
#         for name, _npz, color in MODELS:
#             tw = _model_field(name, f"theta_{nm}_per_window")
#             ax.plot(idx, tw[:, j], marker=markers[name], linestyle="-",
#                     color=color, label=name, markersize=5)
#         ax.axhline(float(eq["theta_true"][j]), color="k", linestyle="--",
#                    label=f"true = {eq['theta_true'][j]:.3f}")
#         ax.set_title(f"{nm}: {eq['feature_names'][j]}")
#         ax.set_xlabel("window index"); ax.set_ylabel("coefficient")
#         ax.grid(alpha=0.3); ax.legend(fontsize=8)
#     for j in range(K, rows * cols):
#         axes[j // cols, j % cols].axis("off")
#     fig.suptitle(f"Per-window coefficients for equation {nm}")
#     fig.tight_layout()
#     fig.savefig(PLOTS_DIR / f"coef_per_window_{nm}.png", dpi=150)
#     plt.close(fig)


# ----- Fig 4: cv-trainable trained vs post-hoc OLS (one figure per variant per equation)
for trainable_name, trainable_npz in _cv_trainable_models:
    for eq in EQUATIONS:
        nm = eq["name"]
        K = eq["k"]
        tw_trained = np.asarray(trainable_npz[f"theta_{nm}_per_window"])
        tw_ols     = np.asarray(trainable_npz[f"theta_{nm}_per_window_ols"])
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
        fig.suptitle(f"{trainable_name}: trained vs post-hoc OLS theta for {nm}")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{trainable_name}_theta_vs_ols_{nm}.png", dpi=150)
        plt.close(fig)


# ----- Fig 5: phase portraits (theta_i vs omega_i)
panels = [("data", th1_data, REFERENCE["omega1_data"],
                 th2_data, REFERENCE["omega2_data"], "k")]
for name, _npz, color in MODELS:
    panels.append((
        name,
        _model_field(name, "theta1_pred"), _model_field(name, "omega1_pred"),
        _model_field(name, "theta2_pred"), _model_field(name, "omega2_pred"),
        color,
    ))

ncols = len(panels)
fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7))
sl_tr = slice(0, N_TRAIN)
sl_te = slice(N_TRAIN - 1, None)   # -1 for visual continuity at the join
TEST_COLOR = "C3"                  # contrasting color for the extrapolation segment
for c, (title, th1, w1, th2, w2, color) in enumerate(panels):
    ax = axes[0, c]
    ax.plot(th1[sl_tr], w1[sl_tr], "-",  linewidth=0.7, alpha=0.85,
            color=color, label="train")
    ax.plot(th1[sl_te], w1[sl_te], "--", linewidth=0.7, alpha=0.85,
            color=TEST_COLOR, label="test")
    ax.plot(th1[N_TRAIN - 1], w1[N_TRAIN - 1], "o",
            color="k", markersize=4, label="split")
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\omega_1$")
    ax.set_title(f"th1 phase ({title})")
    ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
    ax = axes[1, c]
    ax.plot(th2[sl_tr], w2[sl_tr], "-",  linewidth=0.7, alpha=0.85,
            color=color, label="train")
    ax.plot(th2[sl_te], w2[sl_te], "--", linewidth=0.7, alpha=0.85,
            color=TEST_COLOR, label="test")
    ax.plot(th2[N_TRAIN - 1], w2[N_TRAIN - 1], "o",
            color="k", markersize=4, label="split")
    ax.set_xlabel(r"$\theta_2$"); ax.set_ylabel(r"$\omega_2$")
    ax.set_title(f"th2 phase ({title})")
    ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
fig.suptitle(f"Phase portraits (train: model color, test: red dashed; split at t={T_SPLIT:.2f})")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "phase_portrait.png", dpi=150)
plt.close(fig)


# ----- Fig 6: training loss curves (Adam only; L-BFGS refinement not shown)
def _has_hist(name):
    npz = _NPZ_BY_NAME[name]
    return "loss_history_iter" in npz.files


models_with_hist = [m for m in MODELS if _has_hist(m[0])]
if models_with_hist:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    titles = [
        ("loss_history_total", "total loss"),
        ("loss_history_phys",  "physics loss"),
        ("loss_history_cv",    "CV loss"),
        ("loss_history_data",  "data loss"),
    ]
    for ax, (key, title) in zip(axes.flat, titles):
        for name, _npz, color in models_with_hist:
            iters = _model_field(name, "loss_history_iter")
            vals  = _model_field(name, key)
            # Skip series that are all zero (e.g. baseline has no CV).
            if np.max(np.abs(vals)) == 0.0:
                continue
            ax.plot(iters, np.clip(vals, 1e-30, None),
                    color=color, label=name, linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("Adam iteration")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Training loss curves (Adam only; sampled every 1000 iter)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "loss_curves.png", dpi=150)
    plt.close(fig)
    extra_loss_fig = 1
else:
    print("[skip] loss_curves.png — no model artifact has loss_history_iter "
          "(rerun PINN scripts to capture history)")
    extra_loss_fig = 0


n_main = 3 + len(EQUATIONS) + extra_loss_fig
n_extra = len(EQUATIONS) * len(_cv_trainable_models)
print(f"\nWrote {n_main + n_extra} figures and comparison_stats.txt to {PLOTS_DIR}/")
