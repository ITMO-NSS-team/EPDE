"""Visualize vcoef *instability* on Allen-Cahn for three structural variants:

  1. missing diffusion  (the true ``u_xx`` term dropped),
  2. the true equation,
  3. spurious            (an extra ``u_x * u_xx`` term added).

For each variant we fit the varying-coefficient model and reconstruct, per RHS
term, the coefficient field ``beta_j(x,t) = sum_b gamma_{j,b} B_b(x,t)`` over the
grid. A TRUE term fits a near-CONSTANT coefficient (flat field -> instability
score ~ 0); a misspecified equation forces some coefficient(s) to VARY across
the domain (curved field -> large score). Each panel shows the relative
deviation ``(beta - beta0)/|beta0|`` (flat/white = stable, colored = unstable)
annotated with the per-term instability score s = (Var(gamma0)+NC_deb)/gamma0^2.

Writes three PNGs to projects/thesis/plots/.
"""
from __future__ import annotations
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import epde.globals as gv
from epde.interface.equation_translator import translate_equation
from epde.operators.common.stability import VaryingCoefSetup
from kdv_sindy_test import build_pool_only, _normalize_grid_labels
from thesis_runner import load_config, pipeline_settings, _set_seeds

OUT = os.path.join(_THIS, "plots")
os.makedirs(OUT, exist_ok=True)

_UXX = "d^2u/dx1^2{power: 1.0}"
_U3 = "u{power: 3.0}"
_U = "u{power: 1.0}"
_UX_UXX = "du/dx1{power: 1.0} * d^2u/dx1^2{power: 1.0}"
_TARGET = "du/dx0{power: 1.0}"

VARIANTS = [
    ("1_missing_uxx", "Missing diffusion  (no $u_{xx}$)",
     f"-5.0 * {_U3} + 5.0 * {_U} = {_TARGET}"),
    ("2_true", "True Allen-Cahn",
     f"0.0001 * {_UXX} + -5.0 * {_U3} + 5.0 * {_U} = {_TARGET}"),
    ("3_spurious_ux_uxx", "Spurious  ($+\\,u_x\\,u_{xx}$)",
     f"0.0001 * {_UXX} + -5.0 * {_U3} + 5.0 * {_U} + 0.0001 * {_UX_UXX} = {_TARGET}"),
]

_BASE = {"d^2u/dx1^2": "u_{xx}", "du/dx1": "u_x", "du/dx0": "u_t", "u": "u"}


def pretty(name: str) -> str:
    parts = []
    for fac in name.split(" * "):
        base = fac.split("{")[0].strip()
        power = 1.0
        if "power:" in fac:
            try:
                power = float(fac.split("power:")[1].split(",")[0].split("}")[0])
            except ValueError:
                power = 1.0
        b = _BASE.get(base, base)
        if base == "u" and power != 1.0:
            b = f"{b}^{int(power)}"
        parts.append(b)
    return "$" + r" \cdot ".join(parts) + "$"


def setup_pool():
    cfg = load_config("ac")
    _set_seeds(0)
    search = build_pool_only(cfg, pipeline_settings("new"))
    sw = np.asarray(gv.grid_cache.g_func[gv.grid_cache.g_func_mask]).reshape(-1)
    gshape = tuple(int(n) for n in gv.grid_cache.inner_shape)
    return search, sw, gshape


def analyze(search, sw, gshape, eq_str):
    soeq = translate_equation(_normalize_grid_labels(eq_str), search.pool,
                              all_vars=["u"])
    eq = soeq.vals["u"]
    eq.main_var_to_explain = "u"
    eq.weights_internal = np.ones(len(eq.structure) - 1)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    # raw (un-normalized) features so gamma0 is in physical coefficient units;
    # the score / relative-deviation field are scale-invariant either way.
    _, target, feats = eq.evaluate(normalize=False, return_val=False)
    feats = np.asarray(feats, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    feat_terms = [t for i, t in enumerate(eq.structure) if i != eq.target_idx]

    setup = VaryingCoefSetup(feats, y, sw, gshape, main_var="u",
                             fit_intercept=False)
    sol = setup._solve_gammas(None)
    gamma, B, mk = sol["gamma"], sol["B"], sol["mk"]
    is_const = mk == 0
    Bvals = setup._Bvals
    scores = np.asarray(setup.score(None), dtype=float)

    rows = []
    for i, term in enumerate(feat_terms):
        block = gamma[i * B:(i + 1) * B]
        g0 = float(block[is_const][0])
        beta = (Bvals @ block).reshape(gshape)
        rows.append({"label": pretty(term.name), "g0": g0,
                     "beta": beta, "score": float(scores[i])})
    return rows


def plot_variant(title, rows, path):
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 4.6), squeeze=False)
    axes = axes[0]
    norm = Normalize(vmin=-1.0, vmax=1.0)
    total = float(np.sum([r["score"] for r in rows]))
    im = None
    for ax, r in zip(axes, rows):
        rel = (r["beta"] - r["g0"]) / (abs(r["g0"]) + 1e-30)
        im = ax.imshow(rel, origin="lower", aspect="auto", cmap="coolwarm",
                       norm=norm)
        ax.set_title(f"{r['label']}\n$\\beta_0$={r['g0']:.2g},  "
                     f"score={r['score']:.1e}", fontsize=10)
        ax.set_xlabel("$x$"); ax.set_ylabel("$t$")
        ax.set_xticks([]); ax.set_yticks([])
    # explicit layout so the 2-line panel titles never collide with suptitle
    fig.subplots_adjust(left=0.06, right=0.87, top=0.72, bottom=0.10, wspace=0.28)
    cax = fig.add_axes([0.89, 0.12, 0.015, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"relative deviation $(\beta-\beta_0)/|\beta_0|$")
    fig.suptitle(f"{title}     total instability  $\\Sigma$ = {total:.2e}",
                 fontsize=13, y=0.93)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sum_field(title, rows, path):
    """One heatmap per case: the SUM over terms of each term's coefficient-field
    relative deviation ``(beta_j - beta0_j)/|beta0_j|`` over the (x,t) grid (each
    term clipped to +-1 first, so a near-zero-coefficient spurious term cannot
    blow up the scale). Flat/white = every coefficient is ~constant (stable);
    colored = some coefficient is forced to vary there. The title carries the
    total instability Sigma = sum_j score_j."""
    gshape = rows[0]["beta"].shape
    field = np.zeros(gshape, dtype=float)
    for r in rows:
        rel = (r["beta"] - r["g0"]) / (abs(r["g0"]) + 1e-30)
        field += np.clip(rel, -1.0, 1.0)
    total = float(np.sum([r["score"] for r in rows]))
    terms = " + ".join(r["label"] for r in rows)
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    im = ax.imshow(field, origin="lower", aspect="auto", cmap="coolwarm",
                   norm=Normalize(vmin=-1.5, vmax=1.5))
    ax.set_xlabel("$x$"); ax.set_ylabel("$t$")
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\sum_j$ clip$\,[(\beta_j-\beta_{0j})/|\beta_{0j}|,\ \pm1]$")
    ax.set_title(f"{title}\n{terms}\ntotal instability $\\Sigma$ = {total:.2e}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    search, sw, gshape = setup_pool()
    print(f"AC grid (inner) = {gshape},  N = {int(np.prod(gshape))} points\n")
    for tag, title, eq_str in VARIANTS:
        try:
            rows = analyze(search, sw, gshape, eq_str)
        except Exception as e:
            import traceback
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        path = os.path.join(OUT, f"ac_instability_{tag}.png")
        plot_sum_field(title, rows, path)
        tot = float(np.sum([r["score"] for r in rows]))
        print(f"[{tag}] {title}  (Sigma={tot:.3e})")
        for r in rows:
            print(f"    {r['label']:18s} beta0={r['g0']:+.3e}  score={r['score']:.3e}")
        print(f"    saved -> {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
