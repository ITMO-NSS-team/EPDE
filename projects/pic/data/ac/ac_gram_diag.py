"""Gram-matrix diagnostic: AC true equation vs AC + one spurious term.

Builds the AC pool exactly as the thesis pipeline does (FD preprocessor,
max_deriv_order=(2,4), data_fun_pow=3), evaluates each library term on the
inner (boundary-masked) grid, and reports the quantities the stability
subsystem actually sees:

  * raw Gram  G = Phi^T W Phi  (W = masked g_func; = I by default)
  * normalized correlation (unit-diagonal Gram)  -> collinearity
  * condition number of the FEATURE Gram (target column excluded)
  * per-term vcoef instability score  NC_j / gamma_{j,0}^2  (VaryingCoefSetup)

AC truth:  du/dx0 = 0.0001*d^2u/dx1^2 + 5*u - 5*u^3
Axis map (meshgrid(t, x, 'ij')):  dx0 = t,  dx1 = x.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              '../../../..')))

import numpy as np

import epde.globals as global_var
from epde import globals as gv
from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation
from epde.operators.common.stability import VaryingCoefSetup, vc_stability_total_lr

np.set_printoptions(precision=4, suppress=True, linewidth=140)

# ----- AC library terms (factor strings the translator understands) -------
TARGET = 'du/dx0{power: 1.0}'                       # u_t  (LHS / regression target)
TRUE = [
    'd^2u/dx1^2{power: 1.0}',                       # u_xx   (diffusion)
    'u{power: 3.0}',                                # u^3    (cubic reaction)
    'u{power: 1.0}',                                # u      (linear reaction)
]
SPURIOUS = {
    'u_x   (du/dx1)':        'du/dx1{power: 1.0}',          # advection (sym-forbidden)
    'u^2':                   'u{power: 2.0}',               # even nonlinearity
    'u_xxx (d^3u/dx1^3)':    'd^3u/dx1^3{power: 1.0}',      # dispersion
    'u*u_x (Burgers)':       'du/dx1{power: 1.0} * u{power: 1.0}',
    'u_tt  (d^2u/dx0^2)':    'd^2u/dx0^2{power: 1.0}',      # wave-like
}


def build_pool(data_fun_pow=3):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = np.linspace(0., 1., 51)
    x = np.linspace(-1., 0.984375, 128)
    data = np.load(os.path.join(os.path.dirname(__file__), 'ac_data.npy'))
    grid = np.meshgrid(t, x, indexing='ij')
    # boundary = 10% per axis, matching thesis_runner._boundary_for
    boundary = tuple(max(1, n // 10) for n in data.shape)
    search = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True,
                        boundary=boundary, coordinate_tensors=grid, device=device)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    search.create_pool(data=[data], variable_names=['u'], max_deriv_order=(2, 4),
                       additional_tokens=[], data_fun_pow=data_fun_pow, deriv_fun_pow=2)
    return search


def design_matrix(term_strings, search):
    """Evaluate each term string on the inner grid -> (Phi (N,K), labels)."""
    lhs = " + ".join(f"1.0 * {t}" for t in term_strings)
    eq_str = f"{lhs} = {TARGET}"
    soeq = translate_equation(eq_str, search.pool, all_vars=['u'])
    eq = soeq.vals['u']
    cols, labels = [], []
    tgt_idx = eq.target_idx
    order = [i for i in range(len(eq.structure)) if i != tgt_idx] + [tgt_idx]
    for i in order:
        term = eq.structure[i]
        cols.append(np.asarray(term.evaluate(False), dtype=float).reshape(-1))
        labels.append(term.name)
    Phi = np.stack(cols, axis=1)         # last column is the target
    return Phi, labels


def gram_report(tag, term_strings, search, w, grid_shape):
    Phi, labels = design_matrix(term_strings, search)
    K = Phi.shape[1]
    short = [_short(l) for l in labels]
    # weighted Gram  G = Phi^T diag(w) Phi
    G = Phi.T @ (w[:, None] * Phi)
    d = np.sqrt(np.clip(np.diag(G), 1e-300, None))
    C = G / np.outer(d, d)               # normalized correlation (unit diagonal)

    print(f"\n================  {tag}  ================")
    print(f"  columns ({K}): " + ", ".join(f"[{i}]{s}" for i, s in enumerate(short))
          + "   (last = target u_t)")
    print(f"  N inner points = {Phi.shape[0]}   (sum w = {w.sum():.0f})")
    print("\n  column L2 energy  sqrt(diag G):")
    for s, dv in zip(short, d):
        print(f"      {s:<14} {dv:14.6g}")
    print("\n  RAW Gram matrix  G = Phi^T W Phi  (rows=cols=columns above):")
    _print_matrix(G, short, fmt='sci')
    print("\n  normalized Gram / correlation matrix (rows=cols=columns above):")
    _print_matrix(C, short)

    # feature-only conditioning (drop target column = last)
    Gf = G[:-1, :-1]
    Cf = C[:-1, :-1]
    print(f"\n  FEATURE Gram conditioning (target excluded):")
    print(f"      cond(normalized feature Gram) = {np.linalg.cond(Cf):12.4g}")
    print(f"      cond(raw        feature Gram) = {np.linalg.cond(Gf):12.4g}")
    print(f"      max |corr| among features      = "
          f"{np.max(np.abs(Cf - np.eye(Cf.shape[0]))):.4f}")

    # vcoef instability (the live stability objective)
    feats = Phi[:, :-1]
    target = Phi[:, -1]
    setup = VaryingCoefSetup(feats, target, w, grid_shape, main_var='u',
                             fit_intercept=False)
    scores = setup.score(None)
    total = vc_stability_total_lr(feats, target, w, grid_shape, main_var='u',
                                  fit_intercept=False)
    print(f"\n  vcoef instability score  NC_j / gamma0_j^2  (lower = stable/true):")
    for s, sc in zip(short[:-1], scores):
        print(f"      {s:<14} {sc:14.6g}")
    print(f"      {'SUM (objective)':<14} {total:14.6g}")
    return {'labels': short, 'C': C, 'cond_norm': float(np.linalg.cond(Cf)),
            'scores': scores, 'total': float(total),
            'maxcorr': float(np.max(np.abs(Cf - np.eye(Cf.shape[0]))))}


def _short(label):
    label = (label.replace('{power: 1.0}', '').replace('{power: 2.0}', '^2')
                  .replace('{power: 3.0}', '^3').replace(' ', ''))
    return (label.replace('d^2u/dx1^2', 'u_xx').replace('d^3u/dx1^3', 'u_xxx')
                 .replace('d^2u/dx0^2', 'u_tt').replace('du/dx1', 'u_x')
                 .replace('du/dx0', 'u_t'))


def _print_matrix(M, labels, fmt='corr'):
    if fmt == 'sci':
        print("        " + "".join(f"{l[:11]:>13}" for l in labels))
        for i, row in enumerate(M):
            print(f"  {labels[i][:6]:>6}" + "".join(f"{v:13.4e}" for v in row))
    else:
        print("        " + "".join(f"{l[:8]:>9}" for l in labels))
        for i, row in enumerate(M):
            print(f"  {labels[i][:6]:>6}" + "".join(f"{v:9.4f}" for v in row))


def main():
    gv.set_gram_config('vcoef')
    search = build_pool()
    w = (global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask]
         .reshape(-1).astype(float))
    grid_shape = tuple(int(n) for n in global_var.grid_cache.inner_shape)
    print(f"inner_shape = {grid_shape}   (W is {'all-ones' if np.allclose(w, 1) else 'NON-uniform'})")

    true_res = gram_report("AC TRUE  [u_xx, u^3, u ; target u_t]", TRUE, search,
                           w, grid_shape)

    summary = []
    for name, spur in SPURIOUS.items():
        res = gram_report(f"AC + SPURIOUS  {name}", TRUE + [spur], search, w, grid_shape)
        # correlation of the spurious column (2nd-to-last, since target is last)
        spur_idx = len(TRUE)           # position of spurious feature in column order
        corr_with_true = res['C'][spur_idx, :len(TRUE)]
        summary.append((name, res, corr_with_true))

    print("\n\n################  SUMMARY  ################")
    print(f"  TRUE baseline:  cond(normGram)={true_res['cond_norm']:.3g}   "
          f"sum_instab={true_res['total']:.4g}   max|corr|={true_res['maxcorr']:.3f}")
    print(f"\n  {'spurious term':<22}{'cond(normGram)':>16}{'sum_instab':>14}"
          f"{'spur_instab':>14}{'max|corr|truefeat':>20}")
    for name, res, corr_true in summary:
        spur_instab = res['scores'][len(TRUE)]
        print(f"  {name:<22}{res['cond_norm']:>16.4g}{res['total']:>14.4g}"
              f"{spur_instab:>14.4g}{np.max(np.abs(corr_true)):>20.4f}")


if __name__ == "__main__":
    import torch
    print('cuda available:', torch.cuda.is_available())
    main()
