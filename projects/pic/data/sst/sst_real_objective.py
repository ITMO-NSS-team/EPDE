"""Seed the PHYSICALLY-EXPECTED ("truth-candidate") PDE forms on the REAL SST field
and read their EPDE objective vectors, to compare against the degenerate composite-
target forms that won the live real-data Pareto front (sst.py --metric scale_invariant).

On real data there is no exact ground truth, so 'truth' here means the physically
expected heat-transport law; coefficients are refit by the pipeline so the contrast is
purely structural:
  DIFFUSION  dT/dt = k*(T_xx + T_yy)                       (pure heat equation)
  ADV_DIFF   dT/dt = k*(T_xx + T_yy) + cx*T_x + cy*T_y      (advection-diffusion)
vs the live winners:
  REAL_DIAG_Ty    T_y  = a*T_t + b*T*T_y + c*T*T_t          (single-axis diagnostic)
  REAL_COMP_Tyy   T*T_yy = ...                              (composite multiply-by-T target)

The question: is the physical form OUT-COMPETED by the degeneracies (objectives problem)
or merely NOT REACHED by the collapsed search (search/RPS problem)?

Axis map (meshgrid(t,y,x,'ij')): dx0=t, dx1=y, dx2=x -> T_t=dT/dx0, T_yy=d2T/dx1^2,
T_xx=d2T/dx2^2.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

import epde.globals as global_var
from epde import globals as gv
from epde.interface.interface import EpdeSearch
from epde.operators.common.objectives import WAPEDiscrepancy, ScaleInvariantDiscrepancy, FitContext

from sst import load_sst, REGION, COARSEN, N_TIME, TIME_STRIDE
from sst_truth_objective import make_fit_operator, evaluate

DATA = os.path.join(os.path.dirname(__file__), 'sst_l4.nc')

# Physically expected forms (coefficients refit by the pipeline).
DIFFUSION = "1.0 * d^2T/dx2^2{power: 1.0} + 1.0 * d^2T/dx1^2{power: 1.0} = dT/dx0{power: 1.0}"
ADV_DIFF = ("1.0 * d^2T/dx2^2{power: 1.0} + 1.0 * d^2T/dx1^2{power: 1.0} + "
            "1.0 * dT/dx2{power: 1.0} + 1.0 * dT/dx1{power: 1.0} = dT/dx0{power: 1.0}")
# Degenerate forms that topped the live real-data front (target NOT a bare dT/dt).
REAL_DIAG_Ty = ("1.0 * dT/dx0{power: 1.0} + 1.0 * dT/dx1{power: 1.0} * T{power: 1.0} + "
                "1.0 * dT/dx0{power: 1.0} * T{power: 1.0} = dT/dx1{power: 1.0}")
REAL_COMP_Tyy = ("1.0 * T{power: 1.0} + 1.0 * dT/dx1{power: 1.0} + "
                 "1.0 * T{power: 1.0} * dT/dx1{power: 1.0} = T{power: 1.0} * d^2T/dx1^2{power: 1.0}")


def build_pool_real(prep='poly'):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coords, field = load_sst(DATA, REGION, COARSEN, N_TIME, TIME_STRIDE)
    grid = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
    search = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True,
                        boundary=[2, 3, 3], coordinate_tensors=grid, device=device)
    search.set_preprocessor(default_preprocessor_type=prep, preprocessor_kwargs={})
    search.create_pool(data=[field], variable_names=['T'], max_deriv_order=(1, 2, 2),
                       additional_tokens=[], data_fun_pow=1)
    return search


def score(prep):
    gv.set_gram_config('vcoef')
    search = build_pool_real(prep)
    fit_op = make_fit_operator()
    g_fun = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask].reshape(-1)
    ctx = FitContext(g_fun_vals=g_fun, data_shape=global_var.grid_cache.inner_shape,
                     penalty_coeff=0.5, for_rps=False)
    wape_f, sinv_f = WAPEDiscrepancy(), ScaleInvariantDiscrepancy()
    forms = [("DIFFUSION  dT/dt=k(T_xx+T_yy)", DIFFUSION, 'phys'),
             ("ADV_DIFF   +cx*T_x +cy*T_y", ADV_DIFF, 'phys'),
             ("REAL win: T_y diagnostic", REAL_DIAG_Ty, 'degen'),
             ("REAL win: T*T_yy composite", REAL_COMP_Tyy, 'degen')]
    print(f"\n================ REAL SST, preprocessor = {prep} ================")
    print(f"  {'form':<30} {'WAPE':>9} {'scale-inv':>10} {'instab':>10}")
    rows = {}
    for tag, sym, kind in forms:
        eq = evaluate(sym, search, fit_op)
        rows[tag] = (float(wape_f.compute(eq, ctx)), float(sinv_f.compute(eq, ctx)),
                     float(eq.coefficients_stability), kind, eq.text_form)
        print(f"  {tag:<30} {rows[tag][0]:>9.5f} {rows[tag][1]:>10.5f} {rows[tag][2]:>10.6f}")
        print(f"      -> {eq.text_form}")
    # rank on the LIVE discrepancy axis (scale-inv) + the instab axis
    for name, i in [('scale-inv', 1), ('instab', 2)]:
        order = sorted(rows.items(), key=lambda kv: kv[1][i])
        print(f"    rank {name:>10}: " + "  <  ".join(f"{k.split()[0]}({v[i]:.4f})" for k, v in order))
    # domination: is each physical form dominated (>= on BOTH scale-inv and instab) by a degenerate one?
    print("    --- Pareto domination on [scale-inv, instab] ---")
    for tag, v in rows.items():
        if v[3] != 'phys':
            continue
        dominators = [k for k, w in rows.items()
                      if w[3] == 'degen' and w[1] <= v[1] and w[2] <= v[2] and (w[1] < v[1] or w[2] < v[2])]
        verdict = f"DOMINATED by {dominators}" if dominators else "NON-DOMINATED (physical form survives)"
        print(f"      {tag.split()[0]:<10} [{v[1]:.4f}, {v[2]:.5f}] -> {verdict}")


def main():
    for prep in ('poly', 'FD'):
        score(prep)


if __name__ == "__main__":
    main()
