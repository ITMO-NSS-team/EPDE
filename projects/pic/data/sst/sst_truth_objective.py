"""Evaluate the EPDE objective vector of the EXACT truth PDE on the synthetic SST
heat-kernel field, to compare against the discovered Pareto front.

The search reports a 2-vector per single-variable equation: [WAPEDiscrepancy,
Instability]. Here we seed candidate structures with translate_equation, run the
same solver-free fitness (WAPE discrepancy + instability, VWSR sparsity + LinReg
refit), and print [discrepancy, instability] for:
  - TRUTH      dT/dt = 0.5*(T_xx + T_yy)
  - TRUNCATED  dT/dt = 0.5*T_xx          (one diffusion term missing)
  - OVERCOMPLETE dT/dt = a*T_xx + b*T_yy + c*T_x + d*T_y   (spurious advection)
Coefficients are refit by the pipeline, so the contrast is purely structural.

Axis map (coordinate_tensors = meshgrid(t,y,x,'ij')): dx0=t, dx1=y, dx2=x, so
T_t=dT/dx0, T_yy=d^2T/dx1^2, T_xx=d^2T/dx2^2.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

import epde.globals as global_var
from epde import globals as gv
from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation
from epde.operators.common.fitness import SolverFreeFitness
from epde.operators.common.objectives import WAPEDiscrepancy, Instability
from epde.operators.common.sparsity import VWSRSparsity
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.utils.default_parameter_loader import EvolutionaryParams

from sst import _synthetic_field


TRUTH = "0.5 * d^2T/dx2^2{power: 1.0} + 0.5 * d^2T/dx1^2{power: 1.0} = dT/dx0{power: 1.0}"
TRUNCATED = "0.5 * d^2T/dx2^2{power: 1.0} = dT/dx0{power: 1.0}"
OVERCOMPLETE = ("0.5 * d^2T/dx2^2{power: 1.0} + 0.5 * d^2T/dx1^2{power: 1.0} + "
                "0.0 * dT/dx2{power: 1.0} + 0.0 * dT/dx1{power: 1.0} = dT/dx0{power: 1.0}")
# Same physics, multiplied through by T: target becomes T*dT/dt instead of dT/dt.
DEGENERATE = ("0.5 * T{power: 1.0} * d^2T/dx2^2{power: 1.0} + "
              "0.5 * T{power: 1.0} * d^2T/dx1^2{power: 1.0} = "
              "T{power: 1.0} * dT/dx0{power: 1.0}")
# The LIVE winner: T x truth PLUS small residual-fitting terms (a 5-term overfit
# with a composite T*dT/dt target). At baseline it scored [0.00586, 0.00014] and
# Pareto-dominated the truth.
MIXED = ("0.0046 * dT/dx0{power: 1.0} + "
         "0.5276 * d^2T/dx2^2{power: 1.0} * T{power: 1.0} + "
         "-0.0024 * d^2T/dx2^2{power: 1.0} + "
         "-0.0024 * d^2T/dx1^2{power: 1.0} + "
         "0.5276 * d^2T/dx1^2{power: 1.0} * T{power: 1.0} = "
         "dT/dx0{power: 1.0} * T{power: 1.0}")


def build_pool(prep='FD'):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coords, field, kappa = _synthetic_field()
    grid = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
    search = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True,
                        boundary=[2, 3, 3], coordinate_tensors=grid, device=device)
    search.set_preprocessor(default_preprocessor_type=prep, preprocessor_kwargs={})
    search.create_pool(data=[field], variable_names=['T'], max_deriv_order=(1, 2, 2),
                       additional_tokens=[], data_fun_pow=1)
    return search, kappa


def make_fit_operator():
    params = EvolutionaryParams()
    op_params = params.get_default_params_for_operator('SolverFreeFitness')
    disc = WAPEDiscrepancy()
    fit_op = SolverFreeFitness(list(op_params.keys()),
                               objectives=[disc, Instability()], primary=disc)
    fit_op.params = op_params
    fit_op.set_suboperators({'sparsity': VWSRSparsity(),
                             'coeff_calc': LinRegBasedCoeffsEquation()})
    return fit_op


def scale_invariant_disc(eq):
    """Pointwise cancellation residual: mean_x |sum_k c_k phi_k| / sum_k |c_k phi_k|.

    Invariant under multiplying the whole equation by any field g(x) (both sums
    pick up |g(x)| pointwise) and under target choice (same term set). Bounded [0,1].
    """
    _, target, features = eq.evaluate(normalize=True, return_val=False)
    wf = np.asarray(eq.weights_final, dtype=float)
    if features is None:
        return 0.0
    if eq.weights_internal[-1]:
        coefs, intercept = wf[:-1], wf[-1]
    else:
        coefs, intercept = wf, 0.0
    contribs = features * coefs[None, :]                       # weighted term values (Npts, K)
    r = target - contribs.sum(axis=1) - intercept             # homogeneous residual
    term_mass = np.abs(target) + np.abs(contribs).sum(axis=1) + abs(intercept)
    rho = np.abs(r) / (term_mass + 1e-12)
    return float(np.mean(rho))


def evaluate(symbolic, search, fit_op):
    soeq = translate_equation(symbolic, search.pool, all_vars=['T'])
    eq = soeq.vals['T']
    eq.main_var_to_explain = 'T'
    eq.metaparameters = {('sparsity', 'T'): {'optimizable': False, 'value': 1e-6}}
    eq.weights_internal = np.ones(len(eq.structure) - 1)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    fit_op.apply(eq, {}, force_out_of_place=True)          # sparsity + refit
    eq.fitness_calculated = False
    eq.stability_calculated = False
    fit_op.apply(eq, {}, force_out_of_place=False)         # populate objectives
    return eq


def score_forms(prep):
    from epde.operators.common.objectives import ScaleInvariantDiscrepancy, FitContext
    gv.set_gram_config('vcoef')
    search, kappa = build_pool(prep=prep)
    fit_op = make_fit_operator()
    forms = [("TRUTH  dT/dt=.5(T_xx+T_yy)", TRUTH),
             ("TRUNCATED (no T_yy)", TRUNCATED),
             ("DEGENERATE pure xT", DEGENERATE),
             ("MIXED 5-term overfit", MIXED)]
    g_fun = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask].reshape(-1)
    ctx = FitContext(g_fun_vals=g_fun, data_shape=global_var.grid_cache.inner_shape,
                     penalty_coeff=0.5, for_rps=False)
    wape_f, sinv_f = WAPEDiscrepancy(), ScaleInvariantDiscrepancy()
    print(f"\n================ preprocessor = {prep} "
          f"(truth: dT/dt = {kappa}*(T_xx+T_yy)) ================")
    print(f"  {'form':<26} {'WAPE':>9} {'scale-inv':>10} {'instab':>10}")
    rows = {}
    for tag, sym in forms:
        eq = evaluate(sym, search, fit_op)
        rows[tag.split()[0]] = (float(wape_f.compute(eq, ctx)),
                                float(sinv_f.compute(eq, ctx)),
                                float(eq.coefficients_stability))
        print(f"  {tag:<26} {rows[tag.split()[0]][0]:>9.5f} "
              f"{rows[tag.split()[0]][1]:>10.5f} {rows[tag.split()[0]][2]:>10.6f}")
    for name, i in [('WAPE', 0), ('scale-inv', 1)]:
        order = sorted(rows.items(), key=lambda kv: kv[1][i])
        print(f"    rank {name:>10}: " + "  <  ".join(f"{k}({v[i]:.4f})" for k, v in order))
    mix = rows['MIXED']; tr = rows['TRUTH']
    print(f"    MIXED vs TRUTH: WAPE {mix[0]:.4f}/{tr[0]:.4f}  scale-inv {mix[1]:.4f}/{tr[1]:.4f}"
          f"  instab {mix[2]:.5f}/{tr[2]:.5f}  -> "
          f"{'MIXED dominates' if mix[0] < tr[0] and mix[2] < tr[2] else 'MIXED no longer dominates both'}")


def main():
    for prep in ('FD', 'poly'):
        score_forms(prep)


def _decompose_wape():
    """Expose WHY T*E=0 scores below E=0 under WAPE = sum|r| / sum|target|.

    Multiplying the equation by T maps target->T*target and residual->T*r, so
    WAPE becomes a |T|-reweighted mean of the pointwise relative error. The FD
    error is worst in the low-signal tails (small |T|), which the extra |T|
    weight suppresses -> the ratio drops though the physics is identical.
    """
    coords, field, _ = _synthetic_field()
    t, y, x = coords
    T = field
    Tt = np.gradient(T, t, axis=0)
    Tyy = np.gradient(np.gradient(T, y, axis=1), y, axis=1)
    Txx = np.gradient(np.gradient(T, x, axis=2), x, axis=2)
    r = Tt - 0.5 * (Txx + Tyy)                       # residual of the true PDE
    s = slice(2, -2)
    T, Tt, r = T[:, s, s][2:-2], Tt[:, s, s][2:-2], r[:, s, s][2:-2]   # trim FD edges

    wape_clean = np.sum(np.abs(r)) / np.sum(np.abs(Tt))
    wape_deg = np.sum(np.abs(T * r)) / np.sum(np.abs(T * Tt))
    relerr = np.abs(r) / (np.abs(Tt) + 1e-12)
    hi = T >= np.median(T)                            # high-signal core vs tail
    print("\n--- WAPE non-invariance under x T (np.gradient illustration) ---")
    print(f"  clean   WAPE = sum|r|/sum|T_t|       = {wape_clean:.4f}")
    print(f"  x T     WAPE = sum|T r|/sum|T T_t|   = {wape_deg:.4f}   (lower, same physics)")
    print(f"  mean rel-err  core(|T|>=med)={relerr[hi].mean():.3f}  "
          f"tail(|T|<med)={relerr[~hi].mean():.3f}  "
          f"<- x T downweights the worse tail")


if __name__ == "__main__":
    main()
