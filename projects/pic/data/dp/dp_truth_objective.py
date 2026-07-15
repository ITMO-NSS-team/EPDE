"""Seed the FULL 4-term double-pendulum EOM (and partial forms) on the real Trial-1
data and read each form's EPDE objective vector, to confirm the complete equation is
the TOP-scoring structure -- i.e. that the discovery's failure to assemble all 4 terms
in one chromosome is a SEARCH-convergence gap, not an objectives flaw. Mirrors
sst_truth_objective.py but for the 2-variable 2nd-order system.

Per equation we compare (coefficients refit by the pipeline, so the contrast is purely
structural):
  FULL  th_i'' = a*cos(D)*th_j'' + b*sin(D)*th_j'^2 + c*sin th_i   (true 4-term EOM)
  DISC  th_i'' = a*cos(D)*th_j'' + c*sin th_i                      (what EPDE found: no centrifugal)
  GRAV  th_i'' = c*sin th_i                                        (uncoupled pendulum)

A 2-var SoEq needs BOTH equations, so the equation under test is paired with the other
variable's FULL form as a placeholder (each equation's objective depends only on its own
structure+target). If FULL has the lowest discrepancy for BOTH equations, the objectives
already rank the true EOM best.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

import epde.globals as global_var
from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation
from epde.operators.common.fitness import SolverFreeFitness
from epde.operators.common.objectives import (WAPEDiscrepancy, Instability,
                                              ScaleInvariantDiscrepancy, FitContext)
from epde.operators.common.sparsity import VWSRSparsity
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.utils.default_parameter_loader import EvolutionaryParams
from epde import CacheStoredTokens

import dp_real_discovery as disc

# d^2theta1/dx0^2 == th1'';  dtheta1/dx0{power:2.0} == th1'^2;  cos_delta == cos(th1-th2)
T1, T2 = 'd^2theta1/dx0^2{power: 1.0}', 'd^2theta2/dx0^2{power: 1.0}'
W1sq, W2sq = 'dtheta1/dx0{power: 2.0}', 'dtheta2/dx0{power: 2.0}'
COS, SIN = 'cos_delta{power: 1.0}', 'sin_delta{power: 1.0}'
S1, S2 = 'sin_th1{power: 1.0}', 'sin_th2{power: 1.0}'

FULL1 = f"1.0 * {COS} * {T2} + 1.0 * {SIN} * {W2sq} + 1.0 * {S1} = {T1}"
FULL2 = f"1.0 * {COS} * {T1} + 1.0 * {SIN} * {W1sq} + 1.0 * {S2} = {T2}"
OTHER_FULL = {'theta1': FULL1, 'theta2': FULL2}

FORMS = {
    'theta1': {
        "FULL [cosD*th2dd + sinD*th2d^2 + sin th1]": FULL1,
        "DISC [cosD*th2dd + sin th1]":               f"1.0 * {COS} * {T2} + 1.0 * {S1} = {T1}",
        "GRAV [sin th1 only]":                        f"1.0 * {S1} = {T1}",
    },
    'theta2': {
        "FULL [cosD*th1dd + sinD*th1d^2 + sin th2]": FULL2,
        "DISC [cosD*th1dd + sin th2]":               f"1.0 * {COS} * {T1} + 1.0 * {S2} = {T2}",
        "GRAV [sin th2 only]":                        f"1.0 * {S2} = {T2}",
    },
}


def build_pool(start=2000, n=8000, poly_window=51, boundary=60):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t, th1, th2 = disc.load_angles('Trial1', start, n)
    dim = th1.ndim - 1
    D = th1 - th2
    search = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True,
                        boundary=boundary, coordinate_tensors=(t,), device=device)
    search.set_preprocessor(default_preprocessor_type='poly',
                            preprocessor_kwargs={'polynomial_window': poly_window, 'poly_order': 4})
    trig_state = CacheStoredTokens(token_type='trig_state', token_labels=['sin_th1', 'sin_th2'],
                                   token_tensors={'sin_th1': np.sin(th1), 'sin_th2': np.sin(th2)},
                                   params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                                   dimensionality=dim, meaningful=True)
    coupling = CacheStoredTokens(token_type='coupling', token_labels=['cos_delta', 'sin_delta'],
                                 token_tensors={'cos_delta': np.cos(D), 'sin_delta': np.sin(D)},
                                 params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                                 dimensionality=dim, meaningful=False)
    search.create_pool(data=[th1, th2], variable_names=['theta1', 'theta2'],
                       max_deriv_order=(2,), additional_tokens=[trig_state, coupling],
                       data_fun_pow=1, deriv_fun_pow=2)
    return search


def make_fit_operator():
    params = EvolutionaryParams()
    op_params = params.get_default_params_for_operator('SolverFreeFitness')
    d = WAPEDiscrepancy()
    fit_op = SolverFreeFitness(list(op_params.keys()), objectives=[d, Instability()], primary=d)
    fit_op.params = op_params
    fit_op.set_suboperators({'sparsity': VWSRSparsity(), 'coeff_calc': LinRegBasedCoeffsEquation()})
    return fit_op


def evaluate(form, var, search, fit_op):
    # SoEq assigns equations by POOL POSITION, not var_to_explain -> always build the
    # dict in pool order (theta1, theta2), test form in `var`'s slot, FULL in the other.
    forms_d = dict(OTHER_FULL)
    forms_d[var] = form
    ordered = {'theta1': forms_d['theta1'], 'theta2': forms_d['theta2']}
    soeq = translate_equation(ordered, search.pool, all_vars=['theta1', 'theta2'])
    eq = soeq.vals[var]
    eq.main_var_to_explain = var
    eq.metaparameters = {('sparsity', v): {'optimizable': False, 'value': 1e-6} for v in ('theta1', 'theta2')}
    eq.weights_internal = np.ones(len(eq.structure) - 1)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    fit_op.apply(eq, {}, force_out_of_place=True)
    eq.fitness_calculated = False
    eq.stability_calculated = False
    fit_op.apply(eq, {}, force_out_of_place=False)
    return eq


def main():
    from epde import globals as gv
    gv.set_gram_config('vcoef')
    search = build_pool()
    fit_op = make_fit_operator()
    g_fun = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask].reshape(-1)
    ctx = FitContext(g_fun_vals=g_fun, data_shape=global_var.grid_cache.inner_shape,
                     penalty_coeff=0.5, for_rps=False)
    wape_f, sinv_f = WAPEDiscrepancy(), ScaleInvariantDiscrepancy()
    for var, forms in FORMS.items():
        print(f"\n================ equation for {var} (target = {var}'') ================")
        print(f"  {'form':<44} {'WAPE':>9} {'scale-inv':>10} {'instab':>10}")
        rows = {}
        for tag, sym in forms.items():
            eq = evaluate(sym, var, search, fit_op)
            rows[tag] = (float(wape_f.compute(eq, ctx)), float(sinv_f.compute(eq, ctx)),
                         float(eq.coefficients_stability))
            print(f"  {tag:<44} {rows[tag][0]:>9.5f} {rows[tag][1]:>10.5f} {rows[tag][2]:>10.6f}")
            print(f"      -> {eq.text_form}")
        best = min(rows.items(), key=lambda kv: kv[1][0])[0]
        print(f"    lowest WAPE: {best}")


if __name__ == "__main__":
    import torch
    print('cuda', torch.cuda.is_available())
    main()
