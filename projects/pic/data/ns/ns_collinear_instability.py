"""Does the instability / coefficient-stability objective break the collinear tie
in the NS v-momentum equation that OLS magnitude (contribution share) and necessity
(leave-one-out dR2) both fail on?

Setup (from term_contribution.py --hard): in eq[1] (dv/dx0) the spurious term
``v * du/dx2`` (v*u_x) grabbed 14% contribution share -- MORE than the true
``v * dv/dx1`` (v*v_y, 0.4%) -- because CONTINUITY u_x = -v_y (the 3rd NS truth eq)
makes ``v*u_x == -v*v_y`` collinear. So the "spurious" substitution is actually a
VALID IDENTITY, not a truly-wrong term. This script measures it directly.

We score eq 'v' (target dv/dx0) in 5 structural forms (coeffs refit by the pipeline,
so the contrast is purely structural), reading [WAPE, scale-inv, instability]:
  TRUE      ... -1*v*v_y ...                (the true v-momentum balance)
  IDENT     ... +1*v*u_x ...                (continuity substitution: SAME physics)
  WRONG_uux ... +1*u*u_x ...                (u-advection in the v-eq: truly wrong)
  WRONG_v2  ... +1*v^2   ...                (nonlinear reaction: truly wrong)
  DROP      (v*v_y removed)                 (truncated)

Expectation if the objectives are well-posed:
  - TRUE ~= IDENT on ALL THREE metrics (valid identity must NOT be separated);
  - WRONG_* / DROP separated by discrepancy (and ideally also instability).
The question for the instability objective: does it add a signal beyond discrepancy
on the truly-wrong forms, and does it (correctly) stay silent on IDENT?
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../thesis')))

import numpy as np

import epde.globals as global_var
from epde import globals as gv
from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation, parse_factor
from epde.structure.main_structures import Term
from epde.operators.common.fitness import SolverFreeFitness
from epde.operators.common.objectives import (WAPEDiscrepancy, Instability,
                                              ScaleInvariantDiscrepancy, FitContext)
from epde.operators.common.sparsity import VWSRSparsity
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.utils.default_parameter_loader import EvolutionaryParams

from adapters import ns as ns_adapter

VARS = ['u', 'v', 'p']

U_EQ = ("-1.0 * u{power: 1.0} * du/dx2{power: 1.0} + -1.0 * v{power: 1.0} * du/dx1{power: 1.0} + "
        "-1.0 * dp/dx2{power: 1.0} + 0.01 * d^2u/dx2^2{power: 1.0} + 0.01 * d^2u/dx1^2{power: 1.0} "
        "= du/dx0{power: 1.0}")
CONT_EQ = "-1.0 * dv/dx1{power: 1.0} = du/dx2{power: 1.0}"

# eq 'v': vary the 2nd term only (@T2@ placeholder -- str.format would clash
# with the literal {power: ...} braces).
_V = ("-1.0 * u{power: 1.0} * dv/dx2{power: 1.0} + @T2@ + -1.0 * dp/dx1{power: 1.0} + "
      "0.01 * d^2v/dx2^2{power: 1.0} + 0.01 * d^2v/dx1^2{power: 1.0} = dv/dx0{power: 1.0}")
V_FORMS = {
    'TRUE      [-v*v_y]':  _V.replace('@T2@', '-1.0 * v{power: 1.0} * dv/dx1{power: 1.0}'),
    'IDENT     [+v*u_x]':  _V.replace('@T2@', '1.0 * v{power: 1.0} * du/dx2{power: 1.0}'),
    'WRONG_uux [+u*u_x]':  _V.replace('@T2@', '1.0 * u{power: 1.0} * du/dx2{power: 1.0}'),
    'WRONG_v2  [+v^2]':    _V.replace('@T2@', '1.0 * v{power: 2.0}'),
    'DROP      [no term]': ("-1.0 * u{power: 1.0} * dv/dx2{power: 1.0} + -1.0 * dp/dx1{power: 1.0} + "
                            "0.01 * d^2v/dx2^2{power: 1.0} + 0.01 * d^2v/dx1^2{power: 1.0} "
                            "= dv/dx0{power: 1.0}"),
}
OTHER = {'u': U_EQ, 'p': CONT_EQ}


def build_pool():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coords, data, variable_names, dim = ns_adapter.load_data()
    boundary = tuple(max(1, n // 10) for n in np.asarray(coords[0]).shape)
    search = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True,
                        boundary=boundary, coordinate_tensors=coords, device=device)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    search.create_pool(data=data, variable_names=variable_names, max_deriv_order=(1, 2, 2),
                       additional_tokens=[], data_fun_pow=3, deriv_fun_pow=2)
    return search


def make_fit_operator():
    params = EvolutionaryParams()
    op_params = params.get_default_params_for_operator('SolverFreeFitness')
    disc = WAPEDiscrepancy()
    fit_op = SolverFreeFitness(list(op_params.keys()), objectives=[disc, Instability()], primary=disc)
    fit_op.params = op_params
    fit_op.set_suboperators({'sparsity': VWSRSparsity(), 'coeff_calc': LinRegBasedCoeffsEquation()})
    return fit_op


def evaluate(vform, search, fit_op):
    forms = dict(OTHER); forms['v'] = vform
    ordered = {k: forms[k] for k in VARS}
    soeq = translate_equation(ordered, search.pool, all_vars=VARS)
    eq = soeq.vals['v']
    eq.main_var_to_explain = 'v'
    eq.metaparameters = {('sparsity', x): {'optimizable': False, 'value': 1e-6} for x in VARS}
    eq.weights_internal = np.ones(len(eq.structure) - 1)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    fit_op.apply(eq, {}, force_out_of_place=True)
    eq.fitness_calculated = False
    eq.stability_calculated = False
    fit_op.apply(eq, {}, force_out_of_place=False)
    return eq


def _field(factor_strs, pool):
    t = Term(pool, passed_term=[parse_factor(fs, pool, VARS) for fs in factor_strs],
             collapse_powers=False)
    return np.asarray(t.evaluate(False)).reshape(-1).astype(np.float64)


def main():
    gv.set_gram_config('vcoef')
    search = build_pool()

    # Direct identity check: is v*u_x == -v*v_y on the data (FD continuity)?
    vux = _field(['v{power: 1.0}', 'du/dx2{power: 1.0}'], search.pool)
    vvy = _field(['v{power: 1.0}', 'dv/dx1{power: 1.0}'], search.pool)
    corr = np.corrcoef(vux, -vvy)[0, 1]
    rel = np.sqrt(np.mean((vux + vvy) ** 2)) / np.sqrt(np.mean(vvy ** 2))
    print(f"\n[identity] corr(v*u_x, -v*v_y) = {corr:.5f}   ||v*u_x + v*v_y|| / ||v*v_y|| = {rel:.4f}")
    print("           (corr~1 / rel~0  =>  v*u_x is the SAME field as -v*v_y via continuity"
          " => a VALID IDENTITY, not a wrong term)")

    fit_op = make_fit_operator()
    g_fun = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask].reshape(-1)
    ctx = FitContext(g_fun_vals=g_fun, data_shape=global_var.grid_cache.inner_shape,
                     penalty_coeff=0.5, for_rps=False)
    wape_f, sinv_f = WAPEDiscrepancy(), ScaleInvariantDiscrepancy()

    print(f"\n  {'form':<22}{'WAPE':>11}{'scale-inv':>11}{'instab':>12}")
    rows = {}
    for tag, vform in V_FORMS.items():
        eq = evaluate(vform, search, fit_op)
        rows[tag] = (float(wape_f.compute(eq, ctx)), float(sinv_f.compute(eq, ctx)),
                     float(eq.coefficients_stability))
        print(f"  {tag:<22}{rows[tag][0]:>11.5f}{rows[tag][1]:>11.5f}{rows[tag][2]:>12.6f}")
        print(f"      refit: {eq.text_form}")

    print("\n  ranking (lower = better):")
    for name, i in [('WAPE', 0), ('scale-inv', 1), ('instab', 2)]:
        order = sorted(rows.items(), key=lambda kv: kv[1][i])
        print(f"    {name:>10}: " + "  <  ".join(f"{k.split()[0]}({v[i]:.4f})" for k, v in order))

    t, ide = rows['TRUE      [-v*v_y]'], rows['IDENT     [+v*u_x]']
    print(f"\n  TRUE vs IDENT (valid identity -- should be ~equal on ALL three): "
          f"WAPE {t[0]:.5f}/{ide[0]:.5f}  sinv {t[1]:.5f}/{ide[1]:.5f}  instab {t[2]:.6f}/{ide[2]:.6f}")
    print("  -> If TRUE!=IDENT on instability, the objective wrongly penalises a valid identity;")
    print("     if WRONG_* are separated above TRUE while IDENT stays tied, the stack is well-posed.")


if __name__ == '__main__':
    import torch
    print('cuda', torch.cuda.is_available())
    main()
