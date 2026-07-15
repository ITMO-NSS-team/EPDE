"""Verify L2RelativeDiscrepancy as a primary discrepancy_metric (an L2 analogue
of WAPE / ScaleInvariant). Confirms:
  1. EpdeSearch(discrepancy_metric='l2_relative') builds (strategy switch wired).
  2. The filler computes ||target - sum c_j phi_j||_2 / ||target||_2 as the
     primary fitness, side-by-side with WAPE (L1) and ScaleInvariant.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import numpy as np
from epde import globals as gv
gv.set_gram_config('vcoef')

from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation
from epde.operators.common.fitness import SolverFreeFitness
from epde.operators.common.objectives import (L2RelativeDiscrepancy, WAPEDiscrepancy,
                                              ScaleInvariantDiscrepancy)
from epde.operators.common.sparsity import VWSRSparsity
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.utils.default_parameter_loader import EvolutionaryParams

UXX = 'd^2u/dx1^2{power: 1.0}'
U1, U2, U3, U4 = ('u{power: 1.0}', 'u{power: 2.0}', 'u{power: 3.0}', 'u{power: 4.0}')
UT, UX = 'du/dx0{power: 1.0}', 'du/dx1{power: 1.0}'

FORMS = {
    "TRUE  u_t = D u_xx + 5u - 5u^3":   f"1e-4 * {UXX} + -5.0 * {U3} + 5.0 * {U1} = {UT}",
    "TRUE + u_x (spurious)":            f"1e-4 * {UXX} + -5.0 * {U3} + 5.0 * {U1} + 0.0 * {UX} = {UT}",
    "TRUNC u_t = 5u - 5u^3":            f"-5.0 * {U3} + 5.0 * {U1} = {UT}",
    "DEGEN xu  u*u_t = ...":           f"1e-4 * {U1} * {UXX} + 5.0 * {U2} + -5.0 * {U4} = {U1} * {UT}",
}


def build_pool(discrepancy_metric='l2_relative'):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = np.linspace(0., 1., 51)
    x = np.linspace(-1., 0.984375, 128)
    data = np.load(os.path.join(os.path.dirname(__file__), 'ac_data.npy'))
    grid = np.meshgrid(t, x, indexing='ij')
    boundary = tuple(max(1, n // 10) for n in data.shape)
    search = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True,
                        boundary=boundary, coordinate_tensors=grid, device=device,
                        discrepancy_metric=discrepancy_metric)   # <-- exercises the switch
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    search.create_pool(data=[data], variable_names=['u'], max_deriv_order=(2, 4),
                       additional_tokens=[], data_fun_pow=4, deriv_fun_pow=2)
    return search


def make_fit(primary):
    params = EvolutionaryParams()
    op_params = params.get_default_params_for_operator('SolverFreeFitness')
    fit_op = SolverFreeFitness(list(op_params.keys()), objectives=[primary], primary=primary)
    fit_op.params = op_params
    fit_op.set_suboperators({'sparsity': VWSRSparsity(),
                             'coeff_calc': LinRegBasedCoeffsEquation()})
    return fit_op


def evaluate(symbolic, search, fit_op):
    eq = translate_equation(symbolic, search.pool, all_vars=['u']).vals['u']
    eq.main_var_to_explain = 'u'
    eq.metaparameters = {('sparsity', 'u'): {'optimizable': False, 'value': 1e-6}}
    eq.weights_internal = np.ones(len(eq.structure) - 1)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    fit_op.apply(eq, {}, force_out_of_place=True)
    eq.fitness_calculated = False
    fit_op.apply(eq, {}, force_out_of_place=False)
    return float(eq.fitness_value)


def main():
    search = build_pool('l2_relative')
    print("  EpdeSearch(discrepancy_metric='l2_relative') built OK\n")
    fit_l2r = make_fit(L2RelativeDiscrepancy())
    fit_wape = make_fit(WAPEDiscrepancy())
    fit_sinv = make_fit(ScaleInvariantDiscrepancy())
    print(f"  {'form':<34}{'L2-relative':>13}{'WAPE (L1)':>12}{'scale-inv':>12}")
    for tag, sym in FORMS.items():
        l2r = evaluate(sym, search, fit_l2r)
        wape = evaluate(sym, search, fit_wape)
        sinv = evaluate(sym, search, fit_sinv)
        print(f"  {tag:<34}{l2r:>13.5f}{wape:>12.5f}{sinv:>12.5f}")


if __name__ == "__main__":
    import torch
    print('cuda available:', torch.cuda.is_available())
    main()
