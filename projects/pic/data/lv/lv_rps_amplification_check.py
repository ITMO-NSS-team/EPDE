"""Check a suspect LV system against the RPS amplification guard.

Rebuilds the exact pool of lv.py (t_20/data_20, end=150, boundary=15, poly
preprocessor, max_deriv_order=(2,), trig+grid tokens, data_fun_pow=3),
translates the pasted front system onto it, refits each equation the way the
EqRightPartSelector term-sweep does (force_out_of_place VWSR+OLS fit of the
fixed structure+target), and reports the guard ratio

    A = sum_j |c_j| * ||col_j|| / ||target col||

against ``global_var.rps_amplification_cap`` -- i.e. whether the sweep would
have DECLINED this (structure, target) candidate. Run across a range of sparsity
values (1e-8 .. 1e-1 below) to show the sparsity-floor effect. Note that under
the default VWSR sparsity the search itself no longer reads a seeding interval;
the values below are set directly on the ('sparsity', var) metaparameter.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

import epde.globals as global_var
from epde import TrigonometricTokens, GridTokens
from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation
from epde.operators.common.fitness import SolverFreeFitness
from epde.operators.common.objectives import Discrepancy, Instability
from epde.operators.common.right_part_selection import amplification_ratio
from epde.operators.common.sparsity import VWSRSparsity
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.utils.default_parameter_loader import EvolutionaryParams

EQ_U = ("-22178.99924165544 * dv/dx0{power: 1.0} + "
        "-22155.867880447346 * du/dx0{power: 1.0} + "
        "-443224.80403364147 * v{power: 1.0} + "
        "443245.9256850578 * u{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}")
EQ_V = ("16652.998354806732 * x_0{power: 1.0, dim: 0.0} + "
        "2306.0441164471567 * u{power: 1.0} + "
        "-83774.24313642483 * x_0{power: 2.0, dim: 0.0} + 0.0 = "
        "d^2v/dx0^2{power: 1.0}")


def build_pool():
    here = os.path.dirname(__file__)
    t = np.load(os.path.join(here, 't_20.npy'))[:150]
    data = np.load(os.path.join(here, 'data_20.npy'))[:150]
    x, y = data[:, 0], data[:, 1]
    dim = x.ndim - 1
    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8), dimensionality=dim)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dim, max_power=2)
    search = EpdeSearch(use_solver=False, multiobjective_mode=True,
                        verbose_params={'show_iter_idx': False}, device='cpu')
    _, domain = search.createDomain((t,), boundary_width=15, ID=0)
    search.set_preprocessor(default_preprocessor_type='poly', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory({'u': x, 'v': y}, domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(2,), data_fun_pow=3,
                       additional_tokens=[trig_tokens, grid_tokens])
    return search


def make_fit_operator():
    params = EvolutionaryParams()
    op_params = params.get_default_params_for_operator('SolverFreeFitness')
    d = Discrepancy('wape')
    fit_op = SolverFreeFitness(list(op_params.keys()), objectives=[d, Instability()], primary=d)
    fit_op.params = op_params
    # The host is a pure scorer now (no suboperators); ``fit_like_rps`` below
    # supplies the fit that EqRightPartSelector performs before every score.
    return fit_op


def fit_like_rps(equation):
    """The fit EqRightPartSelector runs per candidate target: support
    selection, then physical magnitudes. Kept beside the scorer so this probe
    reproduces the sweep exactly rather than an approximation of it."""
    VWSRSparsity().apply(equation, {})
    LinRegBasedCoeffsEquation().apply(equation, {})


def check(search, fit_op, sparsity_value):
    soeq = translate_equation({'u': EQ_U, 'v': EQ_V}, search.pool, all_vars=['u', 'v'])
    cap = global_var.rps_amplification_cap
    print(f"\n--- sweep-style refit at sparsity = {sparsity_value:g} "
          f"(cap = {cap:g}) ---")
    for var in ('u', 'v'):
        eq = soeq.vals[var]
        eq.main_var_to_explain = var
        eq.metaparameters = {('sparsity', v): {'optimizable': False, 'value': sparsity_value}
                             for v in ('u', 'v')}
        eq.weights_internal = np.append(np.ones(len(eq.structure) - 1), 0.0)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        fit_like_rps(eq)
        wape = fit_op.apply(eq, {}, force_out_of_place=True)
        ratio = amplification_ratio(eq)
        verdict = 'DECLINED by guard' if (cap is not None and ratio > cap) else 'passes guard'
        # None = the host declined the candidate outright (zeroed /
        # degenerate / amplified) -- the sweep would never consider it.
        wape_str = 'None (declined by host)' if wape is None else f'{wape:.6f}'
        print(f"\n  eq[{var}]  target_idx={eq.target_idx}  WAPE={wape_str}")
        terms = [t.name for i, t in enumerate(eq.structure) if i != eq.target_idx]
        coefs = np.asarray(eq.weights_internal, dtype=float)
        for name, c in zip(terms, coefs[:len(terms)]):
            print(f"    {c:+.6e} * {name}")
        print(f"    intercept {coefs[-1]:+.6e}")
        print(f"    amplification A = {ratio:.3e}  ->  {verdict}")
        # In-place verdict: the fitness backstop must condemn an amplified
        # fit (stamp LOSS_NAN_VAL) no matter which path installed the target.
        eq.fitness_calculated = False
        eq.stability_calculated = False
        fit_op.apply(eq, {}, force_out_of_place=False)
        condemned = eq.fitness_value >= 1e7
        print(f"    in-place fitness_value = {eq.fitness_value:.6g}  ->  "
              f"{'CONDEMNED (backstop)' if condemned else 'kept'}")


def main():
    global_var_cap = global_var.rps_amplification_cap
    print(f"rps_amplification_cap = {global_var_cap}")
    search = build_pool()
    fit_op = make_fit_operator()
    for sp in (1e-8, 1e-4, 1e-1):
        check(search, fit_op, sp)


if __name__ == '__main__':
    main()
