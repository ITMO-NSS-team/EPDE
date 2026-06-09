"""Diagnose why the spurious wave solution beats the true wave equation.

Spurious (Pareto-0): 48.679*u_xx*u_tt - 591.98*u_tt^2 = u_xx^2   (target = u_xx^2)
True:                0.04*u_xx = u_tt                            (u_tt = c^2 u_xx, c^2=0.04)

Hypothesis: on the wave solution u_tt = 0.04 u_xx, the three quadratic-in-2nd-
derivative terms u_xx^2, u_xx*u_tt, u_tt^2 are all proportional, so the spurious
equation is an EXACT algebraic identity -> fits ~perfectly with constant coefs,
dominating the (FD-limited) true equation on (fitness, stability). Run once, delete.
"""
import os, sys
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import epde.globals as gv
from epde.interface.equation_translator import translate_equation
from kdv_sindy_test import build_pool_only, make_fit_operator, _normalize_grid_labels
from thesis_runner import load_config, pipeline_settings, _set_seeds

DISCOVERED = ("48.67869238111131 * d^2u/dx1^2{power: 1.0} * d^2u/dx0^2{power: 1.0} "
              "+ -591.9797844706457 * d^2u/dx0^2{power: 2.0} + 0.0 = d^2u/dx1^2{power: 2.0}")
TRUE = "0.04 * d^2u/dx1^2{power: 1.0} = d^2u/dx0^2{power: 1.0}"


def evaluate_eq(search, fit_op, eq_str):
    soeq = translate_equation(_normalize_grid_labels(eq_str), search.pool, all_vars=["u"])
    eq = soeq.vals["u"]
    eq.main_var_to_explain = "u"
    eq.weights_internal = np.ones(len(eq.structure) - 1)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    fit_op.apply(eq, {}, force_out_of_place=True)
    return eq


def main():
    cfg = load_config("wave")
    _set_seeds(0)
    gv.set_gram_config("vcoef")
    search = build_pool_only(cfg, pipeline_settings("new"))
    fit_op = make_fit_operator()

    print("=" * 70)
    print("OBJECTIVE VECTORS  (both objectives: lower = better)")
    print("=" * 70)
    for name, s in [("DISCOVERED (spurious identity)", DISCOVERED), ("TRUE wave", TRUE)]:
        eq = evaluate_eq(search, fit_op, s)
        print(f"\n{name}")
        print(f"  fitness_value          = {getattr(eq, 'fitness_value', None)!r}")
        print(f"  coefficients_stability = {getattr(eq, 'coefficients_stability', None)!r}")
        print(f"  obj_fun                = {getattr(eq, 'obj_fun', None)!r}")

    # ---- identity check on the actual solution fields ----
    soeq = translate_equation(_normalize_grid_labels(TRUE), search.pool, all_vars=["u"])
    teq = soeq.vals["u"]
    teq.main_var_to_explain = "u"
    teq.weights_internal = np.ones(len(teq.structure) - 1)
    teq.weights_internal_evald = True
    teq.weights_final_evald = True
    _, t, f = teq.evaluate(normalize=False, return_val=False)
    u_tt = np.asarray(t, float).reshape(-1)
    f = np.asarray(f, float)
    u_xx = f[:, 0] if f.ndim > 1 else f.reshape(-1)

    a = float(np.linalg.lstsq(u_xx[:, None], u_tt, rcond=None)[0][0])
    A, B, C = u_xx ** 2, u_xx * u_tt, u_tt ** 2
    M = np.column_stack([B, C])
    coef, *_ = np.linalg.lstsq(M, A, rcond=None)
    resid = A - M @ coef
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((A - A.mean()) ** 2)

    print("\n" + "=" * 70)
    print("IDENTITY CHECK on the wave solution")
    print("=" * 70)
    print(f"  u_tt = a * u_xx   ->  a = {a:.6f}   (truth c^2 = 0.04)")
    print(f"  u_xx^2 = b1*(u_xx*u_tt) + b2*(u_tt^2):")
    print(f"     b1 = {coef[0]:.4f}  (discovered 48.679)")
    print(f"     b2 = {coef[1]:.4f}  (discovered -591.98)")
    print(f"  R^2 = {r2:.8f}    ||resid||/||u_xx^2|| = {np.linalg.norm(resid)/np.linalg.norm(A):.3e}")
    print(f"  arithmetic: 48.6787*0.04 - 591.98*0.04^2 = "
          f"{48.6787*0.04 - 591.98*0.04**2:.6f}   (should be ~1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
