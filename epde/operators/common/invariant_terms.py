"""Terms an exact algebraic invariant of the variables makes redundant: plain mode's
``system_rps_params['invariant_terms']='refuse'``.

Variables that satisfy a polynomial identity at every sample -- a spinning robot's heading vector (c, s) on
``c**2 + s**2 = 1`` -- carry exact identities into the term space. The invariant and its derivatives
(``c c' + s s' = 0``, ``c'**2 + c c'' + s'**2 + s s'' = 0``, ...) let an equation write its target through itself
(``dc/dt = c' s**2 - c s s'``) or one law in many forms (``dc/dt = -w s (c**2 + s**2)``). On derivatives estimated
from data such a form can fit better than the law, and it fills the fronts.

The rule, once per token pool:

1. The invariants are read off the data: every polynomial of degree at most :data:`DEGREE` in the system's
   variables that vanishes at every sample of every trajectory (``raw_fields``, the observed fields before any
   preprocessing), to the machine-exactness tolerance of the exact-dependence tests
   (``right_part_selection._LIN_DEP_RTOL``) on unit-normalised monomial columns. They are put in reduced echelon
   form over the monomials (each led by its largest monomial), the coefficients rounded to fractions (denominator
   at most :data:`MAX_DENOMINATOR`) and checked again on the data; one that fails the check is dropped and warned.
2. Each invariant is differentiated along every grid axis, again and again, while every derivative the result
   names is a token of the pool.
3. The Groebner basis of these polynomials (sympy, grevlex; a higher derivative order ranks above a lower one, then
   a later variable above an earlier one, then the larger per-axis orders) gives leading monomials. A term whose
   monomial in the variables and their derivatives -- every other factor left out -- is divisible by one of them is
   REFUSED: the plain fit leaves its column out of the support and the fitted version drops it
   (``plain_fit._versions``).

The monomials left (the standard monomials) are linearly independent modulo the identities and span every
polynomial of the variables on the data: no identity can be written, and a polynomial law keeps exactly one form,
its normal form -- ``dc/dt = -w s`` is one; a law in ``s**2`` is written with ``1 - c**2``. Data without such an
invariant leave every term alone. The test reads only a term's structure, so it holds on weak rows too.

sympy is imported only when the rule is on.
"""
import warnings
import weakref
from fractions import Fraction
from itertools import combinations_with_replacement
from typing import List, Optional

import numpy as np

import epde.globals as global_var

#: The highest degree of an invariant looked for.
DEGREE = 2
#: The largest denominator an invariant's rounded coefficients may have.
MAX_DENOMINATOR = 1 << 20

_RULES = {}       # id(pool) -> (weak reference to the pool, the pool's rule or None)
_REPORTED = set()


def reset() -> None:
    """Forget every pool's rule and the warnings already given."""
    _RULES.clear()
    _REPORTED.clear()


def _warn_once(message: str) -> None:
    if message not in _REPORTED:
        _REPORTED.add(message)
        warnings.warn(message, global_var.EPDEUsageWarning, stacklevel=3)


def _param(factor, name, default):
    for idx, description in factor.params_description.items():
        if description['name'] == name:
            return float(factor.params[idx])
    return default


def _orders(deriv_code, ndim: int) -> tuple:
    """Per-axis derivative orders of a ``deriv_code`` (``[None]`` is the field itself)."""
    orders = [0] * ndim
    for axis in deriv_code:
        if axis is not None:
            orders[axis] += 1
    return tuple(orders)


def _grevlex_key(exps: tuple):
    """Sort key of an exponent tuple (generators listed from the largest): graded reverse lexicographic."""
    return sum(exps), tuple(-e for e in reversed(exps))


def _monomials(n: int, degree: int) -> List[tuple]:
    out = [(0,) * n]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n), d):
            exps = [0] * n
            for i in combo:
                exps[i] += 1
            out.append(tuple(exps))
    return out


def detect(values: List[np.ndarray], degree: int = DEGREE) -> List[dict]:
    """The exact polynomial invariants of the 1-D sample arrays ``values`` (one per variable, generators from
    the largest): reduced-echelon polynomials ``{exponent tuple: Fraction}`` of degree at most ``degree`` that
    vanish at every sample (module docstring, step 1). Rows with a non-finite value are left out."""
    from epde.operators.common.right_part_selection import _LIN_DEP_RTOL
    V = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in values], axis=1)
    V = V[np.all(np.isfinite(V), axis=1)]
    if V.shape[0] == 0:
        return []
    monos = sorted(_monomials(V.shape[1], degree), key=_grevlex_key, reverse=True)
    M = np.stack([np.prod(V ** np.array(e), axis=1) for e in monos], axis=1)
    norms = np.linalg.norm(M, axis=0)
    if np.any(norms == 0):
        return []
    _, sv, vt = np.linalg.svd(M / norms, full_matrices=False)
    null = vt[sv <= _LIN_DEP_RTOL * sv[0]]
    if null.shape[0] == 0:
        return []
    C = null / norms
    rows, col = [], 0
    for col in range(C.shape[1]):              # reduced echelon form, pivots in monomial order
        if len(rows) == C.shape[0]:
            break
        r = len(rows)
        pivot = r + int(np.argmax(np.abs(C[r:, col])))
        if abs(C[pivot, col]) <= _LIN_DEP_RTOL * np.max(np.abs(C[r:])):
            continue
        C[[r, pivot]] = C[[pivot, r]]
        C[r] = C[r] / C[r, col]
        for k in range(C.shape[0]):
            if k != r:
                C[k] = C[k] - C[k, col] * C[r]
        rows.append(r)
    out = []
    for r in rows:
        coefs = {monos[j]: Fraction(float(C[r, j])).limit_denominator(MAX_DENOMINATOR)
                 for j in range(len(monos)) if C[r, j] != 0}
        coefs = {e: q for e, q in coefs.items() if q != 0}
        raw = np.array([float(coefs.get(e, 0)) for e in monos])
        scale = float(np.sum(np.abs(raw) * norms))
        if scale > 0 and np.linalg.norm(M @ raw) <= _LIN_DEP_RTOL * scale:
            out.append(coefs)
        else:
            _warn_once('invariant_terms: an exact invariant of the variables has coefficients that are no '
                       'fractions with a denominator up to {0}; it is left out.'.format(MAX_DENOMINATOR))
    return out


class _Rule(object):
    """The leading monomials of one pool's invariants (module docstring)."""

    def __init__(self, variables, ndim, gens, lms, text):
        self.variables, self.ndim, self.lms, self.text = frozenset(variables), ndim, lms, text
        self.index = {key: i for i, key in enumerate(gens)}

    def reducible(self, term) -> bool:
        exps = [0] * len(self.index)
        for factor in term.structure:
            if factor.deriv_code is None or getattr(factor, 'variable', None) not in self.variables:
                continue
            pos = self.index.get((factor.variable, _orders(factor.deriv_code, self.ndim)))
            if pos is None:
                continue
            power = _param(factor, 'power', 1.0)
            if power < 0 or power != int(power):
                return False
            exps[pos] += int(power)
        return any(all(e >= l for e, l in zip(exps, lm)) for lm in self.lms)


def _pool_symbols(pool, variables, ndim) -> set:
    """(variable, per-axis orders) of every field and derivative token of ``variables`` in ``pool``."""
    keys = set()
    for family in pool.families:
        if not getattr(family, 'family_of_derivs', False) or family.variable not in variables:
            continue
        for code in getattr(family, 'derivs_ords', {}).values():
            keys.add((family.variable, _orders(code, ndim)))
    return keys


def _build(pool) -> Optional[_Rule]:
    try:
        import sympy
    except ImportError as exc:      # pragma: no cover - sympy ships with the scientific stack
        raise ImportError("system_rps_params['invariant_terms']='refuse' needs sympy (its Groebner bases).") \
            from exc
    manager = global_var.samples_manager
    ids = list(manager.trajecatoryIDs)
    variables = list(manager.raw_fields(ids[0]))
    ndim = manager.ndim
    keys = _pool_symbols(pool, variables, ndim)
    variables = [v for v in variables if (v, (0,) * ndim) in keys]
    if not variables:
        return None
    # generators from the largest: a higher total derivative order, then a later variable, then larger orders
    gens_keys = sorted(keys, key=lambda k: (sum(k[1]), variables.index(k[0]), k[1]), reverse=True)
    sym = {k: sympy.Symbol('g%d' % i) for i, k in enumerate(gens_keys)}
    base = sorted(variables, key=variables.index, reverse=True)
    stacked = [np.concatenate([np.asarray(manager.raw_fields(i)[v], dtype=float).reshape(-1) for i in ids])
               for v in base]
    found = detect(stacked)
    if not found:
        _warn_once("invariant_terms='refuse': the variables {0} satisfy no exact polynomial identity of degree "
                   "up to {1} on the data; every term is kept.".format(variables, DEGREE))
        return None
    polys = []
    for coefs in found:
        expr = sympy.Integer(0)
        for exps, q in coefs.items():
            mono = sympy.Integer(1)
            for v, e in zip(base, exps):
                mono = mono * sym[(v, (0,) * ndim)] ** e
            expr = expr + sympy.Rational(q.numerator, q.denominator) * mono
        polys.append(sympy.expand(expr))

    def derivative(expr, axis):
        out = sympy.Integer(0)
        for s in expr.free_symbols:
            key = next(k for k, v in sym.items() if v == s)
            up = (key[0], tuple(o + (a == axis) for a, o in enumerate(key[1])))
            if up not in sym:
                return None
            out = out + sympy.diff(expr, s) * sym[up]
        return sympy.expand(out)

    closure, frontier = list(polys), list(polys)
    while frontier:
        nxt = []
        for expr in frontier:
            for axis in range(ndim):
                d = derivative(expr, axis)
                if d is not None and d != 0 and all(sympy.expand(d - c) != 0 for c in closure):
                    closure.append(d)
                    nxt.append(d)
        frontier = nxt
    gens = [sym[k] for k in gens_keys]
    basis = sympy.groebner(closure, *gens, order='grevlex')
    lms = [sympy.Poly(g, *gens).monoms(order='grevlex')[0] for g in basis.exprs]
    names = {v: k for k, v in sym.items()}

    def show(expr):
        return str(expr.subs({s: sympy.Symbol(_label(*names[s])) for s in expr.free_symbols}))

    text = '; '.join('%s = 0' % show(p) for p in polys)
    leading = ', '.join(show(sympy.Mul(*[g ** e for g, e in zip(gens, lm)])) for lm in lms)
    _warn_once("invariant_terms='refuse': the data satisfy {0} exactly; terms divisible by {1} are refused."
               .format(text, leading))
    return _Rule(variables, ndim, gens_keys, lms, text)


def _label(variable, orders) -> str:
    if not any(orders):
        return variable
    return 'd%s_%s' % (variable, ''.join('x%d' % a * o for a, o in enumerate(orders)))


def rule(pool) -> Optional[_Rule]:
    """``pool``'s rule (built on first use), or None when the data have no exact invariant."""
    entry = _RULES.get(id(pool))
    if entry is not None and entry[0]() is pool:
        return entry[1]
    built = _build(pool)
    _RULES[id(pool)] = (weakref.ref(pool), built)
    return built


def refused(equation, terms) -> List[bool]:
    """Per term of ``terms`` (terms of ``equation``): whether ``invariant_terms='refuse'`` refuses it. All False
    under ``'keep'`` (nothing is computed) and on data without an exact invariant."""
    from epde.operators.common.plain_fit import plain_params
    if plain_params()['invariant_terms'] != 'refuse':
        return [False] * len(terms)
    found = rule(equation.pool)
    if found is None:
        return [False] * len(terms)
    return [found.reducible(term) for term in terms]
