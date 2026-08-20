"""Scoring of discovered equations against the analytic ground truth.

An EPDE equation prints as ``c1 * f{..} * g{..} + c2 * h{..} + c0 = target{..}``.
To compare two such strings we move everything onto one side, drop numerically
zero terms, and canonicalise each remaining term into a hashable signature.
Two equations are then structurally equal when their signature sets coincide,
and numerically equal when, after scaling both coefficient vectors to unit
norm, every coefficient agrees to within a tolerance.
"""

import re
from typing import Dict, Tuple

import numpy as np

_FACTOR_RE = re.compile(r'([^{}*]+)\{([^{}]*)\}')
#: Factor parameters that take part in the identity of a term.  ``freq`` and
#: other continuous parameters are excluded: two trigonometric tokens with
#: frequencies 1.9999 and 2.0001 denote the same term.
_ID_PARAMS = ('power', 'dim')


def _canonical_factor(text: str) -> Tuple:
    match = _FACTOR_RE.fullmatch(text.strip())
    if match is None:
        return (text.strip(), )
    name = match.group(1).strip()
    params = {}
    for chunk in match.group(2).split(','):
        if ':' not in chunk:
            continue
        key, value = chunk.split(':', 1)
        key = key.strip()
        if key in _ID_PARAMS:
            try:
                params[key] = round(float(value), 6)
            except ValueError:
                params[key] = value.strip()
    return (name, tuple(sorted(params.items())))


def _split_term(term: str):
    """``'0.04 * a{..} * b{..}'`` -> ``(0.04, frozenset({canonical factors}))``."""
    parts = [p for p in term.split('*') if p.strip()]
    coefficient = 1.0
    factors = []
    for idx, part in enumerate(parts):
        part = part.strip()
        if idx == 0 and '{' not in part:
            try:
                coefficient = float(part)
                continue
            except ValueError:
                pass
        factors.append(_canonical_factor(part))
    return coefficient, tuple(sorted(factors))


def equation_signature(text: str, rel_tol: float = 1e-8) -> Dict[Tuple, float]:
    """Canonicalise ``lhs = rhs`` into ``{term signature: coefficient}``.

    The target term moves to the left with a flipped sign, so an equation and
    the same equation with a different term chosen as the right part produce
    the same signature up to an overall factor.
    """
    text = text.split('\n')[0]
    if '=' not in text:
        raise ValueError(f'Not an equation: {text!r}')
    lhs, rhs = text.split('=', 1)
    terms: Dict[Tuple, float] = {}
    for chunk in lhs.split('+'):
        chunk = chunk.strip()
        if not chunk:
            continue
        coefficient, factors = _split_term(chunk)
        if not factors:                       # bare intercept
            factors = (('__intercept__',),)
        terms[factors] = terms.get(factors, 0.0) + coefficient
    _, target = _split_term(rhs.strip())
    terms[target] = terms.get(target, 0.0) - 1.0

    scale = max(abs(v) for v in terms.values()) or 1.0
    return {k: v for k, v in terms.items() if abs(v) > rel_tol * scale}


def strip_common_factor(signature: Dict[Tuple, float]) -> Dict[Tuple, float]:
    """Divide the equation through by any factor common to all of its terms.

    Both discovery engines regularly return the true law multiplied by a
    redundant token, e.g. ``u^3 * (u_tt - 0.04 u_xx) = 0``. That is the same
    PDE in a degenerate parameterisation, and a term-set comparison would call
    it a miss. Cancelling the common factor lets the caller ask the weaker,
    physically meaningful question.
    """
    term_sets = [set(term) for term in signature if term != (('__intercept__',),)]
    if not term_sets:
        return signature
    common = set.intersection(*term_sets)
    if not common:
        return signature
    reduced = {}
    for term, coefficient in signature.items():
        if term == (('__intercept__',),):
            reduced[term] = coefficient
            continue
        rest = tuple(sorted(set(term) - common))
        if not rest:                      # the whole term was the common factor
            rest = (('__intercept__',),)
        reduced[rest] = reduced.get(rest, 0.0) + coefficient
    return reduced


def compare(candidate: str, reference: str, coef_tol: float = 0.1,
            up_to_factor: bool = False):
    """Compare a discovered equation with the reference.

    Returns ``(structure_match, coefficient_error)``.  The coefficient error is
    the max absolute difference between the two unit-norm coefficient vectors
    (sign-aligned), and is ``inf`` when the structures differ.  With
    ``up_to_factor`` both sides are first divided through by any factor common
    to all their terms, so the true law multiplied by a redundant token counts
    as a match.
    """
    cand = equation_signature(candidate)
    ref = equation_signature(reference)
    if up_to_factor:
        cand = strip_common_factor(cand)
        ref = strip_common_factor(ref)
    if set(cand) != set(ref):
        return False, float('inf')
    keys = sorted(ref, key=repr)
    a = np.array([cand[k] for k in keys], dtype=float)
    b = np.array([ref[k] for k in keys], dtype=float)
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    if float(a @ b) < 0:
        a = -a
    error = float(np.max(np.abs(a - b)))
    return True, error


def system_text_forms(system) -> list:
    """Per-equation text forms of an EPDE ``SoEq``."""
    return [equation.text_form for equation in system.vals]


def score_forms(forms: list, ground_truth: list, coef_tol: float = 0.1):
    """Score a list of per-equation text forms against the reference system."""
    if len(forms) != len(ground_truth):
        return dict(structure_match=False, coef_error=float('inf'),
                    match_up_to_factor=False)
    matches, errors, loose = [], [], []
    for form, reference in zip(forms, ground_truth):
        try:
            ok, err = compare(form, reference, coef_tol)
        except Exception:
            ok, err = False, float('inf')
        try:
            ok_loose, _ = compare(form, reference, coef_tol, up_to_factor=True)
        except Exception:
            ok_loose = False
        matches.append(ok)
        errors.append(err)
        loose.append(ok_loose or ok)
    return dict(structure_match=all(matches),
                coef_error=max(errors) if all(matches) else float('inf'),
                match_up_to_factor=all(loose),
                per_equation=list(zip(matches, errors)))


def score_system(system, ground_truth: list, coef_tol: float = 0.1):
    """Best-effort match of a discovered system against the reference system.

    Equations are matched positionally (EPDE orders them by
    ``vars_to_describe``, the same order the scenarios list the reference in).
    """
    return score_forms(system_text_forms(system), ground_truth, coef_tol)


def equation_lines(text_form: str) -> list:
    """Per-equation lines of a system's ``text_form``.

    Single-equation systems print as ``<equation>\\n<metaparameters>``; systems
    print each equation on its own line prefixed by ``/``, ``|`` or ``\\``.
    Metaparameter lines start with ``{`` and carry no ``=``.
    """
    lines = []
    for raw in text_form.split('\n'):
        line = raw.strip()
        if not line or line.startswith('{') or '=' not in line:
            continue
        lines.append(line.lstrip('/|\\ ').strip())
    return lines


def snapshot_hits(snapshot, ground_truth: list) -> bool:
    """Is the ground truth present in this ``{'text_form', ...}`` snapshot?"""
    for entry in snapshot:
        forms = equation_lines(entry['text_form'])
        try:
            if score_forms(forms, ground_truth)['structure_match']:
                return True
        except Exception:
            continue
    return False


def best_match(systems, ground_truth: list, coef_tol: float = 0.1):
    """Scan a population/front for the closest system to the ground truth."""
    best = dict(structure_match=False, coef_error=float('inf'),
                match_up_to_factor=False, index=None)
    for idx, system in enumerate(systems):
        score = score_system(system, ground_truth, coef_tol)
        better = (score['structure_match'] and not best['structure_match']) or \
                 (score['structure_match'] == best['structure_match'] and
                  score['coef_error'] < best['coef_error'])
        loose = best['match_up_to_factor'] or score['match_up_to_factor']
        if better:
            best = dict(score, index=idx)
        best['match_up_to_factor'] = loose
    return best
