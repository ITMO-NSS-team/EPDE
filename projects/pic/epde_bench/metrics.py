"""Structural metrics: does a discovered system have the right terms?

Adapted from projects/thesis/thesis_metrics.py (the group's former
benchmark), retaining its term-set comparison and adding numerical
coefficient errors and a truth-free compromise selection.

Equations are compared as SETS OF TERMS: coefficients, term order and the
choice of target side are ignored, factor parameters (power, freq, dim) are
compared rounded to 3 digits. hamming counts terms that differ;
structural_success_any is Hamming 0 against any accepted form.
"""


from __future__ import annotations

import math
import re
from collections import Counter
from itertools import permutations
from typing import Iterable, List, Sequence

# Factor pattern: ``name{key1: val1, key2: val2, ...}`` where ``name`` can
# contain letters, digits, and the symbol characters EPDE uses for
# derivative tokens (``d``, ``u``, ``/``, ``^``, digits) and trig product
# tokens (e.g. ``cos(t)sin(x)``).
_FACTOR_RE = re.compile(r'([A-Za-z0-9_\^/\(\)]+)\s*\{([^}]*)\}')
_PARAM_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^,]+)')
_PARAM_ROUND_DIGITS = 3


def _strip_system_prefix(text: str) -> str:
    """Remove EPDE's leading system brace decoration, keeping numeric signs."""
    return re.sub(r'^\s*[/|\\]+\s*', '', text)


def _split_sum(text: str):
    """Split EPDE's additive terms without splitting positive exponents."""
    return re.split(r'(?<![eE])\+', text)


def _round_param(value: str):
    value = value.strip()
    try:
        return round(float(value), _PARAM_ROUND_DIGITS)
    except ValueError:
        return value


_GRID_COORD_NAME_RE = re.compile(r'^x_\d+$')


def _parse_factor(text: str):
    """Return ``(name, frozenset_of_param_items)`` or None if no factor."""
    m = _FACTOR_RE.search(text)
    if m is None:
        return None
    name = m.group(1)
    params_str = m.group(2)
    params = {}
    for pm in _PARAM_RE.finditer(params_str):
        params[pm.group(1)] = _round_param(pm.group(2))
    # Grid coordinate tokens like ``x_0``, ``x_1``, ... are redundant
    # labels in EPDE's library: the actual coordinate is fully specified
    # by the ``dim`` (axis) and ``power`` parameters, and the ``x_N``
    # prefix is an artifact of how the token was registered. Collapse to
    # a single canonical name so ``x_0{dim:1,power:1}`` and
    # ``x_1{dim:1,power:1}`` compare as the same factor.
    if _GRID_COORD_NAME_RE.match(name):
        name = 'x'
    return (name, frozenset(params.items()))


def _parse_term(term_text: str):
    """Parse a single ``c * f1{...} * f2{...}`` term into a frozenset of factors.

    Pure-constant terms (e.g. ``0.0``) and terms whose leading coefficient
    is numerically zero are filtered out by returning None.
    """
    pieces = [p.strip() for p in term_text.split('*')]
    factors = []
    coef = 1.0
    coef_seen = False
    for piece in pieces:
        if not piece:
            continue
        factor = _parse_factor(piece)
        if factor is None:
            # piece is a bare numeric coefficient (or unparseable scalar).
            try:
                val = float(piece)
                coef *= val
                coef_seen = True
                continue
            except ValueError:
                # Unrecognised piece: skip rather than crash; the canonical
                # set will simply omit it (and Hamming will reflect that).
                continue
        factors.append(factor)

    if not factors:
        # Pure-constant or unparseable term -> drop.
        return None
    if coef_seen and abs(coef) < 1e-12:
        # Zero coefficient -> term doesn't actually appear in the equation.
        return None
    return frozenset(factors)


def _canonical_equation(eq_text: str):
    """Parse one equation ``rhs_sum = target`` into a canonical tuple.

    Returns ``(target_term, frozenset_of_rhs_terms)`` or None if no ``=``.
    """
    if '=' not in eq_text:
        return None
    left, right = _strip_system_prefix(eq_text).split('=', 1)
    target_term = _parse_term(right)
    rhs_terms = []
    for term_text in _split_sum(left):
        term = _parse_term(term_text)
        if term is not None:
            rhs_terms.append(term)
    return (target_term, frozenset(rhs_terms))


def canonical_tokens(eq_texts: Sequence[str]) -> tuple:
    """Convert a list of equation text strings into a canonical structure.

    Each equation contributes one element to the returned tuple: the
    **unordered set of all its terms** -- target (LHS) and RHS combined
    into one frozenset. This makes the canonical form
    *target-side-independent*, so e.g. the wave equation
    ``c^2 * d^2u/dx^2 = d^2u/dt^2`` matches its inverted form
    ``(1/c^2) * d^2u/dt^2 = d^2u/dx^2`` (both have the same set of two
    derivative terms).

    Returns a **sorted tuple of frozensets**, not a frozenset of
    frozensets: when a system contains multiple equations with
    identical canonical forms (e.g., two equations in a coupled
    system that both reduce to the same term-set under the
    target-side-independent rule), every one is preserved. The prior
    ``frozenset(out)`` silently dropped duplicates, so the bipartite
    permutation matcher in :func:`hamming` would compare a
    deduplicated discovered system against a deduplicated truth and
    miss the multiplicity-driven contribution to the cost.

    Sorted by ``(len, repr)`` so equal canonical systems have
    identical tuples (hashable, ==-comparable, Counter-friendly).

    The canonicalisation still ignores coefficient magnitudes, term
    ordering, and factor ordering within terms; it preserves factor
    names + parameters (powers, freqs, dims) rounded to
    :data:`_PARAM_ROUND_DIGITS` digits.
    """
    out = []
    for eq in eq_texts:
        if not eq.strip():
            continue
        canon = _canonical_equation(eq)
        if canon is None:
            continue
        target_term, rhs_terms = canon
        full_terms = set(rhs_terms)
        if target_term is not None:
            full_terms.add(target_term)
        if full_terms:
            out.append(frozenset(full_terms))
    return tuple(sorted(out, key=lambda eq: (len(eq), repr(sorted(eq, key=repr)))))


def _eq_pair_cost(a: frozenset, b: frozenset) -> int:
    """Symmetric-difference term count between two equation term-sets."""
    return len(a.symmetric_difference(b))


def hamming(discovered, truth) -> int:
    """Term-level structural distance between two canonical equation systems.

    Each equation is an unordered set of terms (see :func:`canonical_tokens`).
    The systems are tuples of equation-term-sets WITH multiplicity --
    duplicate canonical equations are preserved (not deduplicated). The
    bipartite pairing brute-forces over permutations of equation
    indices, and the cost of a matched pair is the cardinality of the
    symmetric difference of their term sets. An unmatched equation
    contributes ``len(eq)`` to the total.

    Examples (Lorenz first equation only):
        truth = ({du/dt, a, b, c},), discovered = ({du/dt, a, b},)
            -> hamming = 1   (one term missing)
        truth = ({du/dt, a, b},), discovered = ({dv/dt, a, b},)
            -> hamming = 2   (du/dt missing, dv/dt extra)
        wave truth = ({d2u/dt2, d2u/dx2},), target-flipped discovered
        with the same two terms -> hamming = 0.
        truth = (E, E) (same canonical equation twice), discovered = (E,)
            -> hamming = len(E)   (multiplicity matters)
    """
    disc_eqs = list(discovered)
    truth_eqs = list(truth)
    if not disc_eqs and not truth_eqs:
        return 0
    if not disc_eqs:
        return sum(len(eq) for eq in truth_eqs)
    if not truth_eqs:
        return sum(len(eq) for eq in disc_eqs)

    # Pad the shorter side with empty equation sets so we can iterate
    # full bijections; an empty paired against a real equation costs
    # ``len(real_eq)`` via symmetric_difference, matching the "unmatched"
    # contribution.
    n = max(len(disc_eqs), len(truth_eqs))
    empty = frozenset()
    disc_padded = disc_eqs + [empty] * (n - len(disc_eqs))
    truth_padded = truth_eqs + [empty] * (n - len(truth_eqs))

    best = None
    for perm in permutations(range(n)):
        cost = sum(_eq_pair_cost(disc_padded[i], truth_padded[perm[i]])
                   for i in range(n))
        if best is None or cost < best:
            best = cost
    return best


def hamming_best(discovered, truth_alternatives) -> int:
    """Minimum Hamming across alternative canonical truth systems.

    Some systems admit multiple algebraically-distinct but
    mathematically-equivalent structural forms (e.g. an inviscid
    Burgers solution that satisfies both the PDE ``du/dt + u du/dx = 0``
    AND the similarity-solution identity ``u = x du/dx`` for the family
    ``u(x,t) = x/(t+c)``). Per-system YAMLs declare a primary truth in
    ``truth_equations`` and optional alternatives in
    ``truth_alternatives``; this helper returns the lowest Hamming
    distance to any of them, so EPDE is credited for discovering any
    valid form.

    ``truth_alternatives`` must be a non-empty iterable of canonical
    truth tuples (each one a tuple of frozensets, as produced by
    :func:`canonical_tokens`).
    """
    alternatives = list(truth_alternatives)
    if not alternatives:
        raise ValueError('hamming_best requires at least one truth alternative')
    return min(hamming(discovered, alt) for alt in alternatives)


def structural_success_any(discovered, truth_alternatives) -> bool:
    """True iff ``discovered`` matches any alternative canonical truth.

    Companion to :func:`hamming_best`; equivalent to
    ``hamming_best(...) == 0``.
    """
    return hamming_best(discovered, truth_alternatives) == 0


def structural_success(discovered, truth) -> bool:
    """True iff ``discovered`` equals ``truth`` as a canonical system.

    Match is target-side-independent (see :func:`canonical_tokens`):
    e.g. the wave equation matches regardless of whether EPDE chose the
    time or the space second-derivative as the RHS target.
    """
    return hamming(discovered, truth) == 0


def consistency_rate(reps_canonical: Iterable[frozenset]) -> float:
    """Fraction of reps whose canonical system equals the modal canonical system."""
    reps = list(reps_canonical)
    if not reps:
        return 0.0
    counts = Counter(reps)
    modal_count = counts.most_common(1)[0][1]
    return modal_count / len(reps)


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Coefficient-error metric
#
# Companion to the structural Hamming metric: once a rep matches the truth
# structurally (canonical equality), how close are its numerical coefficients
# to the truth coefficients? Target-flip robust: both sides are written as
# ``sum(c_i * t_i) - target = 0`` and re-normalised so the truth's target
# term has coefficient 1, then per-term relative errors are averaged.
# ---------------------------------------------------------------------------


def _parse_term_with_coef(term_text: str):
    """Parse one ``c * f1{...} * f2{...}`` term into ``(term_canonical, coef)``.

    ``term_canonical`` is the same ``frozenset(factors)`` :func:`_parse_term`
    produces. Returns ``None`` for pure-constant terms (``0.0``,
    ``-0.5``) and zero-coef terms, mirroring the structural metric's
    drop rule so coefficient comparison stays aligned with structure.
    """
    pieces = [p.strip() for p in term_text.split('*')]
    factors = []
    coef = 1.0
    coef_seen = False
    for piece in pieces:
        if not piece:
            continue
        factor = _parse_factor(piece)
        if factor is None:
            try:
                coef *= float(piece)
                coef_seen = True
            except ValueError:
                continue
        else:
            factors.append(factor)
    if not factors:
        return None
    if coef_seen and abs(coef) < 1e-12:
        return None
    if not coef_seen:
        coef = 1.0
    return (frozenset(factors), coef)


def _equation_term_coefs(eq_text: str):
    """Parse ``sum_terms = target`` into ``(coef_by_term, target_key)``.

    Equation is rewritten as ``sum_terms - target = 0``; ``coef_by_term``
    holds the signed coefficient of every canonical term in that form
    (target term gets ``-target_coef`` so the dict is in ``Σ c_i t_i = 0``
    form). Same factor / param canonicalisation as :func:`_parse_term`.
    """
    if '=' not in eq_text:
        return None
    lhs, rhs = _strip_system_prefix(eq_text).split('=', 1)
    target = _parse_term_with_coef(rhs)
    if target is None:
        return None
    target_key, target_coef = target
    coef_by_term: dict = {}
    for term_text in _split_sum(lhs):
        parsed = _parse_term_with_coef(term_text)
        if parsed is None:
            continue
        key, coef = parsed
        coef_by_term[key] = coef_by_term.get(key, 0.0) + coef
    coef_by_term[target_key] = coef_by_term.get(target_key, 0.0) - target_coef
    return coef_by_term, target_key


def _equation_relative_coef_error(disc_eq_text: str, truth_eq_text: str) -> float:
    """Mean per-term relative coefficient error between two equations.

    Both equations are written in ``Σ c_i t_i = 0`` form, anchored at the
    truth's target term so both sides have anchor coef 1, then matched
    term-by-term on the canonical factor set. The relative error for a
    matched term ``t_i`` is ``|c_disc - c_truth| / |c_truth|``; missing
    terms (in discovered or truth) contribute 1.0 each. Returns
    ``float('nan')`` if either equation fails to parse or the truth's
    anchor term is absent from / has zero coefficient in the discovered
    equation (target-flip unresolvable).
    """
    disc = _equation_term_coefs(disc_eq_text)
    truth = _equation_term_coefs(truth_eq_text)
    if disc is None or truth is None:
        return float('nan')
    disc_coefs, _ = disc
    truth_coefs, anchor = truth
    truth_anchor_coef = truth_coefs.get(anchor, 0.0)
    disc_anchor_coef = disc_coefs.get(anchor, 0.0)
    if abs(truth_anchor_coef) < 1e-12 or abs(disc_anchor_coef) < 1e-12:
        return float('nan')
    truth_norm = {k: v / truth_anchor_coef for k, v in truth_coefs.items()}
    disc_norm = {k: v / disc_anchor_coef for k, v in disc_coefs.items()}
    keys = set(truth_norm) | set(disc_norm)
    errors: List[float] = []
    for k in keys:
        if k == anchor:
            continue  # both 1.0 by construction
        tc = truth_norm.get(k)
        dc = disc_norm.get(k)
        if tc is None:
            errors.append(1.0)  # extra term in discovered
            continue
        if dc is None:
            errors.append(1.0)  # missing term in discovered
            continue
        if abs(tc) < 1e-12:
            errors.append(0.0 if abs(dc) < 1e-12 else 1.0)
            continue
        errors.append(abs(dc - tc) / abs(tc))
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def _system_coef_error(discovered_eq_texts: Sequence[str],
                       truth_eq_texts: Sequence[str]) -> float:
    """Bipartite coef-error matching between two equation systems.

    Pads the shorter side with empty equations (each empty pair scores
    1.0) and brute-forces over permutations to minimise the average
    per-equation :func:`_equation_relative_coef_error`. Returns
    ``float('nan')`` if every permutation contains an unparseable pair.
    """
    disc = [s for s in discovered_eq_texts if isinstance(s, str) and s.strip()]
    truth = [s for s in truth_eq_texts if isinstance(s, str) and s.strip()]
    if not disc or not truth:
        return float('nan')
    n = max(len(disc), len(truth))
    pad_d = list(disc) + [''] * (n - len(disc))
    pad_t = list(truth) + [''] * (n - len(truth))
    best = float('nan')
    for perm in permutations(range(n)):
        total = 0.0
        valid = True
        for i in range(n):
            de, te = pad_d[i], pad_t[perm[i]]
            if not de or not te:
                total += 1.0  # unmatched eq counts as fully-wrong
                continue
            err = _equation_relative_coef_error(de, te)
            if err != err:  # nan
                valid = False
                break
            total += err
        if not valid:
            continue
        avg = total / n
        if best != best or avg < best:
            best = avg
    return best


def coefficient_error_best(discovered_eq_texts: Sequence[str],
                           truth_alternatives_text_lists) -> float:
    """Lowest mean coef error across all declared truth alternatives.

    ``truth_alternatives_text_lists`` is an iterable of equation-string
    lists -- the primary truth followed by each alternative. The minimum
    is taken across alternatives so a target-flipped / identity-based
    discovery is scored against the closest valid analytical form (same
    convention as :func:`hamming_best`).
    """
    best = float('nan')
    for truth_alt in truth_alternatives_text_lists:
        err = _system_coef_error(discovered_eq_texts, truth_alt)
        if err != err:
            continue
        if best != best or err < best:
            best = err
    return best


# ---------------------------------------------------------------------------
# Per-run scoring (new)
# ---------------------------------------------------------------------------


def select_compromise(objectives) -> int | None:
    """Pick the smallest sum of min-max normalised finite objective values.

    Invalid or missing vectors are excluded; indices refer to the original
    front. Empty vectors or inconsistent dimensions have no valid compromise,
    and an entirely invalid front returns None. Ties keep front order.
    """
    vectors = []
    for index, vector in enumerate(objectives):
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError, OverflowError):
            continue
        if values and all(math.isfinite(value) for value in values):
            vectors.append((index, values))
    if not vectors:
        return None
    width = len(vectors[0][1])
    if any(len(values) != width for _, values in vectors):
        return None
    scores = [0.0] * len(vectors)
    for k in range(width):
        column = [values[k] for _, values in vectors]
        lo, hi = min(column), max(column)
        # Scaling first avoids overflow for finite values near float limits.
        scale = max(abs(lo), abs(hi)) or 1.0
        lo, hi = lo / scale, hi / scale
        span = hi - lo
        for i, value in enumerate(column):
            scores[i] += 0.0 if span <= 0 else (value / scale - lo) / span
    best = min(range(len(scores)), key=lambda i: scores[i])
    return vectors[best][0]


def score_run(front: Sequence[Sequence[str]], objectives: Sequence, truth_systems) -> dict:
    """All structural metrics of one run.

    ``front``: the Pareto-0 solutions, each a list of equation strings (one
    per variable). ``objectives``: their objective vectors (same order).
    ``truth_systems``: the accepted systems (primary truth first), each a
    list of equation strings; empty when the truth is unknown.
    """
    canon = [canonical_tokens(sol) for sol in front]
    selected = select_compromise(objectives) if front else None
    if selected is not None and selected >= len(front):
        selected = None
    out = {'front_size': len(front), 'selected_index': selected}
    if not truth_systems:
        return out
    alts = [canonical_tokens(sys_) for sys_ in truth_systems]
    if not canon:
        out.update(success_front=False, success_selected=False, hamming_min=None,
                   hamming_selected=None, best_index=None, coef_error=None)
        return out
    hammings = [hamming_best(c, alts) for c in canon]
    best = min(range(len(hammings)), key=lambda i: hammings[i])
    coef = coefficient_error_best(front[best], truth_systems) if hammings[best] == 0 else None
    out.update(success_front=hammings[best] == 0,
               success_selected=selected is not None and hammings[selected] == 0,
               hamming_min=hammings[best],
               hamming_selected=hammings[selected] if selected is not None else None,
               best_index=best,
               coef_error=None if coef is None or coef != coef else coef)
    return out
