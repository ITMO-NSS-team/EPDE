"""
Structural metrics for the thesis Section 4.5 EPDE comparison.

The metric pipeline is text-based: equations are read as strings (the
form produced by EPDE's :meth:`equations(only_str=True)`), parsed into a
canonical token representation that ignores coefficient values and term
ordering, then compared via Hamming distance / equality / modal-set
agreement across repetitions.
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
    left, right = eq_text.split('=', 1)
    target_term = _parse_term(right)
    rhs_terms = []
    for term_text in left.split('+'):
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
    lhs, rhs = eq_text.split('=', 1)
    target = _parse_term_with_coef(rhs)
    if target is None:
        return None
    target_key, target_coef = target
    coef_by_term: dict = {}
    for term_text in lhs.split('+'):
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


if __name__ == '__main__':
    # Quick self-check: round-trip the Lorenz triple and confirm Hamming == 0
    # against itself, then perturb one term and confirm Hamming == 2.
    lorenz_truth = [
        '10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} = du/dx0{power: 1.0}',
        '28.0 * u{power: 1.0} + -1.0 * u{power: 1.0} * w{power: 1.0} + -1.0 * v{power: 1.0} = dv/dx0{power: 1.0}',
        '1.0 * u{power: 1.0} * v{power: 1.0} + -2.6666666666666665 * w{power: 1.0} = dw/dx0{power: 1.0}',
    ]
    canon_truth = canonical_tokens(lorenz_truth)
    print('canon_truth size:', len(canon_truth))
    assert hamming(canon_truth, canon_truth) == 0
    assert structural_success(canon_truth, canon_truth)

    perturbed = list(lorenz_truth)
    # Drop the -10*u term from the first equation -> one rhs term missing.
    perturbed[0] = '10.0 * v{power: 1.0} = du/dx0{power: 1.0}'
    canon_perturbed = canonical_tokens(perturbed)
    h = hamming(canon_perturbed, canon_truth)
    print('hamming(1 term missing) =', h)
    assert h == 1, f"expected 1, got {h}"

    # Swap one rhs term for a different one: 1 removed + 1 added = 2.
    perturbed2 = list(lorenz_truth)
    perturbed2[0] = ('10.0 * v{power: 1.0} + -10.0 * u{power: 2.0} '
                     '= du/dx0{power: 1.0}')
    h2 = hamming(canonical_tokens(perturbed2), canon_truth)
    print('hamming(1 term swapped) =', h2)
    assert h2 == 2, f"expected 2, got {h2}"

    # Adding a pure-constant `+ 0.0` term must NOT change the canonical form.
    with_zero = list(lorenz_truth)
    with_zero[0] = '10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
    h_zero = hamming(canonical_tokens(with_zero), canon_truth)
    print('hamming(+0.0 added) =', h_zero)
    assert h_zero == 0, f"expected 0, got {h_zero}"

    # Drop a whole equation -> target + its 2 rhs terms = 3.
    perturbed3 = list(lorenz_truth[:2])
    h3 = hamming(canonical_tokens(perturbed3), canon_truth)
    print('hamming(1 equation missing) =', h3)
    assert h3 == 3, f"expected 3, got {h3}"

    # Target-flip equivalence: wave equation can be written with either
    # the time or the space second-derivative as the target term. Both
    # forms must canonicalise to the same term set.
    wave_truth = ['1.0 * d^2u/dx1^2{power: 1.0} = d^2u/dx0^2{power: 1.0}']
    wave_flipped = ['1.0 * d^2u/dx0^2{power: 1.0} = d^2u/dx1^2{power: 1.0}']
    h_wave = hamming(canonical_tokens(wave_flipped), canonical_tokens(wave_truth))
    print('hamming(wave target flipped) =', h_wave)
    assert h_wave == 0, f"expected 0, got {h_wave}"
    assert structural_success(canonical_tokens(wave_flipped),
                              canonical_tokens(wave_truth))

    # KdV target-flip: original sees ``du/dt = -6 u du/dx - u_xxx``
    # discovered sees ``-0.169 u_xxx - 0.165 du/dt = u du/dx`` -- same
    # three terms, different target. Hamming should be 0 even though
    # the old metric scored this as 6.
    kdv_truth = ['-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + '
                 '-1.0 * d^3u/dx1^3{power: 1.0} = du/dx0{power: 1.0}']
    kdv_flipped = ['-0.169 * d^3u/dx1^3{power: 1.0} + '
                   '-0.165 * du/dx0{power: 1.0} = '
                   'du/dx1{power: 1.0} * u{power: 1.0}']
    h_kdv = hamming(canonical_tokens(kdv_flipped), canonical_tokens(kdv_truth))
    print('hamming(kdv target flipped) =', h_kdv)
    assert h_kdv == 0, f"expected 0, got {h_kdv}"

    # x_N grid-coordinate collapse: x_0{dim:1,power:1} and
    # x_1{dim:1,power:1} are the same coordinate token, only the prefix
    # differs (an EPDE library labelling artifact). They must hash to
    # the same canonical factor.
    coord_a = ['1.0 * du/dx1{power: 1.0} * x_0{power: 1.0, dim: 1.0} '
               '= u{power: 1.0}']
    coord_b = ['1.0 * du/dx1{power: 1.0} * x_1{power: 1.0, dim: 1.0} '
               '= u{power: 1.0}']
    h_coord = hamming(canonical_tokens(coord_a), canonical_tokens(coord_b))
    print('hamming(x_0 vs x_1 same (dim,power)) =', h_coord)
    assert h_coord == 0, f"expected 0, got {h_coord}"

    # Multiplicity: two equations with identical canonical form must
    # NOT be deduplicated. Previously ``frozenset(out)`` collapsed
    # duplicates and Hamming under-counted -- e.g. a 3-eq system where
    # 2 equations had the same term set would canonicalize to 2 unique
    # frozensets, and bipartite pairing against a 3-eq truth would pad
    # discovered with empty and double-pay only one of the duplicates.
    same_eq = ('10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} '
               '= du/dx0{power: 1.0}')
    different_eq = ('1.0 * u{power: 1.0} * v{power: 1.0} '
                    '+ -2.667 * w{power: 1.0} = dw/dx0{power: 1.0}')
    truth_dup = canonical_tokens([same_eq, same_eq, different_eq])
    print('|truth_dup| =', len(truth_dup), '(expected 3, not 2)')
    assert len(truth_dup) == 3, (
        f"multiplicity dropped: got {len(truth_dup)} equations, expected 3"
    )
    # Discovered has only one copy; hamming should be len(same_eq's terms).
    disc_one = canonical_tokens([same_eq, different_eq])
    h_dup = hamming(disc_one, truth_dup)
    expected = len(canonical_tokens([same_eq])[0])  # one equation, term count
    print('hamming(2-eq vs 3-eq-with-1-dup) =', h_dup,
          '(expected', expected, '— missing duplicate of', same_eq[:30], ')')
    assert h_dup == expected, f"expected {expected}, got {h_dup}"

    # Coefficient-error metric: identical equations -> 0.
    truth_eq = ('10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} '
                '= du/dx0{power: 1.0}')
    ce_zero = _equation_relative_coef_error(truth_eq, truth_eq)
    print('coef_err(identical) =', ce_zero)
    assert ce_zero == 0.0, f"expected 0, got {ce_zero}"

    # 10% perturbation on one term -> 10% / 2 matched non-anchor terms = 5%.
    perturbed_eq = ('11.0 * v{power: 1.0} + -10.0 * u{power: 1.0} '
                    '= du/dx0{power: 1.0}')
    ce_perturb = _equation_relative_coef_error(perturbed_eq, truth_eq)
    print('coef_err(+10% on v term) =', ce_perturb)
    assert abs(ce_perturb - 0.05) < 1e-9, f"expected 0.05, got {ce_perturb}"

    # Target-flip robustness: wave-style flip with scale factor.
    wave_truth_eq = '0.04 * d^2u/dx1^2{power: 1.0} = d^2u/dx0^2{power: 1.0}'
    wave_flipped_eq = '25.0 * d^2u/dx0^2{power: 1.0} = d^2u/dx1^2{power: 1.0}'
    ce_wave = _equation_relative_coef_error(wave_flipped_eq, wave_truth_eq)
    print('coef_err(wave 25 vs 0.04 flipped) =', ce_wave)
    assert abs(ce_wave) < 1e-9, f"expected 0, got {ce_wave}"

    # System-level bipartite pairing for LV: swap the two equations'
    # order on the discovered side; result must be unchanged.
    lv_truth = [
        ('20.0 * u{power: 1.0} + -20.0 * u{power: 1.0} * v{power: 1.0} '
         '= du/dx0{power: 1.0}'),
        ('20.0 * u{power: 1.0} * v{power: 1.0} + -20.0 * v{power: 1.0} '
         '= dv/dx0{power: 1.0}'),
    ]
    lv_disc_swapped = list(reversed(lv_truth))
    ce_lv = _system_coef_error(lv_disc_swapped, lv_truth)
    print('coef_err(LV swapped order) =', ce_lv)
    assert abs(ce_lv) < 1e-9, f"expected 0, got {ce_lv}"

    # coefficient_error_best: picks the matching alternative.
    burgers_truth_alts = [
        ['-1.0 * u{power: 1.0} * du/dx1{power: 1.0} = du/dx0{power: 1.0}'],
        ['1.0 * u{power: 1.0} = x{power: 1.0, dim: 1.0} * du/dx1{power: 1.0}'],
    ]
    # Discovered the similarity-solution identity with 5% coef drift.
    disc_burgers = ['1.05 * u{power: 1.0} = x{power: 1.0, dim: 1.0} * du/dx1{power: 1.0}']
    ce_best = coefficient_error_best(disc_burgers, burgers_truth_alts)
    print('coef_err_best(burgers alt) =', ce_best)
    assert abs(ce_best - 0.05) < 1e-9, f"expected 0.05, got {ce_best}"

    print('thesis_metrics self-check OK')
