"""``system_rps_params['product_rule_decline']`` of plain mode: decline a fitted
version whose target is one member of a product-rule pair.

On weak rows every column is a windowed integral, and for a window ``psi``
integration by parts gives ``int psi (f' g + f g') = -int psi' f g``. The
window's derivative is one order smaller than the window, so the columns of
``f' g`` and ``f g'`` nearly cancel: ``weak(f' g) + weak(f g') ~ 0`` is a
relation of the data that is not a law (on KdV,
``u_xx^2 = -0.99999 u_x u_xxx + 0.0316 u_tt``). A version solved for one
member of such a pair, with the coefficients in the product-rule pattern, is
declined like a reducible support: it is not built, takes no part in the
S / S+{1} comparison, and an equation left without any version is re-rolled.

GROUPS. Terms are read as monomials of atoms, one per unit power: a field
``u``, a derivative of a field along one axis (``d^2u/dx1^2``), a coordinate
(the grid tokens ``t``, ``x``, ``y``, ``z`` with their axis) and, in the
``'trig'`` mode, ``sin`` / ``cos`` of one axis. ``D_a`` raises a field or an
``a``-derivative by one order along axis ``a`` (another axis would make a
mixed derivative, which no token spells: no group), takes ``x_a`` to 1 and
another coordinate to 0, and turns ``sin(w x_a)`` into ``w cos(w x_a)`` and
``cos(w x_a)`` into ``-w sin(w x_a)``. For a monomial ``F`` the group
``G(F, a)`` holds the products ``mult(A) * (F / A) * D_a A`` over the atoms
``A`` of ``F``, equal ones collected with their multiplicities ``m_g``; it
counts with at least two members and without the bare constant. Any other
factor (a non-integer power, another token) is opaque: a term holding one
belongs to no group. A group lies in a version when all its members are
terms of the version.

THE RULE. The version is declined when its target is a member of a group of
its term set whose coefficients, the target's moved to the same side
(``{target: 1, k: -c_k}``), divided by the multiplicities, have one strict
sign over the group. A zero or missing coefficient is not the pattern. The
sign is invariant under the gauge of the relation and under positive column
scalings. The rule reads the version's structure and fitted coefficients
only, draws no random number, and keeps every other version unchanged.

Modes: ``'off'``; ``'plain'`` (fields, derivatives and coordinates); ``'trig'``
(also ``sin`` / ``cos``). The rule was measured on weak rows only, so any mode
but ``'off'`` needs ``domain.evaluation_form='weak'``.

Terms are compared by a text key, ``*``-joined factor keys
``label[^power][~freq][@axis]``, the frequency rounded to four digits (only
its sign enters the rule).
"""
import math
import re
from collections import OrderedDict

import numpy as np

#: The modes of ``system_rps_params['product_rule_decline']`` once resolved.
DECLINE_MODES = ('off', 'plain', 'trig')

ONE = '1'
_DHEAD = re.compile(r'^d(\^(\d+))?([a-z])/dx(\d+)(.*)$')
COORD_LABELS = ('t', 'x', 'y', 'z')
_COORD_SUFFIX = re.compile(r'^@(\d+)$')
TRIG_LABELS = ('sin', 'cos')
_TRIG_SUFFIX = re.compile(r'^~([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)@(\d+)$')
#: Bound of the group memo (memory only: a group set is a pure function of the term set).
_GROUPS_MAX = 500_000
_GROUPS = OrderedDict()


# ------------------------------------------------------------------ keys
def factor_key(factor) -> str:
    params = {d['name']: float(factor.params[i]) for i, d in factor.params_description.items()}
    key = factor.label
    power = params.get('power', 1.0)
    if power != 1.0:
        key += '^%g' % power
    if 'freq' in params:
        key += '~1' if abs(params['freq'] - 1.0) < 0.01 else '~%.4g' % params['freq']
    if 'dim' in params:
        key += '@%d' % int(round(params['dim']))
    return key


def term_key(term) -> str:
    return '*'.join(sorted(factor_key(f) for f in term.structure))


def parse_factor(piece):
    """A factor key -> (base, power); base = (label, suffix), the suffix ``~freq@dim`` kept as text."""
    piece = piece.strip()
    suffix = ''
    if '@' in piece:
        piece, d = piece.rsplit('@', 1)
        suffix = '@' + d
    if '~' in piece:
        piece, f = piece.split('~', 1)
        suffix = '~' + f + suffix
    m = _DHEAD.match(piece)
    if m:
        order = int(m.group(2) or 1)
        rest = m.group(5)
        label = piece[:len(piece) - len(rest)]
        if m.group(1):
            tag = '^%d' % order
            if not rest.startswith(tag):
                raise ValueError('derivative label %r' % piece)
            label += tag
            rest = rest[len(tag):]
        power = 1.0
        if rest:
            if not rest.startswith('^'):
                raise ValueError('derivative label %r' % piece)
            power = float(rest[1:])
        return (label, suffix), power
    if '^' in piece:
        base, p = piece.rsplit('^', 1)
        return (base, suffix), float(p)
    return (piece, suffix), 1.0


def mono(key) -> dict:
    """A term key -> monomial {base: power}; '1' -> {}."""
    if key == ONE:
        return {}
    out = {}
    for piece in key.split('*'):
        b, p = parse_factor(piece)
        out[b] = out.get(b, 0.0) + p
    return out


def key_of(m) -> str:
    """A monomial -> its term key; {} -> '1'."""
    if not m:
        return ONE
    return '*'.join(sorted(label + ('' if p == 1 else '^%g' % p) + suffix for (label, suffix), p in m.items()))


def _deriv_info(label):
    m = _DHEAD.match(label)
    if not m:
        return None
    return m.group(3), int(m.group(4)), int(m.group(2) or 1)          # var, axis, order


def _dlabel(var, axis, order) -> str:
    if order == 0:
        return var
    if order == 1:
        return 'd%s/dx%d' % (var, axis)
    return 'd^%d%s/dx%d^%d' % (order, var, axis, order)


def _is_field(base) -> bool:
    label, suffix = base
    return suffix == '' and re.fullmatch(r'[a-z]', label) is not None


# ------------------------------------------------------------------ atoms
def atom_kind(base, trig=False):
    """('field', var) | ('deriv', var, axis, order) | ('coord', axis) | ('opaque',); with ``trig`` also
    ('trig', label, w, axis) for ``sin~<w>@<axis>`` / ``cos~<w>@<axis>``."""
    label, suffix = base
    if trig and label in TRIG_LABELS:
        m = _TRIG_SUFFIX.match(suffix)
        if m:
            try:
                w = float(m.group(1))
            except ValueError:
                return ('opaque',)
            return ('trig', label, w, int(m.group(2)))
        return ('opaque',)
    if _is_field(base):
        return ('field', label)
    if suffix == '':
        di = _deriv_info(label)
        if di is not None:
            var, axis, order = di
            if _dlabel(var, axis, order) == label:          # the canonical spelling only
                return ('deriv', var, axis, order)
        return ('opaque',)
    m = _COORD_SUFFIX.match(suffix)
    if m and label in COORD_LABELS:
        return ('coord', int(m.group(1)))
    return ('opaque',)


def _int_power(p):
    p = float(p)
    return int(p) if p.is_integer() and p >= 1 else None


def _mono_or_none(key, trig=False):
    """A key -> monomial {base: int power} of readable atoms only, else None (opaque, '1', unparsable)."""
    if key == ONE:
        return None
    try:
        m = mono(key)
    except ValueError:
        return None
    out = {}
    for base, p in m.items():
        q = _int_power(p)
        if q is None or atom_kind(base, trig)[0] == 'opaque':
            return None
        out[base] = q
    return out or None


def _d_atom(base, a, trig=False):
    """D_a of one atom: (new atom, coefficient), 'unit' (D_a x_a = 1), 'zero' (D_a x_b = 0) or None (a mixed
    derivative or an opaque atom)."""
    kind = atom_kind(base, trig)
    if kind[0] == 'field':
        return (_dlabel(kind[1], a, 1), ''), 1
    if kind[0] == 'deriv':
        _, var, axis, order = kind
        return ((_dlabel(var, a, order + 1), ''), 1) if axis == a else None
    if kind[0] == 'coord':
        return 'unit' if kind[1] == a else 'zero'
    if kind[0] == 'trig':
        _, label, w, axis = kind
        if axis != a:
            return 'zero'
        return (('cos' if label == 'sin' else 'sin', base[1]), w if label == 'sin' else -w)
    return None


def _lower(base, a, trig=False):
    """The atom whose D_a is ``base`` (an a-derivative; with ``trig`` also a sin / cos of axis a), else None."""
    kind = atom_kind(base, trig)
    if kind[0] == 'trig':
        return ('cos' if kind[1] == 'sin' else 'sin', base[1]) if kind[3] == a else None
    if kind[0] != 'deriv' or kind[2] != a:
        return None
    _, var, axis, order = kind
    return (_dlabel(var, a, order - 1), '')


def _add_unit(m, base, sign=1) -> dict:
    out = dict(m)
    out[base] = out.get(base, 0) + sign
    if out[base] == 0:
        del out[base]
    return out


def _key(m) -> str:
    return key_of({b: float(p) for b, p in m.items()})


# ------------------------------------------------------------------ groups
def group_of(F, a, trig=False):
    """G(F, a) for a monomial F: {member key: multiplicity}, or None (fewer than two members, a member '1', a
    mixed-derivative member, an opaque atom)."""
    members = OrderedDict()
    for base, p in sorted(F.items()):
        d = _d_atom(base, a, trig)
        if d is None:
            return None
        if d == 'zero':
            continue
        rest = _add_unit(F, base, -1)
        if d == 'unit':
            mem, coef = rest, 1
        else:
            mem, coef = _add_unit(rest, d[0], +1), d[1]
        k = _key(mem)
        if k == ONE:
            return None
        members[k] = members.get(k, 0) + p * coef
    if trig:
        members = OrderedDict((k, m) for k, m in members.items() if m != 0)     # cancelled members are no members
    if len(members) < 2:
        return None
    return dict(members)


def groups_in(keys, trig=False) -> tuple:
    """Every group G(F, a) contained in the term set ``keys`` ('1' ignored), as a sorted tuple of
    (F, axis, ((member key, multiplicity), ...)). The candidate F's are the terms with one a-derivative (or,
    with ``trig``, one sin / cos of axis a) lowered by one order."""
    tr = bool(trig)
    S = frozenset(k for k in keys if k != ONE)
    ck = (S, tr)
    hit = _GROUPS.get(ck)
    if hit is not None:
        return hit
    seen, out = set(), []
    for T in sorted(S):
        m = _mono_or_none(T, tr)
        if m is None:
            continue
        for base in sorted(m):
            kind = atom_kind(base, tr)
            if kind[0] == 'deriv':
                a = kind[2]
            elif kind[0] == 'trig':
                a = kind[3]
            else:
                continue
            low = _lower(base, a, tr)
            F = _add_unit(_add_unit(m, base, -1), low, +1)
            fk = _key(F)
            if (fk, a) in seen:
                continue
            seen.add((fk, a))
            g = group_of(F, a, tr)
            if g is None or not set(g) <= S:
                continue
            out.append((fk, a, tuple(sorted(g.items()))))
    res = tuple(sorted(out))
    if len(_GROUPS) >= _GROUPS_MAX:
        _GROUPS.popitem(last=False)
    _GROUPS[ck] = res
    return res


def target_groups(keys, target, trig=False) -> tuple:
    """The groups of ``keys`` holding ``target`` as a member."""
    return tuple(g for g in groups_in(keys, trig) if target in (k for k, _ in g[2]))


def relation_from_fit(target, coefs) -> dict:
    """target = sum c_k key_k (+ c_0 '1') -> the relation {target: 1, key_k: -c_k}."""
    rel = {k: -float(c) for k, c in coefs.items() if k != target}
    rel[target] = 1.0
    return rel


def sign_pattern(group, relation) -> bool:
    """c_g / m_g of one strict sign over the group's members (missing / zero / non-finite -> False)."""
    vals = []
    for k, m in group[2]:
        c = relation.get(k)
        if c is None:
            return False
        c = float(c)
        if not math.isfinite(c) or c == 0.0:
            return False
        vals.append(c / m)
    return all(v > 0 for v in vals) or all(v < 0 for v in vals)


# ------------------------------------------------------------------ one fitted version
def version_keys(variant):
    """(target key, term keys with '1' when the constant is fitted, {key: fitted coefficient}) of a fitted,
    pruned version, or None when two of its terms share a key (the rule cannot read such a set)."""
    tgt = variant.target_idx
    w = np.asarray(variant.weights_internal, dtype=float)
    tkey = term_key(variant.structure[tgt])
    feats, coefs = set(), {}
    for i, term in enumerate(variant.structure):
        if i == tgt:
            continue
        k = term_key(term)
        if k in feats or k == tkey:
            return None
        feats.add(k)
        coefs[k] = float(w[variant.weight_index(i, tgt)])
    const = bool(w[-1] != 0)
    if const:
        coefs[ONE] = float(w[-1])
    full = sorted(set([tkey]) | feats | ({ONE} if const else set()))
    return tkey, full, coefs


def declines(variant, mode) -> bool:
    """True when ``mode`` ('plain' / 'trig') declines the fitted ``variant`` (see the module docstring)."""
    if mode == 'off':
        return False
    if mode not in DECLINE_MODES:
        raise ValueError("product_rule_decline must be one of %s, instead got %r" % (list(DECLINE_MODES), mode))
    read = version_keys(variant)
    if read is None:
        return False
    target, full, coefs = read
    rel = relation_from_fit(target, coefs)
    return any(sign_pattern(g, rel) for g in target_groups(full, target, mode == 'trig'))
