"""Slot-aware rows for the scratch arms (Nov 5 2026, USER "I implemented weak approach to epde, can we enhance it
with what we got so far?"; plan we-compared-different-instability-vast-unicorn.md, Stage 0.1).

EPDE's weak mode (domain.evaluation_form='weak') evaluates every equation slot on its own window lattice: a term's
column is ``epde.structure.weak_rows.term_columns(term, slot)``, the row weights are
``samples_manager.row_weights(slot, mode)`` (ones) and the lattice ``row_shapes(slot)``; ``gFunc('dm')``,
``inner_shapes`` and ``Term.evaluate()`` without a slot raise. This module is the one place the arms and the offline
scripts read rows from under weak mode. Under strong rows the arms keep their own, unchanged code (so strong runs stay
bit-identical); the strong branches here return exactly what that code reads (``gFunc``, ``inner_shapes``).

The weak constant column is exactly 1 (weak_form.py divides each window's weights by its mass), so the arms' ones
columns stay valid; ``one_check`` measures it per slot against N_acc*eps (N_acc = samples a window accumulates).
Nothing here patches EPDE or caches columns: EPDE's WeakRows caches them by (slot, factors_labels); ``stats`` reads
that cache's size."""
import hashlib
from typing import Dict, List, Optional

import numpy as np

ONE_CHECK: Dict[tuple, dict] = {}     # (manager id, key, slot) -> verdict, computed once


def _manager():
    import epde.globals as global_var
    return global_var.samples_manager


def weak_mode() -> bool:
    """Whether the current fit evaluates weak rows."""
    return bool(getattr(_manager(), 'weak_mode', False))


def form() -> str:
    return 'weak' if weak_mode() else 'strong'


def slot(equation) -> Optional[str]:
    """The equation's row slot: its main_var_to_explain under weak rows, None under strong rows."""
    from epde.structure.weak_rows import slot_of
    return slot_of(equation)


def keys() -> List:
    """The trajectory keys of the current rows."""
    manager = _manager()
    if weak_mode():
        return list(manager.weak.keys)
    return list(manager.inner_shapes.keys())


def weights(slot_name, key, mode: str = 'dm') -> np.ndarray:
    """Row weights of ``slot_name`` on trajectory ``key`` (strong: gFunc(mode)[key])."""
    return np.asarray(_manager().row_weights(slot_name, mode)[key], dtype=float).reshape(-1)


def shape(slot_name, key):
    """Row lattice of ``slot_name`` on trajectory ``key``: a tuple of ints under weak rows, the strong inner shape
    object otherwise."""
    value = _manager().row_shapes(slot_name)[key]
    return tuple(int(v) for v in np.ravel(value)) if weak_mode() else value


def column(term, slot_name, key) -> np.ndarray:
    """``term``'s weak column on ``slot_name``'s rows of trajectory ``key`` (read-only float vector). Weak rows only:
    a strong caller keeps its own column code."""
    if not weak_mode():
        raise RuntimeError('weakio.column is the weak-row path; strong callers keep their own column code')
    from epde.structure.weak_rows import term_columns
    col = np.asarray(term_columns(term, slot_name)[key], dtype=float).reshape(-1)
    col.flags.writeable = False
    return col


def slot_design(var, terms=()) -> dict:
    """Offline helper: the rows of slot ``var`` (single trajectory) and, for ``terms``, their weak columns."""
    ks = keys()
    if len(ks) != 1:
        raise NotImplementedError('weakio.slot_design handles one trajectory; got %s' % ks)
    key = ks[0]
    sw = weights(var, key)
    out = dict(key=key, n=sw.size, sw=sw, shape=shape(var, key), var=var)
    if weak_mode():
        out['plan'] = repr(_manager().weak.plans[(key, var)])
    if terms:
        out['X'] = np.column_stack([column(t, var, key) for t in terms])
    return out


def one_check(slot_name, key=None) -> dict:
    """The weak column of the constant 1 on ``slot_name``'s rows, against N_acc*eps (N_acc = samples one window
    accumulates). Computed once per (weak store, key, slot) -- a new fit builds a new WeakRows, so its windows get a
    fresh verdict; returns {'ok', 'max_dev', 'bound'}; never raises."""
    if not weak_mode():
        return dict(ok=True, max_dev=0.0, bound=0.0, form='strong')
    manager = _manager()
    key = keys()[0] if key is None else key
    ck = (id(manager.weak), key, slot_name)
    if ck in ONE_CHECK:
        return ONE_CHECK[ck]
    try:
        from epde.preprocessing import weak_form as WF
        weak = manager.weak
        plan = weak.plans[(key, slot_name)]
        fields = weak.fields[key]
        D = fields.D
        monos = WF.WeakTerm([], (0,) * D).monomials(psi_cap=plan.q)
        M, _ = WF.weak_matrix([monos], [()], fields, plan.cut, plan.ms, plan.ss, plan.q)
        n_acc = float(np.prod([2 * int(m) + 1 for m in plan.ms]))
        bound = n_acc * np.finfo(float).eps
        dev = float(np.max(np.abs(M[:, 0] - 1.0)))
        verdict = dict(ok=bool(dev <= bound), max_dev=dev, bound=bound, n_acc=n_acc)
    except Exception as exc:  # noqa: BLE001 -- a check, not a gate on the run
        verdict = dict(ok=False, max_dev=None, bound=None, error='%s: %s' % (type(exc).__name__, exc))
    ONE_CHECK[ck] = verdict
    return verdict


def noise_digest(fields: Dict[str, np.ndarray]) -> str:
    """blake2b over the sorted field names and their float64 bytes: equal across arms and forms per (system, seed)."""
    h = hashlib.blake2b(digest_size=12)
    for name in sorted(fields):
        h.update(str(name).encode())
        h.update(np.ascontiguousarray(np.asarray(fields[name], dtype=float)).tobytes())
    return h.hexdigest()


def stats() -> dict:
    """Size of EPDE's weak column cache (read-only)."""
    manager = _manager()
    weak = getattr(manager, 'weak', None)
    if weak is None:
        return dict(form='strong')
    cols = getattr(weak, '_columns', {})
    nbytes = sum(int(np.asarray(v).nbytes) for per_key in cols.values() for v in per_key.values())
    return dict(form='weak', columns=len(cols), mb=round(nbytes / 2 ** 20, 2))
