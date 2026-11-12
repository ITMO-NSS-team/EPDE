"""ARM PV -- per-variable NSGA-II selection + assembly crossover (EXPERIMENT, not in EPDE; patches EPDE at install()).

Every equation is ranked in its own (discrepancy, instability) plane only; nothing is summed or averaged across
equations, only plane ranks and the count of plane fronts held are compared. Same structure label = same equation:
one REPRESENTATIVE per (plane, label), the copy minimising (own plane level, sorted partner-plane levels), the
NEWEST copy (highest index) on a tie, so a fresh (law, law) assembled from tied parents represents both laws;
non-representatives rank after every representative of the plane; crowding per plane level over representatives,
identical points sharing one distance.

install(mode='pv') patches (idempotent through NSGAOptimizer._pv_installed):
  epde.optimizers.nsga.optimizer.NSGAOptimizer._truncate          pv_truncate (two-phase round-robin over the planes)
  epde.optimizers.nsga.operators.NSGASelection.apply             pv_select (plane tournaments -> objective.pv_plan)
  epde.operators.multiobjective.variation.ParetoLevelsCrossover.apply   pv_crossover_apply (assembly child + stock child)
  epde.operators.multiobjective.mutations.SystemMutation.apply   the assembled child skips its first mutation only
  epde.operators.multiobjective.moeadd_specific._FirstLevelCount PerPlaneFirstLevelCount (at most half on level 0 of
                                                                 EVERY plane; twins count, as in the stock rule)
  epde.operators.common.plain_fit.apply_plain,
  moeadd_specific.OffspringUpdater._swap_arrangement,
  epde.optimizers.nsga.operators.AppendUpdater.apply             counting wrappers only (they chain with R2's)
install(mode='ja') (arm_ja.py): ja_select (stock joint tournaments on (level, -plane_crowding), kA uniform) feeding
the same plan, assembly crossover, mutation skip and counters; truncation and the level-0 rule stay stock.
Systems with fewer than 2 equations take the original functions everywhere.
"""
import os
import weakref
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

#: Set False to skip the first-use config check (stub self-tests).
CHECK_CONFIG = True
COUNTS = Counter()
_STATE = {}
_ORIG = {}
REG = {}            # id(assembled child) -> its record, while its first pass may still be running
PLANNED = Counter()  # assembled children by planned-label outcome: main / sibling / not_placed / pending
SIB = {}            # id(sibling of a first fit) -> (weakref, the child's record)
SURVIVAL = {'truncations': 0, 'overflow': 0, 'front_count_changed': 0, 'H': [], 'F': [], 'all_front': [],
            'all_front_raw': [], 'n': []}
_REFUSE_ENV = ('TRACE_ARCHIVE_MATING', 'EPDE_NICHING', 'EPDE_MATING')


# ------------------------------------------------------------------ pure helpers
def _different(labels) -> np.ndarray:
    """ne[j, i] = labels[j] != labels[i]."""
    try:
        ids = {}
        codes = np.array([ids.setdefault(label, len(ids)) for label in labels], dtype=int)
        return codes[:, None] != codes[None, :]
    except TypeError:
        return np.array([[a != b for b in labels] for a in labels], dtype=bool).reshape(len(labels), len(labels))


def plane_levels_fast(points, labels) -> np.ndarray:
    """``moeadd_specific.plane_levels``, vectorised: D[j, i] = j dominates i with a different label."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    levels = np.full(n, -1, dtype=int)
    if n == 0:
        return levels
    if P.ndim == 1:
        P = P[:, None]
    D = (np.all(P[:, None, :] <= P[None, :, :], axis=2) & np.any(P[:, None, :] < P[None, :, :], axis=2)
         & _different(labels))
    remaining = np.ones(n, dtype=bool)
    level = 0
    while remaining.any():
        current = remaining & ~(D & remaining[:, None]).any(axis=0)
        if not current.any():           # unreachable: strict dominance has no cycle
            current = remaining.copy()
        levels[current] = level
        remaining &= ~current
        level += 1
    return levels


def _first_per_label(labels, order) -> list:
    """The first index of ``order`` for each distinct label."""
    try:
        best = {}
        for i in order:
            best.setdefault(labels[i], i)
        return list(best.values())
    except TypeError:
        seen, out = [], []
        for i in order:
            if not any(labels[i] == s for s in seen):
                seen.append(labels[i])
                out.append(i)
        return out


def plane_table(population):
    """Per-plane ranking of ``population`` (see the module docstring). Arrays are n_eq x N."""
    from epde.operators.multiobjective.moeadd_specific import _crowding_distances, equation_planes
    keys = list(population[0].vals.equation_keys)
    n_eq, N = len(keys), len(population)
    planes, labels = equation_planes(population)
    lev = np.array([plane_levels_fast(planes[e], labels[e]) for e in range(n_eq)], dtype=int).reshape(n_eq, N)
    partner = [[tuple(sorted(int(lev[f, i]) for f in range(n_eq) if f != e)) for i in range(N)]
               for e in range(n_eq)]
    rep = np.zeros((n_eq, N), dtype=bool)
    for e in range(n_eq):
        order = sorted(range(N), key=lambda i: (lev[e, i], partner[e][i], -i))
        rep[e, _first_per_label(labels[e], order)] = True
    adj = lev.copy()
    crowd = np.zeros((n_eq, N))
    for e in range(n_eq):
        if N:
            adj[e, ~rep[e]] = lev[e].max() + 1 + lev[e, ~rep[e]]
        for level in np.unique(lev[e, rep[e]]):
            idx = np.flatnonzero(rep[e] & (lev[e] == level))
            unique, inverse = np.unique(np.asarray(planes[e])[idx], axis=0, return_inverse=True)
            crowd[e, idx] = _crowding_distances(unique)[np.asarray(inverse).reshape(-1)]
    crowd = np.nan_to_num(crowd, nan=0.0, posinf=np.inf, neginf=-np.inf)
    front = rep & (lev == 0)
    # Every equation on its own plane's first level, whichever copy represents it there (USER Oct 27: "i dont
    # want to remove true equations" -- after one copy of each front equation, a system made of front equations
    # is kept before any deeper one, so it is not cut for losing a tie on copies).
    all_front = (lev == 0).all(axis=0) if N else np.zeros(0, dtype=bool)
    return SimpleNamespace(keys=keys, n_eq=n_eq, n=N, planes=planes, labels=labels, lev=lev, rep=rep, adj=adj,
                           crowd=crowd, partner=partner, front=front, front_count=front.sum(axis=0).astype(int),
                           all_front=all_front)


def _front_queues(T, by_count=True) -> list:
    def key(e, i):
        head = (-int(T.front_count[i]),) if by_count else ()
        return head + (-T.crowd[e, i], T.partner[e][i], i)
    return [sorted((int(i) for i in np.flatnonzero(T.front[e])), key=lambda i: key(e, i)) for e in range(T.n_eq)]


def _all_front_queues(T) -> list:
    """The systems whose every equation is on its plane's first level, per plane by (-crowd, index)."""
    members = [int(i) for i in np.flatnonzero(T.all_front)]
    return [sorted(members, key=lambda i: (-T.crowd[e, i], i)) for e in range(T.n_eq)]


def _deeper_queues(T) -> list:
    queues = []
    for e in range(T.n_eq):
        reps = sorted((int(i) for i in np.flatnonzero(T.rep[e] & ~T.front[e])),
                      key=lambda i: (T.lev[e, i], -T.crowd[e, i], T.partner[e][i], i))
        rest = sorted((int(i) for i in np.flatnonzero(~T.rep[e])),
                      key=lambda i: (T.lev[e, i], T.partner[e][i], i))
        queues.append(reps + rest)
    return queues


def _fill(phases, mu, order) -> list:
    """Round-robin over the planes in ``order``, phase by phase; a plane with nothing left skips its turn."""
    kept, seen = [], set()
    for queues in phases:
        ptr = [0] * len(queues)
        while len(kept) < mu:
            moved = False
            for e in order:
                if len(kept) >= mu:
                    break
                q, p = queues[e], ptr[e]
                while p < len(q) and q[p] in seen:
                    p += 1
                if p < len(q):
                    kept.append(q[p])
                    seen.add(q[p])
                    p += 1
                    moved = True
                ptr[e] = p
            if not moved:
                break
    return kept


def survivors(T, mu, start=0):
    """(kept indices in pick order, stats). Phase 0: plane-front representatives (one copy of every front
    equation), queues by (-front_count, -crowd, partner, index); then the other systems whose every equation is
    on its plane's first level (``T.all_front``); phase 1: deeper representatives by (level, -crowd, partner,
    index), then non-representatives. ``front_count_changed``: the kept set differs from phase-0 queues without
    the front_count key (computed only when the front representatives overflow mu)."""
    n_eq, N = T.n_eq, T.lev.shape[1]
    H = int(T.front.any(axis=0).sum())
    stats = dict(n=N, mu=mu, H=H, F=[int(x) for x in T.front.sum(axis=1)],
                 all_front=int((T.front_count == n_eq).sum()), all_front_raw=int(T.all_front.sum()),
                 overflow=bool(H > mu), front_count_changed=False)
    if N <= mu:
        return list(range(N)), stats
    order = [(start + r) % n_eq for r in range(n_eq)]
    all_front, deeper = _all_front_queues(T), _deeper_queues(T)
    kept = _fill([_front_queues(T, True), all_front, deeper], mu, order)
    if stats['overflow']:
        stats['front_count_changed'] = set(_fill([_front_queues(T, False), all_front, deeper], mu, order)) != set(kept)
    return kept, stats


def plane_layers(T, population) -> list:
    """``population`` partitioned by min over planes of the adjusted plane rank; empty layers dropped."""
    if not len(population):
        return []
    best = T.adj.min(axis=0)
    return [[population[i] for i in range(len(population)) if best[i] == r] for r in sorted(set(best.tolist()))]


def tournament(T, e, n) -> int:
    """Binary tournament in plane ``e`` on (adjusted level, -crowding); the first drawn wins a full tie."""
    first, second = int(np.random.randint(n)), int(np.random.randint(n))
    key = lambda i: (T.adj[e, i], -T.crowd[e, i])
    return second if key(second) < key(first) else first


def _eq_label(eq):
    label = getattr(eq, 'candidate_label', None)
    if label is None:
        label = getattr(eq, 'structure_label', None)
    return eq.terms_labels if label is None else label


def predicted_label(A, kA, B, kB, keys, from_b=frozenset()) -> tuple:
    """The equations_labels the assembled child carries as planned: B's labels at kB (and ``from_b``), A's else."""
    return tuple(_eq_label((B if (k == kB or k in from_b) else A).vals[k]) for k in keys)


def _replace(vals, key, value) -> None:
    if hasattr(vals, 'replace_gene'):
        vals.replace_gene(gene_key=key, value=value)
    else:
        vals[key] = value


def _ref(obj):
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


def _entry(obj):
    entry = REG.get(id(obj))
    if entry is not None and entry['ref']() is not obj:
        REG.pop(id(obj), None)
        return None
    return entry


def assemble(A, kA, B, kB, keys, from_b=None):
    """deepcopy(A) with gene kB a fresh copy of B's; for n > 2 each other gene from B by a fair coin (``from_b``
    when given). No fit, no target; flagged to skip its first mutation; registered for the fate counters."""
    rest = [k for k in keys if k not in (kA, kB)]
    if from_b is None:
        from_b = frozenset(k for k in rest if np.random.random() < 0.5)
    child = deepcopy(A)
    _replace(child.vals, kB, deepcopy(B.vals[kB]))
    for k in rest:
        if k in from_b:
            _replace(child.vals, k, deepcopy(B.vals[k]))
    if hasattr(child, 'reset_state'):
        child.reset_state(True)
    child.__dict__['_pv_skip_mutation'] = True
    for key in [key for key, entry in REG.items() if entry['ref']() is None]:
        del REG[key]
    record = dict(ref=_ref(child), predicted=predicted_label(A, kA, B, kB, keys, from_b), state='assembled',
                  fit_seen=False, swap_called=False, done=False, planned=None)
    REG[id(child)] = record
    PLANNED['pending'] += 1
    COUNTS['assembled'] += 1
    return child


def _bucket(record) -> str:
    return record['planned'] or ('not_placed' if record['done'] else 'pending')


def _settle(record, done=False, planned=None) -> None:
    """Mark ``record`` done and/or its planned label placed (first outcome kept), moving its PLANNED bucket."""
    before = _bucket(record)
    record['done'] = record['done'] or done
    if planned and record['planned'] is None:
        record['planned'] = planned
    after = _bucket(record)
    if after != before:
        PLANNED[before] -= 1
        PLANNED[after] += 1


class PerPlaneFirstLevelCount(object):
    """``moeadd_specific._FirstLevelCount`` per plane: each plane's level 0 is the members no member with a
    different structure label dominates in that plane (twins count and share a level). holds() iff
    2 * |level 0 of plane k| <= members for every plane k. One-equation systems use the stock count."""

    def __init__(self):
        self.members = []
        self._joint = None
        self._points, self._labels, self.dominators = None, None, None

    def place(self, system) -> None:
        if self._joint is None and not self.members and len(system.vals) < 2:
            self._joint = _stock_first_level_count()()
        if self._joint is not None:
            self._joint.place(system)
            self.members.append(system)
            return
        _check_config()
        from epde.operators.multiobjective.moeadd_specific import equation_planes
        planes, labels = equation_planes([system])
        if self._points is None:
            n_eq = len(planes)
            self._points, self._labels, self.dominators = ([[] for _ in range(n_eq)] for _ in range(3))
        for e in range(len(planes)):
            p, lab = np.asarray(planes[e][0], dtype=float), labels[e][0]
            count = 0
            if self._points[e]:
                Q = np.asarray(self._points[e])
                other = np.array([label != lab for label in self._labels[e]], dtype=bool)
                count = int((np.all(Q <= p, axis=1) & np.any(Q < p, axis=1) & other).sum())
                for k in np.flatnonzero(np.all(p <= Q, axis=1) & np.any(p < Q, axis=1) & other):
                    self.dominators[e][k] += 1
            self._points[e].append(p)
            self._labels[e].append(lab)
            self.dominators[e].append(count)
        self.members.append(system)
        _STATE['init'] = dict(members=len(self.members), plane_sizes=self.plane_sizes())

    def plane_sizes(self) -> list:
        if self._joint is not None:
            return [self._joint.size]
        return [sum(1 for d in dom if d == 0) for dom in (self.dominators or [])]

    @property
    def size(self) -> int:
        return max(self.plane_sizes()) if self.members else 0

    def holds(self) -> bool:
        if self._joint is not None:
            return self._joint.holds()
        return all(2 * s <= len(self.members) for s in self.plane_sizes())


def _stock_first_level_count():
    if 'flc' in _ORIG:
        return _ORIG['flc']
    from epde.operators.multiobjective import moeadd_specific as MS
    return MS._FirstLevelCount


# ------------------------------------------------------------------ patched functions
def _check_config() -> None:
    if not CHECK_CONFIG or not _STATE.get('mode') or _STATE.get('checked'):
        return
    from epde.interface.search_config import active_config
    cfg = active_config()
    if cfg.search_space.system_rps_mode != 'plain' or cfg.evolution.engine != 'nsga2':
        raise SystemExit("arm %s needs system_rps_mode='plain' and engine='nsga2' (got %r, %r)"
                         % (_STATE['mode'], cfg.search_space.system_rps_mode, cfg.evolution.engine))
    _STATE['checked'] = True


def _plan(self, objective, plane_a, pick):
    """The mating plan: per pair (A, kA, B, kB), B redrawn up to 3 times when the predicted child label is A's,
    B's or in the history. Both parents' counters rise."""
    population = objective.population
    n = len(population)
    keys = list(population[0].vals.equation_keys)
    n_eq = len(keys)
    parents = int(n * self.params['parents_fraction'])
    parents = min(max(2, parents + parents % 2), n + n % 2)
    pairs = parents // 2
    history = getattr(objective, 'history', None) or ()
    plan, coins = [], []
    for p in range(pairs):
        eA = plane_a(p)
        others = [e for e in range(n_eq) if e != eA]
        eB = others[int(np.random.randint(n_eq - 1))]
        kA, kB = keys[eA], keys[eB]
        A = population[pick(eA)]
        from_b = frozenset(k for k in keys if k not in (kA, kB) and np.random.random() < 0.5)
        for attempt in range(4):
            B = population[pick(eB)]
            predicted = predicted_label(A, kA, B, kB, keys, from_b)
            if predicted in (A.equations_labels, B.equations_labels):
                reason = 'clone'
            elif predicted in history:
                reason = 'history'
            else:
                break
            if attempt == 3:
                COUNTS['plan_redraw_exhausted'] += 1
                break
            COUNTS['plan_%s_redraws' % reason] += 1
        plan.append((A, kA, B, kB))
        coins.append(from_b)
        A.incr_counter()
        B.incr_counter()
    objective.forced_pair = None
    objective.pv_plan = plan
    objective.pv_coins = coins
    COUNTS['plan_pairs'] += pairs
    return objective


def pv_select(self, objective, arguments):
    """Per-plane binary tournaments: pair p draws A in plane (turn + p) mod n_eq, B in another plane."""
    population = objective.population
    if not population or len(population[0].vals) < 2:
        return _ORIG['select'](self, objective, arguments)
    _check_config()
    self.parse_suboperator_args(arguments=arguments)
    T = plane_table(population)
    n_eq, n = T.n_eq, len(population)
    turn = _STATE.get('sel_turn', 0)
    _STATE['sel_turn'] = turn + 1
    return _plan(self, objective, lambda p: (turn + p) % n_eq, lambda e: tournament(T, e, n))


def ja_select(self, objective, arguments):
    """Stock joint tournaments (operators.py: level index, -plane_crowding) feeding the same plan; kA uniform."""
    from epde.optimizers.nsga.operators import plane_crowding
    population = objective.population
    if not population or len(population[0].vals) < 2:
        return _ORIG['select'](self, objective, arguments)
    _check_config()
    self.parse_suboperator_args(arguments=arguments)
    rank, crowding = {}, {}
    for level_idx, level in enumerate(objective.levels):
        for member, distance in zip(level, plane_crowding(level)):
            rank[id(member)] = level_idx
            crowding[id(member)] = distance
    n, n_eq = len(population), len(population[0].vals)

    def key(idx):
        member = population[idx]
        return (rank.get(id(member), np.inf), -crowding.get(id(member), 0.0))

    def pick(_e):
        first, second = int(np.random.randint(n)), int(np.random.randint(n))
        return second if key(second) < key(first) else first
    return _plan(self, objective, lambda p: int(np.random.randint(n_eq)), pick)


def pv_crossover_apply(self, objective, arguments):
    """Per planned pair: the assembled child, and the A-side child of the stock chromosome crossover."""
    plan, coins = getattr(objective, 'pv_plan', None), getattr(objective, 'pv_coins', None)
    objective.pv_plan = objective.pv_coins = None
    if not plan:
        return _ORIG['crossover'](self, objective, arguments)
    _check_config()
    _, subop_args = self.parse_suboperator_args(arguments=arguments)
    for solution in objective.population:
        solution.reset_counter()
    objective.forced_pair = None
    keys = list(objective.population[0].vals.equation_keys)
    offspring = []
    for p, (A, kA, B, kB) in enumerate(plan):
        assert '_pv_skip_mutation' not in A.__dict__ and '_pv_skip_mutation' not in B.__dict__
        child1 = assemble(A, kA, B, kB, keys, from_b=coins[p] if coins else None)
        a1, b1 = deepcopy(A), deepcopy(B)
        a1.reset_state(True)
        b1.reset_state(True)
        child2, _ = self.suboperators['chromosome_crossover'].apply(objective=(a1, b1),
                                                                    arguments=subop_args['chromosome_crossover'])
        for eq_key in keys:
            for system in (child1, child2, A, B):
                assert len(system.vals[eq_key].terms_labels) == len(system.vals[eq_key].structure)
        offspring.extend([child1, child2])
    objective.unplaced_candidates = offspring
    return objective


def pv_system_mutation(self, objective, arguments):
    """The assembled child's first call returns it unmutated (the flag is popped); every other call is stock."""
    flags = getattr(objective, '__dict__', None)
    if flags is not None and flags.pop('_pv_skip_mutation', False):
        COUNTS['mutation_skipped'] += 1
        entry = _entry(objective)
        if entry is not None:
            entry['state'] = 'first_attempt'
        return objective
    entry = _entry(objective)
    if entry is not None:
        if entry['state'] == 'first_attempt':
            try:
                from epde.operators.multiobjective.moeadd_specific import has_degenerate_equation
                COUNTS['first_failed'] += 1
                COUNTS['first_failed_' + ('duplicate' if entry['swap_called'] else
                                          'degenerate' if has_degenerate_equation(objective) else 'other')] += 1
            except Exception:
                COUNTS['counter_errors'] += 1
            _settle(entry, done=True)
        entry['state'] = 'retry'
        REG.pop(id(objective), None)
    return _ORIG['mutation'](self, objective, arguments)


def pv_truncate(self):
    """mu + lambda -> mu by ``survivors``; the population keeps its order, levels = ``plane_layers``."""
    levels_obj = self.pareto_levels
    population = list(levels_obj.population)
    if not population or len(population[0].vals) < 2:
        return _ORIG['truncate'](self)
    _check_config()
    T = plane_table(population)
    turn = self.__dict__.get('_pv_turn', 0)
    self._pv_turn = turn + 1
    kept_idx, stats = survivors(T, self._mu, turn % T.n_eq)
    kept = [population[i] for i in sorted(kept_idx)]
    levels_obj.population = kept
    levels_obj.levels = (plane_layers(plane_table(kept), kept) if kept else []) or [[]]
    SURVIVAL['truncations'] += 1
    SURVIVAL['overflow'] += int(stats['overflow'])
    SURVIVAL['front_count_changed'] += int(stats['front_count_changed'])
    for name in ('H', 'F', 'all_front', 'all_front_raw', 'n'):
        SURVIVAL[name].append(stats[name])


# ------------------------------------------------------------------ counting wrappers
def _counting_apply_plain(orig):
    def pv_apply_plain(objective, eq_selector, eq_args):
        entry = _entry(objective)
        if entry is None or entry['state'] != 'first_attempt':
            return orig(objective, eq_selector, eq_args)
        if entry['fit_seen']:
            COUNTS['first_regenerated'] += 1
            return orig(objective, eq_selector, eq_args)
        entry['fit_seen'] = True
        keys = list(objective.vals.equation_keys)
        try:
            before = {k: frozenset(objective.vals[k].terms_labels) for k in keys}
        except Exception:
            before = None
            COUNTS['counter_errors'] += 1
        out = orig(objective, eq_selector, eq_args)
        try:
            _count_first_fit(objective, entry, keys, before)
        except Exception:
            COUNTS['counter_errors'] += 1
        return out
    return pv_apply_plain


def _count_first_fit(objective, entry, keys, before) -> None:
    import epde.operators.common.plain_fit as PF
    COUNTS['first_fit_seen'] += 1
    hand = PF._SIBLINGS.get(id(objective))
    siblings = hand[1] if hand is not None and hand[0]() is objective else []
    for key in [key for key, (ref, _) in SIB.items() if ref() is None]:
        del SIB[key]
    for sibling in siblings:
        SIB[id(sibling)] = (_ref(sibling), entry)
    if objective.equations_labels == entry['predicted']:
        COUNTS['fit_label_as_planned'] += 1
    elif any(sibling.equations_labels == entry['predicted'] for sibling in siblings):
        COUNTS['fit_label_planned_as_sibling'] += 1
    else:
        COUNTS['fit_label_changed'] += 1
    if before is None:
        return
    targets = {}
    for k in keys:
        try:
            target = objective.vals[k].target
        except Exception:
            target = None
        targets[k] = None if target is None else target.factors_labels
    removed = 0
    for k in keys:
        partners = {sig for j, sig in targets.items() if j != k and sig is not None}
        removed += len((before[k] - frozenset(objective.vals[k].terms_labels)) & partners)
    if removed:
        COUNTS['step2_altered_children'] += 1
        COUNTS['step2_removed_terms'] += removed


def _counting_swap(orig):
    def pv_swap_arrangement(self, offspring, objective, subop_args):
        out = orig(self, offspring, objective, subop_args)
        entry = _entry(offspring)
        if entry is not None and entry['state'] == 'first_attempt':
            entry['swap_called'] = True
            if out:
                COUNTS['first_swapped'] += 1
                entry['state'] = 'swapped'
                _settle(entry, True, 'main' if offspring.equations_labels == entry['predicted'] else None)
                REG.pop(id(offspring), None)
        return out
    return pv_swap_arrangement


def _counting_append(orig):
    def pv_append(self, objective, arguments):
        try:
            offspring = objective[0]
            entry = _entry(offspring)
            if entry is not None:
                if entry['state'] == 'first_attempt':
                    COUNTS['first_placed'] += 1
                    _settle(entry, True, 'main' if offspring.equations_labels == entry['predicted'] else None)
                entry['state'] = 'placed'
                REG.pop(id(offspring), None)
            else:
                held = SIB.pop(id(offspring), None)
                if held is not None and held[0]() is offspring:
                    COUNTS['first_siblings_placed'] += 1
                    if offspring.equations_labels == held[1]['predicted']:
                        _settle(held[1], planned='sibling')
        except Exception:
            COUNTS['counter_errors'] += 1
        return orig(self, objective, arguments)
    return pv_append


# ------------------------------------------------------------------ install / report
def install(mode='pv') -> None:
    if mode not in ('pv', 'ja'):
        raise ValueError("arm_pv.install: mode must be 'pv' or 'ja', got %r" % (mode,))
    for var in _REFUSE_ENV:
        if os.environ.get(var):
            raise SystemExit('arm %s owns selection/crossover: %s is set (archive mating / niching / mating '
                             'refused)' % (mode, var))
    import epde.operators.common.plain_fit as PF
    import epde.operators.multiobjective.moeadd_specific as MS
    import epde.operators.multiobjective.mutations as MUT
    import epde.operators.multiobjective.variation as VAR
    import epde.optimizers.nsga.operators as NOPS
    import epde.optimizers.nsga.optimizer as NO
    installed = getattr(NO.NSGAOptimizer, '_pv_installed', None)
    if installed == mode:
        return
    if installed:
        raise SystemExit('arm_pv: %s is installed already; pv and ja cannot both be installed' % installed)
    _ORIG.update(truncate=NO.NSGAOptimizer._truncate, select=NOPS.NSGASelection.apply,
                 crossover=VAR.ParetoLevelsCrossover.apply, mutation=MUT.SystemMutation.apply,
                 flc=MS._FirstLevelCount, apply_plain=PF.apply_plain, swap=MS.OffspringUpdater._swap_arrangement,
                 append=NOPS.AppendUpdater.apply)
    if mode == 'pv':
        NO.NSGAOptimizer._truncate = pv_truncate
        NOPS.NSGASelection.apply = pv_select
        MS._FirstLevelCount = PerPlaneFirstLevelCount
    else:
        NOPS.NSGASelection.apply = ja_select
    VAR.ParetoLevelsCrossover.apply = pv_crossover_apply
    MUT.SystemMutation.apply = pv_system_mutation
    PF.apply_plain = _counting_apply_plain(_ORIG['apply_plain'])
    MS.OffspringUpdater._swap_arrangement = _counting_swap(_ORIG['swap'])
    NOPS.AppendUpdater.apply = _counting_append(_ORIG['append'])
    NO.NSGAOptimizer._pv_installed = mode
    _STATE['mode'] = mode


def _summary(values) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return dict(mean=round(float(arr.mean()), 3), max=float(arr.max()))


def report() -> dict:
    F = SURVIVAL['F']
    return dict(
        mode=_STATE.get('mode'),
        counts=dict(sorted(COUNTS.items())),
        planned_label=dict(placed_main=PLANNED['main'], placed_sibling=PLANNED['sibling'],
                           not_placed=PLANNED['not_placed'], pending=PLANNED['pending']),
        survival=dict(truncations=SURVIVAL['truncations'], overflow=SURVIVAL['overflow'],
                      front_count_changed=SURVIVAL['front_count_changed'], H=_summary(SURVIVAL['H']),
                      F=[_summary([f[e] for f in F]) for e in range(len(F[0]))] if F else [],
                      all_front=_summary(SURVIVAL['all_front']),
                      all_front_raw=_summary(SURVIVAL['all_front_raw']), n=_summary(SURVIVAL['n'])),
        initial=_STATE.get('init'))
