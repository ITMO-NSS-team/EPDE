"""``evolution.niching='per_variable'``: NSGA-II with every equation ranked in
its own (discrepancy, instability) plane.

Nothing is summed or averaged across equations: only plane ranks and the
number of plane fronts a system holds are compared. The same structure is
the same equation, so each plane has one REPRESENTATIVE per structure label:
the copy minimising (its level in that plane, its sorted levels in the other
planes), the newest copy on a tie, so a freshly assembled system holding two
tied laws represents both. Non-representatives rank after every
representative of the plane; crowding is measured per plane level over the
representatives, identical points sharing one distance.

* Truncation (:func:`survivors`): a round-robin over the planes, starting at
  a plane that turns with every generation. First one copy of every front
  equation (systems holding more plane fronts first, then the less crowded),
  then the other systems whose every equation is on its plane's first level,
  then the deeper representatives by level and crowding, then the rest.
* Selection (:func:`select`): pair p draws parent A by a binary tournament in
  plane (turn + p) mod n and parent B in another plane drawn uniformly; B is
  redrawn up to three times while the assembled child would repeat A, B or a
  system in the history.
* Crossover (:func:`crossover`): per pair, the ASSEMBLED child -- A with B's
  equation of B's plane (and, for more than two equations, each other
  equation from B by a fair coin) -- which skips its first mutation, and the
  A-side child of the usual chromosome crossover.
* Initial population (:class:`PerPlaneFirstLevelCount`): at most half the
  systems on level 0 of EVERY plane.

Systems of one equation use the joint rules everywhere.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

#: The attribute an assembled child carries until its first mutation.
SKIP_FIRST_MUTATION = '_skip_first_mutation'
#: Redraws of parent B while the assembled child repeats A, B or the history.
PLAN_REDRAWS = 3


def niching_per_variable(population) -> bool:
    """True when ``evolution.niching`` resolves to ``'per_variable'`` and the systems hold several equations."""
    from epde.interface.search_config import active_config, resolve_niching
    config = active_config()
    return (resolve_niching(config.evolution.niching, config.search_space.system_rps_mode,
                            config.evolution.engine) == 'per_variable'
            and len(population) > 0 and len(population[0].vals) > 1)


# ------------------------------------------------------------------ plane ranking
def _different(labels) -> np.ndarray:
    """ne[j, i] = labels[j] != labels[i]."""
    try:
        ids = {}
        codes = np.array([ids.setdefault(label, len(ids)) for label in labels], dtype=int)
        return codes[:, None] != codes[None, :]
    except TypeError:
        return np.array([[a != b for b in labels] for a in labels], dtype=bool).reshape(len(labels), len(labels))


def plane_levels_fast(points, labels) -> np.ndarray:
    """``moeadd_specific.plane_levels``, vectorised: j dominates i in the plane when their labels differ."""
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
    """The per-plane ranking of ``population`` (module docstring); arrays are n_eq x N."""
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
    # A system whose every equation is on its plane's first level, whichever copy represents it there: after
    # one copy of each front equation, a system made of front equations is kept before any deeper one.
    all_front = (lev == 0).all(axis=0) if N else np.zeros(0, dtype=bool)
    return SimpleNamespace(keys=keys, n_eq=n_eq, n=N, planes=planes, labels=labels, lev=lev, rep=rep, adj=adj,
                           crowd=crowd, partner=partner, front=front, front_count=front.sum(axis=0).astype(int),
                           all_front=all_front)


def _front_queues(T) -> list:
    return [sorted((int(i) for i in np.flatnonzero(T.front[e])),
                   key=lambda i: (-int(T.front_count[i]), -T.crowd[e, i], T.partner[e][i], i))
            for e in range(T.n_eq)]


def _all_front_queues(T) -> list:
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


def survivors(T, mu, start=0) -> list:
    """The indices kept by the truncation to ``mu`` (module docstring), in pick order."""
    if T.lev.shape[1] <= mu:
        return list(range(T.lev.shape[1]))
    order = [(start + r) % T.n_eq for r in range(T.n_eq)]
    return _fill([_front_queues(T), _all_front_queues(T), _deeper_queues(T)], mu, order)


def plane_layers(T, population) -> list:
    """``population`` partitioned by the best adjusted plane rank over the planes; empty layers dropped."""
    if not len(population):
        return []
    best = T.adj.min(axis=0)
    return [[population[i] for i in range(len(population)) if best[i] == r] for r in sorted(set(best.tolist()))]


def truncate(optimizer) -> None:
    """``NSGAOptimizer._truncate`` under per-variable niching: mu + lambda -> mu by :func:`survivors`; the
    population keeps its order, the levels are :func:`plane_layers`."""
    levels_obj = optimizer.pareto_levels
    population = list(levels_obj.population)
    T = plane_table(population)
    turn = optimizer.__dict__.get('_pv_turn', 0)
    optimizer._pv_turn = turn + 1
    kept = [population[i] for i in sorted(survivors(T, optimizer._mu, turn % T.n_eq))]
    levels_obj.population = kept
    levels_obj.levels = (plane_layers(plane_table(kept), kept) if kept else []) or [[]]


# ------------------------------------------------------------------ selection and crossover
def _tournament(T, e, n) -> int:
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
    """The equation labels the assembled child carries as planned: B's at kB (and ``from_b``), A's elsewhere."""
    return tuple(_eq_label((B if (k == kB or k in from_b) else A).vals[k]) for k in keys)


def select(operator, objective) -> None:
    """``NSGASelection`` under per-variable niching: the mating plan ``objective.pv_plan`` -- per pair
    (A, kA, B, kB) -- and the coins ``objective.pv_coins`` (module docstring). Both parents' counters rise."""
    population = objective.population
    T = plane_table(population)
    n_eq, n = T.n_eq, len(population)
    turn = objective.__dict__.get('_pv_turn', 0)
    objective._pv_turn = turn + 1
    keys = list(population[0].vals.equation_keys)
    parents = int(n * operator.params['parents_fraction'])
    parents = min(max(2, parents + parents % 2), n + n % 2)
    history = getattr(objective, 'history', None) or ()
    plan, coins = [], []
    for p in range(parents // 2):
        eA = (turn + p) % n_eq
        others = [e for e in range(n_eq) if e != eA]
        eB = others[int(np.random.randint(n_eq - 1))]
        kA, kB = keys[eA], keys[eB]
        A = population[_tournament(T, eA, n)]
        from_b = frozenset(k for k in keys if k not in (kA, kB) and np.random.random() < 0.5)
        for _ in range(PLAN_REDRAWS + 1):
            B = population[_tournament(T, eB, n)]
            predicted = predicted_label(A, kA, B, kB, keys, from_b)
            if predicted not in (A.equations_labels, B.equations_labels) and predicted not in history:
                break
        plan.append((A, kA, B, kB))
        coins.append(from_b)
        A.incr_counter()
        B.incr_counter()
    objective.forced_pair = None
    objective.pv_plan = plan
    objective.pv_coins = coins


def _replace(vals, key, value) -> None:
    if hasattr(vals, 'replace_gene'):
        vals.replace_gene(gene_key=key, value=value)
    else:
        vals[key] = value


def assemble(A, kA, B, kB, keys, from_b):
    """deepcopy(A) with B's equation at kB and at every key of ``from_b``; no fit, no target; it skips its
    first mutation."""
    child = deepcopy(A)
    _replace(child.vals, kB, deepcopy(B.vals[kB]))
    for k in keys:
        if k not in (kA, kB) and k in from_b:
            _replace(child.vals, k, deepcopy(B.vals[k]))
    if hasattr(child, 'reset_state'):
        child.reset_state(True)
    child.__dict__[SKIP_FIRST_MUTATION] = True
    return child


def crossover(operator, objective, plan, coins, subop_args) -> None:
    """``ParetoLevelsCrossover`` under per-variable niching: per planned pair the assembled child and the
    A-side child of the chromosome crossover, into ``objective.unplaced_candidates``."""
    for solution in objective.population:
        solution.reset_counter()
    objective.forced_pair = None
    keys = list(objective.population[0].vals.equation_keys)
    offspring = []
    for (A, kA, B, kB), from_b in zip(plan, coins):
        child1 = assemble(A, kA, B, kB, keys, from_b)
        a1, b1 = deepcopy(A), deepcopy(B)
        a1.reset_state(True)
        b1.reset_state(True)
        child2, _ = operator.suboperators['chromosome_crossover'].apply(
            objective=(a1, b1), arguments=subop_args['chromosome_crossover'])
        for eq_key in keys:
            for system in (child1, child2, A, B):
                assert len(system.vals[eq_key].terms_labels) == len(system.vals[eq_key].structure)
        offspring.extend([child1, child2])
    objective.unplaced_candidates = offspring


def skip_first_mutation(system) -> bool:
    """True (once) for an assembled child that has not been mutated yet; the flag is removed."""
    flags = getattr(system, '__dict__', None)
    return bool(flags is not None and flags.pop(SKIP_FIRST_MUTATION, False))


# ------------------------------------------------------------------ the initial population
class PerPlaneFirstLevelCount(object):
    """``moeadd_specific._FirstLevelCount`` per plane: each plane's level 0 is the members no member with a
    different structure label dominates in that plane (twins share a level). ``holds()`` iff at most half the
    members are on level 0 of every plane. Systems of one equation use the joint count."""

    def __init__(self):
        self.members = []
        self._joint = None
        self._points, self._labels, self.dominators = None, None, None

    def place(self, system) -> None:
        if self._joint is None and not self.members and len(system.vals) < 2:
            from epde.operators.multiobjective.moeadd_specific import _FirstLevelCount
            self._joint = _FirstLevelCount()
        if self._joint is not None:
            self._joint.place(system)
            self.members.append(system)
            return
        from epde.operators.multiobjective.moeadd_specific import equation_planes
        planes, labels = equation_planes([system])
        if self._points is None:
            self._points, self._labels, self.dominators = ([[] for _ in range(len(planes))] for _ in range(3))
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
