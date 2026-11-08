"""

Supplementary procedures for the moeadd optimizer.

Contains:
---------

**check_dominance(target, compared_with) -> bool** : Function to check, if one
solution is dominated by another;

**ndl_update(new_solution, levels) -> list** : Function to add a new solution into
the existing levels;

**fast_non_dominated_sorting(population) -> list** : Sorting of a population into
the non-dominated levels. The faster alternative to the **slow_non_dominated_sorting**.
Computational complexity :math:`O(MN^2)`, where *N* is the population size, and *M* is 
the number of objective functions.

**slow_non_dominated_sorting(population) -> list** : Sorting of a population into
the non-dominated levels. Computational complexity :math:`O(MN^3)`, where *N* is 
the population size, and *M* is the number of objective functions.

**acute_angle(vector_a, vector_b) -> float** : calculation of acute angle between two vectors.

**Constraint** abstract class with **Equality** and **Inequality** implementations.

"""

from copy import deepcopy
import numpy as np
from abc import ABC, abstractmethod

from epde.supplementary import rts

def _structure_label(equation):
    """``Equation.structure_label``; ``terms_labels`` for an equation-like
    object without one."""
    label = getattr(equation, 'structure_label', None)
    return equation.terms_labels if label is None else label


def check_dominance(target, compared_with) -> bool:
    """

    Function to check, if one solution is dominated by another.

    Args:
        target (`src.moeadd.moeadd_solution_template.MOEADDSolution`):  case-specific subclass object
            The individual solution on the pareto levels, compared with the other element.
        compared_with (`src.moeadd.moeadd_solution_template.MOEADDSolution`):  case-specific subclass object
            The individual solution on the pareto levels, with which the target is compared.

    Returns:
        domiated (`bool`): Function returns True, if the **target** dominates (has at least one objective
            functions with less values, while the others are the same) the **compared_with**; 
            False in all other cases.

    """
    return _view_dominates(_dominance_view(target), _dominance_view(compared_with))


def _dominance_view(system):
    """What :func:`check_dominance` reads of a system, read once: its equation
    keys, its objective vector (the property rebuilds it on every access) and
    each equation's structure label. A sort compares every pair, so it reads
    each member once instead of once per comparison."""
    keys = system.vals.equation_keys
    return keys, system.obj_fun, {key: _structure_label(system.vals[key]) for key in keys}


def _view_dominates(target, compared_with) -> bool:
    """:func:`check_dominance` on two :func:`_dominance_view` s."""
    flag = False

    eq_keys, t_obj, t_labels = target
    _, c_obj, c_labels = compared_with
    for obj_fun_idx in range(len(t_obj)):
        # Deliberate noise guard: structurally identical equation genes can
        # still differ in objective values through numerical noise (e.g.
        # re-fitted weights); such objectives must not create spurious
        # dominance, so they are excluded from the comparison entirely.
        # ``structure_label`` is ``terms_labels`` itself outside
        # system_rps_mode='plain', which adds the constant (S and S+{1} differ).
        key = eq_keys[obj_fun_idx % len(eq_keys)]
        if t_labels[key] == c_labels[key]:
            continue
        if t_obj[obj_fun_idx] < c_obj[obj_fun_idx]:
            flag = True
        elif t_obj[obj_fun_idx] > c_obj[obj_fun_idx]:
            return False
        # Equal values (between different structures) are a tie: they
        # neither establish nor block dominance -- canonical Pareto
        # semantics ("no objective worse, at least one strictly better").
    return flag


class DominanceRows:
    """:func:`check_dominance` between one system and many at once.

    Row ``r`` holds what :func:`_dominance_view` reads of member ``r``: its
    objective vector, and per objective the structure label of the equation
    the objective belongs to (objective ``i`` belongs to ``keys[i % len(keys)]``,
    as in :func:`_view_dominates`), interned as an integer id -- equal ids are
    equal labels, so a column whose ids agree is skipped exactly as
    :func:`_view_dominates` skips it. NaN compares as a tie both ways there
    and here. A third integer per row interns the member's
    ``equations_labels``. Every member must share the equation keys and the
    objective count (:meth:`fits`). A label first seen on a system that is not
    added reads as id -1, which no member has."""

    def __init__(self, keys, n_objectives: int):
        self.keys = tuple(keys)
        self._columns = np.arange(n_objectives) % len(self.keys)
        self._label_ids = [{} for _ in self.keys]
        self._system_ids = {}
        self.obj = np.empty((0, n_objectives))
        self.labels = np.empty((0, n_objectives), dtype=np.int64)
        self.systems = np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.systems)

    def fits(self, view) -> bool:
        keys, obj, _ = view
        return tuple(keys) == self.keys and len(obj) == self.obj.shape[1]

    def interned(self) -> int:
        """How many labels the id tables hold (evicted members' included)."""
        return len(self._system_ids) + sum(len(ids) for ids in self._label_ids)

    def row(self, view, system_label):
        """``(objectives, label ids, system id)`` of a view; -1 for labels no
        member has had."""
        _, obj, labels = view
        per_equation = [ids.get(labels[key], -1) for ids, key in zip(self._label_ids, self.keys)]
        return (np.asarray(obj, dtype=float), np.asarray(per_equation, dtype=np.int64)[self._columns],
                self._system_ids.get(system_label, -1))

    def append(self, view, system_label) -> None:
        _, obj, labels = view
        per_equation = [ids.setdefault(labels[key], len(ids)) for ids, key in zip(self._label_ids, self.keys)]
        system_id = self._system_ids.setdefault(system_label, len(self._system_ids))
        self.obj = np.vstack([self.obj, np.asarray(obj, dtype=float)[None, :]])
        self.labels = np.vstack([self.labels, np.asarray(per_equation, dtype=np.int64)[self._columns][None, :]])
        self.systems = np.append(self.systems, system_id)

    def keep(self, mask) -> None:
        self.obj, self.labels, self.systems = self.obj[mask], self.labels[mask], self.systems[mask]

    def twins(self, system_id) -> np.ndarray:
        """Members with these ``equations_labels``, as a boolean mask."""
        return self.systems == system_id if system_id >= 0 else np.zeros(len(self), dtype=bool)

    def dominating(self, obj, label_ids) -> np.ndarray:
        """``check_dominance(member, system)`` for every member."""
        differ = self.labels != label_ids
        return (np.any((self.obj < obj) & differ, axis=1)
                & ~np.any((self.obj > obj) & differ, axis=1))

    def dominated(self, obj, label_ids) -> np.ndarray:
        """``check_dominance(system, member)`` for every member."""
        differ = self.labels != label_ids
        return (np.any((obj < self.obj) & differ, axis=1)
                & ~np.any((obj > self.obj) & differ, axis=1))

    def holds(self, r, view, system_label) -> bool:
        """Row ``r`` still reads as the member's current view (objectives
        compared NaN-aware)."""
        obj, label_ids, system_id = self.row(view, system_label)
        return bool(np.array_equal(obj, self.obj[r], equal_nan=True)
                    and np.array_equal(label_ids, self.labels[r]) and system_id == self.systems[r])


def ndl_update(new_solution, levels) -> list:   # efficient_ndl_update
    """
    Computationally-cheap method of adding new solution into the existing Pareto levels.

    Args:
        new_solution (`src.moeadd.moeadd_solution_template.MOEADDSolution`):  case-specific subclass object
            The solution, that is to be added onto the non-dominated levels.
        levels (`list`): List of lists of ``src.moeadd.moeadd_solution_template.MOEADDSolution`` case-specific subclass 
            object, representing the input non-dominated levels.

    Returns:
        new_levels (`list`): List of lists of ``src.moeadd.moeadd_solution_template.MOEADDSolution`` case-specific subclass 
            object, containing the solution from input parameter *level* with *new_solution*, added to it.

    Notes:
        The idea for this method was introduced in *K. Li, K. Deb, Q. Zhang, and S. Kwong, 
        “Efficient non-domination level update approach for steady-state evolutionary 
        multiobjective optimization,” Dept. Electr. Comput. Eng., Michigan State Univ., 
        East Lansing, MI, USA, Tech. Rep. COIN No. 2014014, 2014.*

    """
    # A list, not a set: SoEq hashing/equality is structural, so a set could
    # collapse distinct objects with equal structure, and set iteration order
    # would make level construction nondeterministic.
    moving_set = [new_solution]
    # Shallow per-level copy: ndl_update only mutates the outer list (slice
    # assignment, append, extend) and the inner level lists (append, replace);
    # the MOEADDSolution objects themselves are never mutated, so cloning them
    # via deepcopy is pure overhead (per-call on every individual added to the
    # non-dominated levels). Aliasing ``levels`` directly DOES corrupt the input
    # because of the in-place slice assignment below -- the comprehension below
    # gives us a fresh outer list and fresh inner lists while preserving the
    # original solution-object identities.
    new_levels = [list(lvl) for lvl in levels]

    for level_idx in np.arange(len(levels)):
        moving_set_new = []
        for moving_set_elem in moving_set:
            level_new = new_levels[level_idx]
            # Compute each direction of dominance against the (possibly already
            # mutated) new level exactly once instead of re-running the same
            # list-comp inside up to three branches.
            dom_over_me = [check_dominance(s, moving_set_elem) for s in level_new]
            dom_by_me = [check_dominance(moving_set_elem, s) for s in level_new]
            if any(dom_over_me):
                moving_set_new.append(moving_set_elem)
            elif not any(dom_by_me):
                # Falls through from branch 1, so `not any(dom_over_me)` already holds:
                # incomparable with every existing element, append to this level.
                level_new.append(moving_set_elem)
            elif all(dom_by_me):
                # ``dom_by_me`` is taken against the CURRENT level content, not
                # the original ``levels`` snapshot. Reading the snapshot here was
                # wrong: elements appended to this level earlier in the same
                # moving-set pass are invisible to it, so an element that
                # dominated the whole PRE-UPDATE layer inserted a new layer above
                # it and pushed those freshly-placed siblings down with it.
                # Minimal counterexample (2 objectives) --
                #   levels: L0 = [(.36,.94), (.65,.59), (.69,.17)], L1 = [(.83,.33)]
                #   insert (.48,.03), which dominates the last three
                # (.65,.59) is placed on L1, then (.69,.17) -- which dominates
                # all of the ORIGINAL L1 = [(.83,.33)] but not (.65,.59) -- opened
                # a new layer and demoted (.65,.59) to L2, though its only
                # dominator sits on L0. Pinned by
                # ``test_ndl_update.py::TestNDLUpdateEquivalence``.
                temp_levels = new_levels[level_idx:]
                new_levels[level_idx:] = []
                new_levels.append([moving_set_elem,])
                new_levels.extend(temp_levels)
            else:
                # Partial domination: keep non-dominated elements + me at this
                # level; bump dominated elements down via moving_set_new.
                new_levels[level_idx] = [le for le, dom in zip(level_new, dom_by_me) if not dom]
                new_levels[level_idx].append(moving_set_elem)
                for le, dom in zip(level_new, dom_by_me):
                    if dom:
                        moving_set_new.append(le)
        moving_set = moving_set_new
        if not len(moving_set):
            break
    if len(moving_set):
        # Mixed-origin leftovers (an element descending past the last level
        # plus elements pushed down by a sibling) can contain dominating
        # pairs; dumping them as a single level would violate the
        # non-domination invariant. Sort them into proper sub-levels --
        # one sub-level in the common case, so no behavior change there.
        new_levels.extend(fast_non_dominated_sorting(moving_set))
    if len(new_levels[len(new_levels)-1]) == 0:
        _ = new_levels.pop()
    return new_levels


def fast_non_dominated_sorting(population) -> list:
    """
    Procedure of separating points from the general population into non-dominated levels.
    This function is a faster alternative to the ``slow_non_dominated_sorting``, but requires 
    a little more memory to store indexes of elements, dominated by every solution. This 
    method was introduced in *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast 
    and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
    vol. 6, no. 2, pp. 182–197, Apr. 2002.* The computational complexity of the method is 
    :math:`O(MN^2)`, where *N* is the population size, and *M* is the number of objective 
    functions in comparisson with :math:`O(MN^3)` of the straightforward way.


    Args:
        population (`list`): The input population, represented as a list of individuals.

    Returns:
        levels (`list`): List of lists of population elements. The outer index is the number of a layer 
            (e.g. 0-th is the current Pareto frontier), while the inner is the index of an element on a level.

    """

    return _sorted_levels(population)[0]


# The dominance matrix replaces the pairwise loops from this many members on,
# up to the second bound (a bool per pair: 256 MiB there). The blocks it is
# computed in hold about _MATRIX_BLOCK pair-objective comparisons.
_MATRIX_MIN_MEMBERS = 8
_MATRIX_MAX_MEMBERS = 16384
_MATRIX_BLOCK = 4_000_000


def _dominance_matrix(views):
    """``D[i, j] = _view_dominates(views[i], views[j])`` for every pair, or
    None when the views do not share the equation keys and the objective
    count, when a label is unhashable, or when there are too few or too many.

    The structure labels are interned per equation, so ``!=`` of two ids is
    ``!=`` of the labels; objective ``i`` reads equation ``keys[i % len(keys)]``.
    The two directions exclude each other, so ``D[i, j]`` is also Deb's
    "``i`` dominates ``j``" (``fast_non_dominated_sorting`` tests ``j`` over
    ``i`` first)."""
    n = len(views)
    if n == 0 or not _MATRIX_MIN_MEMBERS <= n <= _MATRIX_MAX_MEMBERS:
        return None
    keys, n_obj = tuple(views[0][0]), len(views[0][1])
    if any(tuple(view[0]) != keys or len(view[1]) != n_obj for view in views):
        return None
    ids = np.empty((n, len(keys)), dtype=np.int64)
    try:
        for e, key in enumerate(keys):
            table = {}
            ids[:, e] = [table.setdefault(view[2][key], len(table)) for view in views]
    except TypeError:
        return None
    labels = ids[:, np.arange(n_obj) % len(keys)]
    obj = np.array([np.asarray(view[1], dtype=float) for view in views]).reshape(n, n_obj)
    D = np.empty((n, n), dtype=bool)
    step = max(1, _MATRIX_BLOCK // max(1, n * n_obj))
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        differ = labels[lo:hi, None, :] != labels[None, :, :]
        mine, theirs = obj[lo:hi, None, :], obj[None, :, :]
        D[lo:hi] = np.any((mine < theirs) & differ, axis=2) & ~np.any((mine > theirs) & differ, axis=2)
    return D


def _matrix_levels(population, D) -> list:
    """``fast_non_dominated_sorting`` read off the dominance matrix, member
    for member and in the same order: a member joins the next level when the
    last of its dominators on the current level is processed (Deb's loop
    takes the level in order, each one's dominated members by index)."""
    count = D.sum(axis=0)
    current = np.flatnonzero(count == 0)
    levels = [[population[i] for i in current]]
    while len(current):
        block = D[current]
        count = count - block.sum(axis=0)
        reached = np.flatnonzero((count == 0) & block.any(axis=0))
        if len(reached):
            last = len(current) - 1 - np.argmax(block[::-1][:, reached], axis=0)
            reached = reached[np.lexsort((reached, last))]
            levels.append([population[i] for i in reached])
        current = reached
    return levels


def _sorted_levels(population):
    """``(levels, D, views)``: Deb's levels, with the dominance matrix and the
    views they came from (D None on the pairwise path)."""
    views = [_dominance_view(member) for member in population]
    D = _dominance_matrix(views)
    if D is not None:
        return _matrix_levels(population, D), D, views
    levels = []
    ranks = np.empty(len(population))
    # Число элементов, доминирующих над i-ым кандидиатом
    domination_count = np.zeros(len(population))
    # Индексы элементов, над которыми доминирует i-ый кандидиат
    dominated_solutions = [[] for elem_idx in np.arange(len(population))]
    current_level_idxs = []
    for main_elem_idx in np.arange(len(population)):
        for compared_elem_idx in np.arange(len(population)):
            if main_elem_idx == compared_elem_idx:
                continue
            if _view_dominates(views[compared_elem_idx], views[main_elem_idx]):
                domination_count[main_elem_idx] += 1
            elif _view_dominates(views[main_elem_idx], views[compared_elem_idx]):
                dominated_solutions[main_elem_idx].append(compared_elem_idx)
        if domination_count[main_elem_idx] == 0:
            current_level_idxs.append(main_elem_idx)
            ranks[main_elem_idx] = 0
    levels.append([population[elem_idx] for elem_idx in current_level_idxs])

    level_idx = 0
    while len(current_level_idxs) > 0:
        new_level_idxs = []
        for main_elem_idx in current_level_idxs:
            for dominated_elem_idx in dominated_solutions[main_elem_idx]:
                domination_count[dominated_elem_idx] -= 1
                if domination_count[dominated_elem_idx] == 0:
                    ranks[dominated_elem_idx] = level_idx + 1
                    new_level_idxs.append(dominated_elem_idx)

        if len(new_level_idxs):
            levels.append([population[elem_idx]
                          for elem_idx in new_level_idxs])
        level_idx += 1
        current_level_idxs = new_level_idxs  # deepcopy(new_level_idxs)
    return levels, None, views


def cycle_safe_nds(population) -> list:
    """:func:`fast_non_dominated_sorting` that never loses a member.

    The same-structure guard in :func:`check_dominance` makes dominance
    non-transitive between systems of several equations: A can beat B on one
    equation while B's copy of A's other equation is skipped, and a cycle
    A > B > C > A follows. Deb's sort never ranks a member of a cycle -- its
    domination count never reaches zero -- and drops it silently. Here the
    members it left out are peeled off into further levels, each level the
    ones dominated by the fewest of those still left. Without a cycle the
    result is exactly the fast sort's.
    """
    levels, D, _ = _sorted_levels(population)
    placed = {id(member) for level in levels for member in level}
    rest = [member for member in population if id(member) not in placed]
    if not rest:
        return levels
    levels = [level for level in levels if level]
    if D is not None:
        # A member never dominates itself (D's diagonal is False), so a column
        # sum over the members left is the pairwise count below.
        left = [i for i, member in enumerate(population) if id(member) not in placed]
        while left:
            counts = D[np.ix_(left, left)].sum(axis=0)
            fewest = counts.min()
            levels.append([population[i] for i, count in zip(left, counts) if count == fewest])
            left = [i for i, count in zip(left, counts) if count != fewest]
        return levels
    views = {id(member): _dominance_view(member) for member in rest}
    while rest:
        counts = [sum(_view_dominates(views[id(other)], views[id(member)]) for other in rest if other is not member)
                  for member in rest]
        fewest = min(counts)
        levels.append([member for member, count in zip(rest, counts) if count == fewest])
        rest = [member for member, count in zip(rest, counts) if count != fewest]
    return levels


def cycle_safe_ndl_update(new_solution, levels) -> list:
    """:func:`ndl_update` that never loses a member: when the incremental
    update hands back fewer members than it was given (a dominance cycle, see
    :func:`cycle_safe_nds`), the levels are rebuilt by :func:`cycle_safe_nds`."""
    new_levels = ndl_update(new_solution, levels)
    if sum(len(level) for level in new_levels) != sum(len(level) for level in levels) + 1:
        members = [member for level in levels for member in level] + [new_solution]
        new_levels = cycle_safe_nds(members)
    return new_levels


def slow_non_dominated_sorting(population) -> list:
    """
    Procedure of separating points from the general population into non-dominated levels.
    Operates in the straightforward way: each layer is comprised of elements, that are 
    not dominated by any other element in the population, except the ones, already put into
    the output levels. Computational complexity of this variant of sorting in worst scenario is
    :math:`O(MN^3)`, where *N* is the population size, and *M* is the number of objective functions.

    Args:
        population (`list`): The input population, represented as a list of individuals.

    Returns:
        levels (`list`): List of lists of population elements. The outer index is the number of a layer 
            (e.g. 0-th is the current Pareto frontier), while the inner is the index of an element on a level.

    """
    locked_idxs = []
    levels = []
    levels_elems = 0
    while len(population) > levels_elems:
        processed_idxs = []
        for main_elem_idx in np.arange(len(population)):
            if not main_elem_idx in locked_idxs:
                dominated = False
                for compared_elem_idx in np.arange(len(population)):
                    if main_elem_idx == compared_elem_idx or compared_elem_idx in locked_idxs:
                        continue
                    if check_dominance(population[compared_elem_idx], population[main_elem_idx]):
                        dominated = True
                if not dominated:
                    processed_idxs.append(main_elem_idx)
        locked_idxs.extend(processed_idxs)
        levels_elems += len(processed_idxs)
        levels.append([population[elem_idx] for elem_idx in processed_idxs])
    return levels


def acute_angle(vector_a, vector_b) -> float:
    cos_val = np.dot(vector_a, vector_b)/(np.sqrt(np.dot(vector_a, vector_a))*np.sqrt(np.dot(vector_b, vector_b)))
    if np.abs(cos_val) > 1.:
        cos_val = np.sign(cos_val)
    return np.arccos(cos_val)


class Constraint(ABC):
    """

    The abstract class for the constraint. Noteable subclasses: Inequality & Equality.

    """
    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def __call__(self, *args):
        pass


class Inequality(Constraint):
    """
    The class of the constrain (subclass of Constraint), representing the inequality. 
    The format of inequality is assumed in format :math:`g(x) >= 0`.

    Args:
        g (`function (lambda function)`): The constraint function, which shall penalize the candidate solution, if the value of
            :math:`g(x) >= 0` is not fulfilled (is less, than 0). The penalty is equal to 
            the absolute value of constraint violation.

    Methods:
        __call__(self, x) : returns float
            Overloaded call operator returns the value of constaint violation.

    """

    def __init__(self, g):
        self._g = g

    def __call__(self, x) -> float:
        """
        Method to evaluate the constraint violation of the candidate solution.

        Args:
            x (`np.ndarray`): Values (.vals attribute) of the candidate solution, that represent its gene.

        Returns:
            cv (`float`): Constraint violation value. If the value of :math:`g(x) >= 0` is not 
                fulfilled (is less, than 0), than returns :math:`|g(x)|`, else 0.
        """
        return - self._g(x) if self._g(x) < 0 else 0


class Equality(Constraint):
    """
    The class of the constrain (subclass of Constraint), representing the inequality. 
    The format of inequality is assumed in format :math:`h(x) = 0`.

    Args:
        h (`function (lambda function)`): The constraint function, which shall be penalized, if the value does not match with
            the const

    Methods:
        __call__(self, x) : returns float
            Overloaded call operator returns the value of constaint violation.

    """

    def __init__(self, h):
        self._h = h

    def __call__(self, x) -> float:
        """
        Method to evaluate the constraint violation of the candidate solution.

        Args:
            x (`np.ndarray`): Values (.vals attribute) of the candidate solution, that represent its gene.

        Returns:
            cv (`float`): Constraint violation value. If the value of :math:`h(x) = 0` is not 
                fulfilled, than returns :math:`|g(x)|`, else 0.

        """
        return np.abs(self._h(x))
