"""``search_space.system_rps_mode='plain'``: no subset search.

A candidate equation is fitted ONCE on exactly the terms it holds; the
evolution itself selects the support (``plain_moves``: add, drop and swap a
term, move the target). Nothing is screened, sparsified or enumerated.

* The target is a GENE (``Equation._plain_genes['target']``, a term
  signature) the candidate carries across generations. The fit re-finds that
  term -- by its signature, then by its factor functions alone (a term whose
  parameters were nudged or blended), and draws a new own-derivative term only
  when neither is there. No two equations share a target.
* ``'assignment'`` mode's target rules, as deterministic removals: a term
  equal to another equation's target, or related to the equation's own target
  by division, is dropped from the structure.
* The constant is not a gene: each equation is fitted as S (its terms) and as
  S+{1}. The version that dominates the other on (discrepancy, instability) is
  kept; when neither does, S is installed and S+{1} leaves as a SIBLING
  candidate, which the population operators place as well
  (:func:`take_siblings`).
* Each version goes through ``'assignment'`` mode's own build -- the same
  weighted least squares, candidate rules and fitness host -- so a support
  scores the same objectives in both modes.
* An equation that follows from the system's other equations (the same
  relation restated, ``system_redundancy``) is re-rolled.
"""
import itertools
import weakref
from copy import deepcopy

import numpy as np

import epde.globals as global_var
from epde import _loop_stats
from epde.interface.search_config import active_config, plain_kept_terms_cap
from epde.operators.common.system_redundancy import redundant_equations
from epde.structure.weak_rows import slot_of


# Sibling systems by the id of the system whose fit produced them. Keyed by id
# (with a weak reference to check it) rather than by the system: SoEq hashes and
# compares structurally, so a dict keyed by it would merge distinct systems.
_SIBLINGS = {}


def _hand_over(system, siblings) -> None:
    for key in [key for key, (ref, _) in _SIBLINGS.items() if ref() is None]:
        del _SIBLINGS[key]
    if siblings:
        _SIBLINGS[id(system)] = (weakref.ref(system), list(siblings))
    else:
        _SIBLINGS.pop(id(system), None)


def take_siblings(system) -> list:
    """The sibling systems (S+{1} versions no other version dominates) that the
    last plain fit of ``system`` produced, removed from the hand-over; ``[]``
    for anything else."""
    entry = _SIBLINGS.pop(id(system), None)
    if entry is None or entry[0]() is not system:
        return []
    return entry[1]


def target_gene(term) -> dict:
    """The gene naming ``term`` as an equation's target: its signature, and its
    factor functions alone for a term whose parameters moved."""
    return {'target': term.factors_labels,
            'names': tuple(sorted(factor.label for factor in term.structure))}


def gene_term(equation, taken=frozenset()):
    """The term of ``equation`` its target gene names, among the own-derivative
    terms whose signatures are not in ``taken``: the exact signature first,
    then the same factor functions, else None."""
    genes = getattr(equation, '_plain_genes', None)
    if not genes:
        return None
    options = own_derivative_terms(equation, taken)
    exact = next((term for term in options if term.factors_labels == genes['target']), None)
    if exact is not None:
        return exact
    names = genes.get('names')
    return next((term for term in options
                 if tuple(sorted(factor.label for factor in term.structure)) == names), None)


def own_derivative_terms(equation, taken=frozenset()) -> list:
    """The terms of ``equation`` that can be its target: those holding a
    derivative of its own variable, minus the signatures in ``taken``."""
    var = equation.main_var_to_explain
    return [term for term in equation.structure
            if term.contains_deriv(var) and term.factors_labels not in taken]


def _install_target(equation, term) -> None:
    equation.target_idx = next(i for i, t in enumerate(equation.structure) if t is term)
    equation._plain_genes = target_gene(term)


def _fitted(equation) -> bool:
    return bool(equation.right_part_selected and equation.target is not None
                and getattr(equation, 'weights_final_evald', False)
                and getattr(equation, 'fitness_calculated', False))


def _dominates(a, b, attrs) -> bool:
    va = [getattr(a, attr) for attr in attrs]
    vb = [getattr(b, attr) for attr in attrs]
    return all(x <= y for x, y in zip(va, vb)) and any(x < y for x, y in zip(va, vb))


def _read_only(result):
    """An ``Equation.evaluate`` memo entry ``(targets, features)`` with
    read-only views of its arrays."""
    def view(array):
        out = np.asarray(array).view()
        out.flags.writeable = False
        return out
    targets, features = result
    return ({key: view(value) for key, value in targets.items()},
            None if features is None else {key: view(value) for key, value in features.items()})


def _versions(equation, cap, coeff_calc, coeff_args, host, host_args) -> list:
    """The fitted versions of ``equation`` -- S, and S+{1} when the cap leaves
    room for the constant -- that the fitness host accepts, minus one the other
    dominates. Each is a fitted, pruned and scored copy, built exactly as
    ``right_part_selection._equation_variants`` builds a support."""
    from epde.operators.common.right_part_selection import (
        _pruned_to_reducible, _subset_rank_deficient, _subset_reducible)
    from epde.operators.common.subset_selection import (
        store_fit, subset_table, support_scores, trajectory_gram)
    from epde.operators.common.objectives import LOSS_NAN_VAL

    cfg = active_config()
    metric = cfg.objectives.instability_metric
    gram_mode = cfg.objectives.gram_mode
    discrepancy = cfg.objectives.discrepancy_metric
    amp_cap = cfg.search_space.rps_amplification_cap
    gfuncs = global_var.samples_manager.row_weights(slot_of(equation), 'dm')
    shapes = global_var.samples_manager.row_shapes(slot_of(equation))

    target, features = equation.evaluate()
    if features is None:
        return []
    finite = []
    for key in target.keys():
        X = np.asarray(features[key], dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(target[key], dtype=float).reshape(-1)
        sw = np.asarray(gfuncs[key], dtype=float).reshape(-1)
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            continue
        finite.append((key, X, y, sw))
    if not finite:
        return []
    p = finite[0][1].shape[1]
    every_feature = (1 << p) - 1
    masks = [every_feature] + ([every_feature | (1 << p)] if p + 1 <= cap else [])

    samples = []
    for key, X, y, sw in finite:
        gram = trajectory_gram(None, X, y, sw, shapes[key], gram_mode,
                               equation.main_var_to_explain, key)
        # The table's own solve gives each version's coefficients (what
        # ``subset_coefficients`` would solve again, byte for byte).
        disc_tab, amp_tab, _, coef_tab = subset_table(X, y, sw, discrepancy=discrepancy, subsets=masks,
                                                      return_coefs=True)
        samples.append((key, X, y, sw, gram, disc_tab, amp_tab, coef_tab))
    disc = np.mean([s[5] for s in samples], axis=0)
    amp = np.mean([s[6] for s in samples], axis=0)

    work_target = equation.target
    feature_terms = [term for i, term in enumerate(equation.structure) if i != equation.target_idx]
    X0 = samples[0][1]
    norms = np.linalg.norm(X0, axis=0)
    unit_cols = X0 / np.where(norms > 0, norms, 1.0)
    unit_constant = np.full(X0.shape[0], 1.0 / np.sqrt(X0.shape[0]))
    cols = list(range(p))

    inst_attr = host.objectives[1].value_attr if len(host.objectives) > 1 else None
    versions = []
    for position, S in enumerate(masks):
        with_constant = bool((S >> p) & 1)
        if not np.isfinite(disc[position]) or (amp_cap is not None and amp[position] > amp_cap):
            continue
        if any(norms[j] == 0 for j in cols) \
                or _subset_reducible(work_target, feature_terms, cols, with_constant) \
                or _subset_rank_deficient(unit_cols, unit_constant, cols, with_constant):
            continue
        coefs, sw_weights, vc_scores = [], {}, {}
        for key, X, y, sw, gram, disc_tab, amp_tab, coef_tab in samples:
            full = coef_tab[position]
            coefs.append(full)
            vc_scores[key], sw_weights[key] = support_scores(gram, full != 0, metric)
        variant = deepcopy(equation)
        # The same structure and target: the parent's evaluate() memo serves
        # the variant until a prune wipes it -- read-only, as the variant may
        # only read it.
        variant._eval_cache = {tgt: _read_only(result) for tgt, result in equation._eval_cache.items()}
        store_fit(variant, np.mean(np.stack(coefs, axis=1), axis=1),
                  sw_weights, vc_scores)
        coeff_calc.apply(variant, coeff_args)
        variant.remove_zero_terms()
        if _pruned_to_reducible(variant):
            continue
        host.apply(variant, host_args)
        _loop_stats.record('SoEqRPS.plain.fit', 1, 1)
        if variant.fitness_value is None or variant.fitness_value >= LOSS_NAN_VAL:
            continue
        # The host caps an infinite instability at the largest float, where the
        # assignment mode's table keeps inf and refuses the subset.
        if inst_attr is not None and not (np.isfinite(getattr(variant, inst_attr))
                                          and getattr(variant, inst_attr) < np.finfo(float).max):
            continue
        variant.right_part_selected = True
        versions.append(variant)

    if len(versions) == 2:
        attrs = [filler.value_attr for filler in host.objectives]
        if _dominates(versions[1], versions[0], attrs):
            versions = [versions[1]]
        elif _dominates(versions[0], versions[1], attrs):
            versions = [versions[0]]
    return versions


def apply_plain(objective, eq_selector, eq_args) -> None:
    """Fit every equation of ``objective`` (a SoEq) that needs it, in place.

    1. Targets. An equation that is still fitted keeps its target; the others
       re-find the term their gene names (:func:`gene_term`), then draw among
       their free own-derivative terms, restoring one when there is none. No
       two equations share a target.
    2. Every equation drops the terms equal to another equation's target and
       those related to its own target by division; a fitted equation that
       loses a term is refitted.
    3. Each equation to fit gets its versions (:func:`_versions`). With none,
       only that equation is marked degenerate, so the population operators
       re-roll it. Otherwise the first is installed, and the other
       combinations of the kept versions become siblings
       (:func:`take_siblings`) -- mutually non-dominated, since the equations
       they share are the same structures.
    4. An equation that follows from the system's others
       (``system_redundancy.redundant_equations``) is marked degenerate, so it
       is re-rolled; a sibling holding one is not handed over.
    """
    from epde.operators.common.right_part_selection import (
        _ensure_own_derivative, _mark_unassignable, _related_to_target)

    _, eq_subargs = eq_selector.parse_suboperator_args(arguments=eq_args)
    coeff_calc = eq_selector.suboperators['coeff_calc']
    host = eq_selector.suboperators['fitness_calculation']
    coeff_args = eq_subargs['coeff_calc']
    host_args = eq_subargs['fitness_calculation']
    space = active_config().search_space
    cap = plain_kept_terms_cap(space.equation_terms_max_number)

    keys = list(objective.vals.equation_keys)
    equations = list(objective)
    fitted = [_fitted(eq) for eq in equations]
    degenerate = [False] * len(equations)
    targets = [None] * len(equations)
    taken = set()

    # 1. Targets: fitted equations first, then the genes, then fresh draws.
    for i, eq in enumerate(equations):
        if fitted[i] and eq.target.factors_labels not in taken:
            targets[i] = eq.target
            taken.add(eq.target.factors_labels)
            if getattr(eq, '_plain_genes', None) is None:
                eq._plain_genes = target_gene(eq.target)
        else:
            fitted[i] = False
    for i, eq in enumerate(equations):
        if targets[i] is not None:
            continue
        if not _ensure_own_derivative(eq):
            degenerate[i] = True
            continue
        term = gene_term(eq, taken)
        if term is not None:
            targets[i] = term
            taken.add(term.factors_labels)
    for i, eq in enumerate(equations):
        if targets[i] is not None or degenerate[i]:
            continue
        options = own_derivative_terms(eq, taken)
        if not options:
            degenerate[i] = True
            continue
        targets[i] = options[np.random.randint(len(options))]
        taken.add(targets[i].factors_labels)

    # 2. The target rules, as removals.
    signatures = [None if term is None else term.factors_labels for term in targets]
    for i, eq in enumerate(equations):
        if degenerate[i]:
            continue
        others = {sig for j, sig in enumerate(signatures) if j != i and sig is not None}
        kept = [term for term in eq.structure
                if term is targets[i]
                or (term.factors_labels not in others and not _related_to_target(term, targets[i]))]
        if len(kept) != len(eq.structure):
            eq.structure = kept
            eq._invalidate_label_cache()
            fitted[i] = False
        if len(kept) < 2:
            degenerate[i] = True
            continue
        if not fitted[i]:
            eq.reset_state(True)
            _install_target(eq, targets[i])

    # 3. Fit.
    per_equation = []
    for i, eq in enumerate(equations):
        if degenerate[i]:
            per_equation.append([])
        elif fitted[i]:
            per_equation.append([eq])
        else:
            per_equation.append(_versions(eq, cap, coeff_calc, coeff_args, host, host_args))
    for i, versions in enumerate(per_equation):
        if not versions:
            _mark_unassignable([equations[i]], host)
            if targets[i] is not None:
                equations[i]._plain_genes = target_gene(targets[i])
    for key, eq, versions, was_fitted in zip(keys, equations, per_equation, fitted):
        if versions and not was_fitted:
            versions[0].metaparameters = eq.metaparameters
            objective.vals.replace_gene(gene_key=key, value=versions[0])

    siblings = []
    if all(per_equation) and any(len(versions) > 1 for versions in per_equation):
        for combo in itertools.product(*[range(len(versions)) for versions in per_equation]):
            if not any(combo):
                continue
            # The redundancy test first, on the equations the sibling would
            # hold in its order (``list(sibling)``): it reads their structures
            # and weights only, which the copies below keep, so a refused combo
            # is never copied.
            chosen = {key: versions[k] if k else objective.vals[key]
                      for key, versions, k in zip(keys, per_equation, combo)}
            if redundant_equations([chosen[key] for key in objective.vars_to_describe]):
                continue
            sibling = deepcopy(objective)
            for key, versions, k in zip(keys, per_equation, combo):
                if k:
                    variant = deepcopy(versions[k])
                    variant.metaparameters = sibling.vals[key].metaparameters
                    sibling.vals.replace_gene(gene_key=key, value=variant)
            siblings.append(sibling)
        _loop_stats.record('SoEqRPS.plain.siblings', len(siblings), len(siblings))
    # 4. An equation that follows from the others (system_redundancy) is re-rolled like one without a
    #    version; a sibling holding one was not kept above.
    if all(per_equation):
        for i in redundant_equations(list(objective)):
            _loop_stats.record('SoEqRPS.plain.redundant', 1, 1)
            _mark_unassignable([objective.vals[keys[i]]], host)
    _hand_over(objective, siblings)
