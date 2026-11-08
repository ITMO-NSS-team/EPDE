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

Plain-mode settings (``search_space.system_rps_params``; the ``'auto'`` values
resolve by ``search_config.resolve_plain_params``: on under
``evolution.engine='nsga2'``, off otherwise):

* ``plain_target`` -- every target choice is restricted to a PLAIN
  own-derivative token: one factor, a derivative of the equation's own
  variable (any axis and order), power 1, no other parameter
  (:func:`is_plain_term`). Decorated terms (squares, products, trig-weighted
  derivatives) stay allowed on the right-hand side; a gene drifting onto one
  no longer lets a near relation of the data pose as a law.
* ``variable_order='random'`` -- the order in which the equations claim
  their targets (step 1) and are tested for redundancy (step 4) is a random
  permutation per fit, drawn from a generator of its own
  (:func:`start_variable_order`), so the global random stream moves only when
  the order changes an outcome. With the fixed order, the first equation
  always wins a contested target and always loses a restated relation.
* ``product_rule_decline`` -- a version solved for one member of a
  product-rule pair is declined (``product_rule``).
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
    derivative of its own variable (plain ones only under ``plain_target``,
    :func:`is_plain_term`), minus the signatures in ``taken``."""
    var = equation.main_var_to_explain
    plain = plain_params()['plain_target']
    return [term for term in equation.structure
            if term.contains_deriv(var) and term.factors_labels not in taken
            and (not plain or is_plain_term(term, var))]


def plain_params() -> dict:
    """The active plain-mode settings, ``'auto'`` resolved."""
    from epde.interface.search_config import resolve_plain_params
    return resolve_plain_params(active_config())


def is_plain_term(term, var) -> bool:
    """A plain own-derivative token of ``var``: a single factor that is a
    derivative of ``var`` (any axis, any order), with power 1 (read off the
    factor's parameter, never off a label) and no other parameter."""
    if len(term.structure) != 1 or not term.contains_deriv(var):
        return False
    factor = term.structure[0]
    names = {d['name']: float(factor.params[i]) for i, d in factor.params_description.items()}
    return set(names) <= {'power'} and names.get('power', 1.0) == 1.0


#: (id(pool), variable) -> (weak reference to the pool, whether its library holds a plain own-derivative
#: token). The reference is checked on lookup: a new pool at a freed pool's address is checked anew.
_PLAIN_TOKENS = {}
#: ``restore_property`` attempts before an equation without a plain target is given up (as
#: ``right_part_selection._ensure_own_derivative``).
ENSURE_MAX_ITER = 100


def _check_plain_tokens(equation) -> None:
    """Fail loud when the pool can draw no plain own-derivative token for the equation's variable: under
    ``plain_target`` such an equation could never have a target."""
    var = equation.main_var_to_explain
    pool = equation.pool
    key = (id(pool), var)
    entry = _PLAIN_TOKENS.get(key)
    if entry is not None and entry[0]() is pool:
        return
    from epde.operators.common.fill_prune import factors_setting, library_terms
    library = library_terms(pool, factors_setting(equation))
    found = any(is_plain_term(term, var) for term in library)
    _PLAIN_TOKENS[key] = (weakref.ref(pool), found)
    if not found:
        raise ValueError(
            "system_rps_params['plain_target']: the token pool can draw no plain derivative of {0!r} (a single "
            "derivative factor of power 1), so no equation for it could have a target. Add such a token or set "
            "plain_target=False.".format(var))


def _ensure_target(equation, plain: bool) -> bool:
    """Give ``equation`` a term that can be its target, through the same
    ``restore_property`` step ``right_part_selection._ensure_own_derivative``
    uses. Under ``plain_target`` the term must be plain: an equation without
    one draws an own derivative up to :data:`ENSURE_MAX_ITER` times, its
    structure rolled back after every draw that brought no plain one. False
    when none appeared."""
    from epde.operators.common.right_part_selection import _ensure_own_derivative
    if not plain:
        return _ensure_own_derivative(equation)
    var = equation.main_var_to_explain
    _check_plain_tokens(equation)
    if any(is_plain_term(term, var) for term in equation.structure):
        return True
    snapshot = list(equation.structure)
    for _ in range(ENSURE_MAX_ITER):
        equation.restore_property(mandatory_family=False, deriv=True)
        if any(is_plain_term(term, var) for term in equation.structure):
            return True
        equation.structure = list(snapshot)
        equation._invalidate_label_cache()
    return False


# ------------------------------------------------------------------ the variable order
#: The variable order's generator: one per fit (:func:`start_variable_order`).
_ORDER = {'rng': None}
_ORDER_TAG = 0x5256


def start_variable_order(seed=None) -> None:
    """A new generator for ``variable_order='random'``: from ``seed``, or, for
    None, from the global random state (read, not drawn from)."""
    if seed is None:
        entropy = [int(x) for x in np.random.get_state()[1]] + [_ORDER_TAG]
    else:
        entropy = [int(seed), _ORDER_TAG]
    _ORDER['rng'] = np.random.default_rng(entropy)


def _claims(equations, fitted, order):
    """Step 1's deterministic claims under ``order``, read without changing
    anything: (kind per equation -- ('fit' | 'gene', signature) or ('ensure',
    None) for an equation holding no own derivative --, the signatures taken,
    the equations left to draw, the 'ensure' ones)."""
    taken, kinds = set(), {}
    fit = list(fitted)
    for i in order:
        eq = equations[i]
        if fit[i]:
            sig = eq.target.factors_labels
            if sig not in taken:
                kinds[i] = ('fit', sig)
                taken.add(sig)
                continue
        fit[i] = False
    for i in order:
        if i in kinds:
            continue
        eq = equations[i]
        var = eq.main_var_to_explain
        if not any(term.contains_deriv(var) for term in eq.structure):
            kinds[i] = ('ensure', None)
            continue
        term = gene_term(eq, taken)
        if term is not None:
            kinds[i] = ('gene', term.factors_labels)
            taken.add(term.factors_labels)
    draw = [i for i in order if i not in kinds]
    ensure = [i for i in order if i in kinds and kinds[i][0] == 'ensure']
    return kinds, taken, draw, ensure


def _variable_order(equations, fitted, params):
    """(order of step 1, order of step 4 or None) for one plain fit.

    Under ``variable_order='random'`` a permutation is drawn for every fit of
    two or more equations. Step 4 tests in that order. Step 1 uses it only
    where it can matter -- its deterministic claims differ from the fixed
    order's, or two equations left to draw could draw the same term and the
    permutation reorders them, or an equation without an own derivative needs
    the drawing repair -- and the fixed order otherwise: every outcome the
    permutation could give then has the same law, and the global random stream
    stays the fixed order's."""
    n = len(equations)
    ident = list(range(n))
    if n < 2 or params['variable_order'] != 'random':
        return ident, None
    if _ORDER['rng'] is None:
        start_variable_order(params['variable_order_seed'])
    kinds, taken, draw, ensure = _claims(equations, fitted, ident)
    overlap = False
    if len(draw) > 1:
        seen = {}
        for i in draw:
            for term in own_derivative_terms(equations[i], taken):
                seen.setdefault(term.factors_labels, set()).add(i)
        overlap = any(len(owners) > 1 for owners in seen.values())
    perm = [int(x) for x in _ORDER['rng'].permutation(n)]
    kinds_pi = _claims(equations, fitted, perm)[0] if perm != ident else kinds
    changed = any(kinds_pi.get(i) != kinds.get(i) for i in range(n))
    reorders = overlap and [i for i in perm if i in draw] != draw
    unmodelled = bool(ensure) and len(draw) + len(ensure) >= 2 and perm != ident
    return (perm if (changed or reorders or unmodelled) else ident), perm


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
    from epde.operators.common.product_rule import declines

    cfg = active_config()
    decline = plain_params()['product_rule_decline']
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
        # A version the candidate rules refuse, or one solved for a member of a
        # product-rule pair (product_rule_decline), is not built.
        if _pruned_to_reducible(variant) or declines(variant, decline):
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
       two equations share a target. The equations claim in the order of
       :func:`_variable_order`.
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
       (``system_redundancy.redundant_equations``, tested in the order of
       :func:`_variable_order`) is marked degenerate, so it is re-rolled; a
       sibling holding one is not handed over.
    """
    from epde.operators.common.right_part_selection import _mark_unassignable, _related_to_target

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
    params = plain_params()
    order, test_order = _variable_order(equations, fitted, params)

    # 1. Targets: fitted equations first, then the genes, then fresh draws.
    for i in order:
        eq = equations[i]
        if fitted[i] and eq.target.factors_labels not in taken:
            targets[i] = eq.target
            taken.add(eq.target.factors_labels)
            if getattr(eq, '_plain_genes', None) is None:
                eq._plain_genes = target_gene(eq.target)
        else:
            fitted[i] = False
    for i in order:
        eq = equations[i]
        if targets[i] is not None:
            continue
        if not _ensure_target(eq, params['plain_target']):
            degenerate[i] = True
            continue
        term = gene_term(eq, taken)
        if term is not None:
            targets[i] = term
            taken.add(term.factors_labels)
    for i in order:
        eq = equations[i]
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
        for i in redundant_equations(list(objective), order=test_order):
            _loop_stats.record('SoEqRPS.plain.redundant', 1, 1)
            _mark_unassignable([objective.vals[keys[i]]], host)
    _hand_over(objective, siblings)
