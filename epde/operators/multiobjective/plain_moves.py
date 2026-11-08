"""``search_space.system_rps_mode='plain'``: the equation mutation.

A plain candidate is fitted on exactly the terms it holds, so the mutation is
what selects the support. Each call makes ONE move, drawn uniformly among the
moves that apply to the equation:

* ``swap``   -- one non-target term is re-drawn (re-rolled while it duplicates
  another term, and left as it was when that does not stop);
* ``add``    -- one new term, while the equation keeps fewer than the plain
  cap (``search_config.plain_kept_terms_cap``) besides its target;
* ``drop``   -- one non-target term, while at least two remain;
* ``target`` -- the target gene moves to another term holding a derivative of
  the equation's own variable;
* ``nudge``  -- one continuous, ungridded parameter of a non-target term moves
  by a Gaussian step (``TermParameterMutation``'s ``multiplier`` of its range).

The target is the term the equation's gene names (``plain_fit.gene_term``),
so swap, drop and nudge never touch it. The constant has no move: the fit
evaluates every candidate with and without it.

Two settings of ``search_space.system_rps_params`` add moves (on by default
under ``evolution.engine='nsga2'``, ``search_config.resolve_plain_params``):

* ``factor_move_probability`` -- with this probability, drawn first, one
  factor of a non-target term is edited instead: its power moves by +-1
  within its bounds, a random single factor joins a one-factor term, or one
  factor of a two-factor term leaves. An edit that changes nothing or
  duplicates another term is undone and the move above runs.
* ``fill_probability`` -- with this probability, drawn after the move, the
  move is replaced by the fill-and-prune move (``fill_prune``); when that
  installs nothing, the drawn move runs.
"""
from contextlib import contextmanager

import numpy as np

from epde import _loop_stats
from epde.interface.search_config import active_config, plain_kept_terms_cap
from epde.operators.common.plain_fit import gene_term, own_derivative_terms, plain_params, target_gene

#: Re-draws of a swapped term that keeps duplicating another (TermMutation's).
MAX_REGEN_RETRIES = 3
#: Draws of a nudge step before giving up on a parameter that does not move.
MAX_NUDGE_RETRIES = 10
_FIXED_PARAMS = ('dim', 'power')


def _continuous_params(term) -> list:
    """``(factor, param_idx)`` of every continuous, ungridded parameter of ``term``."""
    found = []
    for factor in term.structure:
        for param_idx, description in factor.params_description.items():
            lo, hi = description['bounds'][0], description['bounds'][1]
            if description['name'] in _FIXED_PARAMS or 'grid' in description:
                continue
            if isinstance(lo, (float, np.floating)) and hi > lo:
                found.append((factor, param_idx))
    return found


def _duplicated(equation) -> bool:
    equation._invalidate_label_cache()
    return len({term.factors_labels for term in equation.structure}) != len(equation.structure)


def _swap(equation, term) -> bool:
    original = term.structure
    for _ in range(MAX_REGEN_RETRIES):
        term.randomize()
        term.resetSavedState()
        if not _duplicated(equation):
            return True
    term.structure = original
    term.resetSavedState()
    equation._invalidate_label_cache()
    return False


def _nudge(equation, term) -> bool:
    options = _continuous_params(term)
    factor, param_idx = options[np.random.randint(len(options))]
    lo, hi = factor.params_description[param_idx]['bounds'][:2]
    multiplier = active_config().evolution.operators.get(
        'TermParameterMutation', {}).get('multiplier', 0.1)
    before = factor.params.copy()
    # A step inside the parameter's equality bucket names the same factor (and
    # the same cached column), so only a step that changes the label counts.
    before_label = factor.structural_label
    for _ in range(MAX_NUDGE_RETRIES):
        params = before.copy()
        params[param_idx] = float(np.clip(
            params[param_idx] + np.random.normal(0.0, multiplier * (hi - lo)), lo, hi))
        factor.params = params
        term.resetSavedState()
        if factor.structural_label != before_label and not _duplicated(equation):
            return True
        factor.params = before
        term.resetSavedState()
        equation._invalidate_label_cache()
    return False


#: The system ``SystemMutation`` is mutating (the fill reads the other equations' targets from it).
_MUTATING = {'system': None}


@contextmanager
def mutating_system(system):
    """Name ``system`` as the one whose equations are being mutated."""
    previous = _MUTATING['system']
    _MUTATING['system'] = system
    try:
        yield
    finally:
        _MUTATING['system'] = previous


def _power_step(term) -> bool:
    factor = term.structure[np.random.randint(len(term.structure))]
    for idx, description in factor.params_description.items():
        if description['name'] == 'power':
            lo, hi = description['bounds'][:2]
            new = factor.params[idx] + np.random.choice([-1.0, 1.0])
            if lo <= new <= hi:
                params = factor.params.copy()
                params[idx] = new
                factor.params = params
                return True
    return False


def _add_factor(term) -> bool:
    from epde.structure.main_structures import Term
    if len(term.structure) >= 2:
        return False
    extra = Term(term.pool, max_factors_in_term=1).structure[0]
    if extra.label in {f.label for f in term.structure}:
        return False
    term.structure = term.structure + [extra]
    return True


def _drop_factor(term) -> bool:
    if len(term.structure) < 2:
        return False
    victim = np.random.randint(len(term.structure))
    term.structure = [f for k, f in enumerate(term.structure) if k != victim]
    return True


_FACTOR_MOVES = (('power', _power_step), ('add', _add_factor), ('drop', _drop_factor))


def _factor_move(equation) -> bool:
    """One factor edit of a random non-target term (module docstring); True when it changed the equation,
    which is then reset for a new fit, False (everything undone) otherwise."""
    target = gene_term(equation)
    features = [t for t in equation.structure if t is not target]
    if not features:
        return False
    term = features[np.random.randint(len(features))]
    kind, move = _FACTOR_MOVES[np.random.randint(len(_FACTOR_MOVES))]
    original = list(term.structure)
    saved = [(f, f.params.copy()) for f in original]
    changed = move(term)
    term.resetSavedState()
    if changed and _duplicated(equation):
        changed = False
    if not changed:
        term.structure = original
        for f, params in saved:
            f.params = params
        term.resetSavedState()
        equation._invalidate_label_cache()
        return False
    _loop_stats.record('PlainMutation.factor_' + kind, 1, 1)
    equation._invalidate_label_cache()
    equation.reset_state(reset_right_part=True)
    return True


def plain_mutation(equation):
    """One plain-mode move on ``equation``, in place; the equation is reset
    for a new fit when the move changed it (its target gene survives)."""
    params = plain_params()
    if params['factor_move_probability'] > 0 and np.random.random() < params['factor_move_probability']:
        if _factor_move(equation):
            return equation
    space = active_config().search_space
    cap = plain_kept_terms_cap(space.equation_terms_max_number)
    target = gene_term(equation)
    features = [term for term in equation.structure if term is not target]
    owners = own_derivative_terms(equation)

    moves = []
    if features:
        moves.append('swap')
    if len(features) + 1 <= cap:
        moves.append('add')
    if len(features) >= 2:
        moves.append('drop')
    if target is not None and len(owners) >= 2:
        moves.append('target')
    nudgeable = [term for term in features if _continuous_params(term)]
    if nudgeable:
        moves.append('nudge')
    if not moves:
        return equation
    move = moves[np.random.randint(len(moves))]
    if params['fill_probability'] > 0:
        from epde.operators.common.fill_prune import fill
        if fill(equation, target, features, _MUTATING['system'], params):
            move = 'fill'

    if move == 'fill':
        changed = True
    elif move == 'swap':
        changed = _swap(equation, features[np.random.randint(len(features))])
    elif move == 'add':
        changed = equation.add_random_term()
    elif move == 'drop':
        victim = features[np.random.randint(len(features))]
        equation.structure = [term for term in equation.structure if term is not victim]
        equation._invalidate_label_cache()
        changed = True
    elif move == 'target':
        others = [term for term in owners if term is not target]
        equation._plain_genes = target_gene(others[np.random.randint(len(others))])
        changed = True
    else:
        changed = _nudge(equation, nudgeable[np.random.randint(len(nudgeable))])
    _loop_stats.record('PlainMutation.' + move, int(changed), 1)

    assert len(equation.terms_labels) == len(equation.structure)
    if changed:
        equation.reset_state(reset_right_part=True)
    return equation
