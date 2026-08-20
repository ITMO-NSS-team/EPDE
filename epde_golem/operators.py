"""GOLEM-shaped wrappers around EPDE's own structural operators.

Nothing about the search *space* changes here: the mutation is EPDE's
``SystemMutation`` (per-equation term replacement / term addition plus
metaparameter jitter) and the crossover is EPDE's ``ChromosomeCrossover``
(uniform exchange of equation genes with term-level recombination).  They are
merely given the call signatures GOLEM's ``Mutation`` / ``Crossover``
operators expect, so GOLEM's population machinery can drive them.
"""

import numpy as np

from golem.core.adapter import register_native

from .graph import SoEqGraph, refresh_graph_mirror


def make_mutation(system_mutation, name='epde_system_mutation'):
    """Adapt EPDE's chromosome-level mutation to GOLEM's mutation protocol."""

    @register_native
    def epde_mutation(graph: SoEqGraph, **kwargs) -> SoEqGraph:
        soeq = graph.soeq
        # The single-objective SystemMutation skips anything tagged
        # 'immutable' and reads the attribute unguarded; in the native
        # pipeline it is stamped by the elitism stage, which GOLEM replaces.
        if not hasattr(soeq, 'elite'):
            soeq.elite = 'non-elite'
        # The multi-objective SystemMutation edits in place and returns the
        # same object; the single-objective one deepcopies and returns a new
        # one. Take whatever comes back. (GOLEM already handed over a private
        # deepcopy of the graph, so either is safe.)
        graph.soeq = system_mutation.apply(objective=soeq, arguments={}) or soeq
        graph.obj_values = None
        return refresh_graph_mirror(graph)

    epde_mutation.__name__ = name
    return epde_mutation


def make_add_term_mutation(name='epde_add_term'):
    """A structural "grow" mutation: append one random term to a random equation.

    EPDE's ``EquationMutation`` folds term addition into the same operator as
    term replacement (governed by ``term_addition_prob``).  Exposing growth as
    a separate GOLEM action lets GOLEM's adaptive-mutation agent learn *when*
    growing pays off -- a capability the native engine does not have.
    """

    @register_native
    def epde_add_term(graph: SoEqGraph, **kwargs) -> SoEqGraph:
        soeq = graph.soeq
        keys = list(soeq.vals.equation_keys)
        equation = soeq.vals[keys[np.random.randint(len(keys))]]
        # add_random_term enforces the max_terms_number cap itself and refuses
        # to create a duplicate signature, returning False in either case. The
        # explicit cap check just avoids building a Term that would be thrown
        # away. When nothing is added the graph comes back structurally
        # unchanged, and GOLEM discards the individual.
        max_terms = equation.metaparameters['max_terms_number']['value']
        if len(equation.structure) < max_terms:
            if equation.add_random_term():
                equation.reset_state(reset_right_part=True)
        graph.obj_values = None
        return refresh_graph_mirror(graph)

    epde_add_term.__name__ = name
    return epde_add_term


def make_drop_term_mutation(name='epde_drop_term'):
    """A structural "shrink" mutation: delete one non-target term.

    EPDE's own mutation only *replaces* terms (and drops one as a last-resort
    dedup fallback), so nothing in the native operator set moves a candidate
    down the complexity axis on purpose. On a two-objective front where the
    second axis rewards parsimony this is the missing direction.
    """

    @register_native
    def epde_drop_term(graph: SoEqGraph, **kwargs) -> SoEqGraph:
        soeq = graph.soeq
        keys = list(soeq.vals.equation_keys)
        equation = soeq.vals[keys[np.random.randint(len(keys))]]
        # Keep the two-term floor and never touch the immutable head terms
        # (the right-part anchor and any mandatory-family term).
        removable = list(range(equation.n_immutable, len(equation.structure)))
        if len(equation.structure) <= 2 or not removable:
            return graph
        victim = equation.structure[removable[np.random.randint(len(removable))]]
        equation.structure = [t for t in equation.structure if t is not victim]
        equation._invalidate_label_cache()
        equation.reset_state(reset_right_part=True)
        graph.obj_values = None
        return refresh_graph_mirror(graph)

    epde_drop_term.__name__ = name
    return epde_drop_term


def make_equation_reroll_mutation(name='epde_reroll_equation'):
    """A macro-mutation: randomize one whole equation of the system.

    A restart-in-place. Term-level mutation explores locally; on a landscape
    whose basins are separated by several simultaneous term changes -- which is
    what a different physical law looks like -- local moves rarely cross
    between them.
    """

    @register_native
    def epde_reroll_equation(graph: SoEqGraph, **kwargs) -> SoEqGraph:
        soeq = graph.soeq
        keys = list(soeq.vals.equation_keys)
        equation = soeq.vals[keys[np.random.randint(len(keys))]]
        equation.randomize()
        equation.reset_state(reset_right_part=True)
        graph.obj_values = None
        return refresh_graph_mirror(graph)

    epde_reroll_equation.__name__ = name
    return epde_reroll_equation


def make_sparsity_mutation(param_mutation, name='epde_sparsity_jitter'):
    """Perturb only the sparsity metaparameter, leaving the structure alone.

    Sparsity sets the threshold at which the regression prunes terms, so it
    controls how many terms survive -- it moves a candidate along the
    complexity axis without any structural edit. EPDE folds this into the same
    operator as term replacement; separating it lets GOLEM's bandit agent
    learn when a pure threshold change is the productive move.
    """

    @register_native
    def epde_sparsity_jitter(graph: SoEqGraph, **kwargs) -> SoEqGraph:
        soeq = graph.soeq
        keys = list(soeq.vals.params_keys)
        if not keys:
            return graph
        key = keys[np.random.randint(len(keys))]
        altered = param_mutation.apply(objective=soeq.vals[key], arguments={})
        soeq.vals.replace_gene(gene_key=key, value=altered)
        soeq.vals.pass_parametric_gene(key=key, value=altered)
        for equation in soeq.vals:
            equation.reset_state(reset_right_part=True)
        graph.obj_values = None
        return refresh_graph_mirror(graph)

    epde_sparsity_jitter.__name__ = name
    return epde_sparsity_jitter


def make_crossover(chromosome_crossover, name='epde_chromosome_crossover'):
    """Adapt EPDE's chromosome-level crossover to GOLEM's crossover protocol."""

    @register_native
    def epde_crossover(graph_first: SoEqGraph, graph_second: SoEqGraph,
                       **kwargs):
        # GOLEM deepcopies both parents before the call, matching the
        # ownership contract EPDE's ParetoLevelsCrossover establishes.
        first, second = chromosome_crossover.apply(
            objective=(graph_first.soeq, graph_second.soeq), arguments={})
        graph_first.soeq, graph_second.soeq = first, second
        graph_first.obj_values = graph_second.obj_values = None
        return refresh_graph_mirror(graph_first), refresh_graph_mirror(graph_second)

    epde_crossover.__name__ = name
    return epde_crossover


@register_native
def soeq_is_valid(graph) -> bool:
    """Verification rule: the chromosome must carry at least two terms per
    equation (EPDE's own floor -- a single-term equation has no left-hand
    side to fit) and no duplicate term signatures."""
    soeq = getattr(graph, 'soeq', None)
    if soeq is None:
        raise ValueError('Graph does not carry an EPDE chromosome')
    for equation in soeq.vals:
        if len(equation.structure) < 2:
            raise ValueError('Equation collapsed below the two-term floor')
        signatures = {term.factors_labels for term in equation.structure}
        if len(signatures) != len(equation.structure):
            raise ValueError('Equation contains duplicate terms')
    return True
