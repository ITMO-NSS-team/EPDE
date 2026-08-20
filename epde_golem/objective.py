"""Objective evaluation for the GOLEM-driven EPDE search.

The evaluation chain is EPDE's own, verbatim -- right-part selection,
sparsity-constrained regression, coefficient fitting, degenerate-gene
regeneration -- so a GOLEM run and a MOEA/D run score identical chromosomes
identically.  The only new thing here is the plumbing that exposes the result
as a GOLEM ``Objective``.
"""

from collections import OrderedDict
from copy import deepcopy

import numpy as np

from golem.core.optimisers.objective import Objective

from epde.operators.multiobjective.moeadd_specific import (
    regenerate_degenerate_equations, has_degenerate_equation)

from .graph import refresh_graph_mirror

#: Objective value assigned to a chromosome the EPDE chain could not evaluate.
PENALTY = 1e10


class SystemEvaluator:
    """Runs EPDE's evaluation chain on a ``SoEq`` and caches the objectives.

    Parameters
    ----------
    right_part_selector, chromosome_fitness :
        The very ``CompoundOperator`` instances the EPDE strategy director
        assembled for the native run (pulled off the director's blocks by
        :func:`epde_golem.optimizer.extract_epde_operators`).
    drop_degenerate :
        Whether a chromosome that stays fit-degenerate after regeneration is
        penalised (``True``) or scored as-is.  EPDE's MOEA/D drops such
        offspring outright; GOLEM has no "drop" channel inside the objective,
        so the equivalent is a penalty that keeps it off the Pareto front.
    """

    def __init__(self, right_part_selector, chromosome_fitness, n_objectives,
                 drop_degenerate: bool = True, metric_names=None,
                 cache_size: int = 0):
        self.rps = right_part_selector
        self.fitness = chromosome_fitness
        self.n_objectives = n_objectives
        self.metric_names = list(metric_names) if metric_names else None
        self.drop_degenerate = drop_degenerate
        self.n_evaluations = 0
        self.n_failures = 0
        #: Structure-keyed memo of already-evaluated chromosomes. EPDE's own
        #: ``OffspringUpdater`` keeps a duplicate history and skips the fitness
        #: call for a structure it has already placed; GOLEM has no equivalent,
        #: so an offspring that lands on a previously seen structure pays for
        #: the whole chain again -- including the right-part sweep, which alone
        #: costs one evaluation per candidate target term.
        self._cache = OrderedDict() if cache_size else None
        self._cache_size = cache_size
        self.n_cache_hits = 0
        #: Exceptions raised inside the EPDE chain. GOLEM's ``Objective``
        #: swallows them into a null fitness, which turns a wiring mistake
        #: into a silently empty Pareto front -- keep the first few so the
        #: caller can tell "hard problem" from "broken plumbing".
        self.errors = []

    def evaluate_system(self, soeq, fresh: bool = False) -> np.ndarray:
        """Evaluate ``soeq`` in place and return its objective vector."""
        self.n_evaluations += 1
        if fresh:
            soeq.reset_state(True)
        soeq.reset_moeadd_state()
        self.rps.apply(objective=soeq, arguments={})
        self.fitness.apply(objective=soeq, arguments={})
        regenerate_degenerate_equations(soeq, self.rps, self.fitness, {}, {})
        if self.drop_degenerate and has_degenerate_equation(soeq):
            self.n_failures += 1
            return np.full(self.n_objectives, PENALTY, dtype=float)
        values = np.asarray(soeq.obj_fun, dtype=float)
        values = np.where(np.isfinite(values), values, PENALTY)
        return values

    @staticmethod
    def _cache_key(soeq):
        """Structure plus the metaparameters the fit depends on.

        ``equations_labels`` alone is not enough: sparsity sets the threshold
        at which the regression prunes terms, so the same term set fitted at a
        different sparsity is a different candidate.
        """
        params = tuple(round(float(soeq.vals[key]), 12)
                       for key in soeq.vals.params_keys)
        return soeq.equations_labels, params

    def evaluate_graph(self, graph) -> np.ndarray:
        """Evaluate the chromosome carried by ``graph`` and refresh its mirror.

        The fitness chain physically prunes zero-weight terms, so the node
        mirror is rebuilt afterwards -- GOLEM's dedup and diversity checks
        must see the ACTIVE structure, the same structure EPDE's own
        ``equations_labels`` history keys on.
        """
        key = None
        if self._cache is not None:
            try:
                key = self._cache_key(graph.soeq)
            except Exception:
                key = None
            if key is not None and key in self._cache:
                values, template = self._cache[key]
                self._cache.move_to_end(key)
                self.n_cache_hits += 1
                # Hand back a private copy: the caller mutates its chromosome.
                graph.soeq = deepcopy(template)
                graph.obj_values = tuple(values)
                refresh_graph_mirror(graph)
                return np.asarray(values, dtype=float)
        try:
            values = self.evaluate_system(graph.soeq)
        except Exception:
            if len(self.errors) < 5:
                import traceback
                self.errors.append(traceback.format_exc())
            raise
        graph.obj_values = tuple(values)
        refresh_graph_mirror(graph)
        if key is not None:
            self._cache[key] = (tuple(values), deepcopy(graph.soeq))
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return values


def build_objective(evaluator: SystemEvaluator, metric_names,
                    multi_objective: bool = None) -> Objective:
    """Wrap ``evaluator`` into a GOLEM ``Objective`` over ``metric_names``.

    Only the first metric triggers the (expensive) evaluation; the rest read
    the cached vector.  GOLEM calls the metrics back-to-back on the same
    graph, so one evaluation per candidate is all that happens.

    With ``multi_objective=False`` and several metrics the extra values become
    ``SingleObjFitness`` supplementary values, i.e. lexicographic tie-breakers
    behind the primary one -- which is what EPDE's own single-objective mode
    does when a system has several equations.
    """
    if multi_objective is None:
        multi_objective = len(metric_names) > 1

    def make_metric(idx: int):
        def metric(graph):
            # ``obj_values`` is the cache. Every operator that changes a
            # chromosome clears it, so a non-None value always belongs to the
            # chromosome currently on the graph -- and re-evaluating would only
            # repeat work (or, for the vetted initial population, pay twice).
            if graph.obj_values is None:
                evaluator.evaluate_graph(graph)
            return float(graph.obj_values[idx])
        metric.__name__ = str(metric_names[idx])
        return metric

    quality_metrics = {name: make_metric(idx) for idx, name in enumerate(metric_names)}
    return Objective(quality_metrics=quality_metrics,
                     is_multi_objective=bool(multi_objective))
