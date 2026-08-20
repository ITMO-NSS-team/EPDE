r"""Bridge between EPDE's ``SoEq`` chromosome and GOLEM's ``OptGraph``.

GOLEM operates on directed acyclic graphs.  An EPDE candidate is a *system of
equations*: one ``Equation`` per dependent variable, each equation being a set
of ``Term`` objects (products of ``Factor``s) with one term marked as the right
(target) part.

The mapping used here is a faithful structural mirror:

    term_1 ... term_k   (source nodes, one per term of the equation)
             \  |  /
            eq::<var>   (one node per equation)
                |
            system      (single sink node -- makes the graph connected)

The chromosome itself travels on the graph object (``SoEqGraph.soeq``) rather
than inside node ``params``: GOLEM builds ``descriptive_id`` from node *name*
and *params* only, so keeping the payload out of ``params`` gives a
descriptive id that is exactly the structural signature of the system --
which is what GOLEM's dedup / structural-diversity logic wants.
"""

from copy import deepcopy
from typing import Optional

from golem.core.optimisers.graph import OptGraph, OptNode


def _term_signature(equation, term_idx: int) -> str:
    """Stable textual signature of a term, used as the graph node name."""
    term = equation.structure[term_idx]
    try:
        name = term.name
    except Exception:                                    # pragma: no cover
        name = f'term_{term_idx}'
    if term_idx == getattr(equation, 'target_idx', None):
        name = f'[target]{name}'
    return name


class SoEqGraph(OptGraph):
    """``OptGraph`` that carries an EPDE ``SoEq`` as its payload."""

    def __init__(self, nodes=(), soeq=None):
        super().__init__(nodes)
        self.soeq = soeq
        # Objectives computed by the EPDE evaluation chain; filled in by
        # ``objective.evaluate_system`` and read by the metric callables.
        self.obj_values: Optional[tuple] = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            setattr(new, key, deepcopy(value, memo))
        return new


def _build_nodes(soeq):
    """Build the node mirror of ``soeq``. Returns the list of all nodes."""
    eq_nodes = []
    all_nodes = []
    for var in soeq.vars_to_describe:
        equation = soeq.vals[var]
        term_nodes = []
        for term_idx in range(len(equation.structure)):
            node = OptNode(content={'name': _term_signature(equation, term_idx)})
            term_nodes.append(node)
        eq_node = OptNode(content={'name': f'eq::{var}'}, nodes_from=term_nodes)
        all_nodes.extend(term_nodes)
        all_nodes.append(eq_node)
        eq_nodes.append(eq_node)
    system_node = OptNode(content={'name': 'system'}, nodes_from=eq_nodes)
    all_nodes.append(system_node)
    return all_nodes


def soeq_to_graph(soeq) -> SoEqGraph:
    """Wrap an EPDE ``SoEq`` into a GOLEM-consumable graph."""
    return SoEqGraph(nodes=_build_nodes(soeq), soeq=soeq)


def refresh_graph_mirror(graph: SoEqGraph) -> SoEqGraph:
    """Rebuild the node mirror after the payload chromosome changed in place.

    EPDE's operators mutate the ``SoEq`` directly (and the fitness chain
    physically prunes zero-weight terms), so the node mirror has to be
    re-derived, otherwise GOLEM's ``descriptive_id`` -- and with it dedup,
    structural-diversity checks and history -- would describe a stale system.
    """
    graph.nodes = _build_nodes(graph.soeq)
    return graph
