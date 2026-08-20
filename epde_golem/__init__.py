"""GOLEM-backed evolutionary optimizer for the EPDE equation-discovery framework.

The package swaps EPDE's native population engines -- ``MOEADDOptimizer``
(multi-objective, MOEA/D-DD) and ``SimpleOptimizer`` (single-objective) --
for GOLEM's ``EvoGraphOptimizer`` while keeping EVERY domain-specific
component of EPDE untouched:

* the candidate representation (``SoEq`` -- a system of ``Equation`` objects),
* the initial-population constructor (``SystemsPopulationConstructor``),
* the structural operators (``SystemMutation`` / ``ChromosomeCrossover``),
* the evaluation chain (right-part selection -> sparsity -> coefficient
  regression -> objective readers).

Only the *population-level* machinery differs: sector-wise PBI decomposition
with a Gale-Shapley weight assignment (EPDE) versus SPEA-2 selection with
adaptive operator probabilities and an unbounded Pareto archive (GOLEM).
That makes the two configurations directly comparable.
"""

from .graph import SoEqGraph, soeq_to_graph, refresh_graph_mirror
from .optimizer import GolemEpdeOptimizer, EpdeGolemSearch

__all__ = ['SoEqGraph', 'soeq_to_graph', 'refresh_graph_mirror',
           'GolemEpdeOptimizer', 'EpdeGolemSearch']
