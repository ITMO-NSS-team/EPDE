#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:57:44 2022

@author: maslyaev
"""

import numpy as np
from typing import Callable

from epde.structure.main_structures import SoEq, COMPLEXITY_METAPARAMETER, is_complexity_key

class SystemsPopulationConstructor(object):
    """
    Class for creating population according to the current optimization method/

    Attributes:
        pool (``):
        terms_number (`int`):
        max_factors_in_term (`int`):
        vars_demand_equation (``):
        sparsity_internal (`tuple`):
        complexity_interval (`tuple` or None): integer range of the
            ``('complexity', var)`` budget, or None for no budget at all.
    """
    def __init__(self, pool, second_objective: str = None, terms_number : int = 8,
                 max_factors_in_term : int = 2,
                 obj_functions : Callable = None,
                 sparsity_interval : tuple = (1., 1.),
                 complexity_interval : tuple = None):
        self.pool = pool
        # Which objective occupies the second Pareto axis, or None to defer to
        # the ``epde.globals.second_objective`` global. Replaces the former
        # boolean flag, whose only meaning was this same choice.
        self.second_objective = second_objective 
        self.terms_number = terms_number
        self.max_factors_in_term = max_factors_in_term 
        self.vars_demand_equation = set([family.variable for family in self.pool.families_demand_equation])
        # Log-uniform seeding range for the ('sparsity', var) metaparameter.
        # EpdeSearch.fit always passes it, taking the value from the sparsity
        # operator in use (LASSOSparsity.initial_sparsity_interval) -- only
        # that operator reads the metaparameter. The degenerate default seeds
        # the neutral 1.0 rather than the old (0, 1), whose log10(0) made every
        # unseeded alpha exactly 0 through a -inf.
        self.sparsity_interval = sparsity_interval
        # Integer range (1, Cmax) of the per-equation ('complexity', var)
        # budget -- a property of its own, not a reading of the sparsity alpha.
        # EpdeSearch.fit passes it only when the sparsity operator selects by a
        # budget (KneeSparsity.complexity_interval); None adds no key and draws
        # no random number, so every other run is seeded exactly as before.
        self.complexity_interval = (None if complexity_interval is None
                                    else tuple(int(end) for end in complexity_interval))
        print('self.vars_demand_equation', self.vars_demand_equation)

    def applyToPassed(self, passed_solution: SoEq, **kwargs):
        try:
            passed_solution.set_objective_functions(kwargs['obj_funs'])
        except KeyError:
            passed_solution.use_default_multiobjective_function(self.second_objective)
        # Reconciled in BOTH directions: ``ChromosomeCrossover`` asserts that
        # both parents share one encoding, so a passed system must carry the
        # budget gene exactly when the seeded ones do.
        if self.complexity_interval is not None:
            self._give_budget(passed_solution)
        else:
            self._drop_budget(passed_solution)

    def _give_budget(self, solution: SoEq):
        """Give a passed system the budget gene the seeded ones carry, under
        THIS search's bounds.

        A passed system without ``('complexity', var)`` -- a translated truth,
        say -- would end the search at its first crossover. It joins at the
        CEILING, the loosest budget, which leaves any equation within the
        search's term and factor caps its full support; the mutation then
        moves it like any other.

        A system that already carries the key -- one from an earlier budget
        run -- keeps its value, but read as the integer level it stands for
        and clipped into this search's ``(1, Cmax)``, and its ``'bounds'`` are
        replaced by those: the mutation reflects its step into the stored
        bounds, so an earlier run's wider range would let this system's budget
        wander past the levels every seeded system is confined to.
        """
        # Deferred: mutations imports moeadd.moeadd, which imports this module.
        from epde.operators.multiobjective.mutations import _budget_level
        lo, hi = self.complexity_interval
        rebuild = False
        for variable in solution.vars_to_describe:
            key = (COMPLEXITY_METAPARAMETER, variable)
            current = solution.metaparameters.get(key)
            value = hi if current is None else _budget_level(current.get('value', hi), lo, hi)
            entry = {'optimizable': True, 'value': value, 'bounds': (lo, hi)}
            if current != entry:
                # A fresh entry rather than an in-place edit: a passed entry
                # may be shared with a system this search does not own.
                solution.metaparameters[key] = entry
                rebuild = True
            # The gene writes into the EQUATION's dict, which need not be the
            # system's (each translated equation of a system has its own), so
            # it gets the same entry; the gene is built from the system's.
            equation_mp = solution.vals[variable].metaparameters
            if equation_mp.get(key) != entry:
                equation_mp[key] = dict(entry)
        if rebuild:
            # Rebuild the chromosome so the gene carries the entry's value.
            solution.create(passed_equations=[solution.vals[variable]
                                              for variable in solution.vars_to_describe])

    @staticmethod
    def _drop_budget(solution: SoEq):
        """Strip ``('complexity', var)`` from a passed system in a run without
        budgets.

        A system from an earlier budget run carries the gene, the seeded ones
        here do not, and ``ChromosomeCrossover``'s encoding assertion would
        end the search at their first crossover. Its equations lose the key as
        well, so no stale budget is left for a budget reader to find. A system
        whose own dict has no such key -- every system of a default run --
        carries no budget gene and is not touched at all: no edit, no rebuild.
        """
        system_keys = [key for key in solution.metaparameters if is_complexity_key(key)]
        if not system_keys:
            return
        for key in system_keys:
            del solution.metaparameters[key]
        for variable in solution.vars_to_describe:
            equation_mp = solution.vals[variable].metaparameters
            for key in [key for key in equation_mp if is_complexity_key(key)]:
                del equation_mp[key]
        # The chromosome's genes are built from the system's dict.
        solution.create(passed_equations=[solution.vals[variable]
                                          for variable in solution.vars_to_describe])

    def create(self, **kwargs):
        sparsity = kwargs.get('sparsity', 10 ** (np.random.uniform(low = np.log10(self.sparsity_interval[0]),
                                                                      high = np.log10(self.sparsity_interval[1]),
                                                                      size = len(self.vars_demand_equation))))
        # # nonzero_terms = kwargs.get('nonzero_terms', np.random.randint(low=1,
        #                                                            high=self.terms_number,
        #                                                            size=len(self.vars_demand_equation)))
        # nu = kwargs.get('nu', 10 ** (np.random.uniform(low=np.log10(self.sparsity_interval[0]),
        #                                                              high=np.log10(self.sparsity_interval[1]),
        #                                                              size=len(self.vars_demand_equation))))
        # threshold = kwargs.get('threshold', 10 ** (np.random.uniform(low=np.log10(self.sparsity_interval[0]),
        #                                                            high=np.log10(self.sparsity_interval[1]),
        #                                                            size=len(self.vars_demand_equation))))
        terms_number = kwargs.get('terms_number', self.terms_number)
        max_factors_in_term = kwargs.get('max_factors_in_term', self.max_factors_in_term)
        # Every budget level equally likely, independently per equation -- the
        # way a best-subset study enumerates every complexity. Drawn after the
        # sparsity alpha, so that draw is the same one with or without budgets.
        if self.complexity_interval is not None:
            lo, hi = self.complexity_interval
            budgets = np.random.randint(lo, hi + 1, size=len(self.vars_demand_equation))

        # print(f'Creating new equation, sparsity value {sparsity}')
        metaparameters = {'max_terms_number'    : {'optimizable' : False, 'value' : terms_number},
                          'max_factors_in_term' : {'optimizable' : False, 'value' : max_factors_in_term}}
        for idx, variable in enumerate(self.vars_demand_equation):
            metaparameters[('sparsity', variable)] = {'optimizable' : True, 'value' : sparsity[idx]}
            if self.complexity_interval is not None:
                # A python int: Gene.value's setter silently keeps the old
                # value when a new one is not of the seeded type, and the
                # integer step in multiobjective SystemMutation returns ints.
                # The bounds travel with the value, so the step needs no
                # configuration lookup.
                metaparameters[(COMPLEXITY_METAPARAMETER, variable)] = {
                    'optimizable': True, 'value': int(budgets[idx]),
                    'bounds': (lo, hi)}
            # metaparameters[('nonzero_terms', variable)] = {'optimizable': True, 'value': nonzero_terms[idx]}
            # metaparameters[('threshold', variable)] = {'optimizable': True, 'value': threshold[idx]}
            # metaparameters[('nu', variable)] = {'optimizable': True, 'value': nu[idx]}

        created_solution = SoEq(pool = self.pool, metaparameters = metaparameters)

        try:
            created_solution.set_objective_functions(kwargs['obj_funs'])
        except KeyError:
            created_solution.use_default_multiobjective_function(second_objective=self.second_objective)



        created_solution.create()

        return created_solution
