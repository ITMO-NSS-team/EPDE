#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:57:44 2022

@author: maslyaev
"""

import warnings

import numpy as np
from typing import Callable

import epde.globals as global_var
from epde.structure.main_structures import SoEq

#: Label of the retired ``('complexity', var)`` budget metaparameter, which
#: ``sparsity_cls='knee'``'s ``'complexity'`` realization seeded and evolved
#: until both were removed. Kept only so a system saved from such a run can
#: still be passed in (``SystemsPopulationConstructor._strip_retired_budget``).
_RETIRED_BUDGET_LABEL = 'complexity'

class SystemsPopulationConstructor(object):
    """
    Class for creating population according to the current optimization method/

    Attributes:
        pool (``):
        terms_number (`int`):
        max_factors_in_term (`int`):
        vars_demand_equation (``):
        sparsity_internal (`tuple`):
    """
    def __init__(self, pool, second_objective: str = None, terms_number : int = 8,
                 max_factors_in_term : int = 2,
                 obj_functions : Callable = None,
                 sparsity_interval : tuple = (1., 1.)):
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
        print('self.vars_demand_equation', self.vars_demand_equation)

    def applyToPassed(self, passed_solution: SoEq, **kwargs):
        try:
            passed_solution.set_objective_functions(kwargs['obj_funs'])
        except KeyError:
            passed_solution.use_default_multiobjective_function(self.second_objective)
        self._strip_retired_budget(passed_solution)
        self._seed_target_genes(passed_solution)

    @staticmethod
    def _seed_target_genes(solution: SoEq):
        """system_rps_mode='plain': a passed system keeps the right parts it
        came with. The initial sorting resets every candidate's target before
        its fit, so the target is first written into the equation's target
        gene, which the plain fit reads. No other mode has the gene."""
        from epde.interface.search_config import active_config
        if active_config().search_space.system_rps_mode != 'plain':
            return
        from epde.operators.common.plain_fit import target_gene
        for equation in solution.vals:
            if equation.target is not None and getattr(equation, '_plain_genes', None) is None:
                equation._plain_genes = target_gene(equation.target)

    @staticmethod
    def _strip_retired_budget(solution: SoEq):
        """Strip the retired ``('complexity', var)`` budget from a passed
        system, and say so.

        A system saved from a ``sparsity_cls='knee'`` budget run still carries
        the budget as a gene, which no seeded system has any more:
        ``ChromosomeCrossover``'s encoding assertion would end the search at
        their first crossover, and ``SystemMutation`` would walk the integer
        budget with the alpha's proportional step. Nothing reads the budget
        now, so dropping it changes no fit; the structure and every other
        metaparameter are kept, and the equations lose the key too. A system
        whose own dict has no such key -- every system of a current run -- is
        not touched at all: no edit, no rebuild, no warning.
        """
        def retired(key):
            return (isinstance(key, tuple) and len(key) == 2
                    and key[0] == _RETIRED_BUDGET_LABEL)

        system_keys = [key for key in solution.metaparameters if retired(key)]
        if not system_keys:
            return
        warnings.warn(
            'A passed system carries the {0} metaparameter(s) of the removed '
            "complexity budget (sparsity_cls='knee', realization "
            "'complexity'); they are dropped and the system joins the search "
            'with its structure and its other metaparameters.'.format(
                sorted(system_keys, key=repr)),
            global_var.EPDEUsageWarning, stacklevel=2)
        for key in system_keys:
            del solution.metaparameters[key]
        for variable in solution.vars_to_describe:
            equation_mp = solution.vals[variable].metaparameters
            for key in [key for key in equation_mp if retired(key)]:
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

        # print(f'Creating new equation, sparsity value {sparsity}')
        metaparameters = {'max_terms_number'    : {'optimizable' : False, 'value' : terms_number},
                          'max_factors_in_term' : {'optimizable' : False, 'value' : max_factors_in_term}}
        for idx, variable in enumerate(self.vars_demand_equation):
            metaparameters[('sparsity', variable)] = {'optimizable' : True, 'value' : sparsity[idx]}
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
