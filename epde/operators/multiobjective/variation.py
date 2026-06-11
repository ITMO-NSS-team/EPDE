#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:43:19 2021

@author: mike_ubuntu
"""
import random
from ast import operator
from operator import eq
import numpy as np
from copy import deepcopy

from functools import partial

from epde.structure.structure_template import check_uniqueness
from epde.optimizers.moeadd.moeadd import ParetoLevels

from epde.decorators import HistoryExtender, ResetEquationStatus

from epde.operators.utils.template import CompoundOperator, add_base_param_to_operator
from epde.operators.multiobjective.moeadd_specific import get_basic_populator_updater
from epde.operators.multiobjective.mutations import get_basic_mutation

from epde import _loop_stats


class ParetoLevelsCrossover(CompoundOperator):
    """
    The crossover operator, combining parameter crossover for terms with same 
    factors but different parameters & full exchange of terms between the 
    completely different ones.
    
    Noteable attributes:
    -----------
    suboperators : dict
        Inhereted from the Specific_Operator class. 
        Suboperators, performing tasks of parent selection, parameter crossover, full terms crossover, calculation of weights for each terms & 
        fitness function calculation. Dictionary: keys - strings from 'Selection', 'Param_crossover', 'Term_crossover', 'Coeff_calc', 'Fitness_eval'.
        values - corresponding operators (objects of Specific_Operator class).

    Methods:
    -----------
    apply(population)
        return the new population, created with the noted operators and containing both parent individuals and their offsprings.    
    copy_properties_to
    """
    key = 'ParetoLevelsCrossover'
    
    def apply(self, objective : ParetoLevels, arguments : dict):
        """
        Method to obtain a new population by selection of parent individuals (equations) and performing a crossover between them to get the offsprings.
        
        Attributes:
        -----------
        population : list of Equation objects
            the population, to that the operator is applied;
            
        Returns:
        -----------
        population : list of Equation objects
            the new population, containing both parents and offsprings;
        
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)        
        
        crossover_pool = []
        for solution in objective.population:
            crossover_pool.extend([solution,] * solution.crossover_times())
            solution.reset_counter()

        if len(crossover_pool) == 0:
            raise ValueError('crossover pool not created, probably solution.crossover_selected_times error')
        np.random.shuffle(crossover_pool)
        if len(crossover_pool) % 2:
            crossover_pool = crossover_pool[:-1]
        crossover_pool = np.array(crossover_pool, dtype = object).reshape((-1,2))

        offsprings = []
        for pair_idx in np.arange(crossover_pool.shape[0]):
            # if len(crossover_pool[pair_idx, 0].vals) != len(crossover_pool[pair_idx, 1].vals):
            #     raise IndexError('Equations have diffferent number of terms')
            new_system_1 = deepcopy(crossover_pool[pair_idx, 0])
            new_system_2 = deepcopy(crossover_pool[pair_idx, 1])
            # new_system_1.reset_state(False); new_system_2.reset_state()
            
            new_system_1, new_system_2 = self.suboperators['chromosome_crossover'].apply(objective = (new_system_1, new_system_2),
                                                                                         arguments = subop_args['chromosome_crossover'])

            for eq_key in new_system_1.vals.equation_keys:
                assert len(new_system_1.vals[eq_key].terms_labels) == len(new_system_1.vals[eq_key].structure)
                assert len(new_system_2.vals[eq_key].terms_labels) == len(new_system_2.vals[eq_key].structure)
                assert len(crossover_pool[pair_idx, 0].vals[eq_key].terms_labels) == len(crossover_pool[pair_idx, 0].vals[eq_key].structure)
                assert len(crossover_pool[pair_idx, 1].vals[eq_key].terms_labels) == len(crossover_pool[pair_idx, 1].vals[eq_key].structure)

            if len(new_system_1.vars_to_describe) > 1 and np.random.random() < 0.2:
                key = np.random.choice(new_system_1.vars_to_describe)
                temp = deepcopy(new_system_1.vals.chromosome[key])
                new_system_1.vals.chromosome[key] = new_system_2.vals.chromosome[key]
                new_system_2.vals.chromosome[key] = temp

            offsprings.extend([new_system_1, new_system_2])

        objective.unplaced_candidates = offsprings
        return objective

    def use_default_tags(self):
        self._tags = {'crossover', 'population level', 'contains suboperators', 'standard'}


class ChromosomeCrossover(CompoundOperator):
    key = 'ChromosomeCrossover'
    
    def apply(self, objective : tuple, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
   
        assert objective[0].vals.same_encoding(objective[1].vals)
        offspring_1 = objective[0]; offspring_2 = objective[1]

        eqs_keys = offspring_1.vals.equation_keys; params_keys = offspring_2.vals.params_keys

        if len(eqs_keys) > 1 and random.random() < self.params['equation_exchange_prob']:
            eq_key = random.choice(eqs_keys)
            temp_eq = deepcopy(offspring_1.vals[eq_key])
            offspring_1.vals.replace_gene(gene_key = eq_key, value = offspring_2.vals[eq_key])
            offspring_2.vals.replace_gene(gene_key = eq_key, value = temp_eq)

            return offspring_1, offspring_2

        for eq_key in eqs_keys:
            temp_eq_1, temp_eq_2 = self.suboperators['equation_crossover'].apply(objective = (offspring_1.vals[eq_key],
                                                                                              offspring_2.vals[eq_key]),
                                                                                 arguments = subop_args['equation_crossover'])
            offspring_1.vals.replace_gene(gene_key = eq_key, value = temp_eq_1)
            offspring_2.vals.replace_gene(gene_key = eq_key, value = temp_eq_2)

        # for param_key in params_keys:
        #     temp_param_1, temp_param_2 = self.suboperators['param_crossover'].apply(objective = (offspring_1.vals[param_key],
        #                                                                                          offspring_2.vals[param_key]),
        #                                                                             arguments = subop_args['param_crossover'])
        #     offspring_1.vals.replace_gene(gene_key = param_key, value = temp_param_1)
        #     offspring_2.vals.replace_gene(gene_key = param_key, value = temp_param_2)
        #
        #     offspring_1.vals.pass_parametric_gene(key = param_key, value = temp_param_1)
        #     offspring_2.vals.pass_parametric_gene(key = param_key, value = temp_param_2)

        return offspring_1, offspring_2

    def use_default_tags(self):
        self._tags = {'crossover', 'chromosome level', 'contains suboperators', 'standard'}


class MetaparamerCrossover(CompoundOperator):
    key = 'MetaparamerCrossover'
    
    def apply(self, objective : tuple, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        offspring_1 = objective[0] + self.params['metaparam_proportion'] * (objective[1] - objective[0])
        offspring_2 = objective[0] + (1 - self.params['metaparam_proportion']) * (objective[1] - objective[0])
        return offspring_1, offspring_2

    def use_default_tags(self):
        self._tags = {'crossover', 'gene level', 'no suboperators'}


class EquationCrossover(CompoundOperator):
    key = 'EquationCrossover'

    @HistoryExtender(f'\n -> performing equation crossover', 'ba')
    def apply(self, objective : tuple, arguments : dict):
        """Hybrid random-partition + parameter-blend crossover.

        Parents enter crossover in the post-RPS "non-zero" form: zero-
        weight terms were physically removed by ``remove_zero_terms`` at
        the end of the previous right_part_selector pass, so every term
        in ``parent.structure`` contributed meaningfully to the parent's
        fitness. See project memory ``project_mutation_crossover_non_zero_form``.

        Three-phase build:
          * **Anchor:** terms whose ``factors_labels`` match exactly
            across parents (full structural identity, including bucketed
            params) are preserved unchanged in both offspring.
          * **Param-blend pairs:** among the non-anchor terms, pair up
            across parents by the looser factor-function signature
            (frozenset of ``factor.label`` only, ignoring params). Each
            such pair is passed through ``TermParamCrossover`` to produce
            two distinct blended variants -- one per offspring.
          * **Random partition:** the remaining truly-unique terms (no
            anchor match, no param-blend match) get a coin-flip
            assignment to one offspring or the other.

        The previous design called ``flatten(detect_similar_terms(...))``
        which produced two offspring containing the structural UNION of
        both parents -- i.e. clone offspring with zero diversity. This
        rewrite delivers genuinely-different offspring, activates the
        wired-but-dormant ``term_param_crossover`` sub-operator, and
        keeps the D10 dedup invariant.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        parent1 = objective[0]
        parent2 = objective[1]
        # Snapshot target term identity (not the term itself) -- the
        # actual deepcopy is deferred to ``_ensure_target`` and only
        # pays when the target wasn't already partitioned into the
        # offspring via Phase 1/2/3. Common case (target shared as an
        # anchor) saves both deepcopies entirely.
        p1_target_ref = parent1.structure[parent1.target_idx]
        p2_target_ref = parent2.structure[parent2.target_idx]
        p1_target_labels = p1_target_ref.factors_labels
        p2_target_labels = p2_target_ref.factors_labels

        def factor_signature(term):
            """Factor-function-set signature, ignoring params.

            Two terms have the "same factor functions, different params"
            relation iff they share this signature but differ on
            ``factors_labels``.
            """
            return frozenset(factor.label for factor in term.structure)

        # Phase 1 -- find same-anchor pairs (exact factors_labels match).
        # Pairs are stored as (i, j) so each offspring inherits its own
        # parent's instance of the anchored term: two terms with equal
        # ``factors_labels`` (bucketed structural identity) can still
        # carry slightly different ``factor.params`` within the bucket,
        # and that within-bucket variation is genuine signal we want to
        # preserve per-offspring.
        common_labels = parent1.terms_labels & parent2.terms_labels
        anchor_pairs = []
        e2_used = set()
        unique_e1_idxs = []
        for i, term_e1 in enumerate(parent1.structure):
            if term_e1.factors_labels in common_labels:
                matched = False
                for j, term_e2 in enumerate(parent2.structure):
                    if j in e2_used:
                        continue
                    if term_e2.factors_labels == term_e1.factors_labels:
                        anchor_pairs.append((i, j))
                        e2_used.add(j)
                        matched = True
                        break
                if not matched:
                    unique_e1_idxs.append(i)
            else:
                unique_e1_idxs.append(i)
        unique_e2_idxs = [j for j in range(len(parent2.structure))
                          if j not in e2_used]

        # Phase 2 -- find param-blend pairs (matching factor function set,
        # differing params) among the unique-side terms.
        param_pairs = []
        remaining_e1 = list(unique_e1_idxs)
        remaining_e2 = list(unique_e2_idxs)
        for i in list(remaining_e1):
            sig_i = factor_signature(parent1.structure[i])
            for j in list(remaining_e2):
                if factor_signature(parent2.structure[j]) == sig_i:
                    param_pairs.append((i, j))
                    remaining_e1.remove(i)
                    remaining_e2.remove(j)
                    break

        # Phase 3 -- assemble offspring.
        # Each anchor pair contributes parent1's instance to offspring1
        # and parent2's instance to offspring2 (preserving per-parent
        # within-bucket variation -- see Phase 1 comment).
        offspring1_terms = [deepcopy(parent1.structure[i]) for i, _ in anchor_pairs]
        offspring2_terms = [deepcopy(parent2.structure[j]) for _, j in anchor_pairs]

        for i, j in param_pairs:
            t1 = deepcopy(parent1.structure[i])
            t2 = deepcopy(parent2.structure[j])
            blended1, blended2 = self.suboperators['term_param_crossover'].apply(
                objective=(t1, t2),
                arguments=subop_args['term_param_crossover'],
            )
            offspring1_terms.append(blended1)
            offspring2_terms.append(blended2)

        truly_unique = ([('e1', i) for i in remaining_e1]
                        + [('e2', j) for j in remaining_e2])
        for source, idx in truly_unique:
            src = parent1 if source == 'e1' else parent2
            term = deepcopy(src.structure[idx])
            if np.random.random() < 0.5:
                offspring1_terms.append(term)
            else:
                offspring2_terms.append(term)

        # Phase 4 -- force-include each parent's target term so right-part
        # validity survives the partition. Anchored / partitioned targets
        # are already present; the helper is a no-op in that case. When
        # we DO need to add the target, deepcopy on the spot so the
        # offspring doesn't alias the parent's term.
        def _ensure_target(terms, parent_target_term, target_labels):
            for t in terms:
                if t.factors_labels == target_labels:
                    return terms
            return [deepcopy(parent_target_term)] + terms

        offspring1_terms = _ensure_target(
            offspring1_terms, p1_target_ref, p1_target_labels)
        offspring2_terms = _ensure_target(
            offspring2_terms, p2_target_ref, p2_target_labels)

        # Phase 5 -- D10 post-assembly dedup gate. A param-blend pair can
        # in principle produce a structural_label that collides with an
        # anchor term, and we'd rather revert to parents than emit a
        # duplicate-bearing chromosome.
        eq1_sigs = [t.factors_labels for t in offspring1_terms]
        eq2_sigs = [t.factors_labels for t in offspring2_terms]
        had_duplicate = (len(set(eq1_sigs)) != len(eq1_sigs)
                         or len(set(eq2_sigs)) != len(eq2_sigs))
        _loop_stats.record(
            'EquationCrossover.duplicate_offspring' + ('.FAIL' if had_duplicate else ''),
            1, 1,
        )
        if had_duplicate:
            return objective[0], objective[1]

        # Phase 6 -- build the offspring Equation objects via shell
        # clones (no parent ``structure`` deepcopy). The offspring term
        # list is freshly built above; cloning the full parent and then
        # overwriting ``.structure`` wasted a Term+Factor recursion that
        # was the heaviest single deepcopy in the crossover hot path.
        equation1 = parent1.clone_shell()
        equation2 = parent2.clone_shell()
        equation1.structure = offspring1_terms
        equation2.structure = offspring2_terms

        for i, t in enumerate(equation1.structure):
            if t.factors_labels == p1_target_labels:
                equation1.target_idx = i
                break
        for i, t in enumerate(equation2.structure):
            if t.factors_labels == p2_target_labels:
                equation2.target_idx = i
                break

        equation1._invalidate_label_cache()
        equation2._invalidate_label_cache()
        return equation1, equation2

    def use_default_tags(self):
        self._tags = {'crossover', 'gene level', 'contains suboperators', 'standard'}

class TermParamCrossover(CompoundOperator):
    """
    The crossover exchange between parent terms with the same factor functions, that differ only in the factor parameters. 

    Noteable attributes:
    -----------
    params : dict
        Inhereted from the Specific_Operator class. 
        Main key - 'proportion', value - proportion, in which the offsprings' parameter values are chosen.
        
    Methods:
    -----------
    apply(population)
        return the offspring terms, constructed as the parents' factors with parameter values, selected between the parents' ones.        
    """
    key = 'TermParamCrossover'
        
    def apply(self, objective : tuple, arguments : dict):
        """
        Get the offspring terms, constructed as the parents' factors with parameter values, selected between the parents' ones.
        
        Attributes:
        ------------
        term_1, term_2 : Term objects
            The parent terms.
        
        Returns:
        ------------
        offspring_1, offspring_2 : Term objects
            The offspring terms.
        
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        objective[0].reset_saved_state(); objective[1].reset_saved_state()
        
        if len(objective[0].structure) != len(objective[1].structure):
            print([(token.label, token.params) for token in objective[0].structure], [(token.label, token.params) for token in objective[1].structure])
            raise Exception('Wrong terms passed:')
        for term1_token_idx in np.arange(len(objective[0].structure)):
            term2_token_idx = [i for i in np.arange(len(objective[1].structure)) 
                               if objective[1].structure[i].label == objective[0].structure[term1_token_idx].label][0]
            for param_idx, param_descr in objective[0].structure[term1_token_idx].params_description.items():
                if param_descr['name'] == 'power': power_param_idx = param_idx
                if param_descr['name'] == 'dim': dim_param_idx = param_idx
            
            try:                # TODO: refactor logic
                dim_param_idx
            except:
                dim_param_idx = power_param_idx

            for param_idx in np.arange(objective[0].structure[term1_token_idx].params.size):
                if param_idx != power_param_idx and param_idx != dim_param_idx:
                    factor1 = objective[0].structure[term1_token_idx]
                    factor2 = objective[1].structure[term2_token_idx]
                    try:
                        new_v1 = (factor1.params[param_idx]
                                  + self.params['term_param_proportion']
                                  * (factor2.params[param_idx] - factor1.params[param_idx]))
                        factor1.set_param(new_v1, idx=param_idx)
                    except KeyError:
                        print([(token.label, token.params) for token in objective[0].structure], [(token.label, token.params) for token in objective[1].structure])
                        raise Exception('Wrong set of parameters:', factor1.params_description, factor2.params_description)
                    new_v2 = (factor1.params[param_idx]
                              + (1 - self.params['term_param_proportion'])
                              * (factor2.params[param_idx] - factor1.params[param_idx]))
                    factor2.set_param(new_v2, idx=param_idx)
        objective[0].reset_occupied_tokens(); objective[1].reset_occupied_tokens()
        return objective[0], objective[1]

    def use_default_tags(self):
        self._tags = {'crossover', 'term level', 'exploitation', 'no suboperators', 'standard'}

class TermCrossover(CompoundOperator):
    """
    The crossover exchange between parent terms, done by complete exchange of terms. 

    Noteable attributes:
    -----------
    params : dict
        Inhereted from the Specific_Operator class. 
        Main key - 'crossover_probability', value - probabilty of the term exchange.
        
    Methods:
    -----------
    apply(population)
        return the offspring terms, which are the same parents' ones, but in different order, if the crossover occured.
        .        
    """    
    key = 'TermCrossover'

    def apply(self, objective : tuple, arguments : dict):
        """
        Get the offspring terms, which are the same parents' ones, but in different order, if the crossover occured.
        
        Attributes:
        ------------
        term_1, term_2 : Term objects
            The parent terms.
            
        Returns:
        ------------
        offspring_1, offspring_2 : Term objects
            The offspring terms.
        
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        if (np.random.uniform(0, 1) <= self.params['crossover_probability'] and
            objective[1].descr_variable_marker == objective[0].descr_variable_marker):
                return objective[1], objective[0]
        else:
                return objective[0], objective[1]
        
    def use_default_tags(self):
        self._tags = {'crossover', 'term level', 'exploration', 'no suboperators', 'standard'}


def get_basic_variation(variation_params : dict = {}):
    # TODO: generalize initiation with test runs and simultaneous parameter and object initiation.
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = variation_params)    

    term_param_crossover = TermParamCrossover(['term_param_proportion'])
    add_kwarg_to_operator(operator = term_param_crossover)
    term_crossover = TermCrossover(['crossover_probability'])
    add_kwarg_to_operator(operator = term_crossover)

    equation_crossover = EquationCrossover(['crossover_probability'])
    add_kwarg_to_operator(operator=equation_crossover)
    metaparameter_crossover = MetaparamerCrossover(['metaparam_proportion'])
    add_kwarg_to_operator(operator = metaparameter_crossover)

    chromosome_crossover = ChromosomeCrossover(['equation_exchange_prob'])
    add_kwarg_to_operator(operator = chromosome_crossover)

    pl_cross = ParetoLevelsCrossover([])

    equation_crossover.set_suboperators(operators = {'term_param_crossover' : term_param_crossover,
                                                     'term_crossover' : term_crossover})
    chromosome_crossover.set_suboperators(operators = {'equation_crossover' : equation_crossover,
                                                       'param_crossover' : metaparameter_crossover})
    pl_cross.set_suboperators(operators = {'chromosome_crossover' : chromosome_crossover})
    return pl_cross
