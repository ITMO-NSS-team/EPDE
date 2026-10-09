#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:49:36 2021

@author: mike_ubuntu
"""

import numpy as np
from functools import reduce

from epde.optimizers.single_criterion.optimizer import Population
from epde.optimizers.single_criterion.supplementary import is_stamped
from epde.optimizers.moeadd.moeadd import ParetoLevels

from epde.operators.utils.template import CompoundOperator

# An exact-0 objective is floored to this fraction of the population's smallest
# positive objective: below eps * v_min a value is zero at the population's own
# precision, and the ratio keeps p invariant to a global rescale of the metric.
_ZERO_FLOOR_REL = np.finfo(float).eps
# Absolute backstop for populations living near the float underflow: caps every
# weight at 1e300, so their sum cannot overflow.
_ZERO_FLOOR_ABS = 1e-300


def roulette_probabilities(values, stamped) -> np.ndarray:
    """Parent-selection probabilities ``p_i ~ 1 / v_i`` (lower is better).

    ``values`` are the candidates' scalar objectives, ``stamped`` flags the
    forms the fitness host stamped as failed. Each stamped form gets p = 0:
    its 1e7 stamp is not the worst value for every metric. An exact 0 (the
    truth under most instability metrics on exact data, or WAPE on an exact
    fit) would make 1/v infinite and p NaN, so values are floored at
    ``_ZERO_FLOOR_REL`` times the smallest positive live objective -- the limit
    the 1/v rule already approaches for a near-zero objective, so the zero
    keeps virtually all the mass, tied zeros share it equally, and positive
    candidates keep a non-zero chance. An all-positive population is never
    floored, so its p is unchanged. When no live candidate has a positive
    weight (all stamped, or all infinite) p falls back to uniform over the
    live ones, or over all when every form is stamped. A NaN objective is
    left to propagate, so ``np.random.choice`` still refuses it loudly.
    """
    values = np.asarray(values, dtype=float)
    live = ~np.asarray(stamped, dtype=bool)
    positive = values[live & (values > 0) & np.isfinite(values)]
    floor = (max(_ZERO_FLOOR_REL * positive.min(), _ZERO_FLOOR_ABS)
             if positive.size else 1.0)
    weights = np.where(live, 1 / np.maximum(values, floor), 0.0)
    if np.sum(weights) == 0:
        weights = live.astype(float) if live.any() else np.ones(values.size)
    return weights / np.sum(weights)


class RouletteWheelSelection(CompoundOperator):
    key = 'RouletteWheelSelection'
    def apply(self, objective : Population, arguments: dict):
        # TODO: add docstring
        if isinstance(objective, ParetoLevels):
            raise TypeError('Tring to call method, implemented for Population class objects, to a ParetoLevels object. Must be a wrong type of evolution.')
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        parents_number_counted = int(len(objective.population) * self.params['parents_fraction']) # TODO: Странное упрощение 
        parents_number_counted = parents_number_counted if not parents_number_counted % 2 else parents_number_counted + 1
        
        parents_number = min(max(2, parents_number_counted), len(objective.population))

        # The same scalar simple_sorting ranks by; see roulette_probabilities
        # for the exact-0 floor and the zero share of stamped forms.
        probas = roulette_probabilities([float(np.sum(candidate.obj_fun)) for candidate in objective],
                                        [is_stamped(candidate) for candidate in objective])

        # print(probas.shape)
        candidate_idxs = np.random.choice(range(len(objective.population)), size = parents_number, 
                                          replace = True, p = probas) # Experiment with roulette with replace = False
        for idx in candidate_idxs:
            objective.population[idx].incr_counter() # TODO: direct access
        
        return objective
    
    def use_default_tags(self):
        self._tags = {'selection', 'population level', 'auxilary', 'suboperators', 'standard'}