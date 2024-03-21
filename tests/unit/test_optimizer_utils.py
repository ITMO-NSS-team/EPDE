import unittest
import logging

import numpy as np

from epde.operators.utils.default_parameter_loader import EvolutionaryParams
from epde.operators.utils.template import CompoundOperator, SuboperatorContainer, \
    add_base_param_to_operator

class TestParamsLoader(unittest.TestCase):
    def test_loading(self):
        param_key_expected = {'multi objective'  : ['MOEADDSelection',
                                                    'PopulationUpdater',
                                                    'SortingBasedNeighborSelector',
                                                    'ParetoLevelUpdater',
                                                    'InitialParetoLevelSorting',
                                                    'DiscrepancyBasedFitness',
                                                    'ParetoLevelsCrossover',
                                                    'ChromosomeCrossover',
                                                    'MetaparamerCrossover',
                                                    'EquationCrossover', 
                                                    'EquationExchangeCrossover',
                                                    'TermCrossover',
                                                    'TermParamCrossover',
                                                    'SystemMutation', 
                                                    'EquationMutation',
                                                    'MetaparameterMutation', 
                                                    'TermMutation',
                                                    'TermParameterMutation', 
                                                    'LinRegCoeffCalc',
                                                    'LASSOBasedSparsity'],
                              'single objective' : ['RouletteWheelSelection', 
                                                    'SizeRestriction', 
                                                    'FractionElitism',
                                                    'ParetoLevelUpdater',
                                                    'InitialParetoLevelSorting',
                                                    'DiscrepancyBasedFitness',
                                                    'PopulationLevelCrossover',
                                                    'ChromosomeCrossover',
                                                    'EquationCrossover',
                                                    'TermCrossover',
                                                    'TermParamCrossover',
                                                    'SystemMutation',
                                                    'EquationMutation',
                                                    'MetaparameterMutation',
                                                    'TermMutation',
                                                    'TermParameterMutation',
                                                    'LinRegCoeffCalc',
                                                    'LASSOBasedSparsity']}
        for mode, mode_spec_params in param_key_expected.items():
            params = EvolutionaryParams(mode = mode)
            self.assertTrue(all([param in params._repo for param in mode_spec_params]) and 
                            all([param in mode_spec_params for param in params._repo]),
                            msg = 'Missing necessary parameters for default strategy.')
            params_dupl = EvolutionaryParams(mode = mode)
            self.assertTrue(params_dupl is params, 
                            msg = 'Singleton property of EvolutionaryParams obj is not lost.')
            
            sample_operator = 'DiscrepancyBasedFitness'
            sample_param = 'penalty_coeff'
            sample_value = 0.3
            params.change_operator_param(sample_operator, sample_param, sample_value)
            self.assertEqual(params.get_default_params_for_operator(operator_name=sample_operator)[sample_param],
                             sample_value, msg = 'Parameter update did not work correctly.')
            self.assertEqual(params_dupl.get_default_params_for_operator(operator_name=sample_operator)[sample_param],
                             sample_value, msg = 'Singleton params loader update did not work correctly.')
            del params, params_dupl
            
        #possible test of __enter__ and __exit__
            
class TestCompoundOperator(unittest.TestCase):
    def test_operator(self):
        main_key = 'DummyOperator'
        main_param_keys = ['dummy_param_1', 'dummy_param_2']
        subop_1_key = 'SubOp1'
        subop_1_param_keys = ['subop_param_1']
        subop_2_key = 'SubOp2'
        subop_2_param_keys = []        
        
        param_vals = {main_key: {param : np.random.uniform() for param in main_param_keys},
                      subop_1_key: {param : np.random.uniform() for param in subop_1_param_keys},
                      subop_2_key: {param : np.random.uniform() for param in subop_2_param_keys}}
        
        dummy_operator = CompoundOperator(main_param_keys)
        dummy_operator.key = main_key
        add_base_param_to_operator(operator = dummy_operator, target_dict = param_vals)
        
        subop_1 = CompoundOperator(subop_1_param_keys)
        subop_1.key = subop_1_key
        add_base_param_to_operator(operator = subop_1, target_dict = param_vals)
        
        subop_2 = CompoundOperator(subop_2_param_keys)
        subop_2.key = subop_2_key
        add_base_param_to_operator(operator = subop_2, target_dict = param_vals)
        
        dummy_operator.set_suboperators({subop_1_key : subop_1, subop_2_key : subop_2})
        
        dummy_op