import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import pickle
from typing import Tuple, List
import numpy as np

from epde.interface.prepared_tokens import CustomTokens, PhasedSine1DTokens, ConstantToken, CustomEvaluator
from epde.interface.equation_translator import translate_equation
from epde.interface.interface import EpdeSearch

from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.sparsity import LASSOSparsity

from epde.operators.utils.operator_mappers import map_operator_between_levels
import epde.operators.common.fitness as fitness
from epde.operators.utils.template import CompoundOperator

from epde import TrigonometricTokens, GridTokens, CacheStoredTokens
import epde.globals as global_var

import scipy.io as scio

def load_pretrained_PINN(ann_filename):
    """
    Loads a pre-trained Physics-Informed Neural Network (PINN) from a file.
    
    This function attempts to load a previously trained PINN model, allowing the evolutionary process to start from a potentially good initial guess,
    reducing the computational cost of training from scratch. If no pre-trained model is found, the process continues by retraining the ANN approximation.
    
    Args:
        ann_filename: The filename of the pickled ANN data.
    
    Returns:
        The loaded ANN data if the file exists, otherwise None.
    """
    try:
        with open(ann_filename, 'rb') as data_input_file:
            data_nn = pickle.load(data_input_file)
    except FileNotFoundError:
        print('No model located, proceeding with ann approx. retraining.')
        data_nn = None
    return data_nn


def noise_data(data, noise_level):
    """
    Adds random noise to the input data based on its standard deviation.
    
    This helps to evaluate the robustness of discovered differential equations when data is imperfect.
    
    Args:
        data (np.ndarray): The input data to add noise to.
        noise_level (float): The level of noise to add, as a percentage of the data's standard deviation.
    
    Returns:
        np.ndarray: The data with added noise.
    """
    # add noise level to the input data
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def compare_equations(correct_symbolic: str, eq_incorrect_symbolic: str,
                      search_obj: EpdeSearch, all_vars: List[str] = ['u', ]) -> bool:
    """
    Compares two symbolic equations to assess which one better represents the underlying dynamics.
    
        This method translates symbolic equations into a comparable format, applies a fitting procedure,
        and then evaluates their coefficient stability. The comparison helps determine which equation
        more accurately captures the system's behavior.
    
        Args:
            correct_symbolic: The correct symbolic equation as a string.
            eq_incorrect_symbolic: The incorrect symbolic equation as a string.
            search_obj: An EpdeSearch object containing the search pool.
            all_vars: A list of variable names to consider (default: ['u']).
    
        Returns:
            bool: True if the coefficient stability of the correct equation is
                less than that of the incorrect equation for all variables,
                False otherwise.
    """
    metaparams = {('sparsity', var): {'optimizable': False, 'value': 1E-6} for var in all_vars}

    correct_eq = translate_equation(correct_symbolic, search_obj.pool, all_vars=all_vars)
    for var in all_vars:
        correct_eq.vals[var].main_var_to_explain = var
        correct_eq.vals[var].metaparameters = metaparams
    print(correct_eq.text_form)

    incorrect_eq = translate_equation(eq_incorrect_symbolic, search_obj.pool,
                                      all_vars=all_vars)  # , all_vars = ['u', 'v'])
    for var in all_vars:
        incorrect_eq.vals[var].main_var_to_explain = var
        incorrect_eq.vals[var].metaparameters = metaparams
    print(incorrect_eq.text_form)

    fit_operator.apply(correct_eq, {})
    fit_operator.apply(incorrect_eq, {})
    print([[correct_eq.vals[var].fitness_value, incorrect_eq.vals[var].fitness_value] for var in all_vars])
    print([[correct_eq.vals[var].coefficients_stability, incorrect_eq.vals[var].coefficients_stability] for var in
           all_vars])
    print([[correct_eq.vals[var].aic, incorrect_eq.vals[var].aic] for var in all_vars])

    # print([correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability for var in all_vars])
    return all([correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability for var in
                all_vars])


def prepare_suboperators(fitness_operator: CompoundOperator, operator_params: dict) -> CompoundOperator:
    """
    Prepares the compound fitness operator with necessary sub-operators.
    
        This method configures the sparsity and coefficient calculation sub-operators,
        and sets them within the provided fitness operator. It also maps the operator
        between gene and chromosome levels based on a fitness calculation condition.
        This step is crucial for ensuring that the evolutionary process can effectively
        evaluate the fitness of candidate equations by considering both their sparsity
        and the accuracy of their coefficients.
    
        Args:
          fitness_operator (CompoundOperator): The compound fitness operator to prepare.
          operator_params (dict): A dictionary of parameters for the fitness operator.
    
        Returns:
          CompoundOperator: The modified fitness operator with prepared sub-operators.
    """
    sparsity = LASSOSparsity()
    coeff_calc = LinRegBasedCoeffsEquation()

    # sparsity = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
    # coeff_calc = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')

    fitness_operator.set_suboperators({'sparsity': sparsity,
                                       'coeff_calc': coeff_calc})
    fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
    fitness_operator.params = operator_params
    fitness_operator = map_operator_between_levels(fitness_operator, 'gene level', 'chromosome level',
                                                   objective_condition=fitness_cond)
    return fitness_operator


def lv_discovery(noise_level):
    """
    Discovers the Lotka-Volterra system's governing equations using the EPDE framework.
    
        This method automates the identification of differential equations that describe the dynamics
        of the Lotka-Volterra system. By loading system data and configuring the EPDE search object
        with specific parameters, it performs an equation discovery process, aiming to find the
        equations that best represent the system's behavior. The discovered equations are then printed
        to provide insights into the underlying dynamics. This approach allows for automated modeling
        and analysis of the system's behavior directly from data.
    
        Args:
            noise_level: This parameter is not used in the method.
    
        Returns:
            EpdeSearch: The configured EpdeSearch object after the fitting process, containing the
                discovered equations and related information.
    """
    t_file = os.path.join(os.path.dirname( __file__ ), 't_20.npy')
    t = np.load(t_file)
    data_file = os.path.join(os.path.dirname(__file__), 'data_20.npy')
    data = np.load(data_file)

    end = 150
    t = t[:end]
    x = data[:end, 0]
    y = data[:end, 1]

    dimensionality = x.ndim - 1

    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True, boundary=10,
                                 coordinate_tensors=(t,), verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=30)

    factors_max_number = {'factors_num': [1, 2], 'probas' : [0.8, 0.2]}

    epde_search_obj.fit(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,),
                        equation_terms_max_number=5, data_fun_pow=1, additional_tokens=[trig_tokens, ],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-4, 1e-0))  #

    epde_search_obj.equations(only_print=True, num=1)

    return epde_search_obj


if __name__ == "__main__":
    import torch
    from epde.operators.utils.default_parameter_loader import EvolutionaryParams
    print(torch.cuda.is_available())
    # Operator = fitness.SolverBasedFitness # Replace by the developed PIC-based operator.
    # Operator = fitness.PIC
    Operator = fitness.L2LRFitness
    params = EvolutionaryParams()
    operator_params = params.get_default_params_for_operator('DiscrepancyBasedFitnessWithCV') #{"penalty_coeff": 0.2, "pinn_loss_mult": 1e4}
    print('operator_params ', operator_params)
    fit_operator = prepare_suboperators(Operator(list(operator_params.keys())), operator_params)

    lv_discovery(0)
