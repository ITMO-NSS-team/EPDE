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
    
    This function perturbs the data by adding Gaussian noise scaled by the data's standard deviation and a specified noise level.
    This helps to evaluate the robustness of equation discovery algorithms when dealing with imperfect or noisy data.
    
    Args:
        data (np.ndarray): The input data to which noise will be added.
        noise_level (float): The standard deviation of the noise, expressed as a percentage of the data's standard deviation.
    
    Returns:
        np.ndarray: The data with added noise.
    """
    # add noise level to the input data
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def compare_equations(correct_symbolic: str, eq_incorrect_symbolic: str,
                      search_obj: EpdeSearch, all_vars: List[str] = ['u', ]) -> bool:
    """
    Compares two symbolic equations to determine which better represents the underlying dynamics.
    
        This method translates both equations into a comparable format, applies a fitting procedure,
        and then assesses their coefficient stability. The comparison helps identify equations that
        are more robust and reliable in capturing the system's behavior.
    
        Args:
            correct_symbolic: The correct symbolic equation as a string.
            eq_incorrect_symbolic: The incorrect symbolic equation as a string.
            search_obj: An EpdeSearch object containing the search pool.
            all_vars: A list of variable names to consider (default: ['u']).
    
        Returns:
            bool: True if the correct equation exhibits better coefficient stability
                than the incorrect equation across all variables, indicating a superior
                representation of the system's dynamics; False otherwise.
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
    Prepares the compound fitness operator with necessary sub-operators for equation discovery.
        
        This method initializes and configures the sparsity and coefficient calculation sub-operators,
        then integrates them into the provided compound fitness operator. The operator is mapped
        between gene and chromosome levels based on a condition that checks if the fitness has
        already been calculated. This ensures that the fitness calculation is performed only when
        necessary during the evolutionary process of discovering differential equations.
        
        Args:
          fitness_operator: The compound fitness operator to prepare.
          operator_params: A dictionary of parameters for the fitness operator.
        
        Returns:
          CompoundOperator: The prepared compound fitness operator, ready for use in the equation discovery process.
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

def ac_data(filename: str):
    """
    Loads data and generates corresponding grids for equation discovery.
    
        This function prepares the data by loading it from a file and constructing
        the necessary coordinate grids. These grids represent the independent variables
        over which the solution is defined, and are essential for evaluating candidate
        differential equations.
    
        Args:
            filename (str): The path to the data file ('.npy' format).
    
        Returns:
            tuple: A tuple containing:
                - grids (tuple of ndarray): Meshgrids representing the independent variables (t, x).
                - data (ndarray): The loaded data representing the dependent variable.
    
        Why:
        The grids are created to define the domain over which the differential equations
        will be evaluated and compared against the loaded data.
    """
    t = np.linspace(0., 1., 51)
    x = np.linspace(-1., 0.984375, 128)
    data = np.load(filename)
    # t = np.linspace(0, 1, shape+1); x = np.linspace(0, 1, shape+1)
    grids = np.meshgrid(t, x, indexing = 'ij')  # np.stack(, axis = 2) , axis = 2)
    return grids, data


def AC_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Test scenario to evaluate performance on Allen-Cahn equation. It aims to verify that the system can distinguish between the correct symbolic representation of the Allen-Cahn equation and an incorrect one based on the provided data and configurations. This is crucial for validating the equation discovery process.
    
        Args:
            operator: Compound operator for equation comparison.
            foldername: Name of the folder containing data files.
            noise_level: Level of noise to add to the data (default: 0).
    
        Returns:
            bool: True if the symbolic equation and incorrect equation are different, False otherwise.
    """
    # Test scenario to evaluate performance on Allen-Cahn equation
    eq_ac_symbolic = '0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
    eq_ac_incorrect = '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * u{power: 1.0} + -0.0 = du/dx0{power: 1.0}'

    grid, data = ac_data(os.path.join(foldername, 'ac_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ac_ann_pretrained.pickle'))

    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=((grid[0], grid[1])), verbose_params={'show_iter_idx': True},
                                 device='cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 2),
                                additional_tokens=[], data_nn=data_nn)

    assert compare_equations(eq_ac_symbolic, eq_ac_incorrect, epde_search_obj)


def ac_discovery(foldername, noise_level):
    """
    Discovers the governing equation from data using the EPDE framework.
        
        This method performs equation discovery on data loaded from a specified
        folder, incorporating a given noise level. It leverages a pre-trained
        PINN (Physics-Informed Neural Network) and the EPDE search algorithm
        to identify the underlying equation. The method automates the process of identifying governing differential equations from data,
        allowing users to gain insights into the underlying dynamics of complex systems and build predictive models.
        
        Args:
            foldername: The name of the folder containing the data and PINN.
            noise_level: The level of noise added to the data.
        
        Returns:
            EpdeSearch: The EPDE search object containing the discovered equations.
    """
    grid, data = ac_data(os.path.join(foldername, 'ac_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      use_pic=True, boundary=20,
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=30)

    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-9, 1e-4)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False, data_nn=data_nn) #

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

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

    # Paths
    directory = os.path.dirname(os.path.realpath(__file__))
    ac_folder_name = os.path.join(directory)

    # AC_test(fit_operator, ac_folder_name, 0)
    ac_discovery(ac_folder_name, 0)
