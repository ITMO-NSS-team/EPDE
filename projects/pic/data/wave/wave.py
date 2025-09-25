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
    Loads a pre-trained neural network model from a file.
    
    This function attempts to load a previously trained Physics-Informed Neural Network (PINN)
    from the specified file. This allows to reuse already trained models, avoiding retraining
    and saving computational resources when exploring different equation candidates.
    
    Args:
        ann_filename (str): The filename of the pickled ANN data.
    
    Returns:
        object: The loaded ANN data if the file exists; otherwise, None.
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
    Adds random noise to the input data, scaling it based on the data's standard deviation and a specified noise level.
    
    This helps to evaluate the robustness of discovered differential equations by simulating real-world measurement errors.
    
    Args:
        data (np.ndarray): The input data (e.g., a time series) to which noise will be added.
        noise_level (float):  A percentage determining the magnitude of the noise relative to the data's standard deviation.
    
    Returns:
        np.ndarray: The data with added Gaussian noise, having the same shape as the input data.
    """
    # add noise level to the input data
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def compare_equations(correct_symbolic: str, eq_incorrect_symbolic: str,
                      search_obj: EpdeSearch, all_vars: List[str] = ['u', ]) -> bool:
    """
    Compares two symbolic equations to determine which one better represents the underlying dynamics of the system.
        
        It translates both the correct and incorrect symbolic equations into a form suitable
        for comparison, applies a fitting operator to estimate the coefficients, and then
        assesses their stability. The equation with more stable coefficients is considered
        a better representation of the system's dynamics.
        
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
    
        This method configures the provided compound operator by setting its
        sub-operators for sparsity and coefficient calculation. These sub-operators
        are essential for constructing the overall fitness evaluation process, ensuring
        that the evolutionary algorithm can effectively explore the search space of
        potential equation structures. The operator is then mapped between gene and
        chromosome levels based on a fitness calculation condition to ensure proper
        evaluation during the evolutionary process. This setup is crucial for
        evaluating the fitness of candidate equations within the evolutionary process.
    
        Args:
            fitness_operator (CompoundOperator): The compound fitness operator to prepare.
            operator_params (dict): A dictionary of parameters for the fitness operator.
    
        Returns:
            CompoundOperator: The prepared compound fitness operator.
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


def wave_data(filename):
    """
    Generates spatio-temporal data from a file.
    
    This method loads data representing a physical system from a specified file, 
    creates corresponding time and space grids, and returns both the grids and the loaded data.  
    The data is assumed to be comma-separated values. This is a crucial step 
    in preparing the data for equation discovery, as it establishes the domain 
    over which the differential equations will be defined and evaluated.
    
    Args:
        filename: The name of the file containing the wave data.
    
    Returns:
        tuple: A tuple containing the grids and the data. The grids are a
            numpy array representing the time and space coordinates, and the
            data is a numpy array loaded from the file.
    """
    shape = 80

    # print(os.path.dirname( __file__ ))
    data = np.loadtxt(filename, delimiter=',').T
    t = np.linspace(0, 1, shape + 1);
    x = np.linspace(0, 1, shape + 1)
    grids = np.stack(np.meshgrid(t, x, indexing='ij'), axis=2)
    return grids, data


def wave_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the equation discovery process for the wave equation.
        
        This method validates the equation discovery pipeline on the wave equation.
        It sets up an EPDE search, preprocesses data, and checks if the discovered
        equation matches the expected symbolic form. This ensures the framework
        can accurately identify known relationships within a given dataset.
        
        Args:
            operator: CompoundOperator object for equation comparison.
            foldername: Path to the folder containing the data files
                ('wave_sln_80.csv' and 'ann_pretrained.pickle').
            noise_level: Level of noise to add to the data. Defaults to 0.
        
        Returns:
            bool: True if the discovered equation matches the expected symbolic
                representation, False otherwise.
    """
    # eq_wave_symbolic = '1. * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    eq_wave_symbolic = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    eq_wave_incorrect = '0.04 * d^2u/dx1^2{power: 1} * du/dx0{power: 1} + 0. = d^2u/dx0^2{power: 1} * du/dx0{power: 1}'

    grid, data = wave_data(os.path.join(foldername, 'wave_sln_80.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ann_pretrained.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=5,
                                 coordinate_tensors=(grid[..., 0], grid[..., 1]),
                                 verbose_params={'show_iter_idx': True},
                                 device='cpu')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
    #                                  preprocessor_kwargs={})
    epde_search_obj.set_preprocessor(default_preprocessor_type='spectral',
                                     preprocessor_kwargs={"n":80})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 2),
                                additional_tokens=[])

    assert compare_equations(eq_wave_symbolic, eq_wave_incorrect, epde_search_obj)


def wave_discovery(foldername, noise_level):
    """
    Performs wave equation discovery using an evolutionary search.
        
        This method orchestrates the search for a differential equation that describes
        wave propagation. It involves loading data, adding noise to simulate real-world
        conditions, setting up a preprocessor for data conditioning, and then
        fitting the evolutionary search object to the prepared data. The goal is to
        find the equation that best represents the underlying dynamics of the wave.
        
        Args:
            foldername (str): The name of the folder containing the data files
                ('wave_sln_80.csv' and 'ann_pretrained.pickle').
            noise_level (float): The level of noise to add to the data.
        
        Returns:
            EpdeSearch: The trained EPDE search object, containing the discovered equation.
    """
    grid, data = wave_data(os.path.join(foldername, 'wave_sln_80.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ann_pretrained.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=20,
                                      coordinate_tensors=(grid[..., 0], grid[..., 1]), device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e4})
    # epde_search_obj.set_preprocessor(default_preprocessor_type='spectral',
    #                                  preprocessor_kwargs={"n": 80})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=5)


    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-5, 1e2)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

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
    wave_folder_name = os.path.join(directory)

    wave_test(fit_operator, wave_folder_name, 0)
    # wave_discovery(wave_folder_name, 0)