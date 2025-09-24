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
    Loads a pretrained Physics-Informed Neural Network (PINN) model from a file.
    
    This function attempts to load a previously trained PINN model from disk.
    Loading a pre-trained model can save significant time by avoiding retraining,
    especially when exploring different equation structures or refining existing models.
    
    Args:
        ann_filename (str): The filename of the pickled ANN model.
    
    Returns:
        object: The loaded ANN model if the file exists; otherwise, returns None.
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
    Compares two symbolic equations to determine which one better represents the underlying dynamics of the system.
        
        It translates both the correct and incorrect symbolic equations into a
        form suitable for comparison, applies a fitting operator to estimate the coefficients, and then
        assesses their stability. The equation with more stable coefficients is considered a better
        representation of the system's dynamics.
        
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
    Prepares the compound fitness operator by configuring its sub-operators for sparsity and coefficient calculation.
    
        This method instantiates and configures the sparsity and coefficient calculation sub-operators,
        then sets them within the provided compound fitness operator. It also maps the operator
        between gene and chromosome levels based on a condition that checks if the fitness has already been calculated.
        This ensures that the fitness calculation is performed efficiently and only when necessary during the evolutionary process
        of discovering differential equations.
    
        Args:
            fitness_operator (CompoundOperator): The compound fitness operator to prepare.
            operator_params (dict): A dictionary of parameters for the fitness operator.
    
        Returns:
            CompoundOperator: The prepared compound fitness operator with configured
            sub-operators and level mapping.
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


def ODE_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the ability to identify a known 2nd order ODE from data.
    
        This test evaluates whether the framework can rediscover the equation
        x'' + sin(2t) x' + 4 x = 1.5 t, represented as g1 x'' + g2 x' + g3 x = g4,
        from a dataset generated by its solution. This serves as a validation
        of the core equation discovery process.
    
        Args:
            operator: CompoundOperator object, responsible for equation manipulation.
            foldername: The name of the folder containing the data files ('ode_data.npy' and 'ode_0_ann.pickle').
            noise_level: The level of noise to add to the data, simulating real-world imperfections.
    
        Returns:
            bool: True if the discovered symbolic equation matches the expected equation, indicating successful identification, False otherwise.
    """
    # Test scenario to evaluate performance on simple 2nd order ODE
    # x'' + sin(2t) x' + 4 x = 1.5 t,  written as $g_{1} x'' + g_{2} x' + g_{3} x = g_{4}
    # g1 = lambda x: 1.
    # g2 = lambda x: np.sin(2*x)
    # g3 = lambda x: 4.
    # g4 = lambda x: 1.5*x

    eq_ode_symbolic = '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * x_0{power: 1.0, dim: 0.0} + -4.0 * u{power: 1.0} + -0.0 \
                       = du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'
    eq_ode_incorrect = '1.0 * du/dx0{power: 1.0} + 3.5 * x_0{power: 1.0, dim: 0.0} * u{power: 1.0} + -1.2 \
                        = du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'

    step = 0.05;
    steps_num = 320
    t = np.arange(start=0., stop=step * steps_num, step=step)
    data = np.load(os.path.join(foldername, 'ode_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ode_0_ann.pickle')).cuda()

    dimensionality = 0

    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=[t,], verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u', ], max_deriv_order=(2,),
                                additional_tokens=[grid_tokens, trig_tokens], data_nn=data_nn)

    assert compare_equations(eq_ode_symbolic, eq_ode_incorrect, epde_search_obj)


def ODE_discovery(foldername, noise_level):
    """
    Discovers the underlying ODE from noisy data.
    
    This method orchestrates the search for a differential equation that best describes the provided data, even when the data is corrupted by noise.
    It prepares the data, defines the possible equation components, and then employs an optimization algorithm to find the equation that best fits the data.
    This is done to automatically identify the mathematical relationships governing the observed system's behavior.
    
    Args:
        foldername: The name of the folder containing the 'ode_data.npy' file
            and the 'ode_0_ann.pickle' file, which store the data and a pretrained neural network, respectively.
        noise_level: The level of noise added to the data, influencing the search's robustness.
    
    Returns:
        EpdeSearch: The trained EPDE search object, containing the discovered
            equations and related information, allowing for further analysis and validation of the identified model.
    """
    step = 0.05
    steps_num = 320
    t = np.arange(start=0., stop=step * steps_num, step=step)
    data = np.load(os.path.join(foldername, 'ode_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ode_0_ann.pickle')).cpu()

    dimensionality = 0

    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=[t,], verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=15)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    epde_search_obj.fit(data=[noised_data, ], variable_names=['u', ], max_deriv_order=(2, 3),
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-10, 1e-0)) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)

    # import pickle
    # fname = os.path.join(r'C:\Users\user\EPDE_tests\models', 'ode_0_ann.pickle')
    # with open(fname, 'wb') as output_file:
    #     pickle.dump(global_var.solution_guess_nn, output_file)
    epde_search_obj.visualize_solutions()
    return epde_search_obj

def ODE_simple_discovery(foldername, noise_level):
    """
    Performs a simple ODE discovery using the EPDE framework.
        
        This method demonstrates the core functionality of EPDE by discovering an ordinary
        differential equation (ODE) from a synthetically generated dataset. It showcases
        the process of defining the search space using token sets, configuring the EPDE
        search object with appropriate parameters, and visualizing the discovered solutions.
        The goal is to find a balance between model complexity and accuracy in representing
        the underlying dynamics of the system.
    
        Args:
            foldername: The name of the folder to store results.
            noise_level: The level of noise to add to the data (currently unused in the provided code).
        
        Returns:
            EpdeSearch: The configured and executed EpdeSearch object, containing the discovered equations and related information.
    """
    C = 1.3
    t = np.linspace(0, 4 * np.pi, 200)
    x = np.sin(t) + C * np.cos(t)
    x_dot = np.cos(t) - C * np.sin(t)
    x_dot_dot = -np.sin(t) - C * np.cos(t)
    x_dot_dot_dot = -np.cos(t) + C * np.sin(t)

    dimensionality = x.ndim - 1

    trig_tokens = TrigonometricTokens(freq = (0.999, 1.001),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=[t,], verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=5)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    epde_search_obj.fit(data=[x, ], variable_names=['u', ], max_deriv_order=(2, 3),
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-6, 1e0)) #

    epde_search_obj.equations(only_print=True, num=1)

    # import pickle
    # fname = os.path.join(r'C:\Users\user\EPDE_tests\models', 'ode_0_ann.pickle')
    # with open(fname, 'wb') as output_file:
    #     pickle.dump(global_var.solution_guess_nn, output_file)
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
    ode_folder_name = os.path.join(directory)

    # ODE_test(fit_operator, ode_folder_name, 0)
    # ODE_discovery(ode_folder_name, 0)
    ODE_simple_discovery(ode_folder_name, 0)

