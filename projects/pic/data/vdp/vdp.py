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
    Compares two symbolic equations to determine which better represents the underlying dynamics.
    
        This method translates both equations into a comparable form, applies a fitting
        operator to assess their accuracy, and then evaluates their coefficient stability.
        The comparison helps in identifying the equation that more robustly captures the
        system's behavior.
    
        Args:
            correct_symbolic (str): The correct symbolic equation as a string.
            eq_incorrect_symbolic (str): The incorrect symbolic equation as a string.
            search_obj (EpdeSearch): An EpdeSearch object containing the search pool and
                necessary data for equation fitting.
            all_vars (List[str], optional): A list of variable names to consider. Defaults to ['u'].
    
        Returns:
            bool: True if the correct equation exhibits better coefficient stability
                than the incorrect equation across all variables, indicating a superior
                representation of the system's dynamics. False otherwise.
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


def VdP_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the performance on the Van-der-Pol oscillator.
        
        This method evaluates the ability to identify the Van-der-Pol oscillator equation from data.
        It sets up a test scenario by loading data, adding noise, defining symbolic and incorrect equations,
        creating tokens representing potential equation terms, and then runs a search to compare identified equations
        against the known symbolic form. This helps assess the framework's capability to rediscover governing equations
        from noisy data.
        
        Args:
            operator: The compound operator to be tested.
            foldername: The name of the folder containing the data files.
            noise_level: The level of noise to add to the data. Defaults to 0.
        
        Returns:
            bool: True if the comparison of equations passes, False otherwise.
    """
    # u'' + E (u^2 - 1)u' + u = 0, where $\mathcal{E}$ is a positive constant (in the example we will use $\mathcal{E} = 0.2$)
    # Test scenario to evaluate performance on Van-der-Pol oscillator
    eq_vdp_symbolic = '-0.2 * u{power: 2.0} * du/dx0{power: 1.0} + 0.2 * du/dx0{power: 1.0} + -1.0 * u{power: 1.0} + -0.0 \
                           = d^2u/dx0^2{power: 1.0}'
    eq_vdp_incorrect = '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * x_0{power: 1.0, dim: 0.0} + -4.0 * u{power: 1.0} + -0.0 \
                        = du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'

    # grid, data = load_data(os.path.join(foldername, 'data.npy'))

    step = 0.05; steps_num = 320
    t = np.arange(start = 0., stop = step * steps_num, step = step)
    data = np.load(os.path.join(foldername, 'vdp_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'vdp_ann_pretrained.pickle'))

    dimensionality = 0

    trig_tokens = TrigonometricTokens(freq = (2 - 1e-8, 2 + 1e-8),
                                      dimensionality = dimensionality)
    grid_tokens = GridTokens(['x_0',], dimensionality = dimensionality, max_power = 2)

    epde_search_obj = EpdeSearch(use_solver = False, use_pic=True, boundary = 2,
                                 coordinate_tensors = [t,], verbose_params = {'show_iter_idx' : True},
                                 device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2,),
                                additional_tokens = [grid_tokens, trig_tokens], data_nn = data_nn)

    assert compare_equations(eq_vdp_symbolic, eq_vdp_incorrect, epde_search_obj)


def vdp_discovery(foldername, noise_level):
    """
    Discovers the governing equation of the Van der Pol oscillator from noisy data.
    
    This method automates the process of identifying the underlying differential equation by employing evolutionary algorithms and multi-objective optimization to find equation structures that best fit the provided data. It leverages data preprocessing, custom token definitions, and numerical solvers to achieve accurate discovery. The method is applied to the Van der Pol oscillator to demonstrate its capability in uncovering governing equations from complex systems.
    
    Args:
        foldername: The name of the folder containing the data and pretrained model.
        noise_level: The level of noise to add to the data.
    
    Returns:
        EpdeSearch: The EPDE search object containing the discovered equations and solutions.
    """
    step = 0.05;
    steps_num = 320
    t = np.arange(start=0., stop=step * steps_num, step=step)
    data = np.load(os.path.join(foldername, 'vdp_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'vdp_ann_pretrained.pickle'))

    dimensionality = 0

    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=1,
                                 coordinate_tensors=(t,), verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=30)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    epde_search_obj.fit(data=[noised_data, ], variable_names=['u', ], max_deriv_order=(2, 3),
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-5, 1e-0)) #

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

    vdp_folder_name = os.path.join(directory)

    # VdP_test(fit_operator, vdp_folder_name, 0)
    vdp_discovery(vdp_folder_name, 0)