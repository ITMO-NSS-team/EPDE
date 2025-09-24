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
    Loads a pre-trained Physics-Informed Neural Network (PINN) model from a file.
    
    This function attempts to load a previously trained PINN model from disk.  Loading a pre-trained model can save significant time by avoiding retraining, especially when exploring different equation structures or refining existing models within the EPDE framework.
    
    Args:
        ann_filename: The filename of the pickled ANN model.
    
    Returns:
        The loaded ANN model. Returns None if the file is not found, indicating that a new model may need to be trained from scratch.
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
    Compares two symbolic equations to determine which better represents the underlying dynamics of the system.
    
        It translates both equations into a comparable form, applies a fitting procedure,
        and then assesses their stability based on their coefficients. This helps in
        identifying the equation that more accurately captures the system's behavior.
        
        Args:
            correct_symbolic: The correct symbolic equation as a string.
            eq_incorrect_symbolic: The incorrect symbolic equation as a string.
            search_obj: An EpdeSearch object containing the search pool.
            all_vars: A list of variable names to consider (default: ['u']).
        
        Returns:
            bool: True if the correct equation exhibits better coefficients stability
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
    print("fitness_value: ", [[correct_eq.vals[var].fitness_value, incorrect_eq.vals[var].fitness_value] for var in all_vars])
    print("coefficients_stability: ", [[correct_eq.vals[var].coefficients_stability, incorrect_eq.vals[var].coefficients_stability] for var in
           all_vars])
    print("aic: ", [[correct_eq.vals[var].aic, incorrect_eq.vals[var].aic] for var in all_vars])

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


def kdv_data(filename, shape=80):
    """
    Loads data and creates corresponding grids for equation discovery.
    
    This method reads data from a specified file, and constructs spatial and temporal grids
    to align with the data. These grids are essential for representing the domain
    over which the differential equation is defined. The shape of the grid is fixed to 80.
    This setup facilitates the application of evolutionary algorithms to discover the underlying equation.
    
    Args:
        filename (str): The name of the data file to load (CSV format).
    
    Returns:
        tuple: A tuple containing the grids and the data.
            The grids are a meshgrid of time and space coordinates,
            representing the independent variables of the differential equation.
            The data is a NumPy array loaded from the file, representing the dependent variable.
    """
    shape = 80

    print(os.path.dirname(__file__))
    data = np.loadtxt(filename, delimiter=',').T
    t = np.linspace(0, 1, shape + 1)
    x = np.linspace(0, 1, shape + 1)
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, data


def kdv_data_h(filename, shape=80):
    """
    Loads KdV equation data and prepares it for equation discovery.
    
    This function loads data representing the solution of the KdV equation
    from a specified file. It also generates the corresponding time and
    space grids necessary for representing the data as a function of
    these independent variables. This setup is crucial for algorithms
    aiming to identify the underlying differential equation.
    
    Args:
        filename (str): The path to the .npy file containing the KdV equation data.
        shape (int, optional): An integer intended to define the shape of the data.
            This parameter is overridden internally and does not affect the output. Defaults to 80.
    
    Returns:
        tuple: A tuple containing the time and space grids (t, x) and the loaded data.
            - grids (tuple of numpy.ndarray): A tuple containing two numpy arrays,
              representing the time and space grids, respectively. These grids are
              created using np.meshgrid with t ranging from 0 to 1 (120 points)
              and x ranging from -3 to 3 (480 points).
            - data (numpy.ndarray): The loaded KdV equation data from the specified file.
    
    Why:
        This function prepares the KdV equation data and its corresponding grids
        so that equation discovery algorithms can use them to learn the underlying
        differential equation. The grids provide the coordinate system in which
        the data is defined, which is essential for calculating derivatives and
        evaluating potential equation candidates.
    """
    shape = 119

    print(os.path.dirname(__file__))
    data = np.load(filename)

    t = np.linspace(0, 1, 120)
    x = np.linspace(-3, 3, 480)
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, data


def kdv_data_sga(filename):
    """
    Loads KdV data from a .mat file and prepares it for symbolic equation discovery.
    
    This function loads the KdV data, extracts the solution `u`, spatial grid `x`,
    and temporal grid `t`, and then creates a meshgrid from `t` and `x`. This
    structured data is essential for representing the solution on a grid, which
    is a prerequisite for applying symbolic regression techniques to discover the
    underlying partial differential equation.
    
    Args:
        filename: The name of the .mat file containing the KdV data.
    
    Returns:
        tuple: A tuple containing:
            - grids: A tuple representing the meshgrid of t and x.
            - u: The solution data.
    
    Why:
        The KdV data is loaded and prepared in this way to create a structured
        representation of the solution on a grid. This is necessary for
        subsequent symbolic regression to discover the underlying partial
        differential equation. The meshgrid provides the coordinates for each
        point in the solution, which is used to evaluate candidate equations.
    """
    data = scio.loadmat(filename)
    u = data.get("uu").T
    n, m = u.shape
    x = np.squeeze(data.get("x")).reshape(-1, 1)
    t = np.squeeze(data.get("tt").reshape(-1, 1))
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, u

def kdv_sindy_data(filename):
    """
    Prepares data from a MATLAB file for equation discovery.
        
        Loads data from a specified MATLAB file, extracts the time series,
        spatial coordinates, and solution data, and prepares them for use in
        equation discovery algorithms. This function ensures that the data is
        correctly formatted and accessible for subsequent analysis, enabling
        the identification of potential governing equations.
        
        Args:
            filename: The name of the MATLAB file containing the data.
        
        Returns:
            tuple: A tuple containing:
                - grids: A tuple of meshgrids representing the time and spatial
                  coordinates.
                - u: A NumPy array representing the solution data.
    """
    data = scio.loadmat(filename)
    t = np.ravel(data['t'])
    x = np.ravel(data['x'])
    u = np.real(data['usol'])
    u = np.transpose(u)
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, u

def KdV_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the performance of the EPDE search on the Korteweg-de Vries equation.
        
        This method aims to validate the equation discovery process by attempting to identify the KdV equation from noisy data. 
        It compares the identified equation against known correct and incorrect formulations to assess the accuracy of the search.
        The method employs a finite difference preprocessor to prepare the data for the equation search.
        This test helps ensure that the EPDE framework can reliably extract governing equations from data, even in the presence of noise.
        
        Args:
            operator: The compound operator to use for equation comparison.
            foldername: The name of the folder containing the data files ('data.csv' and 'kdv_0_ann.pickle').
            noise_level: The level of noise to add to the data (default: 0).
        
        Returns:
            bool: True if the identified equation matches the correct equation and
                does not match the incorrect equation, False otherwise.
    """
    # Test scenario to evaluate performance on Korteweg-de Vries equation
    # eq_kdv_symbolic = '-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} + \
    #                        1.0 * sin{power: 1, freq: 1.0, dim: 1} * cos{power: 1, freq: 1.0, dim: 0} + \
    #                        0.0 = du/dx0{power: 1.0}'

    eq_kdv_symbolic = '-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} + \
                           1.0 * cos(t)sin(x){power: 1.0} + \
                           0.0 = du/dx0{power: 1.0}'

    # eq_kdv_symbolic = '-1.0 * du/dx1{power: 1.0} * u{power: 1.0} + -0.0025 * d^3u/dx1^3{power: 1.0} + \
    #                        0.0 = du/dx0{power: 1.0}'

    eq_kdv_incorrect = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'

    grid, data = kdv_data(os.path.join(foldername, 'data.csv'))
    # grid, data = kdv_data(os.path.join(foldername, 'Kdv.mat'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'kdv_0_ann.pickle'))

    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    trig_tokens = TrigonometricTokens(freq=(1 - 1e-8, 1 + 1e-8),
                                      dimensionality=dimensionality)

    epde_search_obj = EpdeSearch(use_solver = False, use_pic=True, boundary = 10,
                                 coordinate_tensors = (grid[0], grid[1]), verbose_params = {'show_iter_idx' : True},
                                 device = 'cuda')

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                      token_labels=['cos(t)sin(x)'],
                                      evaluator=custom_trig_evaluator,
                                      params_ranges=trig_params_ranges,
                                      params_equality_ranges=trig_params_equal_ranges,
                                      meaningful=True, unique_token_type=True)

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                  preprocessor_kwargs={'epochs_max': 1e4}) #'epochs_max': 5e4
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})  # 'epochs_max': 5e4

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2, 3),
                                additional_tokens = [custom_trig_tokens,]) # data_nn

    # np.save(os.path.join(foldername, 'kdv_0_derivs.npy'), epde_search_obj.derivatives)

    assert compare_equations(eq_kdv_symbolic, eq_kdv_incorrect, epde_search_obj)


def KdV_h_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the performance on the Korteweg-de Vries equation.
        
        This method evaluates the ability of the EPDE framework to identify the
        Korteweg-de Vries (KdV) equation from data. It sets up a test scenario
        involving data loading, noise addition, and equation search, ultimately
        assessing whether the identified equation matches the known KdV equation.
        This serves as a benchmark to ensure the framework can accurately
        recover governing equations from noisy data, a crucial aspect of its
        equation discovery capabilities.
        
        Args:
            operator: CompoundOperator object for equation comparison.
            foldername: Path to the folder containing the data files.
            noise_level: Level of noise to add to the data (default: 0).
        
        Returns:
            bool: True if the identified equation matches the expected KdV equation, False otherwise.
    """
    # Test scenario to evaluate performance on Korteweg-de Vries equation
    eq_kdv_symbolic = '-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} + \
                           0.0 = du/dx0{power: 1.0}'

    eq_kdv_incorrect = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'

    grid, data = kdv_data_h(os.path.join(foldername, 'data_kdv_homogen.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'kdv_0_ann.pickle'))

    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    trig_tokens = TrigonometricTokens(freq=(1 - 1e-8, 1 + 1e-8),
                                      dimensionality=dimensionality)

    epde_search_obj = EpdeSearch(use_solver = False, use_pic=True, boundary = 20,
                                 coordinate_tensors = (grid[0], grid[1]), verbose_params = {'show_iter_idx' : True},
                                 device = 'cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={}) #'epochs_max': 5e4

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2, 3),
                                additional_tokens = []) # , data_nn = data_nn

    # np.save(os.path.join(foldername, 'kdv_0_derivs.npy'), epde_search_obj.derivatives)

    assert compare_equations(eq_kdv_symbolic, eq_kdv_incorrect, epde_search_obj)


def KdV_sga_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the performance of the symbolic genetic algorithm (SGA) on the Korteweg-de Vries (KdV) equation.
        
        This method evaluates the SGA's ability to identify the underlying differential equation of the KdV system from data.
        It sets up a controlled experiment by loading KdV data, introducing noise to simulate real-world imperfections, and then employing the SGA to discover the equation.
        The goal is to assess whether the algorithm can accurately extract the governing equation, demonstrating its capability to automate the discovery of differential equations from data.
        
        Args:
            operator: The CompoundOperator instance to be tested.
            foldername: The name of the folder containing the KdV data and pretrained PINN model.
            noise_level: The level of noise to add to the data (default: 0).
        
        Returns:
            bool: True if the discovered equation matches the expected KdV equation, False otherwise.
    """
    # Test scenario to evaluate performance on Korteweg-de Vries equation
    eq_kdv_symbolic = '-1 * du/dx1{power: 1.0} * u{power: 1.0} + -0.0025 * d^3u/dx1^3{power: 1.0} + \
                           0.0 = du/dx0{power: 1.0}'

    eq_kdv_incorrect = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'

    grid, data = kdv_data_sga(os.path.join(foldername, 'Kdv.mat'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'kdv_0_ann.pickle'))

    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    trig_tokens = TrigonometricTokens(freq=(1 - 1e-8, 1 + 1e-8),
                                      dimensionality=dimensionality)

    epde_search_obj = EpdeSearch(use_solver = False, use_pic=True, boundary = 10,
                                 coordinate_tensors = (grid[0], grid[1]), verbose_params = {'show_iter_idx' : True},
                                 device = 'cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={}) #'epochs_max': 5e4

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2, 3),
                                additional_tokens = []) # , data_nn = data_nn

    # np.save(os.path.join(foldername, 'kdv_0_derivs.npy'), epde_search_obj.derivatives)

    assert compare_equations(eq_kdv_symbolic, eq_kdv_incorrect, epde_search_obj)


def kdv_discovery(foldername, noise_level):
    """
    Discovers the KdV equation using the EPDE framework.
        
        This method automates the process of identifying the Korteweg-de Vries (KdV)
        equation from data. It leverages the EPDE framework to explore potential
        equation structures that best represent the provided data, incorporating
        noise handling, custom token definitions, and model fitting. The goal is to
        uncover the underlying dynamics of the KdV equation from data, providing
        a symbolic representation of the system's behavior.
        
        Args:
            foldername: The name of the folder containing the data.csv file and the
                pretrained PINN model.
            noise_level: The level of noise to add to the data.
        
        Returns:
            EpdeSearch: The trained EPDE search object, containing the discovered
                equation and related information.
    """
    grid, data = kdv_data(os.path.join(foldername, 'data.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=5,
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 30

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=20)

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                         token_labels=['cos(t)sin(x)'],
                                         evaluator=custom_trig_evaluator,
                                         params_ranges=trig_params_ranges,
                                         params_equality_ranges=trig_params_equal_ranges,
                                         meaningful=True, unique_token_type=False)

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))
    trig_tokens._token_family.set_status(unique_specific_token=True, unique_token_type=False,
                                  meaningful=True)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-5, 1e-2)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[trig_tokens], #custom_trig_tokens
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    res = epde_search_obj.equations(only_print=False, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


def kdv_h_discovery(foldername, noise_level):
    """
    Discovers the governing equation for the KdV equation from noisy data using EPDE.
    
    This method applies the EPDE framework to identify the Korteweg-de Vries (KdV) equation
    from noisy data. It involves loading data, adding noise, preprocessing it using finite differences,
    and then searching for the equation that best represents the data's dynamics. The goal is to 
    automatically learn the underlying equation without relying on specific prior assumptions 
    about its form, showcasing EPDE's ability to extract governing equations from complex datasets.
    
    Args:
        foldername: The name of the folder containing the data.
        noise_level: The level of noise to add to the data.
    
    Returns:
        EpdeSearch: The EPDE search object containing the discovered equations.
    """
    grid, data = kdv_data_h(os.path.join(foldername, 'data_kdv_homogen.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=20,
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=15)


    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                         token_labels=['cos(t)sin(x)'],
                                         evaluator=custom_trig_evaluator,
                                         params_ranges=trig_params_ranges,
                                         params_equality_ranges=trig_params_equal_ranges,
                                         meaningful=True, unique_token_type=True)

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-5, 1e-0)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

def kdv_sga_discovery(foldername, noise_level):
    """
    Discovers the KdV equation using the EPDE framework with SGA.
    
        This method leverages the EPDE framework's evolutionary algorithms to
        identify the KdV equation from data. It prepares the data, defines the
        search space with custom tokens relevant to the KdV equation, and then
        employs a Sequential Genetic Algorithm (SGA) to explore potential equation
        structures. The discovered equations are then presented and visualized.
        This automated discovery process helps in understanding the underlying
        dynamics of the system by finding the best-fit differential equation.
    
        Args:
            foldername: The name of the folder containing the data and pretrained PINN.
            noise_level: The level of noise to add to the data.
    
        Returns:
            EpdeSearch: The trained EPDE search object.
    """
    grid, data = kdv_data_sga(os.path.join(foldername, 'Kdv.mat'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=20,
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=5)


    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                         token_labels=['cos(t)sin(x)'],
                                         evaluator=custom_trig_evaluator,
                                         params_ranges=trig_params_ranges,
                                         params_equality_ranges=trig_params_equal_ranges,
                                         meaningful=True, unique_token_type=True)

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-3, 1e-0)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.equations(only_print=False, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


def kdv_sindy_discovery(foldername, noise_level):
    """
    Performs KdV SINDy discovery using EPDE.
        
        This method leverages EPDE to identify the underlying KdV equation from noisy data. 
        It automates the equation discovery process by setting up an EPDE search, defining 
        relevant equation terms (including custom trigonometric functions), and fitting the model 
        to the provided data. The goal is to find the equation that best represents the 
        system's dynamics, even in the presence of noise.
        
        Args:
            foldername: The name of the folder containing the data and pretrained PINN.
            noise_level: The level of noise to add to the data.
        
        Returns:
            EpdeSearch: The EPDE search object containing the discovered equations.
    """
    grid, data = kdv_sindy_data(os.path.join(foldername, 'kdv_sindy.mat'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=10,
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=5)

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                         token_labels=['cos(t)sin(x)'],
                                         evaluator=custom_trig_evaluator,
                                         params_ranges=trig_params_ranges,
                                         params_equality_ranges=trig_params_equal_ranges,
                                         meaningful=True, unique_token_type=False)

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))
    trig_tokens._token_family.set_status(unique_specific_token=True, unique_token_type=False,
                                  meaningful=True)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-8, 1e0)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[custom_trig_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    res = epde_search_obj.equations(only_print=False, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


if __name__ == "__main__":
    import torch
    from epde.operators.utils.default_parameter_loader import EvolutionaryParams
    print("CUDA available:", torch.cuda.is_available())
    # Operator = fitness.SolverBasedFitness # Replace by the developed PIC-based operator.
    # Operator = fitness.PIC
    Operator = fitness.L2LRFitness
    params = EvolutionaryParams()
    operator_params = params.get_default_params_for_operator('DiscrepancyBasedFitnessWithCV') #{"penalty_coeff": 0.2, "pinn_loss_mult": 1e4}
    # operator_params = {"penalty_coeff": 0.2, "pinn_loss_mult": 1e4}
    print('operator_params ', operator_params)
    fit_operator = prepare_suboperators(Operator(list(operator_params.keys())), operator_params)

    # Paths
    directory = os.path.dirname(os.path.realpath(__file__))
    kdv_folder_name = os.path.join(directory)

    # KdV_test(fit_operator, kdv_folder_name, 0)
    # KdV_h_test(fit_operator, kdv_folder_name, 0)
    # KdV_sga_test(fit_operator, kdv_folder_name, 0)

    kdv_discovery(kdv_folder_name, 0)
    # kdv_h_discovery(kdv_folder_name, 0)
    # kdv_sga_discovery(kdv_folder_name, 5)
    # kdv_sindy_discovery(kdv_folder_name, 0)