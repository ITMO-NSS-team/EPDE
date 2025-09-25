'''
OLD FILE FOR TESTS. ALL TESTS ARE IN THEIR OWN FILES NOW!
'''
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

# Introduce noise levels, test with complex setups
# np.random.seed(0)

def load_pretrained_PINN(ann_filename):
    """
    Loads a pre-trained Physics-Informed Neural Network (PINN) from a file to potentially accelerate the equation discovery process.
    
        This method attempts to load a pickled PINN model from the specified file.
        If the file is not found, it prints a message indicating that a new model
        will be trained from scratch. Loading a pre-trained model can save
        computational resources by providing a good initial guess for the network's
        parameters, which is especially useful when dealing with complex datasets
        or high-dimensional problems.
        
        Args:
            ann_filename: The filename of the pickled PINN model.
        
        Returns:
            The loaded PINN model if the file is found, otherwise None.
    """
    try:
        with open(ann_filename, 'rb') as data_input_file:  
            data_nn = pickle.load(data_input_file)
    except FileNotFoundError:
        print('No model located, proceeding with ann approx. retraining.')
        data_nn = None
    return data_nn


def compare_equations(correct_symbolic: str, eq_incorrect_symbolic: str, 
                      search_obj: EpdeSearch, all_vars: List[str] = ['u',]) -> bool:
    """
    Compares two symbolic equations to determine which better represents the underlying dynamics.
        
        It translates both equations into a usable format, applies a fitting operator to
        assess their accuracy, and then compares the coefficient stability of the variables
        in both equations. This helps in identifying equations that are more robust and
        reliable representations of the system.
        
        Args:
            correct_symbolic: The symbolic representation of the correct equation.
            eq_incorrect_symbolic: The symbolic representation of the incorrect equation.
            search_obj: An EpdeSearch object containing the search pool.
            all_vars: A list of variable names to consider (default: ['u']).
        
        Returns:
            bool: True if the coefficient stability of all variables in the
                correct equation is less than that of the incorrect equation,
                False otherwise.
    """
    metaparams = {('sparsity', var): {'optimizable': False, 'value': 1E-6} for var in all_vars}

    correct_eq = translate_equation(correct_symbolic, search_obj.pool, all_vars = all_vars)
    for var in all_vars:
        correct_eq.vals[var].main_var_to_explain = var
        correct_eq.vals[var].metaparameters = metaparams
    print(correct_eq.text_form)
 
    incorrect_eq = translate_equation(eq_incorrect_symbolic, search_obj.pool, all_vars = all_vars)   #  , all_vars = ['u', 'v'])
    for var in all_vars:
        incorrect_eq.vals[var].main_var_to_explain = var
        incorrect_eq.vals[var].metaparameters = metaparams
    print(incorrect_eq.text_form)

    fit_operator.apply(correct_eq, {})
    fit_operator.apply(incorrect_eq, {})
    print([[correct_eq.vals[var].fitness_value, incorrect_eq.vals[var].fitness_value] for var in all_vars])
    print([[correct_eq.vals[var].coefficients_stability, incorrect_eq.vals[var].coefficients_stability] for var in all_vars])
    print([[correct_eq.vals[var].aic, incorrect_eq.vals[var].aic] for var in all_vars])

    # print([correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability for var in all_vars])
    return all([correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability for var in all_vars])


def prepare_suboperators(fitness_operator: CompoundOperator, operator_params: dict) -> CompoundOperator:
    """
    Prepares the sub-operators required for the compound fitness operator.
    
        This method configures the sparsity and coefficient calculation sub-operators,
        and sets them within the provided fitness operator. It also maps the operator
        between gene and chromosome levels ensuring compatibility between the symbolic
        representation of equations and the data. This setup is crucial for evaluating
        the fitness of candidate equations during the evolutionary search process.
    
        Args:
          fitness_operator: The compound fitness operator to prepare.
          operator_params: A dictionary of parameters for the fitness operator.
    
        Returns:
          The prepared compound fitness operator, ready for use in the evolutionary process.
    """
    sparsity = LASSOSparsity()
    coeff_calc = LinRegBasedCoeffsEquation()

    # sparsity = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
    # coeff_calc = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')

    fitness_operator.set_suboperators({'sparsity' : sparsity,
                                       'coeff_calc' : coeff_calc})
    fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
    fitness_operator.params = operator_params
    fitness_operator = map_operator_between_levels(fitness_operator, 'gene level', 'chromosome level',
                                                  objective_condition=fitness_cond)
    return fitness_operator


def noise_data(data, noise_level):
    """
    Adds random noise to the input data based on its standard deviation.
    
    This function perturbs the data by adding Gaussian noise scaled 
    by the data's standard deviation and a specified noise level. 
    This is useful for simulating real-world measurement errors or 
    for data augmentation purposes, ensuring the discovered equations 
    are robust to small variations in the input.
    
    Args:
        data (np.ndarray): The input data to which noise will be added.
        noise_level (float): The standard deviation of the noise, expressed as a percentage of the data's standard deviation.
    
    Returns:
        np.ndarray: The data with added noise.
    """
    # add noise level to the input data
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def ODE_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the ability of the EPDE framework to identify a simple 2nd order ODE from data.
    
        This method defines a symbolic representation of the ODE
        x'' + sin(2t) x' + 4 x = 1.5 t, sets up an EPDE search problem, loads data (potentially adding noise),
        and then checks if the identified equation is closer to the correct symbolic representation
        than to an intentionally incorrect one. This verifies that the search process can distinguish
        between plausible and implausible equation forms based on the provided data.
    
        Args:
            operator: CompoundOperator object, not used in the current implementation.
            foldername: Path to the folder containing the data and pretrained PINN model.
            noise_level: Level of noise to add to the data (default: 0).
    
        Returns:
            bool: True if the comparison between the identified equation and the
                incorrect equation passes, indicating the test was successful.
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

    step = 0.05; steps_num = 320
    t = np.arange(start = 0., stop = step * steps_num, step = step)
    data = np.load(os.path.join(foldername, 'ode_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ode_0_ann.pickle')).cuda()

    dimensionality = 0

    trig_tokens = TrigonometricTokens(freq = (2 - 1e-8, 2 + 1e-8),
                                      dimensionality = dimensionality)
    grid_tokens = GridTokens(['x_0',], dimensionality = dimensionality, max_power = 2)

    epde_search_obj = EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 10,
                                 coordinate_tensors = (t,), verbose_params = {'show_iter_idx' : True},
                                 device = 'cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2,),
                                additional_tokens = [grid_tokens, trig_tokens], data_nn = data_nn)

    assert compare_equations(eq_ode_symbolic, eq_ode_incorrect, epde_search_obj)


def VdP_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the performance on the Van-der-Pol oscillator.
        
        This method evaluates the EPDE framework's ability to identify the Van-der-Pol oscillator equation from data. 
        It sets up a test scenario by loading data, adding noise, defining a symbolic representation of the equation, 
        and creating a pool of candidate terms. The method then compares the discovered equation with an incorrect one 
        to assess the search's accuracy. This helps to ensure that the evolutionary algorithm can effectively 
        distinguish the true equation from other possibilities.
        
        Args:
            operator: Compound operator.
            foldername: Name of the folder containing the data.
            noise_level: Level of noise to add to the data (default: 0).
        
        Returns:
            bool: True if the symbolic and incorrect equations are the same, False otherwise.
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

    epde_search_obj = EpdeSearch(use_solver = False, use_pic=True, boundary = 10,
                                 coordinate_tensors = (t,), verbose_params = {'show_iter_idx' : True},
                                 device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2,),
                                additional_tokens = [grid_tokens, trig_tokens], data_nn = data_nn)

    assert compare_equations(eq_vdp_symbolic, eq_vdp_incorrect, epde_search_obj)


def lorenz_discovery(foldername, noise_level):
    """
    Discovers the Lorenz system equations using the EPDE framework.
    
    This method leverages the EPDE framework to identify the underlying equations
    governing the Lorenz attractor directly from data. It automates the equation
    discovery process by configuring an EPDE search object, performing a search
    for the best equation structures, and presenting the discovered equations.
    This is done to provide insights into the dynamics of the Lorenz system
    and build a predictive model.
    
    Args:
        foldername: The name of the folder containing the data. (Not used in the provided code snippet)
        noise_level: The level of noise in the data. (Not used in the provided code snippet)
    
    Returns:
        EpdeSearch: The trained EPDE search object, containing the discovered equations and search results.
    """
    t_file = os.path.join(os.path.dirname( __file__ ), 'data\\lorenz\\t.npy')
    t = np.load(t_file)
    data_file = os.path.join(os.path.dirname(__file__), 'data\\lorenz\\lorenz.npy')
    data = np.load(data_file)

    end = 1000
    t = t[:end]
    x = data[:end, 0]
    y = data[:end, 1]
    z = data[:end, 2]

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
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=100)

    factors_max_number = {'factors_num': [1, 2], 'probas' : [0.8, 0.2]}

    epde_search_obj.fit(data=[x, y, z], variable_names=['u', 'v', 'w'], max_deriv_order=(1,),
                        equation_terms_max_number=5, data_fun_pow=1, additional_tokens=[trig_tokens, ],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-10, 1e-0))  #

    epde_search_obj.equations(only_print=True, num=1)

    return epde_search_obj


def  lv_discovery(foldername, noise_level):
    """
    Discovers the underlying equations governing the Lotka-Volterra system.
    
    This method automates the process of identifying the differential equations
    that best describe the dynamics of the Lotka-Volterra system, given time-series data.
    It configures and executes an EPDE search, leveraging preprocessors and
    multi-objective optimization to find a set of equations that accurately
    capture the relationships between the variables. The method returns the
    trained EPDE search object, which contains the discovered equations and
    related information. The method is doing that to provide insights into the
    underlying dynamics of the Lotka-Volterra system and build predictive models.
    
    Args:
        foldername: The name of the folder where the data is stored (not used).
        noise_level: The noise level in the data (not used).
    
    Returns:
        EpdeSearch: The trained EPDE search object containing the discovered equations.
    """
    t_file = os.path.join(os.path.dirname( __file__ ), 'data\\lv\\t_20.npy')
    t = np.load(t_file)
    data_file = os.path.join(os.path.dirname(__file__), 'data\\lv\\data_20.npy')
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
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=100)

    factors_max_number = {'factors_num': [1, 2], 'probas' : [0.8, 0.2]}

    epde_search_obj.fit(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,),
                        equation_terms_max_number=5, data_fun_pow=1, additional_tokens=[trig_tokens, ],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-4, 1e-0))  #

    epde_search_obj.equations(only_print=True, num=1)

    return epde_search_obj


def ac_data(filename: str):
    """
    Loads data and generates corresponding grid coordinates.
    
    This function is designed to load data from a specified file and create a grid
    representing the coordinate system in which the data is defined. This is a 
    necessary step to work with the data in a structured manner, allowing for 
    further analysis and processing within the framework.
    
    Args:
        filename (str): The path to the file containing the data to be loaded.
    
    Returns:
        tuple: A tuple containing two elements:
            - grids (tuple of numpy.ndarray): A tuple of arrays representing the meshgrid
              of the coordinate system.
            - data (numpy.ndarray): The loaded data from the specified file.
    """
    t = np.linspace(0., 1., 51)
    x = np.linspace(-1., 0.984375, 128)
    data = np.load(filename)
    # t = np.linspace(0, 1, shape+1); x = np.linspace(0, 1, shape+1)
    grids = np.meshgrid(t, x, indexing = 'ij')  # np.stack(, axis = 2) , axis = 2)
    return grids, data    


def AC_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the performance on the Allen-Cahn equation.
        
        This method evaluates the ability to identify the Allen-Cahn equation from data.
        It sets up a test scenario by loading data, adding noise, and then checking
        if the discovered equation matches the known symbolic representation. This
        validation is crucial to ensure the reliability of the equation discovery process.
        
        Args:
            operator: Compound operator to test.
            foldername: Name of the folder containing the data and pretrained model.
            noise_level: Level of noise to add to the data (default: 0).
        
        Returns:
            bool: True if the identified equation matches the expected equation, False otherwise.
    """
    # Test scenario to evaluate performance on Allen-Cahn equation
    eq_ac_symbolic = '0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
    eq_ac_incorrect = '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * u{power: 1.0} + -0.0 = du/dx0{power: 1.0}'
    
    grid, data = ac_data(os.path.join(foldername, 'ac_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ac_ann_pretrained.pickle'))

    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    epde_search_obj = EpdeSearch(use_solver = False, use_pic=True, boundary = 10,
                                 coordinate_tensors = ((grid[0], grid[1])), verbose_params = {'show_iter_idx' : True},
                                 device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2, 2),
                                additional_tokens = [], data_nn = data_nn)

    assert compare_equations(eq_ac_symbolic, eq_ac_incorrect, epde_search_obj)


def wave_data(filename):
    """
    Generates grid coordinates and wave data from a file.
    
    This method loads wave data from a CSV file and creates a corresponding grid of coordinates.
    The grid and data are structured to facilitate the discovery of underlying differential equations.
    
    Args:
        filename (str): The name of the CSV file containing the wave data.
    
    Returns:
        tuple (np.ndarray, np.ndarray): A tuple containing two numpy arrays:
            - grids: A numpy array representing the grid coordinates.
            - data: A numpy array containing the wave data loaded from the file.
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
    Tests the equation comparison within the `EpdeSearch` object to ensure accurate equation discovery.
    
        This test validates the ability of the `compare_equations` function to differentiate
        between a correct symbolic equation representing the wave equation and an incorrect one.
        It's crucial for verifying that the search process can reliably distinguish between
        candidate equations based on their structure and fit to the data.
    
        Args:
            operator: CompoundOperator object.
            foldername: Path to the folder containing the wave data and pretrained ANN.
            noise_level: Level of noise to add to the wave data. Defaults to 0.
    
        Returns:
            bool: True if the comparison passes, otherwise raises an AssertionError.
    """
    # eq_wave_symbolic = '1. * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    eq_wave_symbolic = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    eq_wave_incorrect = '1. * d^2u/dx1^2{power: 1} * du/dx1{power: 1} + 2.3 * d^2u/dx0^2{power: 1} + 0. = du/dx0{power: 1}'

    grid, data = wave_data(os.path.join(foldername, 'wave_sln_80.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ann_pretrained.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=(grid[..., 0], grid[..., 1]),
                                 verbose_params={'show_iter_idx': True},
                                 device='cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 2),
                                additional_tokens=[], data_nn=data_nn)

    assert compare_equations(eq_wave_symbolic, eq_wave_incorrect, epde_search_obj)


def kdv_data(filename, shape = 80):
    """
    Loads data and creates corresponding spatial and temporal grids.
    
        This function is essential for preparing the data into a suitable format
        that can be used for equation discovery. It reads the raw data, and then
        generates the spatial and temporal grids necessary for representing the
        solution domain. This is a preliminary step before the evolutionary
        search can be applied to find the best fitting differential equation.
    
        Args:
            filename (str): The name of the data file to load (CSV format).
            shape (int, optional): The shape of the grid to create. Defaults to 80.
    
        Returns:
            tuple: A tuple containing the spatial and temporal grids and the loaded data.
                The first element is a tuple of two numpy arrays representing the grids.
                The second element is a numpy array containing the loaded data.
    """
    shape = 80
    
    print(os.path.dirname( __file__ ))
    data = np.loadtxt(filename, delimiter = ',').T
    t = np.linspace(0, 1, shape+1)
    x = np.linspace(0, 1, shape+1)
    grids = np.meshgrid(t, x, indexing = 'ij') # np.stack(, axis = 2)
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


def KdV_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the performance on the Korteweg-de Vries equation.
        
        This method evaluates the ability of the EPDE framework to rediscover the
        Korteweg-de Vries (KdV) equation from data. It sets up a controlled experiment
        by loading data, optionally adding noise, and then using the EPDE search
        algorithm to identify the equation that best describes the data. This test
        validates the framework's capacity to automatically identify governing equations.
        
        Args:
            operator: The compound operator to be used in the EPDE search.
            foldername: The name of the folder containing the data files.
            noise_level: The level of noise to add to the data (default: 0).
        
        Returns:
            bool: True if the identified equation matches the expected KdV equation, False otherwise.
    """
    # Test scenario to evaluate performance on Korteweg-de Vries equation
    eq_kdv_symbolic = '-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} + \
                           1.0 * sin{power: 1, freq: 1.0, dim: 1} * cos{power: 1, freq: 1.0, dim: 0} + \
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

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={}) #'epochs_max': 5e4

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2, 3),
                                additional_tokens = [trig_tokens,]) # data_nn

    # np.save(os.path.join(foldername, 'kdv_0_derivs.npy'), epde_search_obj.derivatives)

    assert compare_equations(eq_kdv_symbolic, eq_kdv_incorrect, epde_search_obj)


def kdv_data_h(filename, shape=80):
    """
    Loads KdV equation data and prepares it for equation discovery.
    
    This function loads data representing the solution of the KdV equation
    from a specified file and generates corresponding time and space grids.
    These grids and the data are structured to be used as input for
    discovering the underlying differential equation. The shape parameter
    is overridden internally to ensure consistency with the grid dimensions.
    
    Args:
        filename (str): The name of the file containing the KdV equation data.
        shape (int, optional): An integer intended to define the shape of the data (unused). Defaults to 80.
    
    Returns:
        tuple (np.ndarray, np.ndarray): A tuple containing the grids (t, x) and the loaded data.
            The grids are created using np.meshgrid with 120 time points
            between 0 and 1, and 480 spatial points between -3 and 3.
            The data is loaded from the specified file using np.load.
    
    Why:
        This function prepares the data into a format suitable for the equation discovery process.
        The grids provide the independent variable values (time and space) corresponding to the
        solution data, which is necessary for algorithms to learn the underlying equation.
    """
    shape = 119

    print(os.path.dirname(__file__))
    data = np.load(filename)

    t = np.linspace(0, 1, 120)
    x = np.linspace(-3, 3, 480)
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, data


def KdV_h_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the performance of the EPDE search on the Korteweg-de Vries equation.
        
        This method sets up and executes an EPDE search to identify the KdV equation
        from noisy data. It then validates whether the discovered equation aligns with the known KdV equation, ensuring the framework's ability to accurately identify governing equations from data.
        
        Args:
            operator: CompoundOperator object, not used in the current implementation.
            foldername: Path to the folder containing the data and pretrained PINN.
            noise_level: Level of noise to add to the data.
        
        Returns:
            bool: True if the comparison of equations passes, False otherwise.
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
        It sets up a controlled experiment by loading KdV data, introducing noise, and defining a search space for potential equation candidates.
        The SGA then explores this space to find an equation that accurately represents the system's dynamics. This process demonstrates the framework's ability to discover governing equations from data.
        
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

def darcy_data(filename: str):
    """
    Loads Darcy flow data from a file and prepares it for use in equation discovery.
    
    This method loads a NumPy array representing a Darcy flow solution from the specified file,
    reshapes it to be compatible with the expected input format, and creates a corresponding
    coordinate grid. The coordinate grid and processed data are then returned for use in
    identifying the underlying differential equation.
    
    Args:
        filename: The name of the file containing the Darcy flow data
            in NumPy format (.npy).
    
    Returns:
        A tuple containing:
            - grids: A tuple of NumPy arrays representing the coordinate grid.
            - data: A NumPy array containing the processed Darcy flow data,
                    reshaped for compatibility with the equation discovery process.
    """
    t = np.linspace(0., 1., 2)
    x = np.linspace(0., 1., 128)
    y = np.linspace(0., 1., 128)
    grids = np.meshgrid(t, x, y, indexing='ij')  # np.stack(, axis = 2)
    data = np.load(filename)
    data = data[0]
    data = np.stack([data]*2, axis=0)
    print(data.shape)
    return grids, data

def darcy_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the performance of the equation discovery process on the Darcy equation.
    
    This method evaluates the ability of the EPDE framework to identify the Darcy equation
    from noisy data. It involves data loading, preprocessing, defining custom tokens
    representing domain-specific knowledge, and comparing the identified equation with
    known correct and incorrect forms. The goal is to assess whether the framework can
    accurately extract the underlying equation despite the presence of noise and complexity.
    
    Args:
        operator: The CompoundOperator instance to use for equation comparison.
        foldername: The name of the folder containing the Darcy data.
        noise_level: The level of noise to add to the data (default: 0).
    
    Returns:
        bool: True if the identified equation matches the expected Darcy equation,
            False if it matches the incorrect equation.
    """
    # Test scenario to evaluate performance on darcy equation
    # eq_darcy_symbolic = '-1.0 * du/dx1{power: 1.0} * dnu/dy{power: 1.0} + -1.0 * nu{power: 1.0} * d^2u/dx1^2{power: 1} + \
    #                       -2.0 * nu{power: 1.0} * d^2u/dxdy{power: 1} + \
    #                       -1.0 * du/dx2{power: 1.0} * dnu/dx{power: 1.0} + -1.0 * du/dx2{power: 1.0} * dnu/dy{power: 1.0} + \
    #                       -1.0 * nu{power: 1.0} * d^2u/dx2^2{power: 1} + -1.0 = du/dx1{power: 1.0} * dnu/dx{power: 1.0}'

    eq_darcy_symbolic = '-1.0 * du/dx2{power: 1.0} * dnu/dy{power: 1.0} + -1.0 * nu{power: 1.0} * d^2u/dx1^2{power: 1} + \
                              -1.0 * nu{power: 1.0} * d^2u/dx2^2{power: 1} + -1.0 = du/dx1{power: 1.0} * dnu/dx{power: 1.0}'

    eq_darcy_incorrect = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx2^2{power: 1}'

    grid, data = darcy_data(os.path.join(foldername, 'darcy_1.0.npy'))
    nu = np.load(r'C:\Users\user\PycharmProjects\EPDE\EPDE\projects\pic\data\darcy\darcy_nu_1.0.npy')
    noised_data = noise_data(data, noise_level)
    x = np.linspace(0., 1., 128)
    y = np.linspace(0., 1., 128)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    dimensionality = 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True, boundary=0,
                                 coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    darcy_gradient_nux = np.gradient(nu[0], dx, axis=0, edge_order=2)
    darcy_gradient_nuy = np.gradient(nu[0], dy, axis=1, edge_order=2)
    darcy_gradient_xy = np.gradient(np.gradient(data, dx, axis=1, edge_order=2), dy, axis=2, edge_order=2)

    custom_grid_tokens_nu = CacheStoredTokens(token_type='nu-tensors',
                                              token_labels=['nu', 'dnu/dx', 'dnu/dy'],
                                              token_tensors={'nu': nu[0], 'dnu/dx': darcy_gradient_nux,
                                                             'dnu/dy': darcy_gradient_nuy},
                                              params_ranges={'power': (1, 1)},
                                              params_equality_ranges=None, )

    custom_grid_tokens_xy = CacheStoredTokens(token_type='xy-tensor',
                                              token_labels=['d^2u/dxdy'],
                                              token_tensors={'d^2u/dxdy': darcy_gradient_xy},
                                              params_ranges={'power': (1, 1)},
                                              params_equality_ranges=None,
                                              meaningful=True
                                              )

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(0, 2, 2),
                                additional_tokens = [custom_grid_tokens_nu, custom_grid_tokens_xy]) # , data_nn = data_nn

    # np.save(os.path.join(foldername, 'kdv_0_derivs.npy'), epde_search_obj.derivatives)

    assert compare_equations(eq_darcy_symbolic, eq_darcy_incorrect, epde_search_obj)

def darcy_discovery(foldername, noise_level):
    """
    Discovers a differential equation governing Darcy flow from data using the EPDE framework.
    
        This method automates the identification of the underlying differential equation
        that describes Darcy flow by analyzing provided data. It preprocesses the data,
        defines custom tokens relevant to the flow characteristics (permeability and its
        gradients, mixed derivative of pressure), and employs the EPDE framework to
        search for the equation that best fits the data. The goal is to uncover the
        mathematical relationship governing the flow based on observed data, even in the
        presence of noise.
    
        Args:
            foldername: The name of the folder containing the Darcy flow data ('darcy_1.0.npy') and optionally a pretrained ANN model ('ann_pretrained.pickle').
            noise_level: The level of noise to add to the Darcy flow data.
    
        Returns:
            EpdeSearch: The EPDE search object containing the discovered equations and related information.
    """
    grid, data = darcy_data(os.path.join(foldername, 'darcy_1.0.npy'))
    nu = np.load(r'C:\Users\user\PycharmProjects\EPDE\EPDE\projects\pic\data\darcy\darcy_nu_1.0.npy')
    noised_data = noise_data(data, noise_level)
    x = np.linspace(0., 1., 128)
    y = np.linspace(0., 1., 128)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    # data_nn = load_pretrained_PINN(os.path.join(foldername, 'ann_pretrained.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=0,
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=30)

    darcy_gradient_nux = np.gradient(nu[0], dx, axis=0, edge_order=2)
    darcy_gradient_nuy = np.gradient(nu[0], dy, axis=1, edge_order=2)
    darcy_gradient_xy = np.gradient(np.gradient(data, dx, axis=1, edge_order=2), dy, axis=2, edge_order=2)

    custom_grid_tokens_nu = CacheStoredTokens(token_type='nu-tensors',
                                                token_labels=['nu', 'dnu/dx', 'dnu/dy'],
                                                token_tensors={'nu': nu[0], 'dnu/dx': darcy_gradient_nux, 'dnu/dy': darcy_gradient_nuy},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None,)

    custom_grid_tokens_xy = CacheStoredTokens(token_type='xy-tensor',
                                               token_labels=['d^2u/dxdy'],
                                               token_tensors={'d^2u/dxdy': darcy_gradient_xy},
                                               params_ranges={'power': (1, 1)},
                                               params_equality_ranges=None,
                                               meaningful=True
                                              )

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.5, 0.5]}

    bounds = (1e-12, 1e-2)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(0, 2, 2), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[custom_grid_tokens_nu, custom_grid_tokens_xy],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) #, data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

def ODE_discovery(foldername, noise_level):
    """
    Discovers the underlying ODE from noisy data using the EPDE framework.
        
        This method performs equation discovery on ODE data, leveraging a combination
        of finite difference preprocessing, neural network priors, and evolutionary
        algorithms to identify the governing equations. The method aims to find the best equation structure that accurately represents the dynamics of the system described by the provided data. This is achieved by exploring a space of possible equations and selecting those that minimize the error between the model's predictions and the observed data.
    
        Args:
            foldername: The name of the folder containing the 'ode_data.npy' file
                and the pretrained PINN model 'ode_0_ann.pickle'.
            noise_level: The level of noise to add to the data.
        
        Returns:
            EpdeSearch: The trained EpdeSearch object, containing the discovered equations.
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
                                 coordinate_tensors=(t,), verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=15)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    epde_search_obj.fit(data=[noised_data, ], variable_names=['u', ], max_deriv_order=(2, 2),
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-12, 1e-4), data_nn=data_nn) #

    epde_search_obj.equations(only_print=True, num=1)

    # import pickle
    # fname = os.path.join(r'C:\Users\user\EPDE_tests\models', 'ode_0_ann.pickle')
    # with open(fname, 'wb') as output_file:
    #     pickle.dump(global_var.solution_guess_nn, output_file)
    # epde_search_obj.visualize_solutions()
    return epde_search_obj

def vdp_discovery(foldername, noise_level):
    """
    Discovers the governing equation of the Van der Pol oscillator from noisy data.
    
    This method leverages the EPDE framework to identify the underlying differential equation
    that describes the dynamics of the Van der Pol oscillator. It starts by loading the data,
    introducing noise to simulate real-world conditions, and utilizing a pre-trained neural network
    to approximate the solution. EPDE then explores the space of possible equation structures to
    find the one that best fits the data. This approach allows us to automatically infer the
    mathematical model from the observed behavior of the system.
    
    Args:
        foldername: The name of the folder containing the data and pretrained neural network.
        noise_level: The level of noise to add to the data.
    
    Returns:
        EpdeSearch: The trained EPDE search object, which encapsulates the discovered equation and related information.
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

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=(t,), verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=5)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    epde_search_obj.fit(data=[noised_data, ], variable_names=['u', ], max_deriv_order=(2, 2),
                        equation_terms_max_number=5, data_fun_pow=2,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-12, 1e-4), data_nn=data_nn) #

    epde_search_obj.equations(only_print=True, num=1)

    # import pickle
    # fname = os.path.join(r'C:\Users\user\EPDE_tests\models', 'ode_0_ann.pickle')
    # with open(fname, 'wb') as output_file:
    #     pickle.dump(global_var.solution_guess_nn, output_file)
    # epde_search_obj.visualize_solutions()
    return epde_search_obj


def kdv_discovery(foldername, noise_level):
    """
    Discovers the governing equation for the KdV equation from data using an evolutionary approach.
    
        This method leverages the EPDE framework to identify the Korteweg-de Vries (KdV)
        equation. It begins by loading data and configuring the EPDE search
        object with settings appropriate for the KdV equation. Custom tokens are defined
        to guide the search process, and finally, the model is trained to find the
        equation that best describes the provided data. This automated discovery
        process helps in understanding the underlying dynamics of the system.
    
        Args:
            foldername: The name of the folder containing the data.
            noise_level: The level of noise in the data.
    
        Returns:
            EpdeSearch: The trained EPDE search object, containing the discovered equation.
    """
    grid, data = kdv_data(os.path.join(foldername, 'data.csv'))
    # grid, data = kdv_data(os.path.join(foldername, 'Kdv.mat'))
    # noised_data = noise_data(data, noise_level)
    # data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

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
                                      training_epochs=15)

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
    trig_tokens._token_family.set_status(unique_specific_token=False, unique_token_type=False,
                                  meaningful=True)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-9, 1e-4)
    epde_search_obj.fit(data=data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[custom_trig_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

def kdv_h_discovery(foldername, noise_level):
    """
    Discover the governing equation for the KdV equation with added noise using EPDE.
    
    This method leverages the EPDE framework to identify the underlying equation
    of the KdV system from noisy data. By setting up an EPDE search object,
    defining custom tokens relevant to the KdV equation, and fitting the model
    to the provided data, it aims to find the equation that best describes the system's behavior.
    This is done to provide insights into the system's dynamics and build a predictive model.
    
    Args:
        foldername: The name of the folder containing the data and pretrained PINN.
        noise_level: The level of noise added to the data.
    
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

    bounds = (1e-9, 1e-4)
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
    Discovers the KdV equation by fitting equation structures to data using an evolutionary algorithm.
        
        This method employs the EPDE framework with a Sequential Genetic Algorithm (SGA) to
        identify the governing equation for the KdV equation. It involves loading and
        preprocessing data, adding noise to test robustness, defining custom equation components,
        and then fitting the EPDE model to find the equation that best describes the data.
        This approach automates the equation discovery process, providing insights into the
        system's dynamics by finding a balance between model complexity and accuracy in
        representing the observed data.
        
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

    bounds = (1e-9, 1e-2)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

def wave_discovery(foldername, noise_level):
    """
    Performs wave equation discovery using the EPDE framework.
        
        This method leverages the EPDE search to identify the underlying equation
        governing wave propagation from provided data, even in the presence of noise.
        It prepares the data, configures the search space with appropriate tokens
        and preprocessors, and then fits the EPDE model to the data to find the
        best equation. This is done to automatically extract the mathematical
        representation of the wave phenomenon from the observed data.
        
        Args:
            foldername: The name of the folder containing the data files
                ('wave_sln_80.csv' and 'ann_pretrained.pickle').
            noise_level: The level of noise to add to the data.
        
        Returns:
            EpdeSearch: The trained EPDE search object.
    """
    grid, data = wave_data(os.path.join(foldername, 'wave_sln_80.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ann_pretrained.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=20,
                                      coordinate_tensors=(grid[..., 0], grid[..., 1]), device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=10)


    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-12, 1e-4)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False, data_nn=data_nn) #

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

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
    ode_folder_name = os.path.join(directory, 'data\\ode')
    vdp_folder_name = os.path.join(directory, 'data\\vdp')
    ac_folder_name = os.path.join(directory, 'data\\ac')
    wave_folder_name = os.path.join(directory, 'data\\wave')
    kdv_folder_name = os.path.join(directory, 'data\\kdv')
    darcy_folder_name = os.path.join(directory, 'data\\darcy')

    # Pair-wise tests
    # ODE_test(fit_operator, ode_folder_name, 0)
    # VdP_test(fit_operator, vdp_folder_name, 0)
    # AC_test(fit_operator, ac_folder_name, 0)
    # wave_test(fit_operator, wave_folder_name, 0)
    # KdV_test(fit_operator, kdv_folder_name, 0)
    # KdV_h_test(fit_operator, kdv_folder_name, 0)
    # KdV_sga_test(fit_operator, kdv_folder_name, 0)
    # darcy_test(fit_operator, darcy_folder_name, 0)

    # Full_scale test
    # eso = ODE_discovery(ode_folder_name, 0)
    # eso = vdp_discovery(vdp_folder_name, 0)
    # eso = lorenz_discovery(vdp_folder_name, 0)
    # eso = lv_discovery(vdp_folder_name, 0)
    # eso = ac_discovery(ac_folder_name, 0)
    # eso = wave_discovery(wave_folder_name, 0)
    # eso = kdv_discovery(kdv_folder_name, 0)
    # eso = kdv_h_discovery(kdv_folder_name, 0)
    # eso = kdv_sga_discovery(kdv_folder_name, 0)
    # eso = darcy_discovery(darcy_folder_name, 0)

