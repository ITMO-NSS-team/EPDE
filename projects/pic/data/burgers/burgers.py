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
from scipy.io import loadmat

from epde import TrigonometricTokens, GridTokens, CacheStoredTokens
import epde.globals as global_var

import scipy.io as scio


def load_pretrained_PINN(ann_filename):
    """
    Loads a pretrained Physics-Informed Neural Network (PINN) model from a file.
    
    This function attempts to load a previously trained PINN model from disk.
    This is useful for resuming training or using a pre-trained model
    without retraining, saving computational resources.
    
    Args:
        ann_filename: The filename of the pickled ANN model.
    
    Returns:
        The loaded ANN model if the file exists; otherwise, returns None.
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
                necessary context for equation translation and fitting.
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
    Prepares the compound operator by setting sub-operators for sparsity and coefficient calculation.
    
        This method configures the provided compound operator by setting its
        sub-operators, which are responsible for calculating sparsity and coefficients.
        It then maps the operator between gene and chromosome levels based on a fitness
        calculation condition. This ensures that the evolutionary process can
        effectively explore the search space of possible equation structures.
    
        Args:
            fitness_operator: The compound operator to prepare.
            operator_params: Parameters to be set for the fitness operator.
    
        Returns:
            The modified fitness operator with prepared sub-operators and level mapping.
            This operator is now ready to be used in the evolutionary search for
            differential equations. The mapping ensures compatibility between
            different levels of representation (genes and chromosomes).
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

def burgers_data(filename: str):
    """
    Loads Burgers' equation data from a MATLAB file and prepares it for equation discovery.
    
        This function loads the solution data of Burgers' equation from a specified
        MATLAB file. It extracts the solution `usol`, time `t`, and spatial `x`
        data, transposes the solution data, and creates corresponding time and
        spatial grids. This preprocessing step is crucial for preparing the data
        into a suitable format that the equation discovery algorithms can utilize
        to identify the underlying differential equation.
    
        Args:
            filename: The path to the MATLAB file containing the Burgers' equation data.
    
        Returns:
            A tuple containing:
              - grids: A tuple of arrays representing the time and spatial grids,
                structured for use in equation discovery.
              - data: A NumPy array containing the transposed solution data,
                ready for analysis.
    """
    burg = loadmat(filename)
    t = np.ravel(burg['t'])
    x = np.ravel(burg['x'])
    data = np.real(burg['usol'])
    data = np.transpose(data)
    grids = np.meshgrid(t, x, indexing = 'ij')  # np.stack(, axis = 2) , axis = 2)
    return grids, data


def burgers_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    """
    Tests the EPDE search on the Burgers' equation.
    
    This method evaluates the EPDE framework's ability to identify the Burgers' equation
    from noisy data. It sets up an EPDE search and compares the identified equation
    against known correct and incorrect equations to assess the accuracy of the 
    equation discovery process. This test helps ensure the framework can reliably 
    extract governing equations from data, even in the presence of noise.
    
    Args:
        operator: CompoundOperator object for equation comparison.
        foldername: Path to the folder containing the Burgers' equation data.
        noise_level: Level of noise to add to the data (default: 0).
    
    Returns:
        bool: True if the identified equation matches the correct equation and differs
            from the incorrect equation, False otherwise.
    """
    # Test scenario to evaluate performance on Allen-Cahn equation
    eq_burgers_symbolic = '0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
    eq_burgers_incorrect = '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * u{power: 1.0} + -0.0 = du/dx0{power: 1.0}'

    grid, data = burgers_data(os.path.join(foldername, 'burgers.mat'))
    noised_data = noise_data(data, noise_level)
    # data_nn = load_pretrained_PINN(os.path.join(foldername, 'ac_ann_pretrained.pickle'))

    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=((grid[0], grid[1])), verbose_params={'show_iter_idx': True},
                                 device='cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 2),
                                additional_tokens=[])

    assert compare_equations(eq_burgers_symbolic, eq_burgers_incorrect, epde_search_obj)


def burgers_discovery(foldername, noise_level):
    """
    Performs equation discovery for the Burgers' equation.
        
        This method leverages an evolutionary algorithm to identify the governing equation of the Burgers' equation from noisy data. It sets up and executes an EPDE search, incorporating finite difference (FD) preprocessing to estimate derivatives. Custom grid and trigonometric tokens are defined to enrich the search space. The model is then fitted to the data to find the equation that best describes the underlying dynamics. Finally, the discovered equations are printed and visualized to provide insights into the system's behavior. This automated equation discovery process helps to reveal the fundamental relationships within the data.
        
        Args:
            foldername: The name of the folder containing the data files ('burgers.mat' and 'kdv_{noise_level}_ann.pickle').
            noise_level: The level of noise added to the data.
        
        Returns:
            EpdeSearch: The trained EPDE search object containing the discovered equations and related information.
    """
    grid, data = burgers_data(os.path.join(foldername, 'burgers.mat'))
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
                                      training_epochs=15)

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
                        eq_sparsity_interval=bounds, fourier_layers=False) #

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
    burgers_folder_name = os.path.join(directory)

    # burgers_test(fit_operator, burgers_folder_name, 0)
    burgers_discovery(burgers_folder_name, 0)
