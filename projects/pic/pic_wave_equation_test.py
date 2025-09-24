import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import pickle
from typing import Tuple
import numpy as np

from epde.interface.prepared_tokens import CustomTokens, PhasedSine1DTokens, ConstantToken
from epde.interface.equation_translator import translate_equation
from epde.interface.interface import EpdeSearch

from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.sparsity import LASSOSparsity

from epde.operators.utils.operator_mappers import map_operator_between_levels
import epde.operators.common.fitness as fitness
from epde.operators.utils.template import CompoundOperator

# def load_data(data_filename: str, grid_filename : str = None) -> Tuple[np.ndarray]:
#     if '.npy' in data_filename:
#         data = np.load(data_filename)
#     elif ('.csv' in data_filename) or ('.txt' in data_filename):
#         data = np.loadtxt(data_filename)

#     if grid_filename is None:
#         if '.npy' in grid_filename:
#             grid = np.load(grid_filename)
#         elif ('.csv' in grid_filename) or ('.txt' in grid_filename):
#             grid = np.loadtxt(grid_filename)
#     else:


#     return (data, grid)

def load_data(filename):
    """
    Loads data representing a physical field from a file and creates a corresponding grid.
    
        This function reads data from the specified file, assuming comma-separated values,
        and constructs a grid of coordinates that represents the spatial or temporal domain
        over which the data is defined. This is a preliminary step for discovering
        the underlying differential equation.
    
        Args:
            filename: The name of the file containing the data.
    
        Returns:
            A tuple containing:
              - grids: A NumPy array representing the grid coordinates.
              - data: A NumPy array containing the loaded data.
    """
    shape = 80
    
    # print(os.path.dirname( __file__ ))
    data = np.loadtxt(filename, delimiter = ',').T
    t = np.linspace(0, 1, shape+1); x = np.linspace(0, 1, shape+1)
    grids = np.stack(np.meshgrid(t, x, indexing = 'ij'), axis = 2)
    return grids, data


def load_pretrained_PINN(ann_filename):
    """
    Loads a pre-trained Physics-Informed Neural Network (PINN) from a file to accelerate the equation discovery process.
    
        This method attempts to load a pickled PINN model from the specified file.
        If the file is not found, it prints a message indicating that the model
        will be retrained and returns None. Loading a pre-trained model can significantly reduce the computational cost of identifying the underlying differential equation, especially when starting from scratch.
    
        Args:
            ann_filename (str): The filename of the pickled PINN model.
    
        Returns:
            object: The loaded PINN model if the file is found, otherwise None.
    """
    try:
        with open(ann_filename, 'rb') as data_input_file:  
            data_nn = pickle.load(data_input_file)
    except FileNotFoundError:
        print('No model located, proceeding with ann approx. retraining.')
        data_nn = None
    return data_nn

def prepare_suboperators(fitness_operator: CompoundOperator) -> CompoundOperator:
    """
    Prepares sub-operators required to calculate the fitness of an equation.
    
        This method configures and assigns sparsity and coefficient calculation
        sub-operators to the provided fitness operator. It maps these sub-operators
        between 'gene level' and 'chromosome level' and then sets them as
        sub-operators within the fitness operator. This setup is crucial for evaluating the equation's performance by considering both its complexity (sparsity) and accuracy (coefficient calculation) when fitting the data.
    
        Args:
            fitness_operator (CompoundOperator): The compound operator to prepare.
    
        Returns:
            CompoundOperator: The modified fitness operator with the prepared sub-operators.
    """
    sparsity = LASSOSparsity()
    coeff_calc = LinRegBasedCoeffsEquation()

    sparsity = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
    coeff_calc = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')

    fitness_operator.set_suboperators({'sparsity' : sparsity,
                                       'coeff_calc' : coeff_calc})
    return fitness_operator


if __name__ == "__main__":
    # Operator = fitness.SolverBasedFitness # Replace by the developed PIC-based operator.
    Operator = fitness.PIC  # Replace by the developed PIC-based operator.
    operator_params = {"penalty_coeff" : 0.2, "pinn_loss_mult" : 1e4}
    fit_operator = prepare_suboperators(Operator(list(operator_params.keys())))
    fit_operator.params = operator_params

    directory = os.path.dirname(os.path.realpath(__file__))
    pinn_file_name = os.path.join(directory, 'data/wave/ann_pretrained.pickle') # If neccessary, replace by other filename

    shape = 80
    data_file_name = os.path.join(os.path.dirname( __file__ ), f'wave_sln_{shape}.csv')

    grid, data = load_data(data_file_name)
    data_nn = load_pretrained_PINN(pinn_file_name)

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 10,
                                      coordinate_tensors = (grid[..., 0], grid[..., 1]), verbose_params = {'show_iter_idx' : True},
                                      device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=data, variable_names=['u',], max_deriv_order=(2, 2),
                                additional_tokens = [], data_nn = data_nn)

    # eq_wave_symbolic = '1. * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    eq_wave_symbolic = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    wave_eq = translate_equation(eq_wave_symbolic, epde_search_obj.pool, all_vars = ['u',])
    wave_eq.vals['u'].main_var_to_explain = 'u'
    wave_eq.vals['u'].metaparameters = {('sparsity', 'u'): {'optimizable': False, 'value': 0.5}}
    print(wave_eq.text_form)

    eq_incorrect_symbolic = '1. * d^2u/dx1^2{power: 1} * du/dx1{power: 1} + 2.3 * d^2u/dx0^2{power: 1} + 0. = du/dx0{power: 1}'
    incorrect_eq = translate_equation(eq_incorrect_symbolic, epde_search_obj.pool, all_vars = ['u',])   #  , all_vars = ['u', 'v'])
    incorrect_eq.vals['u'].main_var_to_explain = 'u'
    incorrect_eq.vals['u'].metaparameters = {('sparsity', 'u'): {'optimizable': False, 'value': 0.5}}
    print(incorrect_eq.text_form)

    fit_operator.apply(wave_eq, {})
    fit_operator.apply(incorrect_eq, {})

    print(wave_eq.vals['u'].fitness_value)
    print(incorrect_eq.vals['u'].fitness_value)
    assert wave_eq.vals['u'].fitness_value < incorrect_eq.vals['u'].fitness_value