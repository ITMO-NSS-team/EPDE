'''

'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

from typing import Callable, Union, Dict, List
from functools import singledispatchmethod, singledispatch

from torch.nn import Sequential

from epde.structure.main_structures import Equation, SoEq
import epde.globals as global_var
from epde.evaluators import CustomEvaluator, simple_function_evaluator

from epde.integrate.interface import SystemSolverInterface
from epde.integrate.bop import BOPElement, PregenBOperator

from epde.supplementary import create_solution_net
from epde.solver.data import Domain, Conditions
from epde.solver.data import Equation as SolverEquation
from epde.solver.model import Model
from epde.solver.callbacks import cache, early_stopping, plot, adaptive_lambda
from epde.solver.optimizers.optimizer import Optimizer
from epde.solver.device import solver_device, check_device, device_type
from epde.solver.models import mat_model

'''
Specification of baseline equation solver parameters. Can be separated into its own json file for 
better handling, i.e. manual setup.
'''

BASE_COMPILING_PARAMS = {
                         'mode'                 : 'NN',
                         'lambda_operator'      : 1e1,
                         'lambda_bound'         : 1e4,
                         'normalized_loss_stop' : False,
                         'h'                    : 0.001,
                         'inner_order'          : '1',
                         'boundary_order'       : '2',
                         'weak_form'            : 'None',
                         'tol'                  : 0.005,
                         'derivative_points'    : 3
                         }

ADAM_OPTIMIZER_PARAMS = {
                         'lr'  : 1e-5,
                         'eps' : 1e-6
                         }

SGD_OPTIMIZER_PARAMS = {
                        }

LBFGS_OPTIMIZER_PARAMS = {
                          'lr'       : 1e-2,
                          'max_iter' : 10
                          }

PSO_OPTIMIZER_PARAMS = {
                        'pop_size'   : 30,
                        'b'          : 0.9,
                        'c1'         : 8e-2,
                        'c2'         : 5e-1,
                        'lr'         : 1e-3,
                        'betas'      : (0.99, 0.999),
                        'c_decrease' : False,
                        'variance'   : 1,
                        'epsilon'    : 1e-8,
                        'n_iter'     : 2000
                        }

BASE_OPTIMIZER_PARAMS = {
                         'optimizer'   : 'Adam', # Alternatively, switch to PSO, if it proves to be effective.
                         'gamma'       : 'None',
                         'decay_every' : 'None'
                         }

OPTIMIZERS_MATCHED = {
                      'Adam'  : ADAM_OPTIMIZER_PARAMS,
                      'LBFGS' : LBFGS_OPTIMIZER_PARAMS,
                      'PSO'   : PSO_OPTIMIZER_PARAMS,
                      'SGD'   : SGD_OPTIMIZER_PARAMS
                      }

BASE_CACHE_PARAMS = {
                     'cache_verbose'             : True,
                     'cache_model'               : 'None',
                     'model_randomize_parameter' : 0,
                     'clear_cache'               : False
                     }

BASE_EARLY_STOPPING_PARAMS = {
                              'eps'                     : 1e-7,
                              'loss_window'             : 100,
                              'no_improvement_patience' : 1000,
                              'patience'                : 7,
                              'abs_loss'                : 1e-5,
                              'normalized_loss'         : False,
                              'randomize_parameter'     : 1e-5,
                              'info_string_every'       : 'None',
                              'verbose'                 : False
                              }

try:
    plot_saving_directory = os.path.realpath(__file__)
except NameError:
    plot_saving_directory = 'None'
    
BASE_PLOTTER_PARAMS = {
                       'save_every'  : 1000, 
                       'print_every' : 500,
                       'title'       : 'None',
                       'img_dir'     : plot_saving_directory
                       }


BASE_TRAINING_PARAMS = {
                        'epochs'            : 1e2,  # 1e5
                        'info_string_every' : 'None', #1e4,
                        'mixed_precision'   : False,
                        'save_model'        : False,
                        'model_name'        : 'None'
                        }

def solver_formed_grid(training_grid=None, grid_var_keys=None, device = 'cpu'):
    """
    Forms a grid for solver input.
    
    Transforms training grid data into a PyTorch tensor, preparing it for use
    within the solver by ensuring it has the correct format and resides on the
    appropriate device. This transformation is crucial for efficient
    numerical computation within the equation discovery process. The grid
    represents the independent variable space over which the solution is
    evaluated.
    
    Args:
        training_grid: The training grid data. If None, it's retrieved from cache.
        grid_var_keys: Keys associated with the grid variables. If None and
            training_grid is provided, keys are retrieved from cache.
        device: The device to place the resulting tensor on (e.g., 'cpu', 'cuda').
    
    Returns:
        torch.Tensor: A tensor representing the formed grid, transposed and
        placed on the specified device. The tensor's data type is float32.
    """
    if training_grid is None:
        keys, training_grid = global_var.grid_cache.get_all(mode = 'torch')
    elif grid_var_keys is None:
        keys, _ = global_var.grid_cache.get_all(mode = 'torch')

    assert len(keys) == training_grid[0].ndim, 'Mismatching dimensionalities'

    training_grid = np.array(training_grid).reshape((len(training_grid), -1))
    return torch.from_numpy(training_grid).T.to(device).float()


class SolverAdapter(object):
    """
    A base class for solver adapters, providing a common interface for different solver implementations.
    
        Class Methods:
        - solve_epde_system: Solves the EPDE system based on the provided equations and domain.
        - solve: Solves the equation using the specified method.
        - create_domain: Creates a Domain object from a list of variables and grids.
        - change_parameter: Changes a specified parameter within the object's parameter dictionaries.
        - set_training_params: Sets training parameters for the solver.
        - set_plotting_params: Sets plotting parameters for the solver.
        - set_early_stopping_params: Sets the early stopping parameters for the solver.
        - set_cache_params: Sets the cache parameters for the solver.
        - set_optimizer_params: Sets the optimizer parameters for the solver.
        - set_compiling_params: Sets the compiling parameters for the solver.
        - get_net: Generates a neural network based on the specified mode.
        - set_net: Sets the network attribute of the object.
        - mode: Returns the current compilation mode.
        - __init__: Initializes the class instance.
    """

    def __init__(self, net=None, use_cache: bool = True, device: str = 'cpu'):
        """
        Initializes the SolverAdapter for equation discovery.
        
        The SolverAdapter manages the optimization and training process for identifying differential equations.
        This initialization configures the core components required for the search, including the network,
        device for computation, and default parameters for various stages like compiling, optimization,
        caching, early stopping, plotting, and training. These parameters can be adjusted later to fine-tune
        the discovery process. The adapter needs to be initialized with the network and other parameters
        to properly conduct the search for differential equations.
        
        Args:
            net: The network to be used.
            use_cache: A boolean indicating whether to use cache.
            device: The device to run the computations on.
        
        Returns:
            None.
        """
        self._device = device
        self.set_net(net)
        
        self._compiling_params = dict()
        self.set_compiling_params(**BASE_COMPILING_PARAMS)
        
        self._optimizer_params = dict()
        self.set_optimizer_params(**BASE_OPTIMIZER_PARAMS)
        
        self._cache_params = dict()
        self.set_cache_params(**BASE_CACHE_PARAMS)
        
        self._early_stopping_params = dict()
        self.set_early_stopping_params(**BASE_EARLY_STOPPING_PARAMS)
        
        self._ploter_params = dict()
        self.set_plotting_params(**BASE_PLOTTER_PARAMS)
        
        self._training_params = dict()
        self.set_training_params(**BASE_TRAINING_PARAMS)
        
        self.use_cache = use_cache
    
    # def set_net(self, net, get_net_kwargs: dict):
    #     self.net = net if net is None else self.get_net(**get_net_kwargs)

    @property
    def mode(self):
        """
        Returns the current compilation mode.
        
                This method retrieves the compilation mode, which influences the trade-off between compilation speed and the ability to explore a wider range of equation structures. Different modes might prioritize faster iteration during the equation discovery process or focus on a more thorough search for potentially complex solutions.
        
                Args:
                    self: The object instance.
        
                Returns:
                    str: The current compilation mode.
        """
        return self._compiling_params['mode']
    
    def set_net(self, net: torch.nn.Sequential):
        """
        Sets the neural network model.
        
        This method allows to update the neural network used for approximating solutions of differential equations.
        By setting a new network, the solver can explore different solution spaces and potentially find better approximations.
        
        Args:
            net (torch.nn.Sequential): The new neural network to be assigned.
        
        Returns:
            None.
        
        Class Fields:
            net (torch.nn.Sequential): The neural network model.
        """
        # if self.net is not None and 
        self.net = net

    @staticmethod
    def get_net(equations, mode: str, domain: Domain, use_fourier = True, 
                fft_params: dict = {'L' : [4,], 'M' : [3,]}, device: str = 'cpu'):
        """
        Generates a neural network suitable for solving differential equations, adapting the network structure based on the specified mode.
        
                Args:
                    equations: The equations to be solved.
                    mode: The mode of operation ('mat', 'autograd', or 'NN'). 'mat' returns a simplified model, while others return a full neural network.
                    domain: The domain of the problem.
                    use_fourier: Flag to use Fourier transform in the network.
                    fft_params: Parameters for the Fourier transform.
                    device: The device to use for computation ('cpu' or 'cuda').
        
                Returns:
                    A neural network model. Returns a mat_model if mode is 'mat', otherwise returns a solution network created by create_solution_net.
        
                Why:
                    The method creates different network architectures based on the chosen mode. This allows the framework to explore different solution strategies, from simplified models for initial exploration to more complex neural networks capable of capturing intricate relationships within the data when discovering underlying equations.
        """
        if mode == 'mat':
            return mat_model(domain, equations)
        elif mode in ['autograd', 'NN']:
            return create_solution_net(equations_num=equations.num, domain_dim=domain.dim,
                                       use_fourier=use_fourier, fourier_params=fft_params, device=device)
            

    def set_compiling_params(self, mode: str = None, lambda_operator: float = None, 
                             lambda_bound : float = None, normalized_loss_stop: bool = None,
                             h: float = None, inner_order: str = None, boundary_order: str = None,
                             weak_form: List[Callable] = None, tol: float = None, derivative_points: int = None):
        """
        Sets the compilation parameters to tailor the equation discovery process.
        
                This method configures the solver's behavior by updating its internal
                compilation parameters. It selectively updates parameters if they are
                provided, ensuring that the solver uses the specified settings during
                the equation search. This allows users to fine-tune the search process
                based on their specific problem and data characteristics.
        
                Args:
                    mode: The mode of compilation.
                    lambda_operator: The lambda operator value.
                    lambda_bound: The lambda bound value.
                    normalized_loss_stop: Flag for normalized loss stopping.
                    h: The step size.
                    inner_order: The order of the inner approximation.
                    boundary_order: The order of the boundary approximation.
                    weak_form: The weak form of the equation.
                    tol: The tolerance value.
                    derivative_points: The number of derivative points.
        
                Returns:
                    None
        """
        compiling_params = {'mode' : mode, 'lambda_operator' : lambda_operator, 'lambda_bound' : lambda_bound,
                            'normalized_loss_stop' : normalized_loss_stop, 'h' : h, 'inner_order' : inner_order,
                            'boundary_order' : boundary_order, 'weak_form' : weak_form, 'tol' : tol,
                            'derivative_points' : derivative_points}
        
        for param_key, param_vals in compiling_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._compiling_params[param_key] = None
                    else:
                        self._compiling_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')

    def set_optimizer_params(self, optimizer: str = None, params: Dict[str, float] = None,
                             gamma: float = None, decay_every: int = None):
        """
        Sets the optimizer parameters for the equation discovery process.
        
                This method configures the optimization algorithm used to refine the equation coefficients.
                It updates the internal dictionary of optimizer parameters with the provided values, if they are not None.
                It also performs validation on the optimizer name and sets default parameters if an optimizer is specified.
                This ensures that the equation discovery process uses a valid and properly configured optimization strategy.
        
                Args:
                    optimizer: The name of the optimizer to use (e.g., 'Adam', 'SGD').
                    params: A dictionary of optimizer-specific parameters.
                    gamma: The learning rate decay factor.
                    decay_every: The frequency of learning rate decay.
        
                Returns:
                    None.
        
                Class Fields Initialized:
                    _optimizer_params (Dict[str, Any]): A dictionary storing the optimizer parameters,
                        including the optimizer name, parameters, gamma, and decay frequency.
                        This dictionary is updated by this method.
                
                Why:
                    Configuring the optimizer is crucial for effectively searching the space of possible equation coefficients.
                    This method allows users to specify the optimization algorithm and its parameters,
                    enabling them to fine-tune the equation discovery process for optimal performance.
        """
        optim_params = {'optimizer' : optimizer, 'params' : params, 'gamma' : gamma, 
                        'decay_every' : decay_every}
        
        for param_key, param_vals in optim_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._optimizer_params[param_key] = None
                    else:
                        self._optimizer_params[param_key] = param_vals
                        if param_key == 'optimizer':
                            if param_vals not in ['Adam', 'SGD', 'PSO', 'LBFGS']:
                                raise ValueError(f'Unimplemented optimizer has been selected. Please, use {OPTIMIZERS_MATCHED.keys()}')
                            self._optimizer_params['params'] = OPTIMIZERS_MATCHED[param_vals]
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
    
    def set_cache_params(self, cache_verbose: bool = None, cache_model: Sequential = None, 
                         model_randomize_parameter: Union[int, float] = None, clear_cache: bool = None): # use_cache: bool = None, 

        cache_params = { 'cache_verbose' : cache_verbose, 'cache_model' : cache_model, # 'use_cache' : use_cache,
                        'model_randomize_parameter' : model_randomize_parameter, 'clear_cache' : clear_cache}

        for param_key, param_vals in cache_params.items():
        """
        Sets the cache parameters for the solver.
                
                This method configures the solver's caching behavior, allowing for optimization of repeated equation evaluations. By adjusting parameters like verbosity, cached models, and randomization, the efficiency of the equation discovery process can be significantly improved. This is particularly useful when exploring a large space of potential equation structures.
        
                Args:
                    cache_verbose: Verbosity level for caching.
                    cache_model: The model to be cached (e.g., a neural network).
                    model_randomize_parameter: Parameter to randomize the model.
                    clear_cache: Flag indicating whether to clear the cache.
                
                Returns:
                    None.
        """
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._cache_params[param_key] = None
                    else:
                        self._cache_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
                    
    def set_early_stopping_params(self, eps: float = None, loss_window: int = None, no_improvement_patience: int = None,
                                  patience: int = None, abs_loss: float = None, normalized_loss: bool = None,
                                  randomize_parameter: float = None, info_string_every: int = None, verbose: bool = None):
        """
        Sets the early stopping parameters for the solver.
        
                This method configures the early stopping criteria used during the equation discovery process. By adjusting parameters like patience, loss tolerance, and verbosity, the search can be fine-tuned to stop when satisfactory progress is no longer being made, thus saving computational resources.
        
                Args:
                    eps: The minimum difference between two consecutive loss values to be considered an improvement.
                    loss_window: The number of loss values to consider when calculating the average loss.
                    no_improvement_patience: The number of iterations to wait without improvement before stopping.
                    patience: The number of iterations to wait before stopping.
                    abs_loss: The absolute loss value to reach before stopping.
                    normalized_loss: Whether to use normalized loss for early stopping.
                    randomize_parameter: The parameter to randomize.
                    info_string_every: The frequency at which to print early stopping information.
                    verbose: Whether to print early stopping information.
        
                Returns:
                    None
        """
        early_stopping_params = {'eps' : eps, 'loss_window' : loss_window, 'no_improvement_patience' : no_improvement_patience,
                                 'patience' : patience, 'abs_loss' : abs_loss, 'normalized_loss' : normalized_loss, 
                                 'randomize_parameter' : randomize_parameter, 'info_string_every' : info_string_every,
                                 'verbose' : verbose}
    
        for param_key, param_vals in early_stopping_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._early_stopping_params[param_key] = None
                    else:
                        self._early_stopping_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
                    
    def set_plotting_params(self, save_every: int = None, print_every: int = None, title: str = None,
                            img_dir: str = None):
        """
        Sets the parameters that control how the solver's progress and results are visualized.
        
        This method configures the plotting behavior of the solver by updating its internal plotting parameters.
        It allows users to specify how frequently plots are saved and information is printed, as well as
        the directory where plot images are stored. This is essential for monitoring the solver's progress
        and analyzing the discovered equations.
        
        Args:
            save_every: The interval (in iterations) at which to save plots.
            print_every: The interval (in iterations) at which to print plotting information.
            title: The title to use for the plots. (Not used in the method's body)
            img_dir: The directory to save the plot images.
        
        Returns:
            None.
        """
        plotting_params = {'save_every' : save_every, 'print_every' : print_every, 
                           'print_every' : print_every, 'img_dir' : img_dir}
        for param_key, param_vals in plotting_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._ploter_params[param_key] = None
                    else:
                        self._ploter_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
    
    def set_training_params(self, epochs: int = None, info_string_every: int = None, mixed_precision: bool = None,
                            save_model: bool = None, model_name: str = None):
        """
        Sets training parameters for the solver.
        
                This method configures the training process by allowing users to specify parameters such as the number of epochs,
                information printing frequency, mixed precision usage, model saving behavior, and the model's name.
                These parameters influence how the solver refines its equation discovery process. By adjusting these parameters,
                users can fine-tune the training to achieve optimal equation discovery and model performance.
        
                Args:
                    epochs: The number of training epochs.
                    info_string_every: The frequency of printing training information.
                    mixed_precision: Whether to use mixed precision training.
                    save_model: Whether to save the trained model.
                    model_name: The name to use when saving the model.
        
                Returns:
                    None
        """
        training_params = {'epochs' : epochs, 'info_string_every' : info_string_every, 
                           'mixed_precision' : mixed_precision, 'save_model' : save_model, 'model_name' : model_name}

        for param_key, param_vals in training_params.items():
            if param_vals is not None:
                try:
                    if param_vals == 'None':
                        self._training_params[param_key] = None
                    else:
                        self._training_params[param_key] = param_vals
                except KeyError:
                    print(f'Parameter {param_key} can not be passed into the solver.')
                    
    def change_parameter(self, parameter: str, value, param_dict_key: str = None):
        """
        Changes a specified parameter within the solver's configuration.
        
                This method facilitates dynamic adjustment of solver settings, enabling users
                to fine-tune the equation discovery process without directly modifying the code.
                It updates parameters related to compiling, optimization, caching, early stopping,
                plotting, and training, allowing for experimentation with different configurations
                to improve the accuracy and efficiency of equation discovery.
        
                Args:
                    parameter (str): The name of the parameter to change.
                    value: The new value for the parameter. If None, it's converted to the string 'None'.
                    param_dict_key (str, optional): An optional key specifying which parameter dictionary to target directly.
                        If provided, the method only searches within that specific dictionary.
                        Valid keys are 'compiling_params', 'optimizer_params', 'cache_params',
                        'early_stopping_params', 'plotting_params', and 'training_params'. Defaults to None.
        
                Returns:
                    None: The method modifies the object's internal parameter dictionaries directly.
        
                Why:
                    This method allows users to modify solver parameters, which is essential for
                    adapting the equation discovery process to different datasets and problem settings.
                    By changing parameters, users can influence the search strategy, optimization
                    behavior, and other aspects of the solver, ultimately affecting the quality
                    and speed of the discovered equations.
        """
        setters = {'compiling_params'      : (BASE_COMPILING_PARAMS, self.set_compiling_params), 
                   'optimizer_params'      : (BASE_OPTIMIZER_PARAMS, self.set_optimizer_params),
                   'cache_params'          : (BASE_CACHE_PARAMS, self.set_cache_params),
                   'early_stopping_params' : (BASE_EARLY_STOPPING_PARAMS, self.set_early_stopping_params),
                   'plotting_params'       : (BASE_PLOTTER_PARAMS, self.set_plotting_params),
                   'training_params'       : (BASE_TRAINING_PARAMS, self.set_training_params)}

        if value is None:
            value = 'None'
            
        if param_dict_key is not None: # TODO: Add regular expressions
            param_labeled = {parameter : value}
            setters[param_dict_key][1](**param_labeled)
        else:
            for key, param_elem in setters.items():
                if parameter in param_elem[0].keys():
                    param_labeled = {parameter : value}
                    param_elem[1](**param_labeled)
                
    @staticmethod
    def create_domain(variables: List[str], grids : List[Union[np.ndarray, torch.Tensor]],
                      device: str = 'cpu') -> Domain:
        """
        Creates a `Domain` object representing the problem's spatial or temporal domain.
        
                This method constructs the domain by associating variable names with their corresponding grid values,
                ensuring that the number of variables matches the number of provided grids. The grid values are converted
                to a consistent format (PyTorch tensors) and placed on the specified device. This domain representation
                is crucial for defining the space in which the differential equations are solved.
        
                Args:
                    variables: A list of variable names (strings) that define the dimensions of the domain.
                    grids: A list of grids, where each grid is a NumPy array or a PyTorch tensor
                        representing the values for the corresponding variable. These grids define the
                        discretization of each dimension in the domain.
                    device: The device to move the grids to (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
        
                Returns:
                    Domain: A Domain object representing the domain defined by the variables and grids.
        """
        assert len(variables) == len(grids), f'Number of passed variables {len(variables)} does not \
            match number of grids {len(grids)}.'
        if isinstance(grids[0], np.ndarray):
            assert len(variables) == grids[0].ndim, 'Grids have to be set as a N-dimensional np.ndarrays with dim \
                matching the domain dimensionality'
        domain = Domain('uniform')

        for idx, var_name in enumerate(variables):
            var_grid = grids[idx].to(device) if isinstance(grids[idx], torch.Tensor) else torch.tensor(grids[idx]).to(device)
            var_grid = var_grid.unique().reshape(-1)
            domain.variable(variable_name = var_name, variable_set = var_grid, 
                            n_points = None)
            
        return domain

    def solve_epde_system(self, system: Union[SoEq, dict], grids: list=None, boundary_conditions=None,
                          mode='NN', data=None, use_cache: bool = False, use_fourier: bool = False,
                          fourier_params: dict = None, use_adaptive_lambdas: bool = False,
                          to_numpy: bool = False, grid_var_keys = None, *args, **kwargs):
        """
        Solves the EPDE system based on the provided equations and domain.
                
                This method orchestrates the solution of a system of equations by first discretizing it over a specified domain.
                It then applies necessary boundary conditions and employs a chosen solving mode (e.g., 'NN' for neural network-based solver)
                to find the solution. This process is essential for identifying the underlying differential equations from data
                by comparing the solutions obtained with different equation structures.
                
                Args:
                    system: The system of equations to solve. Can be a SoEq object,
                        a dictionary of equations, or a list of equations.
                    grids: The grids on which to discretize the equations. If None,
                        grids are retrieved from the global grid cache.
                    boundary_conditions: The boundary conditions to apply. If None,
                        default boundary conditions are generated based on the data and grids.
                    mode: The solving mode (e.g., 'NN').
                    data: Data used for generating default boundary conditions if
                        `boundary_conditions` is None.
                    use_cache: Whether to use cached results if available.
                    use_fourier: Whether to use Fourier features.
                    fourier_params: Parameters for the Fourier features.
                    use_adaptive_lambdas: Whether to use adaptive lambda parameters.
                    to_numpy: Whether to convert the solution to a NumPy array.
                    grid_var_keys: Keys for the grid variables. If None, keys are retrieved from the global grid cache.
                    *args: Additional positional arguments passed to the solver.
                    **kwargs: Additional keyword arguments passed to the solver.
                
                Returns:
                    The solution to the EPDE system. The type of the returned object
                    depends on the solver and the `to_numpy` flag.
        """
        solver_device(device = self._device)

        if isinstance(system, SoEq):
            system_interface = SystemSolverInterface(system_to_adapt=system)
            system_solver_forms = system_interface.form(grids = grids, mode = mode)
        elif isinstance(system, dict):
            system_solver_forms = list(system.values())
        elif isinstance(system, list):
            system_solver_forms = system
        else:
            raise TypeError(f'Incorrect type of the equations passed into solver. Expected dict or SoEq, got {type(system)}.')
        
        if boundary_conditions is None:
            raise NotImplementedError('TBD')
            op_gen = PregenBOperator(system=system,
                                     system_of_equation_solver_form=[sf_labeled[1] for sf_labeled
                                                                     in system.values()])
            op_gen.generate_default_bc(vals = data, grids = grids)
            boundary_conditions = op_gen.conditions
            
        bconds_combined = Conditions()
        for cond in boundary_conditions:
            bconds_combined.operator(bnd = cond['bnd_loc'], operator = cond['bnd_op'], 
                                     value = cond['bnd_val'])

        if grids is None:
            grid_var_keys, grids = global_var.grid_cache.get_all(mode = 'torch')
        elif grid_var_keys is None:
            grid_var_keys, _ = global_var.grid_cache.get_all(mode = 'torch')

        domain = self.create_domain(grid_var_keys, grids, self._device)

        return self.solve(equations=system_solver_forms, domain = domain,
                          boundary_conditions = bconds_combined, mode = mode, use_cache = use_cache,
                          use_fourier = use_fourier, fourier_params = fourier_params, 
                          use_adaptive_lambdas = use_adaptive_lambdas, to_numpy = to_numpy)

    def solve(self, equations: Union[List, SoEq, SolverEquation], domain: Domain,
              boundary_conditions = None, mode = 'NN', use_cache: bool = False, 
              use_fourier: bool = False, fourier_params: dict = None, #  epochs = 1e3, 
              use_adaptive_lambdas: bool = False, to_numpy = False, *args, **kwargs):
        """
        Solves the given equation within the specified domain, aiming to find the best possible solution by optimizing the network parameters.
        
                This method orchestrates the solution process, taking into account boundary conditions, network configurations, and optimization strategies to minimize the loss function. It leverages techniques like caching, Fourier features, and adaptive lambdas to enhance the solution's accuracy and efficiency.
        
                Args:
                  equations: The equation(s) to be solved. It can be a single
                    `SolverEquation` object or a list of equations to be added to a
                    `SolverEquation` object.
                  domain: The domain over which the equation is solved.
                  boundary_conditions: Boundary conditions for the equation. Defaults to None.
                  mode: Specifies the solution mode ('NN', 'autograd', or 'mat'). Defaults to 'NN'.
                  use_cache: A boolean indicating whether to use caching. Defaults to False.
                  use_fourier: A boolean indicating whether to use Fourier features. Defaults to False.
                  fourier_params: Parameters for the Fourier features. Defaults to None.
                  use_adaptive_lambdas: A boolean indicating whether to use adaptive
                    lambdas. Defaults to False.
                  to_numpy: A boolean indicating whether to convert the solution to a
                    NumPy array. Defaults to False.
                  *args: Additional positional arguments.
                  **kwargs: Additional keyword arguments.
                
                Returns:
                  tuple: A tuple containing the final loss value and the solution.
                    The solution can be a PyTorch tensor or a NumPy array, depending
                    on the `mode` and `to_numpy` parameters.
        """
    
        if isinstance(equations, SolverEquation):
            equations_prepared = equations
        else:
            equations_prepared = SolverEquation()
            for form in equations:
                equations_prepared.add(form)
        if self.net is None:
            self.net = self.get_net(equations_prepared, mode, domain, use_fourier,
                                    fourier_params, device=self._device)
        
        
        cb_early_stops = early_stopping.EarlyStopping(**self._early_stopping_params)
        callbacks = [cb_early_stops,]
        if use_cache:
            callbacks.append(cache.Cache(**self._cache_params))

        if use_adaptive_lambdas:
            callbacks.append(adaptive_lambda.AdaptiveLambda())
        
        optimizer = Optimizer(**self._optimizer_params)

        self.net.to(device = self._device)        

        model = Model(net = self.net, domain = domain, equation = equations_prepared, 
                      conditions = boundary_conditions)
        
        model.compile(**self._compiling_params)
        loss = model.train(optimizer, callbacks=callbacks, **self._training_params)
        
        grid = domain.build(mode = self.mode)
 
        grid = check_device(grid)
        
        if mode in ['NN', 'autograd'] and to_numpy:
            solution = self.net(grid).detach().cpu().numpy()
        elif mode in ['NN', 'autograd'] and not to_numpy:
            solution = self.net
        elif mode == 'mat' and to_numpy:
            solution = self.net.detach().cpu().numpy()
        elif mode == 'mat' and not to_numpy:
            solution = self.net
        else:
            raise ValueError('Incorrect mode.')
        return loss, solution
