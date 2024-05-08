import numpy as np
import torch

from functools import partial
from typing import Tuple, List, Union
from abc import ABC, abstractmethod

from epde.interface.interface import EpdeMultisample, EpdeSearch, ExperimentCombiner
from epde.interface.solver_integration import SolverAdapter, net 

# Add logic of transforming control function as a fixed equation token into the  neural network 
def get_control_nn(n_indep: int, n_dep: int, n_control: int):
    hidden_neurons = 128
    layers = [torch.nn.Linear(n_indep + n_dep, hidden_neurons),
              torch.nn.Tanh(),
              torch.nn.Linear(hidden_neurons, hidden_neurons),
              torch.nn.Tanh(),
              torch.nn.Linear(hidden_neurons, hidden_neurons),
              torch.nn.Tanh(),
              torch.nn.Linear(hidden_neurons, n_control)]
    return torch.nn.Sequential(*layers)

class BasicDeriv(ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Trying to create abstract differentiation method')
    
    def take_derivative(self, u: torch.Tensor, grid: torch.Tensor, axes: list):
        raise NotImplementedError('Trying to differentiate with abstract differentiation method')

class AutogradDeriv(BasicDeriv):
    def __init__(self):
        pass

    def take_derivative(self, u: torch.nn.Sequential, grid: torch.Tensor, axes: List = [],
                        component: int = 0):
        grid.requires_grad = True
        output_vals = u(grid)[..., component].sum(dim = 0)
        for axis in axes[:-1]:
            output_vals = output_vals.sum(dim = 0)
            output_vals = torch.autograd.grad(outputs = output_vals, inputs = grid)[0][:, axis]
        return output_vals


class ControlConstraint(ABC):
    def __init__(self, val, deriv_method, deriv_axes, **kwargs):
        self._val = val

    @abstractmethod
    def __call__(self, fun_nn):
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

    @abstractmethod
    def loss(self, fun_nn):
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

class ControlConstrEq():
    '''
    Class for constrints of type $c(u^(n)) = f(u) - val = 0$
    '''
    def __init__(self, val : Union[float, torch.Tensor], grid: torch.Tensor, deriv_method: BasicDeriv, 
                 deriv_axes: List = [None,], tolerance: float = 1e-7):
        self._val = val
        self._grid = grid        
        self._eps = tolerance
        self._axes = deriv_axes
        self._deriv_method = deriv_method

    def __call__(self, fun_nn: torch.nn.Sequential) -> Tuple[bool, torch.Tensor]:
        to_compare = self._deriv_method.take_derivative(u = fun_nn, grid=self._grid, axes=self._axes)
        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val)
        else:
            assert to_compare.shape == self._val.shape, 'Incorrect shapes of constraint value tensor'
            val_transformed = self._val
        return torch.isclose(val_transformed, to_compare, r_tol = self._eps), val_transformed - to_compare
        
    def loss(self, fun_nn: torch.nn.Sequential) -> torch.Tensor:
        _, discrepancy = self(fun_nn)
        return torch.norm(discrepancy)

class ControlConstrNEq():
    '''
    Class for constrints of type $c(u, x) = f(u, x) - val < 0$
    '''
    def __init__(self, val: Union[float, torch.Tensor], grid: torch.Tensor, deriv_method: TensorDeriv,
                 deriv_axes: List = [None,]):
        self._val = val
        self._grid = grid
        self._axes = deriv_axes
        self._deriv_method = deriv_method

    def __call__(self, fun_nn: torch.nn.Sequential) -> Tuple[bool, torch.Tensor]:
        to_compare = self._deriv_method.take_derivative(u = fun_nn, grid=self._grid,
                                                        axes=self._axes)
        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val)
        else:
            assert to_compare.shape == self._val.shape, 'Incorrect shapes of constraint value tensor'
            val_transformed = self._val
        return torch.greater(val_transformed, to_compare), torch.nn.functional.relu(val_transformed - to_compare)
        
    def loss(self, fun_nn: torch.nn.Sequential) -> torch.Tensor:
        _, discrepancy = self(fun_nn)
        return torch.norm(discrepancy)

class ConditionalLoss():
    def __init__(self, conditions: List[Tuple[float, ControlConstraint]]):
        self._cond = conditions

    def __call__(self, u: torch.nn.Sequential):
        return sum([cond[1]* cond[1].loss(u) for cond in self._cond])

class ControlExp():
    def __init__(self):
        self._var_net = None
        self._control_net = None
        pass # TODO: parameters? boundary conditions? 

    def train_equation(self):
        # raise NotImplementedError() # TODO: combine input samples, train equations

        res_combiner = ExperimentCombiner(optimal_equations)
        return res_combiner.create_best(self._pool)   

    def train_pinn(self, bc_operators: List[], grids: torch.tensor, epochs: int = 1e4):
        t = 0
        stop_training = False


        # Make preparations for L-BFGS method use
        while t < epochs and not stop_training: # training of control function
            def fix_control(u_net: torch.nn.Sequential, grid_args: torch.):
                '''
                For control function first inputs correspond to independent variables
                while the latter ones are dependent ones. 

                TODO: use functools partial or similar construction to fix the arguments of the NN. 
                '''
                partial()
                return lambda x: 
                # equation
            adapter = SolverAdapter(net = self._var_net, use_cache = False)
            # adapter = SolverAdapter(net = net, use_cache = use_cache) # var_number = len(system.vars_to_describe), 
            
            # Setting various adapater parameters
            adapter.set_compiling_params(**compiling_params)
            
            adapter.set_optimizer_params(**optimizer_params)
            
            adapter.set_cache_params(**cache_params)
            
            adapter.set_early_stopping_params(**early_stopping_params)
            
            adapter.set_plotting_params(**plotting_params)
            
            adapter.set_training_params(**training_params)
            
            adapter.change_parameter('mode', mode, param_dict_key = 'compiling_params')
            print(f'grid.shape is {grid[0].shape}')
            solution_model = adapter.solve_epde_system(system = system, grids = grid, data = data, 
                                                    boundary_conditions = boundary_conditions, 
                                                    mode = mode, use_cache = use_cache, 
                                                    use_fourier = use_fourier, fourier_params = fourier_params,
                                                    use_adaptive_lambdas = use_adaptive_lambdas)          
                