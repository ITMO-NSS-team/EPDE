import numpy as np
import torch

from functools import partial
from typing import Tuple, List, Union
from abc import ABC, abstractmethod

import epde.globals as global_var
from epde.interface.interface import EpdeMultisample, EpdeSearch, ExperimentCombiner
from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.interface.solver_integration import SolverAdapter, BOPElement 
from epde.supplementary import BasicDeriv, AutogradDeriv

from epde.solver.data import Conditions
from epde.solver.optimizers.closure import Closure

# TODO: Later to be refactored into multiple files

# class ControlOptClosure():
#     '''
#     Inspired by https://github.com/ITMO-NSS-team/torch_DE_solver/blob/main/tedeous/optimizers/closure.py
#     '''
#     def __init__(self, model):
#         self.set_model()
#         self.parameters = model.parameters
#         self._optimizer = model.optimizer

#     def get_closure(self):
#         self._optimizer.zero_grad()



class ControlConstraint(ABC):
    def __init__(self, val, deriv_method, deriv_axes, **kwargs):
        self._val = val

    @abstractmethod
    def __call__(self, fun_nn):
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

    @abstractmethod
    def loss(self, fun_nn):
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

class ControlConstrEq(ControlConstraint):
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

class ControlConstrNEq(ControlConstraint):
    '''
    Class for constrints of type $c(u, x) = f(u, x) - val < 0$
    '''
    def __init__(self, val: Union[float, torch.Tensor], grid: torch.Tensor, deriv_method: BasicDeriv,
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
    def __init__(self, conditions: List[Tuple[Union[float, ControlConstraint]]]):
        self._cond = conditions

    def __call__(self, u: torch.nn.Sequential):
        return sum([cond[1]* cond[1].loss(u) for cond in self._cond])

class ControlExp():
    def __init__(self, loss : ConditionalLoss):
        self._var_net = None
        self._control_net = None
        self.loss = loss

    def create_best_equations(self, optimal_equations: Union[list, ParetoLevels]):
        res_combiner = ExperimentCombiner(optimal_equations)
        return res_combiner.create_best(self._pool)   

    @staticmethod
    def create_ode_bop(key, var, term, grid_loc, value):
        bop = BOPElement(axis = 0, key = key, term = term, power = 1, var = var)
        bop_grd_np = np.array([[grid_loc,]])
        bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
        bop.values = torch.from_numpy(np.array([[value,]])).float()
        return bop

    def control_optim_params(self, lr: float = 1e-2, max_iter: int = 10, max_eval = None, tolerance_grad: float = 1e-7,
                          tolerance_change: float = 1e-9, history_size: int = 100, line_search_fn = 'strong_wolfe'):
        self._optim_params = {'lr': lr, 'max_iter': max_iter, 'max_eval': max_eval, 'tolerance_grad': tolerance_grad,
                              'tolerance_change': tolerance_change, 'history_size': history_size, 
                              'line_search_fn': line_search_fn}

    def solver_params(self, mode: str = 'NN', compiling_params: dict = {}, optimizer_params: dict = {},
                      cache_params: dict = {}, early_stopping_params: dict = {}, plotting_params: dict = {}, 
                      training_params: dict = {}, use_cache: bool = False, use_fourier: bool = False, 
                      fourier_params: dict = None, use_adaptive_lambdas: bool = False, device = torch.device('cpu')):
        self._solver_params = {'mode': mode, 
                               'compiling_params': compiling_params, 
                               'optimizer_params': optimizer_params,
                               'cache_params': cache_params,
                               'early_stopping_params': early_stopping_params,
                               'plotting_params': plotting_params,
                               'training_params': training_params,
                               'use_cache': use_cache,
                               'use_fourier': use_fourier,
                               'fourier_params': fourier_params,
                               'use_adaptive_lambdas': use_adaptive_lambdas,
                               'device': device
                               }

    def train_pinn(self, bc_operators: List[BOPElement], grids: List[np.ndarray], control_args = [], 
                   n_control: int = 1, epochs: int = 1e4, net = None):
        # Properly formulate training approach
        t = 0
        min_loss = np.inf
        stop_training = False
        if isinstance(net, torch.nn.Sequential): self._var_net = net

        global_var.reset_control_nn(n_var = len(control_args), n_control=n_control, net = net)
        self._control_net = global_var.control_nn

        grids_merged = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grids])).float().T
        grids_merged.to(device=self._solver_params['device'])

        control_optim_params = {'lr': 1e-2, 'max_iter': 10, 'max_eval': None, 'tolerance_grad': 1e-07,
                                'tolerance_change': 1e-09, 'history_size': 100, 'line_search_fn': 'strong_wolfe'}

        optimizer = torch.optim.LBFGS(params = self._var_net.parameters(), **control_optim_params)
        # Implement closure for loss function

        while t < epochs and not stop_training:
            
            with SolverAdapter(net = self._var_net, use_cache = False) as adapter:
                # Edit solver forms of functions of dependent variable to Callable objects.
                # Setting various adapater parameters
                adapter.set_compiling_params(**self._solver_params['compiling_params'])
                adapter.set_optimizer_params(**self._solver_params['optimizer_params'])
                adapter.set_cache_params(**self._solver_params['cache_params'])
                adapter.set_early_stopping_params(**self._solver_params['early_stopping_params'])
                adapter.set_plotting_params(**self._solver_params['plotting_params'])
                adapter.set_training_params(**self._solver_params['training_params'])
                adapter.change_parameter('mode', self._solver_params['mode'], param_dict_key = 'compiling_params')

                print(f'grid.shape is {grids[0].shape}')
                solver_loss, model = adapter.solve_epde_system(system = self.system, grids = grids, data = None, 
                                                               boundary_conditions = bc_operators, 
                                                               mode = self._solver_params['mode'], 
                                                               use_cache = self._solver_params['use_cache'], 
                                                               use_fourier = self._solver_params['use_fourier'],
                                                               fourier_params = self._solver_params['fourier_params'],
                                                               use_adaptive_lambdas = self._solver_params['use_adaptive_lambdas'])
                loss = self.loss(model, grids_merged)
                loss.backward()
                optimizer.step()

                if loss < min_loss:
                    self._var_net = model
                print(loss)
            
        return self._var_net