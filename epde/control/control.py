import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt
import datetime
import os 
from warnings import warn

import gc
from typing import List, Union
from copy import deepcopy


import epde.globals as global_var
from epde.interface.interface import ExperimentCombiner
from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.integrate import SolverAdapter, OdeintAdapter, BOPElement 

from epde.control.constr import ConditionalLoss
from epde.control.utils import prepare_control_inputs, eps_increment_diff
from epde.control.optim import AdamOptimizer, CoordDescentOptimizer

from epde.supplementary import FDDeriv, AutogradDeriv

class ControlExp():
    '''
    Represents a control experiment.
    
        This class provides a framework for conducting control experiments,
        allowing for different control strategies and optimization methods.
    
        Class Methods:
        - __init__: Initializes the ConditionalControlNetTrainer.
        - create_best_equations: Creates the best equations based on the provided optimal equations.
        - create_ode_bop: Creates a BOPElement for ODE integration.
        - set_solver_params: None
        - get_solver_adapter: Returns the appropriate solver adapter based on the configuration.
        - finite_diff_calculation: Calculate finite-differecnce approximation of gradient in respect to the specified parameter.
        - feedback: Performs feedback control to optimize a system.
        - time_based: Performs a time-based simulation or optimization.
    '''

    def __init__(self, loss : ConditionalLoss, device: str = 'cpu'):
        """
        Initializes the ConditionalControlNetTrainer.
        
        This class prepares the training environment for discovering differential equations. It sets up the necessary components,
        including the loss function and the device for computation, to optimize the control network.
        
        Args:
            loss: The conditional loss function to be minimized during the equation discovery process.
            device: The device to run the training on (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
        
        Fields:
            _device: The device used for training.
            _state_net: The state network (initialized to None).
            _best_control_net: The best control network found during training (initialized to None).
            loss: The conditional loss function.
        
        Returns:
            None.
        
        Why: This initialization is a crucial step in setting up the training process for the control network, which is used to identify the underlying differential equations from the given data by minimizing the specified loss function.
        """
        self._device = device
        self._state_net = None
        self._best_control_net = None
        self.loss = loss

    def create_best_equations(self, optimal_equations: Union[list, ParetoLevels]):
        """
        Creates the best equation by combining optimal equations using an evolutionary algorithm.
        
                This method refines a set of optimal equations by combining them and
                evaluating their performance, ultimately aiming to identify the equation
                that best balances accuracy and complexity. It uses parallel processing
                to efficiently explore the search space of possible combinations.
        
                Args:
                    optimal_equations: A list or ParetoLevels object containing the optimal
                        equations to combine.
        
                Returns:
                    The result of calling `create_best` method on the
                    `ExperimentCombiner` instance, after initialization with provided equations and passing the `pool` to it.
                    The `create_best` method selects the highest complexity values and creates the best equation variant.
        """
        res_combiner = ExperimentCombiner(optimal_equations)
        return res_combiner.create_best(self._pool)

    @staticmethod
    def create_ode_bop(key, var, term, grid_loc, value, device: str = 'cpu'):
        """
        Creates a BOPElement, sets its grid location, and assigns a value to it, preparing it for integration within the equation discovery process.
        
                This method constructs a BOPElement, sets its grid location, and assigns a value to it.
                The BOPElement is configured for use in ODE integration. This is a crucial step in representing the equation terms at specific grid points, enabling the framework to evaluate the equation's fitness against the observed data.
        
                Args:
                    key: The key associated with the BOPElement.
                    var: The variable associated with the BOPElement.
                    term: The term associated with the BOPElement.
                    grid_loc: The grid location for the BOPElement.
                    value: The value to be assigned to the BOPElement.
                    device: The device to which the BOPElement's tensors will be moved (default: 'cpu').
        
                Returns:
                    BOPElement: The created and configured BOPElement.
        """
        bop = BOPElement(axis = 0, key = key, term = term, power = 1, var = var, device = device)
        bop_grd_np = np.array([[grid_loc,]])
        bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor).to(device))
        bop.values = torch.from_numpy(np.array([[value,]])).float().to(device)
        return bop

    def set_solver_params(self, use_pinn: bool = True, mode: str = 'autograd', compiling_params: dict = {}, 
                          optimizer_params: dict = {}, cache_params: dict = {}, early_stopping_params: dict = {}, 
                          plotting_params: dict = {}, training_params: dict = {'epochs': 150,}, 
                          use_cache: bool = False, use_fourier: bool = False, #  5*1e0
                          fourier_params: dict = None, use_adaptive_lambdas: bool = False): #  device: str = 'cpu'
        self._use_pinn = use_pinn
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
                               'device': torch.device(self._device)
                               }

    def get_solver_adapter(self, net: torch.nn.Sequential = None):
        """
        Returns the appropriate solver adapter based on the configuration.
        
                This method determines which solver adapter to use for simulating the system's dynamics. If `use_pinn` is True, a `SolverAdapter` is returned, configured with parameters from `solver_params`. Otherwise, an `OdeintAdapter` is returned, using the specified ODE solving method. The choice depends on whether a neural network-based solver is employed or a traditional ODE solver is preferred.
        
                Args:
                    net: The neural network to be used by the SolverAdapter.
        
                Returns:
                    The solver adapter, either a SolverAdapter or an OdeintAdapter.
        
                Why:
                    This method allows the framework to switch between different numerical integration approaches based on user configuration. The `SolverAdapter` leverages neural networks, while the `OdeintAdapter` relies on traditional ODE solvers. This flexibility enables the framework to handle a wider range of problems and leverage different computational resources.
        """
        if self._use_pinn:
            adapter = SolverAdapter(net = net, use_cache = False, device = self._device)
            # Edit solver forms of functions of dependent variable to Callable objects.
            # Setting various adapater parameters
            adapter.set_compiling_params(**self._solver_params['compiling_params'])
            adapter.set_optimizer_params(**self._solver_params['optimizer_params'])
            adapter.set_cache_params(**self._solver_params['cache_params'])
            adapter.set_early_stopping_params(**self._solver_params['early_stopping_params'])
            adapter.set_plotting_params(**self._solver_params['plotting_params'])
            adapter.set_training_params(**self._solver_params['training_params'])
            adapter.change_parameter('mode', self._solver_params['mode'], param_dict_key = 'compiling_params')
        else:
            try:
                self._solver_params['method']
            except KeyError:
                self._solver_params['method'] = 'Radau'
            adapter = OdeintAdapter(method = self._solver_params['method'])
        return adapter

    @staticmethod
    def finite_diff_calculation(system, adapter, loc, control_loss, state_net: torch.nn.Sequential,
                                bc_operators, grids: list, solver_params: dict, eps: float):
        """
        Calculates the finite difference approximation of the gradient with respect to a specified parameter of the control neural network.
        
                This method perturbs a parameter of the control neural network by a small amount `eps` and calculates the corresponding change in the loss function.
                This approximation is used to estimate the sensitivity of the loss function to changes in the control parameters,
                which is essential for optimizing the control strategy.
        
                Args:
                    system: The EPDE system to be solved.
                    adapter: The solver adapter used to solve the EPDE system.
                    loc (List[Union[str, Tuple[int]]]): A list specifying the location of the parameter to be modified within the control NN's state dictionary.
                    control_loss: The loss function used to evaluate the performance of the control strategy.
                    state_net (torch.nn.Sequential): The state neural network.
                    bc_operators: The boundary condition operators.
                    grids (list): The grids used for solving the EPDE system and evaluating the control loss.
                    solver_params (dict): The parameters for the EPDE solver.
                    eps (float): The perturbation size used for finite difference approximation.
        
                Returns:
                    torch.Tensor: The finite difference approximation of the gradient.
        """
        # Calculating loss in p[i]+eps:
        ctrl_dict_prev = global_var.control_nn.net.state_dict()       
        ctrl_nn_dict = eps_increment_diff(input_params=ctrl_dict_prev,
                                          loc = loc, forward=True, eps=eps)
        global_var.control_nn.net.load_state_dict(ctrl_nn_dict)

        if isinstance(adapter, SolverAdapter):
            adapter.set_net(deepcopy(state_net))
            diff_method = AutogradDeriv
        else:
            diff_method = FDDeriv

        solver_loss_forward, model = adapter.solve_epde_system(system = system, grids = grids[0], data = None,
                                             boundary_conditions = bc_operators,
                                             mode = solver_params['mode'],
                                             use_cache = solver_params['use_cache'],
                                             use_fourier = solver_params['use_fourier'],
                                             fourier_params = solver_params['fourier_params'],
                                             use_adaptive_lambdas = solver_params['use_adaptive_lambdas'])
        
        control_inputs = prepare_control_inputs(model, grids[1], global_var.control_nn.net_args,
                                                diff_method = diff_method)
        loss_forward = control_loss([model, global_var.control_nn.net], [grids[1], control_inputs])

        # Calculating loss in p[i]-eps:
        ctrl_dict_prev = global_var.control_nn.net.state_dict() # deepcopy()
        ctrl_nn_dict = eps_increment_diff(input_params=ctrl_dict_prev,
                                          loc = loc, forward=False, eps=eps)
        global_var.control_nn.net.load_state_dict(ctrl_nn_dict)

        if isinstance(adapter, SolverAdapter):
            adapter.set_net(deepcopy(state_net))
        solver_loss_backward, model = adapter.solve_epde_system(system = system, grids = grids[0], data = None,
                                                       boundary_conditions = bc_operators,
                                                       mode = solver_params['mode'],
                                                       use_cache = solver_params['use_cache'],
                                                       use_fourier = solver_params['use_fourier'],
                                                       fourier_params = solver_params['fourier_params'],
                                                       use_adaptive_lambdas = solver_params['use_adaptive_lambdas'])

        control_inputs = prepare_control_inputs(model, grids[1], global_var.control_nn.net_args,
                                                diff_method = diff_method)
        loss_back = control_loss([model, global_var.control_nn.net], [grids[1], control_inputs])
        
        # Restore values of the control NN parameters
        ctrl_nn_dict = global_var.control_nn.net.state_dict()
        ctrl_nn_dict = eps_increment_diff(input_params=ctrl_nn_dict,
                                          loc = loc, forward=True, eps=eps)
        global_var.control_nn.net.load_state_dict(ctrl_nn_dict)

        loss_max = 1e-3
        if solver_loss_backward > loss_max or solver_loss_forward > loss_max:
            warn(f'High solver loss occured: backward {solver_loss_backward} and forward {solver_loss_forward}.')
        loss_alpha = 1e1

        with torch.no_grad():
            delta = loss_forward - loss_back
            if torch.abs(delta) < solver_loss_forward+solver_loss_backward:
                res = 0*delta
            else:
                res = delta/(2*eps*(1+loss_alpha*(solver_loss_forward+solver_loss_backward)))
        # print(f'loss_forward {loss_forward, solver_loss_forward}, loss_backward {loss_back, solver_loss_backward}, res {res}')
        # print(f'loss_alpha*(solver_loss_forward+solver_loss_backward) {loss_alpha*(solver_loss_forward+solver_loss_backward)}')
        return res
        

    def feedback(self, bc_operators: List[Union[dict, float]], grids: List[Union[np.ndarray, torch.Tensor]], 
                 n_control: int = 1, epochs: int = 1e2, state_net: torch.nn.Sequential = None, 
                 opt_params: List[float] = [0.01, 0.9, 0.999, 1e-8],
                 control_net: torch.nn.Sequential = None, fig_folder: str = None, 
                 LV_exp: bool = True, eps: float = 1e-2, solver_params: dict = {}):
        """
        Performs feedback control to optimize a system by iteratively solving a differential equation and refining a control strategy.
        
                This method alternates between solving the full system and an abridged version to efficiently adjust control parameters based on finite differences. It aims to find the optimal control strategy that minimizes a defined loss function, effectively steering the system towards a desired behavior. The iterative refinement leverages both the full system dynamics and a simplified representation to balance accuracy and computational cost.
        
                Args:
                    bc_operators: List of boundary condition operators, each a dict or float. These define the constraints on the solution at the boundaries of the domain.
                    grids: List of grids (np.ndarray or torch.Tensor) on which to solve the system. These represent the spatial or temporal discretization of the problem domain.
                    n_control: Number of control variables. Defaults to 1. This specifies the dimensionality of the control input.
                    epochs: Number of training epochs. Defaults to 1e2. This determines the number of iterations for the optimization process.
                    state_net: Initial state network (torch.nn.Sequential). Defaults to None. This provides an initial guess for the solution of the differential equation.
                    opt_params: Optimization parameters (learning rate, beta1, beta2, epsilon). Defaults to [0.01, 0.9, 0.999, 1e-8]. These control the behavior of the optimizer used to update the control parameters.
                    control_net: Initial control network (torch.nn.Sequential). Defaults to None. This represents the initial control strategy.
                    fig_folder: Folder to save figures. Defaults to None. If provided, visualizations of the training process will be saved here.
                    LV_exp: Flag to indicate if it is a Lotka-Volterra experiment. Defaults to True. This is a specific flag for a particular type of experiment.
                    eps: Epsilon value for finite difference calculations. Defaults to 1e-2. This determines the step size used to approximate derivatives.
                    solver_params: Dictionary of solver parameters for full and abridged solves. Defaults to {}. These parameters configure the numerical solver used to solve the differential equation.
        
                Returns:
                    tuple: A tuple containing:
                        - The optimized state network (torch.nn.Sequential). This is the refined solution to the differential equation.
                        - The optimized control network (torch.nn.Sequential). This is the refined control strategy.
                        - The control predictions (torch.Tensor). These are the control actions predicted by the optimized control network.
                        - A list of loss values over training history (list). This provides insight into the convergence of the optimization process.
        """
        def modify_bc(operator: dict, scale: Union[float, torch.Tensor]) -> dict:
            noised_operator = deepcopy(operator)
            noised_operator['bnd_val'] = torch.normal(operator['bnd_val'], scale).to(self._device)
            return noised_operator

        # Properly formulate training approach
        t = 0

        loss_hist = []
        stop_training = False

        time = datetime.datetime.now()

        if isinstance(state_net, torch.nn.Sequential): self._state_net = state_net
        global_var.reset_control_nn(n_control = n_control, ann = control_net, 
                                    ctrl_args = global_var.control_nn.net_args, device = self._device)

        # TODO Refactor hook: To optimize the net in global variables is a terrific approach, rethink it

        if isinstance(grids[0], np.ndarray):
            grids_merged = torch.from_numpy(np.array([subgrid.reshape(-1) for subgrid in grids])).float().T.to(self._device)
        elif isinstance(grids[0], torch.Tensor):
            grids_merged = torch.cat([subgrid.reshape(-1, 1) for subgrid in grids], dim = 1).float()
        grids_merged.to(device=self._device)

        grad_tensors = deepcopy(global_var.control_nn.net.state_dict())

        min_loss = np.inf
        self._best_control_params = global_var.control_nn.net.state_dict()

        # optimizer = AdamOptimizer(optimized = global_var.control_nn.net.state_dict(), parameters = opt_params)
        optimizer = CoordDescentOptimizer(optimized = global_var.control_nn.net.state_dict(), parameters = opt_params)

        self.set_solver_params(**solver_params['full'])
        adapter = self.get_solver_adapter(None)
        if isinstance(adapter, SolverAdapter): 
            adapter.set_net(deepcopy(self._state_net))
            diff_method = AutogradDeriv
        else:
            diff_method = FDDeriv

        sampled_bc = [modify_bc(operator, noise_std) for operator, noise_std in bc_operators]
        loss_pinn, model = adapter.solve_epde_system(system = self.system, grids = grids, data = None,
                                                     boundary_conditions = sampled_bc,
                                                     mode = self._solver_params['mode'],
                                                     use_cache = self._solver_params['use_cache'],
                                                     use_fourier = self._solver_params['use_fourier'],
                                                     fourier_params = self._solver_params['fourier_params'],
                                                     use_adaptive_lambdas = self._solver_params['use_adaptive_lambdas'])


        print(f'Model is {type(model)}, while loss requires {[(cond, cond[1]._deriv_method) for cond in self.loss._cond]}')
        control_inputs = prepare_control_inputs(model, grids_merged, global_var.control_nn.net_args,
                                                diff_method = diff_method)
        loss = self.loss([model, global_var.control_nn.net], [grids_merged, control_inputs])
        print('current loss is ', loss, 'model undertrained with loss of ', loss_pinn)

        while t < epochs and not stop_training:
            self.set_solver_params(**solver_params['abridged'])            
            adapter = self.get_solver_adapter(None)
            sampled_bc = [modify_bc(operator, noise_std) for operator, noise_std in bc_operators]

            # self.set_solver_params(**solver_params['full'])
            adapter = self.get_solver_adapter(None)

            if isinstance(adapter, SolverAdapter):            
                adapter.set_net(self._state_net)
            
            loss_pinn, model = adapter.solve_epde_system(system = self.system, grids = grids, data = None,
                                                         boundary_conditions = sampled_bc,
                                                         mode = self._solver_params['mode'],
                                                         use_cache = self._solver_params['use_cache'],
                                                         use_fourier = self._solver_params['use_fourier'],
                                                         fourier_params = self._solver_params['fourier_params'],
                                                         use_adaptive_lambdas = self._solver_params['use_adaptive_lambdas'])

            control_inputs = prepare_control_inputs(model, grids_merged, global_var.control_nn.net_args,
                                                    diff_method = diff_method)
            loss = self.loss([model, global_var.control_nn.net], [grids_merged, control_inputs])            
            self._state_net = model
            self.set_solver_params(**solver_params['abridged'])

            global_var.control_nn.net.load_state_dict(self._best_control_params)
            state_net  = deepcopy(self._state_net)
            print(f'Control function optimization epoch {t}.')
            for param_key, param_tensor in grad_tensors.items():
                print(f'Optimizing {param_key}: shape is {param_tensor.shape}')
                if len(param_tensor.size()) == 1:
                    for param_idx, _ in enumerate(param_tensor):
                        loc = (param_key, param_idx)
                        grad_tensors[loc[0]] = grad_tensors[loc[0]].detach()                  
                        grad_tensors[loc[0]][loc[1:]] = self.finite_diff_calculation(system = self.system,
                                                                                     adapter = adapter,
                                                                                     loc = loc, control_loss = self.loss,
                                                                                     state_net = state_net,
                                                                                     bc_operators = sampled_bc,
                                                                                     grids = [grids, grids_merged],
                                                                                     solver_params = self._solver_params,
                                                                                     eps = eps)
                        if optimizer.behavior == 'Coordinate':
                            state_dict_prev = global_var.control_nn.net.state_dict()
                            state_dict = optimizer.step(gradient = grad_tensors, optimized = state_dict_prev, loc = loc)
                            global_var.control_nn.net.load_state_dict(state_dict)                                            
                elif len(param_tensor.size()) == 2:
                    for param_outer_idx, _ in enumerate(param_tensor):
                        for param_inner_idx, _ in enumerate(param_tensor[0]):
                            loc = (param_key, param_outer_idx, param_inner_idx)
                            grad_tensors[loc[0]] = grad_tensors[loc[0]].detach()
                            grad_tensors[loc[0]][tuple(loc[1:])] = self.finite_diff_calculation(system = self.system,
                                                                                                adapter = adapter,
                                                                                                loc = loc,
                                                                                                control_loss = self.loss,
                                                                                                state_net = state_net,
                                                                                                bc_operators = sampled_bc,
                                                                                                grids = [grids, grids_merged],
                                                                                                solver_params = self._solver_params,
                                                                                                eps = eps)
                            if optimizer.behavior == 'Coordinate':
                                state_dict_prev = global_var.control_nn.net.state_dict()
                                state_dict = optimizer.step(gradient = grad_tensors, optimized = state_dict_prev, loc = loc)
                                global_var.control_nn.net.load_state_dict(state_dict)                               
                else:
                    raise Exception(f'Incorrect shape of weights/bias. Got {param_tensor.size()} tensor.')
            if optimizer.behavior == 'Gradient':
                state_dict_prev = global_var.control_nn.net.state_dict()
                state_dict = optimizer.step(gradient = grad_tensors, optimized = state_dict_prev)
                global_var.control_nn.net.load_state_dict(state_dict)
            del state_dict, state_dict_prev

            self.set_solver_params(**solver_params['full'])
            adapter = self.get_solver_adapter(None)
            if isinstance(adapter, SolverAdapter):
                adapter.set_net(self._state_net)
            loss_pinn, model = adapter.solve_epde_system(system = self.system, grids = grids, data = None,
                                                         boundary_conditions = sampled_bc,
                                                         mode = self._solver_params['mode'],
                                                         use_cache = self._solver_params['use_cache'],
                                                         use_fourier = self._solver_params['use_fourier'],
                                                         fourier_params = self._solver_params['fourier_params'],
                                                         use_adaptive_lambdas = self._solver_params['use_adaptive_lambdas'])

            # var_prediction = model(grids_merged)
            self._state_net = model

            control_inputs = prepare_control_inputs(model, grids_merged, global_var.control_nn.net_args, 
                                                    diff_method = diff_method)
            loss = self.loss([model, global_var.control_nn.net], [grids_merged, control_inputs])
            print('current loss is ', loss, 'model undertrained with loss of ', loss_pinn)

            self._best_control_params = global_var.control_nn.net.state_dict()
            loss_hist.append(loss)
            
            if fig_folder is not None and LV_exp:
                plt.figure(figsize=(11, 6))
                plt.plot(grids_merged.cpu().detach().numpy(), control_inputs.cpu().detach().numpy()[:, 0], color = 'k')
                plt.plot(grids_merged.cpu().detach().numpy(), control_inputs.cpu().detach().numpy()[:, 1], color = 'r')
                plt.plot(grids_merged.cpu().detach().numpy(), global_var.control_nn.net(control_inputs).cpu().detach().numpy(),
                         color = 'tab:orange')
                plt.grid()
                frame_name = f'Exp_{time.month}_{time.day}_at_{time.hour}_{time.minute}_{t}.png'
                plt.savefig(os.path.join(fig_folder, frame_name))
            
            if fig_folder is not None:
                exp_res = {'state'   : control_inputs.cpu().detach().numpy(),
                           'control' : global_var.control_nn.net(control_inputs).cpu().detach().numpy()}
                frame_name = f'Exp_{time.month}_{time.day}_at_{time.hour}_{time.minute}_{t}.pickle'
                with open(os.path.join(fig_folder, frame_name), 'wb') as ctrl_output_file:  
                    pickle.dump(exp_res, file = ctrl_output_file)

            gc.collect()
            t += 1

        control_inputs = prepare_control_inputs(model, grids_merged, global_var.control_nn.net_args,
                                                diff_method = diff_method)
        ctrl_pred = global_var.control_nn.net(control_inputs)

        return self._state_net, global_var.control_nn.net, ctrl_pred, loss_hist

    def time_based(self, bc_operators: List[Union[dict, float]], grids: List[Union[np.ndarray, torch.Tensor]], 
                   n_control: int = 1, epochs: int = 1e2, state_net: torch.nn.Sequential = None, 
                   opt_params: List[float] = [0.01, 0.9, 0.999, 1e-8],
                   control_net: torch.nn.Sequential = None, fig_folder: str = None, 
                   LV_exp: bool = True, eps: float = 1e-2, solver_params: dict = {}):        
        """
        Performs a time-based simulation or optimization.
        
                This method is intended to perform a time-based simulation or
                optimization of a system, but the implementation is not yet
                available. It sets solver parameters, gets a solver adapter,
                and then raises a NotImplementedError. This method would be used to simulate the system's behavior over time, potentially optimizing control parameters to achieve desired outcomes based on the discovered equations.
        
                Args:
                    bc_operators: Boundary condition operators.
                    grids: Spatial grids for the simulation.
                    n_control: Number of control variables. Defaults to 1.
                    epochs: Number of training epochs. Defaults to 1e2.
                    state_net: Neural network for the state. Defaults to None.
                    opt_params: Optimization parameters. Defaults to [0.01, 0.9, 0.999, 1e-8].
                    control_net: Neural network for the control. Defaults to None.
                    fig_folder: Folder to save figures. Defaults to None.
                    LV_exp: Flag for Lotka-Volterra experiment. Defaults to True.
                    eps: A small constant. Defaults to 1e-2.
                    solver_params: Parameters for the solver. Defaults to {}.
        
                Returns:
                    None.
        
                Raises:
                    NotImplementedError: This method is not yet implemented.
        """
        self.set_solver_params(**solver_params['full'])
        adapter = self.get_solver_adapter(None)
        solver_form = self.system

        raise NotImplementedError()
