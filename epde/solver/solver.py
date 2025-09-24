from filecmp import clear_cache
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import datetime
from typing import Union
from torch.optim.lr_scheduler import ExponentialLR

from epde.solver.cache import *
from epde.solver.device import check_device, device_type
from epde.solver.solution import Solution
import epde.solver.input_preprocessing


def grid_format_prepare(coord_list, mode='NN') -> torch.Tensor:
    """
    Prepares the coordinate grid into a standardized tensor format suitable for downstream processing within the equation discovery pipeline. This ensures compatibility with various computational methods used for equation fitting.
    
        Args:
            coord_list (list or torch.Tensor): A list of coordinate arrays or a pre-existing coordinate tensor.  If a list, each element represents the coordinates along a single dimension.
            mode (str, optional):  Specifies the method for grid creation. Options include 'NN', 'autograd', and 'mat', each influencing how the grid is constructed. Defaults to 'NN'.
    
        Returns:
            torch.Tensor: A tensor representing the formatted grid. The shape of the tensor depends on the input `coord_list` and the specified `mode`.
    """
    device = device_type()
    if type(coord_list) == torch.Tensor:
        print('Grid is a tensor, assuming old format, no action performed')
        return check_device(coord_list)
    elif mode == 'NN' or mode == 'autograd':
        if len(coord_list) == 1:
            coord_list = torch.tensor(coord_list).float().to(device)
            grid = coord_list.reshape(-1, 1)
        else:
            coord_list_tensor = []
            for item in coord_list:
                if isinstance(item, (np.ndarray)):
                    coord_list_tensor.append(torch.from_numpy(item).float().to(device))
                else:
                    coord_list_tensor.append(item.to(device))
            grid = torch.cartesian_prod(*coord_list_tensor)
    elif mode == 'mat':
        grid = np.meshgrid(*coord_list, indexing='ij')
        grid = torch.tensor(np.array(grid)).to(device)
    print(f'In grid format prepare: grid shape is {grid.shape}')
    return grid


class Plots():
    """
    A class for plotting solutions obtained from different methods.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, model, grid, mode, tol = 0):
        """
        Initializes the Plots class, configuring it for visualization and analysis.
        
                The initialization process involves setting up the model, grid, mode, and tolerance parameters, which are essential for generating meaningful plots and visualizations that reflect the underlying dynamics of the discovered differential equations.
        
                Args:
                    model: The discovered equation model to be visualized.
                    grid: The spatial or temporal grid over which the model is defined.
                    mode: The plotting mode (e.g., '2D', '3D', 'time_series').
                    tol: The tolerance value used for filtering or smoothing the data (default is 0).
        
                Returns:
                    None
        """
        self.model = model
        self.grid = grid
        self.mode = mode
        self.tol = tol

    def print_nn(self, title: str):
        """
        Solution plot for the learned differential equation by the neural network.
        
                This method visualizes the solution obtained from the neural network model,
                allowing for inspection of the learned behavior across the input grid.
                It helps to understand how well the model approximates the true solution
                of the differential equation.
        
                Args:
                    title (str): The title of the plot.
        
                Returns:
                    None: The plot is displayed using matplotlib.
        """
        try:
            nvars_model = self.model[-1].out_features
        except:
            nvars_model = self.model.model[-1].out_features

        nparams = self.grid.shape[1]
        fig = plt.figure(figsize=(15, 8))
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i + 1)
                if title != None:
                    ax1.set_title(title + ' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.model(self.grid)[:, i].detach().cpu().numpy())

            else:
                ax1 = fig.add_subplot(1, nvars_model, i + 1, projection='3d')
                if title != None:
                    ax1.set_title(title + ' variable {}'.format(i))

                if self.tol != 0:
                    ax1.plot_trisurf(self.grid[:, 1].detach().cpu().numpy(),
                                     self.grid[:, 0].detach().cpu().numpy(),
                                     self.model(self.grid)[:, i].detach().cpu().numpy(),
                                     cmap=cm.jet, linewidth=0.2, alpha=1)
                else:
                    ax1.plot_trisurf(self.grid[:, 0].detach().cpu().numpy(),
                                     self.grid[:, 1].detach().cpu().numpy(),
                                     self.model(self.grid)[:, i].detach().cpu().numpy(),
                                     cmap=cm.jet, linewidth=0.2, alpha=1)
                ax1.set_xlabel("x1")
                ax1.set_ylabel("x2")

    def print_mat(self, title):
        """
        Generates a visualization of the discovered equation's solution compared to the provided data. It creates either a 2D scatter plot or a 3D surface plot, depending on the dimensionality of the input data.
        
                Args:
                    title (str, optional): The title for the plot. Defaults to None.
        
                Returns:
                    None. The function displays the plot using matplotlib.pyplot.show().
        
                Why: This method helps to visually assess how well the discovered equation matches the data. By plotting the equation's solution against the data, we can quickly identify areas where the model performs well or poorly, guiding further refinement of the equation discovery process.
        """

        nparams = self.grid.shape[0]
        nvars_model = self.model.shape[0]
        fig = plt.figure(figsize=(15,8))
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i+1)
                if title != None:
                    ax1.set_title(title+' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.model[i].detach().cpu().numpy().reshape(-1))
            else:
                ax1 = fig.add_subplot(1, nvars_model, i+1, projection='3d')
                
                if title!=None:
                    ax1.set_title(title+' variable {}'.format(i))
                ax1.plot_trisurf(self.grid[0].detach().cpu().numpy().reshape(-1),
                            self.grid[1].detach().cpu().numpy().reshape(-1),
                            self.model[i].detach().cpu().numpy().reshape(-1),
                            cmap=cm.jet, linewidth=0.2, alpha=1)
            ax1.set_xlabel("x1")
            ax1.set_ylabel("x2")

    def dir_path(self, save_dir: str):
        """
        Generates a file path for saving a plot, creating the necessary directories if they don't exist.
        
        This method ensures that plots are saved in a structured manner, either in a default 'img' directory or a user-specified directory.
        It generates a unique file name based on the current timestamp to prevent overwriting previous plots.
        
        Args:
            save_dir (str, optional): The directory where the plot should be saved. If None, the plot is saved to a default 'img' directory.
        
        Returns:
            str: The absolute path to the generated plot file.
        """
        if save_dir == None:
            try:
                img_dir = os.path.join(os.path.dirname(__file__), 'img')
            except:
                current_dir = globals()['_dh'][0]
                img_dir = os.path.join(os.path.dirname(current_dir), 'img')

            if not (os.path.isdir(img_dir)):
                os.mkdir(img_dir)
            directory = os.path.abspath(os.path.join(img_dir,
                                                     str(datetime.datetime.now().timestamp()) + '.png'))
        else:
            if not (os.path.isdir(save_dir)):
                os.mkdir(save_dir)
            directory = os.path.join(save_dir,
                                     str(datetime.datetime.now().timestamp()) + '.png')
        return directory

    def solution_print(self, title=None, solution_print=False,
                       solution_save=False, save_dir=None):
        """
        Prints and saves the discovered equation's solution.
        
                Depending on the representation (`mode`), the solution is printed either as a matrix or a neural network.
                This function also handles saving the visualization of the solution to a file and/or displaying it,
                allowing for inspection and preservation of the identified equation's behavior. The plot is closed after
                display or save.
        
                Args:
                    title: The title of the plot (optional).
                    solution_print: Whether to display the plot (optional, default: False).
                    solution_save: Whether to save the plot to a file (optional, default: False).
                    save_dir: The directory to save the plot to (optional).
        
                Returns:
                    None.
        """

        directory = self.dir_path(save_dir)

        if self.mode == 'mat':
            self.print_mat(title)
        else:
            self.print_nn(title)
        if solution_save:
            plt.savefig(directory)
        if solution_print:
            plt.show()
        plt.close()


class Solver():
    """
    High-level interface for solving equations.
    """


    def __init__(self, grid: torch.Tensor, equal_cls,
                 model: Any, mode: str, weak_form: Union[None, list] = None):
        """
        Initializes the Solver for discovering differential equations from data.
        
        This class provides a high-level interface for setting up the problem.
        It takes the computational grid, equation definition, neural network model,
        calculation mode, and weak form (if applicable) as inputs.
        The Solver then uses these components to find the solution of the problem.
        
        Args:
            grid (torch.Tensor): Array of n-D points representing the computational domain.
            equal_cls: Object defining the equation to be solved.
            model: Neural network model used to approximate the solution.
            mode (str): Calculation method (e.g., "NN", "autograd", "mat").
            weak_form (list, optional): List of basis functions for weak formulation. Defaults to None.
        """
        self.grid = check_device(grid)
        self.equal_cls = equal_cls
        self.model = model
        self.mode = mode
        self.weak_form = weak_form

    def optimizer_choice(self, optimizer: str, learning_rate: float) -> \
            Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS]:
        """
        Sets the optimizer for training the model. The choice of optimizer impacts how the model learns from data and converges to a solution. Different optimizers are suitable for different types of problems and model architectures.
        
                Args:
                   optimizer: The name of the optimizer to use ('Adam', 'SGD', or 'LBFGS').
                   learning_rate: The learning rate for the optimizer, controlling the step size during optimization.
        
                Returns:
                   A torch.optim object (Adam, SGD, or LBFGS) configured with the model parameters and specified learning rate. This optimizer will be used to update the model's weights during the training process, guiding it towards a solution that best fits the data.
        """
        if optimizer == 'Adam':
            torch_optim = torch.optim.Adam
        elif optimizer == 'SGD':
            torch_optim = torch.optim.SGD
        elif optimizer == 'LBFGS':
            torch_optim = torch.optim.LBFGS
        else:
            print('Wrong optimizer chosen, optimization was not performed')
            return self.model

        if self.mode == 'NN' or self.mode == 'autograd':
            optimizer = torch_optim(self.model.parameters(), lr=learning_rate)
        elif self.mode == 'mat':
            optimizer = torch_optim([self.model.requires_grad_()], lr=learning_rate)

        return optimizer

    def str_param(self, inverse_param: dict):
        """
        Generates a string representation of model parameters that are part of the equation search space.
        
                This method is used to create a readable representation of the parameters
                currently being explored by the evolutionary algorithm. This helps in
                tracking the progress of the search and understanding the parameter values
                associated with different equation candidates.
        
                Args:
                    inverse_param (dict): A dictionary whose keys are parameter names to include in the string.
        
                Returns:
                    str: A string containing the name and value of each specified parameter,
                         formatted as "name=value ", or an empty string if no parameters are found.
        """
        param = list(inverse_param.keys())
        for name, p in self.model.named_parameters():
            if name in param:
                try:
                    param_str += name + '=' + str(p.item()) + ' '
                except:
                    param_str = name + '=' + str(p.item()) + ' '
        return param_str

    def solve(self,lambda_operator: Union[float, list] = 1,lambda_bound: Union[float, list] = 10, derivative_points:float=2,
              lambda_update: bool = False, second_order_interactions: bool = True, sampling_N: int = 1, verbose: int = 0,
              learning_rate: float = 1e-4, gamma=None, lr_decay=1000,
              eps: float = 1e-5, tmin: int = 1000, tmax: float = 1e5,
              nmodels: Union[int, None] = None, name: Union[str, None] = None,
              abs_loss: Union[None, float] = None, use_cache: bool = True,
              cache_dir: str = '../cache/', cache_verbose: bool = False,
              save_always: bool = False, print_every: Union[int, None] = 100,
              cache_model: Union[torch.nn.Sequential, None] = None,
              patience: int = 5, loss_oscillation_window: int = 100,
              no_improvement_patience: int = 1000, model_randomize_parameter: Union[int, float] = 0,
              optimizer_mode: str = 'Adam', step_plot_print: Union[bool, int] = False,
              step_plot_save: Union[bool, int] = False, image_save_dir: Union[str, None] = None, tol: float = 0,
              clear_cache: bool  =False, normalized_loss_stop: bool = False, inverse_parameters: dict = None) -> Any:
        """
        High-level interface for solving equations, aiming to find the best model representation of the underlying system.
        
                This method iteratively refines a model by minimizing the loss function, which balances the equation's residual and boundary conditions.
                It employs optimization techniques and caching mechanisms to efficiently explore the solution space and identify the optimal model parameters.
                The training process can be customized with various parameters, including learning rate, regularization coefficients, and stopping criteria.
        
                Args:
                    lambda_operator: Coefficient for the operator part of the loss function, controlling the weight of the equation's residual. Can be a single value or a list for adaptive weighting.
                    lambda_bound: Coefficient for the boundary part of the loss function, controlling the weight of satisfying boundary conditions. Can be a single value or a list for adaptive weighting.
                    derivative_points: Number of points to use for derivative calculations.
                    lambda_update: Enable adaptive update of lambda coefficients during training.
                    second_order_interactions: Consider second-order interactions between variables in the loss calculation.
                    sampling_N: Number of sampling points to use during training.
                    verbose: Level of detail to print during the training process (0 for no output, higher values for more detailed information).
                    learning_rate: Determines the step size at each iteration while moving toward a minimum of a loss function.
                    gamma: Multiplicative factor of learning rate decay.
                    lr_decay: Decays the learning rate of each parameter group by gamma every epoch.
                    eps: Arbitrarily small number that uses for loss comparison criterion.
                    tmin: Minimum execution time.
                    tmax: Maximum execution time.
                    nmodels: Number of cached models to store. If None, caching is disabled.
                    name: Model name for saving and loading from cache.
                    abs_loss: Absolute loss threshold for stopping the training process.
                    use_cache: Whether to use cached models to initialize the training process.
                    cache_dir: Directory where cached models are stored.
                    cache_verbose: Whether to print detailed information about the models in the cache.
                    save_always: Whether to save the trained model even if caching is disabled.
                    print_every: Frequency (in iterations) at which to print the training progress. If None, no printing occurs.
                    cache_model: A pre-trained model to use as the initial model in the cache.
                    patience: Number of iterations to wait for improvement in loss before stopping the training.
                    loss_oscillation_window: Window size for detecting loss oscillations.
                    no_improvement_patience: Number of iterations with no improvement in loss before triggering a model randomization.
                    model_randomize_parameter: Creates a random model parameters (weights, biases) multiplied with a given randomize parameter.
                    optimizer_mode: Optimizer choice (Adam, SGD, LBFGS).
                    step_plot_print: Whether to print a plot of the solution at each given step. Can be a boolean or an integer representing the frequency of plotting.
                    step_plot_save: Whether to save a plot of the solution at each given step. Can be a boolean or an integer representing the frequency of plotting.
                    image_save_dir: Directory where to save the plots.
                    tol: Float constant, influences on error penalty in casual_loss algorithm.
                    clear_cache: Whether to clear the cache directory before starting the training process.
                    normalized_loss_stop: Whether to use normalized loss for stopping criterion.
                    inverse_parameters: Dictionary of parameters to inverse during training.
        
                Returns:
                    The trained model.
        """
        print('before Model_prepare:', self.grid.dtype)
        Cache_class = Model_prepare(self.grid, self.equal_cls,
                                    self.model, self.mode, self.weak_form)
        print('after Model_prepare:', self.grid.dtype)
        Cache_class.change_cache_dir(cache_dir)

        # prepare input data to uniform format
        r = create_random_fn(model_randomize_parameter)

        if clear_cache:
            Cache_class.clear_cache_dir()

        #  use cache if needed
        if use_cache:
            self.model, min_loss = Cache_class.cache(nmodels,
                                                     lambda_operator,
                                                     lambda_bound,
                                                     cache_verbose,
                                                     model_randomize_parameter,
                                                     cache_model,
                                                    return_normalized_loss=normalized_loss_stop)

            Solution_class = Solution(self.grid, self.equal_cls,
                                      self.model, self.mode, self.weak_form,
                                      lambda_operator, lambda_bound, tol, derivative_points)
        else:
            Solution_class = Solution(self.grid, self.equal_cls,
                                      self.model, self.mode, self.weak_form,
                                      lambda_operator, lambda_bound, tol, derivative_points)

            min_loss , _ = Solution_class.evaluate()

        self.plot = Plots(self.model, self.grid, self.mode, tol)

        optimizer = self.optimizer_choice(optimizer_mode, learning_rate)

        if gamma != None:
            scheduler = ExponentialLR(optimizer, gamma=gamma)

        # standard NN stuff
        if verbose:
            print('[{}] initial (min) loss is {}'.format(
                datetime.datetime.now(), min_loss.item()))

        t = 0

        last_loss = np.zeros(loss_oscillation_window) + float(min_loss)
        line = np.polyfit(range(loss_oscillation_window), last_loss, 1)

        def closure():
            nonlocal cur_loss
            optimizer.zero_grad()
            loss, loss_normalized = Solution_class.evaluate(second_order_interactions=second_order_interactions,
                                           sampling_N=sampling_N,
                                           lambda_update=lambda_update)

            loss.backward()
            if normalized_loss_stop:
                cur_loss = loss_normalized.item()
            else:
                cur_loss = loss.item()
            return loss

        stop_dings = 0
        t_imp_start = 0
        # to stop train proceduce we fit the line in the loss data
        # if line is flat enough "patience" times, we stop the procedure
        cur_loss = min_loss
        while stop_dings <= patience:
            optimizer.step(closure)
            if cur_loss != cur_loss:
                print(f'Loss is equal to NaN, something went wrong (LBFGS+high'
                      f'learning rate and pytorch<1.12 could be the problem)')
                break

            last_loss[(t - 1) % loss_oscillation_window] = cur_loss

            if cur_loss < min_loss:
                min_loss = cur_loss
                t_imp_start = t

            if verbose:
                info_string = 'Step = {} loss = {:.6f} normalized loss line= {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                    t, cur_loss, line[0] / cur_loss, line[1] / cur_loss, stop_dings + 1)

            if gamma != None and t % lr_decay == 0:
                scheduler.step()

            if t % loss_oscillation_window == 0:
                line = np.polyfit(range(loss_oscillation_window), last_loss, 1)
                if abs(line[0] / cur_loss) < eps and t > 0:
                    stop_dings += 1
                    if self.mode == 'NN' or self.mode == 'autograd':
                        self.model.apply(r)
                    if verbose:
                        print('[{}] Oscillation near the same loss'.format(
                            datetime.datetime.now()))
                        print(info_string)
                        if inverse_parameters is not None:
                            print(self.str_param(inverse_parameters))
                        if step_plot_print or step_plot_save:
                            self.plot.solution_print(title='Iteration = ' + str(t),
                                                     solution_print=step_plot_print,
                                                     solution_save=step_plot_save,
                                                     save_dir=image_save_dir)

            if (t - t_imp_start) == no_improvement_patience:
                if verbose:
                    print('[{}] No improvement in {} steps'.format(
                        datetime.datetime.now(), no_improvement_patience))
                    print(info_string)
                    if inverse_parameters is not None:
                        print(self.str_param(inverse_parameters))
                    if step_plot_print or step_plot_save:
                        self.plot.solution_print(title='Iteration = ' + str(t),
                                                 solution_print=step_plot_print,
                                                 solution_save=step_plot_save,
                                                 save_dir=image_save_dir)
                t_imp_start = t
                stop_dings += 1
                if self.mode == 'NN' or self.mode == 'autograd':
                    self.model.apply(r)

            if abs_loss != None and cur_loss < abs_loss:
                if verbose:
                    print('[{}] Absolute value of loss is lower than threshold'.format(datetime.datetime.now()))
                    print(info_string)
                    if inverse_parameters is not None:
                        print(self.str_param(inverse_parameters))
                    if step_plot_print or step_plot_save:
                        self.plot.solution_print(title='Iteration = ' + str(t),
                                                 solution_print=step_plot_print,
                                                 solution_save=step_plot_save,
                                                 save_dir=image_save_dir)
                stop_dings += 1
            # print('t',t)
            if print_every != None and (t % print_every == 0) and verbose:
                print('[{}] Print every {} step'.format(
                    datetime.datetime.now(), print_every))
                print(info_string)
                if inverse_parameters is not None:
                    print(self.str_param(inverse_parameters))
                # print('loss', closure().item(), 'loss_norm', cur_loss)
                if step_plot_print or step_plot_save:
                    self.plot.solution_print(title='Iteration = ' + str(t),
                                             solution_print=step_plot_print,
                                             solution_save=step_plot_save,
                                             save_dir=image_save_dir)

            t += 1
            if t > tmax:
                break
        if save_always:
            if self.mode == 'mat':
                Cache_class.save_model_mat(name=name, cache_verbose=cache_verbose)
            else:
                Cache_class.save_model(self.model, self.model.state_dict(),
                                       optimizer.state_dict(),
                                       name=name)
        return self.model
