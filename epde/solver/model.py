import torch
from typing import Union, List, Any
import tempfile
import os

from epde.solver.data import Domain, Conditions, Equation
from epde.solver.input_preprocessing import Operator_bcond_preproc
from epde.solver.callbacks.callback_list import CallbackList
from epde.solver.solution import Solution
from epde.solver.optimizers.optimizer import Optimizer
from epde.solver.utils import save_model_nn, save_model_mat
from epde.solver.optimizers.closure import Closure
from epde.solver.device import device_type
import datetime


class Model():
    """
    class for preprocessing
    """

    def __init__(
            self,
            net: Union[torch.nn.Module, torch.Tensor],
            domain: Domain,
            equation: Equation,
            conditions: Conditions,
            batch_size: int = None):
        """
        Initializes the Model class, setting up the neural network, domain, equation, and boundary conditions.
        
                This setup is crucial for defining the problem to be solved, ensuring that the neural network is properly configured to learn the solution within the specified domain and satisfying the given equation and conditions.
        
                Args:
                    net (Union[torch.nn.Module, torch.Tensor]): The neural network to be trained, or a torch.Tensor for matrix-based approaches.
                    domain (Domain): The spatial or temporal domain over which the equation is defined.
                    equation (Equation): The differential equation to be solved.
                    conditions (Conditions): The boundary or initial conditions that constrain the solution.
                    batch_size (int, optional): The size of the batches used during training. Defaults to None.
        """
        self.net = net
        self.domain = domain
        self.equation = equation
        self.conditions = conditions

        self._check = None
        temp_dir = tempfile.gettempdir()
        folder_path = os.path.join(temp_dir, 'tedeous_cache/')
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            pass
        else:
            os.makedirs(folder_path)
        self._save_dir = folder_path
        self.batch_size = batch_size

    def compile(
            self,
            mode: str,
            lambda_operator: Union[List[float], float],
            lambda_bound: Union[List[float], float],
            normalized_loss_stop: bool = False,
            h: float = 0.001,
            inner_order: str = '1',
            boundary_order: str = '2',
            derivative_points: int = 2,
            weak_form: List[callable] = None,
            tol: float = 0):
        """
        Compiles the model by setting up the computational domain, equation, and boundary conditions based on the specified mode.
        
                This process prepares the model for the training loop by constructing the necessary components for loss calculation and optimization.
                The compilation configures the model to solve the problem using either matrix-based methods, neural networks, or automatic differentiation,
                depending on the chosen mode.
        
                Args:
                    mode (str): Specifies the computational approach (*mat*, *NN*, or *autograd*).
                    lambda_operator (Union[List[float], float]): Weight(s) for the operator term in the loss function.
                        Can be a single float for a single equation or a list of floats for a system of equations.
                    lambda_bound (Union[List[float], float]): Weight(s) for the boundary term in the loss function.
                        Can be a single float for uniform weighting or a list of floats for individual boundary conditions.
                    normalized_loss_stop (bool, optional): If True, the loss is normalized with lambdas set to 1. Defaults to False.
                    h (float, optional): Increment for finite-difference schemes (only for *NN* mode). Defaults to 0.001.
                    inner_order (str, optional): Order of the finite-difference scheme for inner points (*'1'* or *'2'*, only for *NN* mode). Defaults to '1'.
                    boundary_order (str, optional): Order of the finite-difference scheme for boundary points (*'1'* or *'2'*, only for *NN* mode). Defaults to '2'.
                    derivative_points (int, optional): Number of points for finite-difference derivative calculation (*mat* mode).
                        If set to 2, a central difference scheme is used. Defaults to 2.
                    weak_form (List[callable], optional): Basis functions for the weak formulation of the loss. Defaults to None.
                    tol (float, optional): Tolerance for the casual loss term. Defaults to 0.
        
                Returns:
                    None
        """
        self.mode = mode
        self.lambda_bound = lambda_bound
        self.lambda_operator = lambda_operator
        self.normalized_loss_stop = normalized_loss_stop
        self.weak_form = weak_form

        grid = self.domain.build(mode=mode)
        dtype = grid.dtype
        self.net.to(dtype)
        variable_dict = self.domain.variable_dict
        operator = self.equation.equation_lst
        bconds = self.conditions.build(variable_dict)

        self.equation_cls = Operator_bcond_preproc(grid, operator, bconds, h=h, inner_order=inner_order,
                                                   boundary_order=boundary_order).set_strategy(mode)
        if self.batch_size != None:
            if len(grid)<self.batch_size:
                self.batch_size=None

            # if len(grid)<self.batch_size:
            #     self.batch_size=None

        self.solution_cls = Solution(grid, self.equation_cls, self.net, mode, weak_form,
                                     lambda_operator, lambda_bound, tol, derivative_points,
                                     batch_size=self.batch_size)


    def _model_save(
        self,
        save_model: bool,
        model_name: str):
        """
        Saves the trained model to disk.
        
        This function persists the model, allowing for later reuse without retraining.
        The saving format depends on the operational mode (MAT or NN), using either a
        MAT-file format or a neural network-specific format.
        
        Args:
            save_model (bool): A flag indicating whether to save the model.
            model_name (str): The name to use for the saved model file.
        
        Returns:
            None. The function saves the model to a file in the specified directory.
        """
        if save_model:
            if self.mode == 'mat':
                save_model_mat(self._save_dir,
                                model=self.net,
                                domain=self.domain,
                                name=model_name)
            else:
                save_model_nn(self._save_dir, model=self.net, name=model_name)

    def train(self,
              optimizer: Optimizer,
              epochs: int,
              info_string_every: Union[int, None] = None,
              mixed_precision: bool = False,
              save_model: bool = False,
              model_name: Union[str, None] = None,
              callbacks: Union[List, None] = None):
        """
        Trains the model to discover the underlying differential equation.
        
                The training process involves optimizing the model's parameters using the provided optimizer and training data over a specified number of epochs. Callbacks are used to monitor and control the training process, allowing for customization and early stopping. The goal is to minimize the loss function, which represents the difference between the model's predictions and the observed data. This optimization helps to refine the model's structure and parameters, ultimately leading to a more accurate representation of the underlying differential equation.
        
                Args:
                    optimizer (Optimizer): The optimizer object used for updating the model's parameters during training.
                    epochs (int): The number of training epochs to perform.
                    info_string_every (Union[int, None], optional):  Prints the loss value every *info_string_every* epochs. Defaults to None.
                    mixed_precision (bool, optional): Enables mixed precision training for faster computation and reduced memory usage. Defaults to False.
                    save_model (bool, optional): Saves the trained model to the cache. Defaults to False.
                    model_name (Union[str, None], optional): The name to use when saving the model. Defaults to None.
                    callbacks (Union[List, None], optional): A list of callbacks to execute during training, providing hooks for monitoring and control. Defaults to None.
        
                Returns:
                    torch.Tensor: The final loss value achieved during training.
        """

        self.t = 1
        self.stop_training = False

        callbacks = CallbackList(callbacks=callbacks, model=self)

        callbacks.on_train_begin()

        self.net = self.solution_cls.model

        self.optimizer = optimizer.optimizer_choice(self.mode, self.net)

        closure = Closure(mixed_precision, self).get_closure(optimizer.optimizer)

        self.min_loss, _ = self.solution_cls.evaluate()

        self.cur_loss = self.min_loss

        # print('[{}] initial (min) loss is {}'.format(
        #         datetime.datetime.now(), self.min_loss.item()))

        while self.t < epochs and self.stop_training == False:
            callbacks.on_epoch_begin()

            self.optimizer.zero_grad()
            if optimizer.optimizer == 'NNCG' and (self.t-1) % 20 == 0: #Hard-coded preconditional freq
                grads = self.optimizer.gradient(self.cur_loss)
                self.optimizer.update_preconditioner(grads)
            

            iter_count = 1 if self.batch_size is None else self.solution_cls.operator.n_batches
            for _ in range(iter_count): # if batch mod then iter until end of batches else only once
                if device_type() == 'cuda' and mixed_precision:
                    closure()
                else:
                    self.optimizer.step(closure)
                if optimizer.gamma is not None and self.t % optimizer.decay_every == 0:
                    optimizer.scheduler.step()
            callbacks.on_epoch_end()

            self.t += 1
            if info_string_every is not None:
                if self.t % info_string_every == 0:
                    loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                    info = 'Step = {} loss = {:.6f}.'.format(self.t, loss)
                    print(info)

        callbacks.on_train_end()

        self._model_save(save_model, model_name)
        return self.cur_loss # Or min_loss?
