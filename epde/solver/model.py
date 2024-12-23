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
    """class for preprocessing"""
    def __init__(
            self,
            net: Union[torch.nn.Module, torch.Tensor],
            domain: Domain,
            equation: Equation,
            conditions: Conditions,
            batch_size: int = None):
        """
        Args:
            net (Union[torch.nn.Module, torch.Tensor]): neural network or torch.Tensor for mode *mat*
            grid (Domain): object of class Domain
            equation (Equation): object of class Equation
            conditions (Conditions): object of class Conditions
            batch_size (int): size of batch
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
        """ Compile model for training process.

        Args:
            mode (str): *mat, NN, autograd*
            lambda_operator (Union[List[float], float]): weight for operator term.
            It can be float for single equation or list of float for system.
            lambda_bound (Union[List[float], float]): weight for boundary term.
            It can be float for all types of boundary cond-ns or list of float for every condition type.
            normalized_loss_stop (bool, optional): loss with lambdas=1. Defaults to False.
            h (float, optional): increment for finite-difference scheme only for *NN*. Defaults to 0.001.
            inner_order (str, optional): order of finite-difference scheme *'1', '2'* for inner points.
            Only for *NN*. Defaults to '1'.
            boundary_order (str, optional): order of finite-difference scheme *'1', '2'* for boundary points.
            Only for *NN*. Defaults to '2'.
            derivative_points (int, optional): number of points for finite-difference scheme in *mat* mode.
            if derivative_points=2 the central scheme are used. Defaults to 2.
            weak_form (List[callable], optional): basis function for weak loss. Defaults to None.
            tol (float, optional): tolerance for causual loss. Defaults to 0.
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
        print(self.solution_cls.grid)
        print('boop!')


    def _model_save(
        self,
        save_model: bool,
        model_name: str):
        """ Model saving.

        Args:
            save_model (bool): save model or not.
            model_name (str): model name.
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
        """ train model.

        Args:
            optimizer (Optimizer): the object of Optimizer class
            epochs (int): number of epoch for training.
            info_string_every (Union[int, None], optional): print loss state after *info_string_every* epoch. Defaults to None.
            mixed_precision (bool, optional): apply mixed precision for calculation. Defaults to False.
            save_model (bool, optional): save resulting model in cache. Defaults to False.
            model_name (Union[str, None], optional): model name. Defaults to None.
            callbacks (Union[List, None], optional): callbacks for training process. Defaults to None.
        """

        self.t = 1
        self.stop_training = False

        callbacks = CallbackList(callbacks=callbacks, model=self)

        callbacks.on_train_begin()

        self.net = self.solution_cls.model

        self.optimizer = optimizer.optimizer_choice(self.mode, self.net)

        print()
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
