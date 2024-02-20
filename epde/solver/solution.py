from __future__ import annotations

import torch
import numpy as np
from copy import copy, deepcopy
from typing import Tuple, Union

from epde.solver.points_type import Points_type
from epde.solver.derivative import Derivative
from epde.solver.eval import Operator, Bounds
from epde.solver.losses import Losses
from epde.solver.device import device_type, check_device
from epde.solver.input_preprocessing import lambda_prepare, Equation_NN, Equation_mat, Equation_autograd
from epde.solver.utils import *


flatten_list = lambda t: [item for sublist in t for item in sublist]

class Solution():
    """
    class for different loss functions calculation.
    """
    def __init__(self, grid: torch.Tensor, equal_cls: Union[Equation_NN, Equation_mat, Equation_autograd],
                 model: Union[torch.nn.Sequential, torch.Tensor], mode: str, weak_form: Union[None, list[callable]],
                 lambda_operator, lambda_bound, tol: float = 0, derivative_points: int = 2):

        self.grid = check_device(grid)
        if mode == 'NN':
            sorted_grid = Points_type(self.grid).grid_sort()
            self.n_t = len(sorted_grid['central'][:, 0].unique())
        elif mode == 'autograd':
            self.n_t = len(self.grid[:, 0].unique())
        elif mode == 'mat':
            self.n_t = grid.shape[1]
        equal_copy = deepcopy(equal_cls)
        prepared_operator = equal_copy.operator_prepare()
        self.operator_coeff(equal_cls, prepared_operator)
        prepared_bconds = equal_copy.bnd_prepare()
        self.model = model.to(device_type())
        self.mode = mode
        self.weak_form = weak_form
        self.lambda_operator = lambda_operator
        self.lambda_bound = lambda_bound
        self.tol = tol


        self.operator = Operator(self.grid, prepared_operator, self.model,
                                   self.mode, weak_form, derivative_points)
        self.boundary = Bounds(self.grid, prepared_bconds, self.model,
                                   self.mode, weak_form, derivative_points)

        self.loss_cls = Losses(self.mode, self.weak_form, self.n_t, self.tol)
        self.eps = 0
        self.op_list = []
        self.bval_list = []
        self.loss_list = []

    @staticmethod
    def operator_coeff(equal_cls, operator):
        for i in range(len(operator)):
            eq = operator[i]
            for key in eq.keys():
                if isinstance(eq[key]['coeff'], torch.Tensor):
                    eq[key]['coeff'] = eq[key]['coeff'].to(device_type())
#                    try:
#                        eq[key]['coeff'] = equal_cls.operator[i][key]['coeff'].to(device_type())
#                    except:
#                        eq[key]['coeff'] = equal_cls.operator[key]['coeff'].to(device_type())


    def evaluate(self,
                 second_order_interactions: bool = True,
                 sampling_N: int = 1,
                 lambda_update: bool = False,
                 save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes loss.

        Args:
            second_order_interactions: optimizer iteration (serves only for computing adaptive lambdas).
            sampling_N: parameter for accumulation of solutions (op, bcs). The more sampling_N, the more accurate the estimation of the variance.
            lambda_update: update lambda or not.
            tol: float constant, influences on error penalty.
            save_graph: boolean constant, responsible for saving the computational graph.

        Returns:
            loss
        """

        op = self.operator.operator_compute()
        bval, true_bval, bval_keys, bval_length = self.boundary.apply_bcs()

        self.lambda_operator = lambda_prepare(op, self.lambda_operator)
        self.lambda_bound = lambda_prepare(bval, self.lambda_bound)

        loss, loss_normalized = self.loss_cls.compute(op, bval, true_bval,
                                                      self.lambda_operator,
                                                      self.lambda_bound,
                                                      save_graph)

        if lambda_update:
            # TODO refactor this lambda thing to class or function.
            bcs = bcs_reshape(bval, true_bval, bval_length)
            op_length = [op.shape[0]]*op.shape[-1]

            self.op_list.append(torch.t(op).reshape(-1).cpu().detach().numpy())
            self.bval_list.append(bcs.cpu().detach().numpy())
            self.loss_list.append(float(loss_normalized.item()))

            sampling_amount, sampling_D = samples_count(
                        second_order_interactions = second_order_interactions,
                        sampling_N = sampling_N,
                        op_length=op_length,
                        bval_length = bval_length)

            if len(self.op_list) == sampling_amount:
                self.lambda_operator, self.lambda_bound = Lambda(
                                        self.op_list, self.bval_list,
                                        self.loss_list,
                                        second_order_interactions)\
                                        .update(op_length=op_length,
                                                bval_length=bval_length,
                                                sampling_D=sampling_D)
                self.op_list.clear()
                self.bval_list.clear()
                self.loss_list.clear()

                oper_keys = [f'eq_{i}' for i in range(len(op_length))]
                lambda_print(self.lambda_operator, oper_keys)
                lambda_print(self.lambda_bound, bval_keys)


        return loss, loss_normalized

