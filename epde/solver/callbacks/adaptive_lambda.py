import numpy as np
import torch
from typing import Tuple, List
from SALib import ProblemSpec

from epde.solver.callbacks.callback import Callback
from epde.solver.utils import bcs_reshape, samples_count, lambda_print

class AdaptiveLambda(Callback):
    """
    Serves for computing adaptive lambdas.
    """
    def __init__(self,
                 sampling_N: int = 1,
                 second_order_interactions = True):
        """

        Args:
            sampling_N (int, optional): essentially determines how often the lambda will be re-evaluated. Defaults to 1.
            second_order_interactions (bool, optional): Calculate second-order sensitivities. Defaults to True.
        """
        super().__init__()
        self.second_order_interactions = second_order_interactions
        self.sampling_N = sampling_N

    @staticmethod
    def lambda_compute(pointer: int, length_list: list, ST: np.ndarray) -> torch.Tensor:
        """ Computes lambdas.

        Args:
            pointer (int): the label to calculate the lambda for the corresponding parameter.
            length_list (list): dict where values are lengths.
            ST (np.ndarray): result of SALib.ProblemSpec().

        Returns:
            torch.Tensor: calculated lambdas written as vector
        """

        lambdas = []
        for value in length_list:
            lambdas.append(sum(ST) / sum(ST[pointer:pointer + value]))
            pointer += value
        return torch.tensor(lambdas).float().reshape(1, -1)

    def update(self,
               op_length: List,
               bval_length: List,
               sampling_D: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Updates all lambdas (operator and boundary).

        Args:
            op_length (list): list with lengths of operator solution.
            bval_length (list): list with lengths of boundary solution.
            sampling_D (int): sum of op_length and bval_length.

        Returns:
            lambda_op (torch.Tensor): values of lambdas for operator.
            lambda_bound (torch.Tensor): values of lambdas for boundary.
        """

        op_array = np.array(self.op_list)
        bc_array = np.array(self.bval_list)
        loss_array = np.array(self.loss_list)

        X_array = np.hstack((op_array, bc_array))

        bounds = [[-100, 100] for _ in range(sampling_D)]
        names = ['x{}'.format(i) for i in range(sampling_D)]

        sp = ProblemSpec({'names': names, 'bounds': bounds})

        sp.set_samples(X_array)
        sp.set_results(loss_array)
        sp.analyze_sobol(calc_second_order=self.second_order_interactions)

        #
        # To assess variance we need total sensitiviy indices for every variable
        #
        ST = sp.analysis['ST']

        lambda_op = self.lambda_compute(0, op_length, ST)

        lambda_bnd = self.lambda_compute(sum(op_length), bval_length, ST)

        return lambda_op, lambda_bnd

    def lambda_update(self):
        """ Method for lambdas calculation.
        """
        sln_cls = self.model.solution_cls
        bval = sln_cls.bval
        true_bval = sln_cls.true_bval
        bval_keys = sln_cls.bval_keys
        bval_length = sln_cls.bval_length
        op = sln_cls.op if sln_cls.batch_size is None else sln_cls.save_op # if batch mod use accumulative loss else from single eval
        self.op_list = sln_cls.op_list
        self.bval_list = sln_cls.bval_list
        self.loss_list = sln_cls.loss_list

        bcs = bcs_reshape(bval, true_bval, bval_length)
        op_length = [op.shape[0]]*op.shape[-1]

        self.op_list.append(torch.t(op).reshape(-1).cpu().detach().numpy())
        self.bval_list.append(bcs.cpu().detach().numpy())
        self.loss_list.append(float(sln_cls.loss_normalized.item()))

        sampling_amount, sampling_D = samples_count(
                    second_order_interactions = self.second_order_interactions,
                    sampling_N = self.sampling_N,
                    op_length=op_length,
                    bval_length = bval_length)

        if len(self.op_list) == sampling_amount:
            sln_cls.lambda_operator, sln_cls.lambda_bound = \
                self.update(op_length=op_length, bval_length=bval_length, sampling_D=sampling_D)
            self.op_list.clear()
            self.bval_list.clear()
            self.loss_list.clear()

            oper_keys = [f'eq_{i}' for i in range(len(op_length))]
            lambda_print(sln_cls.lambda_operator, oper_keys)
            lambda_print(sln_cls.lambda_bound, bval_keys)

    def on_epoch_end(self, logs=None):
        self.lambda_update()
