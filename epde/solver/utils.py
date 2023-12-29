# this one contain some stuff for computing different auxiliary things.

import torch
import numpy as np
from typing import Tuple
from torch.nn import Module
from torch import Tensor
from SALib import ProblemSpec

def samples_count(second_order_interactions: bool,
                  sampling_N: int,
                  op_length: list,
                  bval_length:list) -> Tuple[int, int]:
    """
    Count samples for variance based sensitivity analysis.

    Args:
        second_order_interactions:
        sampling_N: essentially determines how often the lambda will be re-evaluated.
        op_length: operator values length.
        bval_length: boundary value length.

    Returns:
        sampling_amount: overall sampling value.
        sampling_D: sum of length of grid and boundaries.


    """
    grid_len = sum(op_length)
    bval_len = sum(bval_length)

    sampling_D = grid_len + bval_len

    if second_order_interactions:
        sampling_amount = sampling_N * (2 * sampling_D + 2)
    else:
        sampling_amount = sampling_N * (sampling_D + 2)
    return sampling_amount, sampling_D

def lambda_print(lam, keys) -> None:
    """
    Print lambda value.

    Args:
        dict_: dict with lambdas.
    """
    lam = lam.reshape(-1)
    for val, key in zip(lam, keys):
        print('lambda_{}: {}'.format(key, val.item()))

def bcs_reshape(bval, true_bval, bval_length) \
                                            -> Tuple[dict, dict, dict, dict]:
    """
    Preprocessing for lambda evaluating.

    Args:
        op: dict with operator solution.
        bval: dict with boundary solution.
        true_bval: dict with true boundary solution (i.e. right side of equation).

    Returns:
        op: dict with operator solution.
        bcs: dict with difference of bval and true_bval.
        op_length: dict with lengths of operator solution.
        bval_length: dict with lengths of boundary solution.
    """

    bval_diff = bval - true_bval

    bcs = torch.cat([bval_diff[0:bval_length[i], i].reshape(-1)
                                        for i in range(bval_diff.shape[-1])])

    return bcs


class Lambda:
    """
    Serves for computing adaptive lambdas.
    """
    def __init__(self, op_list: list,
                 bcs_list: list,
                 loss_list: list,
                 sampling_N: int = 1,
                 second_order_interactions = True):
        """
        Args:
            op_list: list with operator solution.
            bcs_list: list with boundary solution.
            loss_list: list with losses.
            sampling_N: parameter for accumulation of solutions (op, bcs). The more sampling_N, the more accurate the estimation of the variance.
            second_order_interactions: computes second order Sobol indices.
        """
        self.second_order_interactions = second_order_interactions
        self.op_list = op_list
        self.bcs_list = bcs_list
        self.loss_list = loss_list
        self.sampling_N = sampling_N

    @staticmethod
    def lambda_compute(pointer: int, length_list: list, ST: np.ndarray) -> dict:
        """
        Computes lambdas.

        Args:
            pointer: the label to calculate the lambda for the corresponding parameter.
            length_dict: dict where values are lengths.
            ST: result of SALib.ProblemSpec().

        Returns:
            dict with lambdas.

        """
        lambdas = []
        for value in length_list:
            lambdas.append(sum(ST) / sum(ST[pointer:pointer + value]))
            pointer += value
        return torch.tensor(lambdas).float().reshape(1, -1)

    def update(self, op_length: list,
               bval_length: list,
               sampling_D: int) -> Tuple[dict, dict]:
        """
        Updates all lambdas (operator and boundary).

        Args:
            op_length: dict with lengths of operator solution.
            bval_length: dict with lengths of boundary solution.
            sampling_D: sum of op_length and bval_length.

        Returns:
            lambda_operator: values of lambdas for operator.
            lambda_bound: values of lambdas for boundary.
        """
        op_array = np.array(self.op_list)
        bc_array = np.array(self.bcs_list)
        loss_array = np.array(self.loss_list)

        X_array = np.hstack((op_array, bc_array))

        bounds = [[-100, 100] for _ in range(sampling_D)]
        names = ['x{}'.format(i) for i in range(sampling_D)]

        sp = ProblemSpec({'names': names, 'bounds': bounds})

        sp.set_samples(X_array)
        sp.set_results(loss_array)
        sp.analyze_sobol(calc_second_order=self.second_order_interactions)

        '''
        To assess variance we need total sensitiviy indices for every variable
        '''
        ST = sp.analysis['ST']

        lambda_op = self.lambda_compute(0, op_length, ST)

        lambda_bnd = self.lambda_compute(sum(op_length), bval_length, ST)

        return lambda_op, lambda_bnd


class PadTransform(Module):
    """Pad tensor to a fixed length with given padding value.

    :param max_length: Maximum length to pad to
    :type max_length: int
    :param pad_value: Value to pad the tensor with
    :type pad_value: bool
    
    src: https://pytorch.org/text/stable/transforms.html#torchtext.transforms.PadTransform
    
    Done to avoid torchtext dependency (we need only this function).
    """

    def __init__(self, max_length: int, pad_value: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: The tensor to pad
        :type x: Tensor
        :return: Tensor padded up to max_length with pad_value
        :rtype: Tensor
        """
        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x