import numpy as np
import torch
from typing import Tuple, List
# from SALib import ProblemSpec

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
        Initializes the AdaptiveLambda module.
        
        This module dynamically adjusts a symbolic expression (lambda) based on data, refining its structure and parameters to improve accuracy.
        
        Args:
            sampling_N (int, optional): Controls the frequency of re-evaluating the lambda expression. Higher values lead to more frequent updates. Defaults to 1.
            second_order_interactions (bool, optional): Enables the calculation of second-order sensitivities, capturing more complex relationships within the data. Defaults to True.
        
        Returns:
            None
        
        Why:
            The sampling_N parameter controls how often the symbolic expression is refined based on the data.
            The second_order_interactions parameter enables the discovery of more complex relationships in the data by considering second-order sensitivities.
        """
        super().__init__()
        self.second_order_interactions = second_order_interactions
        self.sampling_N = sampling_N

    @staticmethod
    def lambda_compute(pointer: int, length_list: list, ST: np.ndarray) -> torch.Tensor:
        """
        Computes adaptive lambda values for each group of parameters.
        
        This function calculates lambda values based on the Sobol indices (ST) obtained from sensitivity analysis.
        These lambdas are used to adaptively adjust the learning rates during the optimization process,
        giving more weight to parameters that have a greater impact on the model's output.
        
        Args:
            pointer (int): Starting index in the ST array for the current group of parameters.
            length_list (list): A list containing the number of parameters in each group.
            ST (np.ndarray): Array of Sobol indices (total effect indices) for all parameters.
        
        Returns:
            torch.Tensor: A tensor containing the calculated lambda values for each group of parameters.
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
        """
        Updates the lambda values for the operator and boundary conditions based on sensitivity analysis.
        
                The method performs a sensitivity analysis using the provided solution lengths and total sampling dimension to determine the influence of each variable on the loss function. These sensitivity indices are then used to update the lambda values, effectively weighting the contribution of each term in the operator and boundary conditions based on its importance in minimizing the loss. This ensures that the optimization process focuses on the most relevant terms, leading to a more efficient and accurate discovery of the underlying differential equation.
        
                Args:
                    op_length (list): A list containing the lengths of each term in the operator solution.
                    bval_length (list): A list containing the lengths of each term in the boundary condition solution.
                    sampling_D (int): The total sampling dimension, which is the sum of `op_length` and `bval_length`.
        
                Returns:
                    lambda_op (torch.Tensor): Updated lambda values for the operator terms, reflecting their sensitivity.
                    lambda_bound (torch.Tensor): Updated lambda values for the boundary condition terms, reflecting their sensitivity.
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
        """
        Calculates and updates the Lagrangian multipliers (lambdas) for the equation and boundary conditions. These lambdas are crucial for adjusting the optimization process, balancing the influence of the equation and boundary conditions in satisfying the underlying differential equation. The method accumulates information about the equation's operator values, boundary values, and loss, and then updates the lambdas when enough samples are collected. This adaptive adjustment ensures that the solution adheres to both the equation and the specified boundary conditions.
        
                Args:
                    None
        
                Returns:
                    None
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
        """
        Updates the lambda value at the end of each epoch.
        
        This update is crucial for adapting the equation discovery process based on the performance of the current population of equations. By adjusting lambda, the algorithm can refine its search strategy and converge towards more accurate and relevant differential equation models.
        
        Args:
            logs: Contains information about the current epoch, such as loss values.
        
        Returns:
            None.
        """
        self.lambda_update()
