"""Module for losses calculation"""

from typing import Tuple, Union
import numpy as np
import torch

from epde.solver.input_preprocessing import lambda_prepare


class Losses():
    """
    Class which contains all losses.
    """

    def __init__(self,
                 mode: str,
                 weak_form: Union[None, list],
                 n_t: int,
                 tol: Union[int, float],
                 n_t_operation: callable = None):
        """
        Initializes the Losses class with parameters defining the loss calculation.
        
                This setup configures how the loss will be computed based on the chosen mode,
                basis functions (if applicable), temporal discretization, and tolerance levels.
                These parameters are essential for tailoring the loss function to the specific
                characteristics of the differential equation being discovered and the data used for training.
        
                Args:
                    mode (str): Calculation mode (*NN*, *autograd*, or *mat*), determining the method for loss computation.
                    weak_form (Union[None, list]): List of basis functions if using a weak formulation of the loss.
                    n_t (int): Number of unique points in the time dimension, influencing the temporal discretization.
                    tol (Union[int, float]): Tolerance value used as a penalty in the causal loss calculation.
                    n_t_operation (callable): Function to calculate `n_t` for each batch, allowing dynamic adjustment of temporal discretization.
        """

        self.mode = mode
        self.weak_form = weak_form
        self.n_t = n_t
        self.n_t_operation = n_t_operation
        self.tol = tol
        # TODO: refactor loss_op, loss_bcs into one function, carefully figure out when bval
        # is None + fix causal_loss operator crutch (line 76).

    def _loss_op(self,
                operator: torch.Tensor,
                lambda_op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the operator loss term, penalizing deviations from the governing equation.
        
        This loss encourages the discovered equation to accurately represent the
        relationships present in the data.  It measures how well the learned
        operator satisfies the equation across the domain.
        
        Args:
            operator (torch.Tensor): The result of applying the discovered operator.
                This represents the equation's residual at each point.
                See `eval` module -> `operator_compute()` for details.
            lambda_op (torch.Tensor): Regularization parameter controlling the
                strength of the operator loss.  This balances the trade-off
                between equation fit and complexity.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - loss_operator (torch.Tensor): The operator loss term, a scalar value.
                - op (torch.Tensor): The mean squared error of the operator on the grid,
                  providing a measure of equation satisfaction.
        """
        if self.weak_form is not None and self.weak_form != []:
            op = operator
        else:
            op = torch.mean(operator**2, 0)

        loss_operator = op @ lambda_op.T
        return loss_operator, op


    def _loss_bcs(self,
                 bval: torch.Tensor,
                 true_bval: torch.Tensor,
                 lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss associated with boundary conditions, penalizing deviations from the true boundary values. This ensures that the discovered equation accurately reflects the system's behavior at its boundaries.
        
        Args:
            bval (torch.Tensor): Calculated values of boundary conditions.
            true_bval (torch.Tensor): True values of boundary conditions.
            lambda_bound (torch.Tensor): Regularization parameter for the boundary term in the loss.
        
        Returns:
            loss_bnd (torch.Tensor): Boundary term in the loss.
            bval_diff (torch.Tensor): Mean squared error of all boundary conditions.
        """

        bval_diff = torch.mean((bval - true_bval)**2, 0)

        loss_bnd = bval_diff @ lambda_bound.T
        return loss_bnd, bval_diff


    def _default_loss(self,
                     operator: torch.Tensor,
                     bval: torch.Tensor,
                     true_bval: torch.Tensor,
                     lambda_op: torch.Tensor,
                     lambda_bound: torch.Tensor,
                     save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the overall loss based on the discrepancy between the predicted and true values, considering both the governing equation and boundary conditions. This loss function guides the optimization process to find the equation that best fits the data.
        
                Args:
                    operator (torch.Tensor): The result of applying the discovered differential operator to the input data.
                    bval (torch.Tensor): The predicted values at the boundaries of the domain.
                    true_bval (torch.Tensor): The true (observed) values at the boundaries of the domain.
                    lambda_op (torch.Tensor): A weighting factor to balance the importance of the operator loss.
                    lambda_bound (torch.Tensor): A weighting factor to balance the importance of the boundary condition loss.
                    save_graph (bool, optional): Whether to save the computational graph for backpropagation. Defaults to True.
        
                Returns:
                    loss (torch.Tensor): The total loss, combining the operator and boundary condition losses.
                    loss_normalized (torch.Tensor): The total loss with regularization parameters set to 1, providing a baseline for comparison.
        """

        if bval is None:
            return torch.sum(torch.mean((operator) ** 2, 0))

        loss_oper, op = self._loss_op(operator, lambda_op)
        dtype = op.dtype
        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1).to(dtype)
        lambda_bound_normalized = lambda_prepare(bval, 1).to(dtype)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T +\
                        bval_diff @ lambda_bound_normalized.T

        # TODO make decorator and apply it for all losses.
        if not save_graph:
            temp_loss = loss.detach()
            del loss
            torch.cuda.empty_cache()
            loss = temp_loss

        return loss, loss_normalized

    def _causal_loss(self,
                    operator: torch.Tensor,
                    bval: torch.Tensor,
                    true_bval: torch.Tensor,
                    lambda_op: torch.Tensor,
                    lambda_bound: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a weighted loss, emphasizing earlier time points in the data.
        
                This loss function is designed to address the challenges of time-dependent
                differential equations, where the influence of initial states on the solution
                decreases over time. By weighting the loss at each time step, the method
                prioritizes accurate modeling of the initial dynamics.
        
                Args:
                    operator (torch.Tensor): The result of applying the differential operator.
                        See `eval module -> operator_compute()` for details.
                    bval (torch.Tensor): Calculated values of boundary conditions.
                    true_bval (torch.Tensor): True values of boundary conditions.
                    lambda_op (torch.Tensor): Regularization parameter for the operator term in the loss.
                    lambda_bound (torch.Tensor): Regularization parameter for the boundary term in the loss.
        
                Returns:
                    loss (torch.Tensor): The total loss, combining operator and boundary losses.
                    loss_normalized (torch.Tensor): The total loss with regularization parameters set to 1.
        """
        if self.n_t_operation is not None: # calculate if batch mod
            self.n_t = self.n_t_operation(operator)
        try:
            res = torch.sum(operator**2, dim=1).reshape(self.n_t, -1)
        except: # if n_t_operation calculate bad n_t then change n_t to batch size
            self.n_t = operator.size()[0]
            res = torch.sum(operator**2, dim=1).reshape(self.n_t, -1)
        m = torch.triu(torch.ones((self.n_t, self.n_t), dtype=res.dtype), diagonal=1).T
        with torch.no_grad():
            w = torch.exp(- self.tol * (m @ res))

        loss_oper = torch.mean(w * res)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)

        loss = loss_oper + loss_bnd

        lambda_bound_normalized = lambda_prepare(bval, 1)
        with torch.no_grad():
            loss_normalized = loss_oper +\
                        lambda_bound_normalized @ bval_diff

        return loss, loss_normalized

    def _weak_loss(self,
                  operator: torch.Tensor,
                  bval: torch.Tensor,
                  true_bval: torch.Tensor,
                  lambda_op: torch.Tensor,
                  lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss function for a weak solution of a differential equation, considering both the equation itself and boundary conditions.
        
                Args:
                    operator (torch.Tensor): The result of applying the differential operator to the predicted solution.  See `eval` module -> `operator_compute()` for details.
                    bval (torch.Tensor): The predicted values at the boundaries of the domain.
                    true_bval (torch.Tensor): The true (target) values at the boundaries of the domain.
                    lambda_op (torch.Tensor): Regularization parameter to weight the importance of satisfying the differential equation.
                    lambda_bound (torch.Tensor): Regularization parameter to weight the importance of satisfying the boundary conditions.
        
                Returns:
                    loss (torch.Tensor): The total loss, combining the equation and boundary condition losses.
                    loss_normalized (torch.Tensor): The total loss with regularization parameters set to 1, providing a baseline loss value.
        
                Why: This function calculates the loss, guiding the optimization process to find a solution that minimizes the error in both satisfying the differential equation and adhering to the specified boundary conditions. The regularization parameters allow weighting the relative importance of these two aspects.
        """

        if bval is None:
            return sum(operator)

        loss_oper, op = self._loss_op(operator, lambda_op)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1)
        lambda_bound_normalized = lambda_prepare(bval, 1)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T +\
                        bval_diff @ lambda_bound_normalized.T

        return loss, loss_normalized

    def compute(self,
                operator: torch.Tensor,
                bval: torch.Tensor,
                true_bval: torch.Tensor,
                lambda_op: torch.Tensor,
                lambda_bound: torch.Tensor,
                save_graph: bool = True) -> Union[_default_loss, _weak_loss, _causal_loss]:
        """
        Selects and applies the appropriate loss calculation method based on the specified mode and tolerances.
        
        This method acts as a dispatcher, choosing between different loss calculation strategies
        depending on whether a weak form is specified or a tolerance level is set. This allows the framework
        to adapt the loss calculation to different problem settings and solution requirements.
        
        Args:
            operator (torch.Tensor): The result of the operator calculation. See `eval module -> operator_compute()` for details.
            bval (torch.Tensor): Calculated values of boundary conditions.
            true_bval (torch.Tensor): True values of boundary conditions.
            lambda_op (torch.Tensor): Regularization parameter for the operator term in the loss.
            lambda_bound (torch.Tensor): Regularization parameter for the boundary term in the loss.
            save_graph (bool, optional): Whether to save the computational graph. Defaults to True.
        
        Returns:
            Union[_default_loss, _weak_loss, _causal_loss]: The calculated loss based on the chosen method.
        
        Why:
            This method centralizes the selection of the appropriate loss calculation strategy,
            allowing the framework to handle different problem formulations (e.g., with or without
            weak forms, with different tolerance requirements) in a modular and adaptable way.
        """

        if self.mode in ('mat', 'autograd'):
            if bval is None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf
        inputs = [operator, bval, true_bval, lambda_op, lambda_bound]

        if self.weak_form is not None and self.weak_form != []:
            return self._weak_loss(*inputs)
        elif self.tol != 0:
            return self._causal_loss(*inputs)
        else:
            return self._default_loss(*inputs, save_graph)
