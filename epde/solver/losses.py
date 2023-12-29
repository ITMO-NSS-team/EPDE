from typing import Tuple, Union

from epde.solver.input_preprocessing import lambda_prepare
from epde.solver.utils import *

class Losses():
    """
    Class which contains all losses.
    """
    def __init__(self,
                 mode: str,
                 weak_form: Union[None, torch.Tensor],
                 n_t: int,
                 tol):
        self.mode = mode
        self.weak_form = weak_form
        self.n_t = n_t
        self.tol = tol
        # TODO: refactor loss_op, loss_bcs into one function, carefully figure out when bval is None + fix causal_loss operator crutch (line 76).

    def loss_op(self, operator, lambda_op) -> torch.Tensor:
        if self.weak_form != None and self.weak_form != []:
            op = operator
        else:
            op = torch.mean(operator**2, 0)
        
        loss_operator = op @ lambda_op.T 
        return loss_operator, op


    def loss_bcs(self, bval, true_bval, lambda_bound) -> torch.Tensor:
        """
        Computes boundary loss for corresponding type.

        Returns:
            boundary loss
        """
        bval_diff = torch.mean((bval - true_bval)**2, 0)

        loss_bnd = bval_diff @ lambda_bound.T
        return loss_bnd, bval_diff


    def default_loss(self, operator, bval, true_bval, lambda_op, lambda_bound, save_graph=True) \
                                        -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes l2 loss.

        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
        Returns:
            model loss.
        """

        if bval == None:
            return torch.sum(torch.mean((operator) ** 2, 0))

        loss_oper, op = self.loss_op(operator, lambda_op)

        loss_bnd, bval_diff = self.loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1)
        lambda_bound_normalized = lambda_prepare(bval, 1)

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

    def causal_loss(self, operator, bval, true_bval, lambda_op, lambda_bound) \
                                                            -> torch.Tensor:
        """
        Computes causal loss, which is calculated with weights matrix:
        W = exp(-tol*(Loss_i)) where Loss_i is sum of the L2 loss from 0
        to t_i moment of time. This loss function should be used when one
        of the DE independent parameter is time.

        Args:
            lambda_bound: an arbitrary chosen constant, influence only convergence speed.
            tol: float constant, influences on error penalty.
        Returns:
            model loss.
        """

        res = torch.sum(operator**2, dim=1).reshape(self.n_t, -1)
        res = torch.mean(res, axis=1).reshape(self.n_t, 1)
        M = torch.triu(torch.ones((self.n_t, self.n_t), dtype=res.dtype), diagonal=1).T
        with torch.no_grad():
            W = torch.exp(- self.tol * (M @ res))

        loss_oper = torch.mean(W * res)

        loss_bnd, bval_diff = self.loss_bcs(bval, true_bval, lambda_bound)

        loss = loss_oper + loss_bnd

        lambda_bound_normalized = lambda_prepare(bval, 1)
        with torch.no_grad():
            loss_normalized = loss_oper +\
                        lambda_bound_normalized @ bval_diff

        return loss, loss_normalized

    def weak_loss(self, operator, bval, true_bval, lambda_op, lambda_bound) \
                                                            -> torch.Tensor:
        """
        Weak solution of O/PDE problem.

        Args:
            weak_form: list of basis functions.
            lambda_bound: const regularization parameter.
        Returns:
            model loss.
        """
        if bval == None:
            return sum(operator)

        loss_oper, op = self.loss_op(operator, lambda_op)

        loss_bnd, bval_diff = self.loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1)
        lambda_bound_normalized = lambda_prepare(bval, 1)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T +\
                        bval_diff @ lambda_bound_normalized.T

        return loss, loss_normalized

    def compute(self, operator, bval, true_bval, lambda_op, lambda_bound, save_graph=True) -> \
                                Union[default_loss, weak_loss, causal_loss]:
            """
            Setting the required loss calculation method.

            Args:
                tol: float constant, influences on error penalty.
            Returns:
                A given calculation method.
            """
            if self.mode == 'mat' or self.mode == 'autograd':
                if bval == None:
                    print('No bconds is not possible, returning infinite loss')
                    return np.inf
            inputs = [operator, bval, true_bval, lambda_op, lambda_bound]

            if self.weak_form != None and self.weak_form != []:
                return self.weak_loss(*inputs)
            elif self.tol != 0:
                return self.causal_loss(*inputs)
            else:
                return self.default_loss(*inputs, save_graph)