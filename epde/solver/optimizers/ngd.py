import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from epde.solver.utils import replace_none_by_zero
from epde.solver.device import check_device


class NGD(torch.optim.Optimizer):
    """
    NGD implementation (https://arxiv.org/abs/2302.13163).
    """

    """NGD implementation (https://arxiv.org/abs/2302.13163).
    """

    def __init__(self, params,
                 grid_steps_number: int = 30):
        """
        Initializes the Natural Gradient Descent optimizer.
        
        This class sets up the optimizer with necessary parameters,
        including the grid resolution for exploration of potential step sizes.
        The grid is constructed in a log scale.
        
        Args:
            params (iterable): Iterable of parameters to optimize.
            grid_steps_number (int, optional): Number of steps in the grid for step size selection.
                A finer grid allows for a more precise step size selection. Defaults to 30.
        """
        defaults = {'grid_steps_number': grid_steps_number}
        super(NGD, self).__init__(params, defaults)
        self.params = self.param_groups[0]['params']
        self.grid_steps_number = grid_steps_number
        self.grid_steps = torch.linspace(0, self.grid_steps_number, self.grid_steps_number + 1)
        self.steps = 0.5**self.grid_steps
        self.cuda_out_of_memory_flag=False
        self.cuda_empty_once_for_test=True

    def grid_line_search_update(self, loss_function: callable, f_nat_grad: torch.Tensor) -> None:
        """
        Update model parameters using a grid line search along the natural gradient direction to minimize the loss.
        
                This method explores different step sizes along the natural gradient to find the one that results in the lowest loss. This ensures a more stable and efficient update compared to using a fixed step size, which is crucial for discovering accurate differential equation models.
        
                Args:
                    loss_function (callable): A callable that computes the loss value. It should return a tuple containing the loss tensor and any auxiliary information.
                    f_nat_grad (torch.Tensor): The natural gradient, a tensor representing the direction of steepest descent in the parameter space, preconditioned by the Fisher information matrix.
        
                Returns:
                    None. The model parameters are updated in place.
        """
        # function to update models paramters at each step
        def loss_at_step(step, loss_function: callable, f_nat_grad: torch.Tensor) -> torch.Tensor:
            params = parameters_to_vector(self.params)
            new_params = params - step * f_nat_grad
            vector_to_parameters(new_params, self.params)
            loss_val, _ = loss_function()
            vector_to_parameters(params, self.params)
            return loss_val

        losses = []
        for step in self.steps:
            losses.append(loss_at_step(step, loss_function, f_nat_grad).reshape(1))
        losses = torch.cat(losses)
        step_size = self.steps[torch.argmin(losses)]

        params = parameters_to_vector(self.params)
        new_params = params - step_size * f_nat_grad
        vector_to_parameters(new_params, self.params)
    
    def gram_factory(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Computes the Gram matrix of the Jacobian of the residuals with respect to the model parameters.
        
        This matrix is used to estimate the inverse of the Fisher Information Matrix,
        which is crucial for preconditioning the gradient during optimization.
        The preconditioning enhances the convergence and stability of the training process.
        
        Args:
            residuals (torch.Tensor): The PDE residuals evaluated at different points.
        
        Returns:
            torch.Tensor: The Gram matrix, a symmetric positive semi-definite matrix.
        """
        # Make Gram matrice.
        def jacobian() -> torch.Tensor:
            jac = []
            for l in residuals:
                j = torch.autograd.grad(l, self.params, retain_graph=True, allow_unused=True)
                j = replace_none_by_zero(j)
                j = parameters_to_vector(j).reshape(1, -1)
                jac.append(j)
            return torch.cat(jac)

        J = jacobian()
        return 1.0 / len(residuals) * J.T @ J


    def gram_factory_cpu(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Computes the Gram matrix of the Jacobian of the residuals with respect to the model parameters.
        
        This matrix is used to quantify the linear dependencies between the gradients of the residuals,
        providing insights into the sensitivity of the model to parameter changes and the identifiability
        of the parameters themselves. It is calculated on the CPU.
        
        Args:
            residuals (torch.Tensor): A tensor containing the PDE residuals.
        
        Returns:
            torch.Tensor: The Gram matrix, a measure of the linear dependence between parameter gradients.
        """
        # Make Gram matrice.
        def jacobian() -> torch.Tensor:
            jac = []
            for l in residuals:
                j = torch.autograd.grad(l, self.params, retain_graph=True, allow_unused=True)
                j = replace_none_by_zero(j)
                j = parameters_to_vector(j).reshape(1, -1)
                jac.append(j)
            return torch.cat(jac)

        J = jacobian().cpu()
        return 1.0 / len(residuals) * J.T @ J


    
    def torch_cuda_lstsq(self, A: torch.Tensor, B: torch.Tensor, tol: float = None) -> torch.Tensor:
        """
        Finds the least-squares solution for a system of linear equations represented by CUDA tensors. This is a crucial step in identifying the underlying differential equations from data by minimizing the error between the model's predictions and the observed data.
        
                Args:
                    A (torch.Tensor): The left-hand side tensor of shape (*, m, n), where * represents zero or more batch dimensions.
                    B (torch.Tensor): The right-hand side tensor of shape (*, m, k), where * represents zero or more batch dimensions.
                    tol (float, optional): Tolerance value used to determine the effective rank of A. Defaults to the machine precision of the dtype of A.
        
                Returns:
                    torch.Tensor: The least-squares solution for A and B.
        """
        tol = torch.finfo(A.dtype).eps if tol is None else tol
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        Spinv = torch.zeros_like(S)
        Spinv[S>tol] = 1/S[S>tol]
        UhB = U.adjoint() @ B
        if Spinv.ndim!=UhB.ndim:
            Spinv = Spinv.unsqueeze(-1)
        SpinvUhB = Spinv * UhB
        return Vh.adjoint() @ SpinvUhB



    def numpy_lstsq(self, A: torch.Tensor, B: torch.Tensor, rcond: float = None) -> torch.Tensor:
        """
        Solves a linear least squares problem using NumPy to estimate natural gradients.
        
                This method leverages NumPy's least squares solver to efficiently compute
                the solution to a linear system, which is then used to approximate the
                natural gradient. By solving this system, the method aims to find the
                optimal update direction for parameters, facilitating efficient learning.
                The input tensors `A` and `B` are detached from the computation graph
                and moved to the CPU as NumPy arrays for compatibility with the solver.
                The solution is then converted back to a PyTorch tensor and placed on
                the appropriate device.
        
                Args:
                    A: The "coefficient" matrix (left-hand side of the equation).
                    B: The "dependent variable" matrix (right-hand side of the equation).
                    rcond:  Cutoff ratio for small singular values of a.
                        For the purposes of rank determination, singular values are treated
                        as zero if they are smaller than rcond times the largest singular
                        value of a.
        
                Returns:
                    torch.Tensor: The least squares solution, as a PyTorch tensor on the
                        correct device.
        """

        A = A.detach().cpu().numpy()
        B = B.detach().cpu().numpy()

        f_nat_grad = np.linalg.lstsq(A, B,rcond=rcond)[0] 

        f_nat_grad=torch.from_numpy(f_nat_grad)

        f_nat_grad = check_device(f_nat_grad)

        return f_nat_grad


    def step(self, closure=None) -> torch.Tensor:
        """
        Performs a single natural gradient descent update step, adjusting model parameters based on the computed natural gradient. This involves assembling the Gramian matrix, solving a least squares problem to obtain the natural gradient, and updating the parameters using a line search to ensure stable convergence.
        
                Args:
                    closure (callable, optional): A closure that reevaluates the model and returns a tuple containing:
                        - int_res (torch.Tensor): Interior residual values.
                        - bval (torch.Tensor): Boundary values.
                        - true_bval (torch.Tensor): True boundary values.
                        - loss (torch.Tensor): The loss value.
                        - loss_function (callable): The loss function itself.
        
                Returns:
                    torch.Tensor: The loss value after the NGD step.
        """

        int_res, bval, true_bval, loss, loss_function = closure()
        grads = torch.autograd.grad(loss, self.params, retain_graph=True, allow_unused=True)
        grads = replace_none_by_zero(grads)
        f_grads = parameters_to_vector(grads)

        bound_res = bval-true_bval

        ## assemble gramian
        #G_int  = self.gram_factory(int_res.reshape(-1))
        #G_bdry = self.gram_factory(bound_res.reshape(-1))
        #G      = G_int + G_bdry

        ## Marquardt-Levenberg
        #Id = torch.eye(len(G))
        #G = torch.min(torch.tensor([loss, 0.0])) * Id + G

        

        # compute natural gradient
        if not self.cuda_out_of_memory_flag:
            try:
                if self.cuda_empty_once_for_test:
                    #print('Initial GPU check')
                    torch.cuda.empty_cache()
                    self.cuda_empty_once_for_test=False
                
                # assemble gramian

                #print('NGD GPU step')

                G_int  = self.gram_factory(int_res.reshape(-1))
                G_bdry = self.gram_factory(bound_res.reshape(-1))
                G      = G_int + G_bdry

                # Marquardt-Levenberg
                Id = torch.eye(len(G))
                G = torch.min(torch.tensor([loss, 0.0])) * Id + G

                f_nat_grad = self.torch_cuda_lstsq(G, f_grads)   
            except torch.OutOfMemoryError:
                print('[Warning] Least square returned CUDA out of memory error, CPU and RAM are used, which is significantly slower')
                self.cuda_out_of_memory_flag=True

                G_int  = self.gram_factory_cpu(int_res.reshape(-1).cpu())
                G_bdry = self.gram_factory_cpu(bound_res.reshape(-1).cpu())
                G      = G_int + G_bdry


                f_nat_grad = self.numpy_lstsq(G, f_grads)
        else:


            #print('NGD CPU step')

            G_int  = self.gram_factory_cpu(int_res.reshape(-1).cpu())
            G_bdry = self.gram_factory_cpu(bound_res.reshape(-1).cpu())
            G      = G_int + G_bdry

            f_nat_grad = self.numpy_lstsq(G, f_grads)

        # one step of NGD
        self.grid_line_search_update(loss_function, f_nat_grad)
        self.param_groups[0]['params'] = self.params

        return loss