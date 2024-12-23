import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from epde.solver.utils import replace_none_by_zero
from epde.solver.device import check_device


class NGD(torch.optim.Optimizer):

    """NGD implementation (https://arxiv.org/abs/2302.13163).
    """

    def __init__(self, params,
                 grid_steps_number: int = 30):
        """The Natural Gradient Descent class.

        Args:
            grid_steps_number (int, optional): Grid steps number. Defaults to 30.
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
        """ Update models paramters by natural gradient.

        Args:
            loss (callable): function to calculate loss.

        Returns:
            None.
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
        """ Make Gram matrice.

        Args:
            residuals (callable): PDE residual.

        Returns:
            torch.Tensor: Gram matrice.
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
        """ Make Gram matrice.

        Args:
            residuals (callable): PDE residual.

        Returns:
            torch.Tensor: Gram matrice.
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
        """ Find lstsq (least-squares solution) for torch.tensor cuda.

        Args:
            A (torch.Tensor): lhs tensor of shape (*, m, n) where * is zero or more batch dimensions.
            B (torch.Tensor): rhs tensor of shape (*, m, k) where * is zero or more batch dimensions.
            tol (float):  used to determine the effective rank of A. By default set to the machine precision of the dtype of A.

        Returns:
            torch.Tensor: solution for A and B.
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

        A = A.detach().cpu().numpy()
        B = B.detach().cpu().numpy()

        f_nat_grad = np.linalg.lstsq(A, B,rcond=rcond)[0] 

        f_nat_grad=torch.from_numpy(f_nat_grad)

        f_nat_grad = check_device(f_nat_grad)

        return f_nat_grad


    def step(self, closure=None) -> torch.Tensor:
        """ It runs ONE step on the natural gradient descent.

        Returns:
            torch.Tensor: loss value for NGD step.
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