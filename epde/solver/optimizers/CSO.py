from typing import Tuple
import torch
from copy import copy
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from epde.solver.device import device_type


class CSO(torch.optim.Optimizer):

    """Custom CSO optimizer.
    """

    def __init__(self,
                 params,
                 pop_size: int = 20,
                 fi: float = 0.0,
                 lr: float = 1e-3,
                 betas: Tuple = (0.99, 0.999),
                 c_decrease: bool = False,
                 variance: float = 1,
                 epsilon: float = 1e-8,
                 n_iter: int = 2000):
        """The Competitive Swarm Optimizer class.

        Args:
            pop_size (int, optional): Population of the CSO swarm. Defaults to 20.
            fi (float, optional): Parameter that controls the influence of mean position value of the
                relevant particles. Defaults to 0 for pop_size < 100.
            lr (float, optional): Learning rate for gradient descent. Defaults to 0.00,
                so there will not be any gradient-based optimization.
            betas (tuple(float, float), optional): same coeff in Adam algorithm. Defaults to (0.99, 0.999).
            c_decrease (bool, optional): Flag for update_pso_params method. Defautls to False.
            variance (float, optional): Variance parameter for swarm creation
                based on model. Defaults to 1.
            epsilon (float, optional): some add to gradient descent like in Adam optimizer.
                Defaults to 1e-8.
        """
        defaults = {'pop_size': pop_size,
                    'fi': fi, 
                    'lr': lr, 'betas': betas,
                    'c_decrease': c_decrease,
                    'variance': variance,
                    'epsilon': epsilon}
        super(CSO, self).__init__(params, defaults)
        self.params = self.param_groups[0]['params']
        self.pop_size = pop_size
        self.fi = fi
        self.epsilon = epsilon
        self.beta1, self.beta2 = betas
        self.lr = lr * np.sqrt(1 - self.beta2) / (1 - self.beta1)
        self.use_grad = True if self.lr != 0 else False
        self.variance = variance
        self.name = "СSO"
        self.n_iter = n_iter
        self.t = 0
        self.V_max = 1.0

        vec_shape = self.params_to_vec().shape
        self.vec_shape = list(vec_shape)[0]

        self.swarm = self.build_swarm()

        self.p = copy(self.swarm).detach()

        self.v = self.start_velocities()
        self.m1 = torch.zeros(self.pop_size, self.vec_shape)
        self.m2 = torch.zeros(self.pop_size, self.vec_shape)

        self.indicator = True

    def params_to_vec(self) -> torch.Tensor:
        """ Method for converting model parameters *NN and autograd*
           or model values *mat* to vector.

        Returns:
            torch.Tensor: model parameters/model values vector.
        """
        if not isinstance(self.params, torch.Tensor):
            vec = parameters_to_vector(self.params)
        else:
            self.model_shape = self.params.shape
            vec = self.params.reshape(-1)

        return vec

    def vec_to_params(self, vec: torch.Tensor) -> None:
        """Method for converting vector to model parameters (NN, autograd)
           or model values (mat)

        Args:
            vec (torch.Tensor): The particle of swarm. 
        """
        if not isinstance(self.params, torch.Tensor):
            vector_to_parameters(vec, self.params)
        else:
            self.params.data = vec.reshape(self.params).data


    def build_swarm(self):
        """Creates the swarm based on solution class model.

        Returns:
            torch.Tensor: The PSO swarm population.
            Each particle represents a neural network (NN, autograd) or model values (mat).
        """
        vector = self.params_to_vec()
        matrix = []
        for _ in range(self.pop_size):
            matrix.append(vector.reshape(1, -1))
        matrix = torch.cat(matrix)
        variance = torch.FloatTensor(self.pop_size, self.vec_shape).uniform_(
            -self.variance, self.variance).to(device_type())
        swarm = matrix + variance
        swarm[0] = matrix[0]
        return swarm.clone().detach().requires_grad_(True)

    def start_velocities(self) -> torch.Tensor:
        """Start the velocities of each particle in the population (swarm) as `0`.

        Returns:
            torch.Tensor: The starting velocities.
        """
        return torch.zeros((self.pop_size, self.vec_shape))

    def gradient(self, loss: torch.Tensor) -> torch.Tensor:
        """ Calculation of loss gradient by model parameters (NN, autograd)
            or model values (mat).

        Args:
            loss (torch.Tensor): result of loss calculation.

        Returns:
            torch.Tensor: calculated gradient vector.
        """
        dl_dparam = torch.autograd.grad(loss, self.params)

        grads = parameters_to_vector(dl_dparam)

        return grads
    
    def update_zero_v(self) -> Tuple[torch.Tensor, torch.Tensor]:

        idx = torch.where(abs(self.v) == 0)
        if len(idx[0]) > 0: 
            rand = torch.rand_like(self.v[idx])  # Генерируем случайные значения той же формы, что и self.v
            self.v[idx] = torch.where(rand > 0.5, rand * self.variance * (-self.V_max), rand * self.variance * self.V_max)

    def get_randoms(self) -> torch.Tensor:
        """Generate random values to update the particles' positions.

        Returns:
            torch.Tensor: random tensor
        """
        return torch.rand((2, 1, self.vec_shape))

    def update_p_best(self) -> None:
        """Updates the *p-best* positions."""

        idx = torch.where(self.loss_swarm < self.f_p)

        self.p[idx] = self.swarm[idx]
        self.f_p[idx] = self.loss_swarm[idx].detach()

    def update_g_best(self) -> None:
        """Update the *g-best* position."""
        self.g_best = self.p[torch.argmin(self.f_p)]

    def gradient_descent(self) -> torch.Tensor:
        """ Gradiend descent based on Adam algorithm.

        Returns:
            torch.Tensor: gradient term in velocities vector.
        """
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * self.grads_swarm
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * torch.square(
            self.grads_swarm)

        update = self.lr * self.m1 / (torch.sqrt(torch.abs(self.m2)) + self.epsilon)

        return update

    def step(self, closure=None) -> torch.Tensor:
        """ It runs ONE step on the competitive swarm optimization.

        Returns:
            torch.Tensor: loss value for best particle of the swarm.
        """  
        self.loss_swarm, self.grads_swarm = closure()
        if self.indicator:
            self.f_p = copy(self.loss_swarm).detach()
            self.g_best = self.p[torch.argmin(self.f_p)]
            self.indicator = False

        fix_attempt=0
        while torch.any(self.loss_swarm!=self.loss_swarm):
            self.swarm=self.swarm+0.001*torch.rand(size=self.swarm.shape)
            self.loss_swarm, self.grads_swarm = closure()
            fix_attempt+=1
            if fix_attempt>5:
                break

        U = list(range(self.pop_size))
        while U:
            i, j = np.random.choice(U, 2, replace=False)
            X1, X2 = self.swarm[i], self.swarm[j]
            
            if self.loss_swarm[i] <= self.loss_swarm[j]:
                Xw, Xl = X1, X2
                iw, il = i, j
            else:
                Xw, Xl = X2, X1
                iw, il = j, i

            R1, R2, R3 = torch.rand(self.vec_shape), torch.rand(self.vec_shape), torch.rand(self.vec_shape)
            self.v[il] = R1 * self.v[il] + R2 * (Xw - Xl) + self.fi * R3 * ((Xl + Xw)/2 - Xl)
                
            with torch.no_grad():
                self.swarm[il] = Xl + self.v[il]
            U.remove(i)
            U.remove(j)

        if self.use_grad: 
            self.swarm = self.swarm -  self.gradient_descent()

        self.update_p_best()
        self.update_g_best()

        if self.t > 150:
            self.update_zero_v()
        self.vec_to_params(self.g_best)
        min_loss =  torch.min(self.f_p)
        self.t += 1
        return min_loss
