from typing import Tuple
import torch
from copy import copy
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from epde.solver.device import device_type


class PSO(torch.optim.Optimizer):
    """
    Custom PSO optimizer.
    """

    """Custom PSO optimizer.
    """

    def __init__(self,
                 params,
                 pop_size: int = 30,
                 b: float = 0.9,
                 c1: float = 8e-2,
                 c2: float = 5e-1,
                 lr: float = 1e-3,
                 betas: Tuple = (0.99, 0.999),
                 c_decrease: bool = False,
                 variance: float = 1,
                 epsilon: float = 1e-8,
                 n_iter: int = 2000):
        """
        Initializes the Particle Swarm Optimizer (PSO).
        
        This method sets up the PSO algorithm with specified hyperparameters,
        preparing it to explore the parameter space of a given model to find optimal configurations.
        The initialization includes setting the swarm size, inertia, cognitive and social coefficients,
        learning rate for gradient descent, and other parameters that control the optimization process.
        The swarm is initialized based on the model parameters, and initial velocities are assigned to each particle.
        The method prepares the optimizer to iteratively update the swarm's positions and velocities
        to minimize a given loss function, effectively searching for the best-fitting differential equation.
        
        Args:
            params (iterable): Iterable of parameters to optimize (typically model parameters).
            pop_size (int, optional): Population size of the PSO swarm. Defaults to 30.
            b (float, optional): Inertia of the particles, controlling their tendency to continue in their current direction. Defaults to 0.9.
            c1 (float, optional): The *p-best* coefficient, influencing the particle's attraction to its best historical position. Defaults to 0.08.
            c2 (float, optional): The *g-best* coefficient, influencing the particle's attraction to the swarm's best historical position. Defaults to 0.5.
            lr (float, optional): Learning rate for gradient descent, used to refine particle positions based on gradient information. Defaults to 1e-3.
            betas (tuple(float, float), optional): Coefficients used for computing running averages of gradient and its square (similar to Adam). Defaults to (0.99, 0.999).
            c_decrease (bool, optional): Flag indicating whether to decrease c1 and c2 over iterations. Defaults to False.
            variance (float, optional): Variance parameter for initializing the swarm based on the model parameters. Defaults to 1.
            epsilon (float, optional): Small value added to the denominator for numerical stability in gradient-based updates. Defaults to 1e-8.
            n_iter (int, optional): Number of iterations. Defaults to 2000.
        
        Returns:
            None
        """
        defaults = {'pop_size': pop_size,
                    'b': b, 'c1': c1, 'c2': c2,
                    'lr': lr, 'betas': betas,
                    'c_decrease': c_decrease,
                    'variance': variance,
                    'epsilon': epsilon}
        super(PSO, self).__init__(params, defaults)
        self.params = self.param_groups[0]['params']
        self.pop_size = pop_size
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.c_decrease = c_decrease
        self.epsilon = epsilon
        self.beta1, self.beta2 = betas
        self.lr = lr * np.sqrt(1 - self.beta2) / (1 - self.beta1)
        self.use_grad = True if self.lr != 0 else False
        self.variance = variance
        self.name = "PSO"
        self.n_iter = n_iter

        vec_shape = self.params_to_vec().shape
        self.vec_shape = list(vec_shape)[0]

        self.swarm = self.build_swarm()

        self.p = copy(self.swarm).detach()

        self.v = self.start_velocities()
        self.m1 = torch.zeros(self.pop_size, self.vec_shape)
        self.m2 = torch.zeros(self.pop_size, self.vec_shape)

        self.indicator = True

    def params_to_vec(self) -> torch.Tensor:
        """
        Converts the model's parameters or values into a single vector.
        
        This is a utility function to represent the model's state in a flattened format,
        which is useful for optimization algorithms that require vector-based inputs.
        
        Args:
            None
        
        Returns:
            torch.Tensor: A 1D tensor containing the model's parameters or values.
        
        Why:
            This method facilitates the application of vector-based optimization techniques
            by providing a unified representation of the model's parameters or values.
        """
        if not isinstance(self.params, torch.Tensor):
            vec = parameters_to_vector(self.params)
        else:
            self.model_shape = self.params.shape
            vec = self.params.reshape(-1)

        return vec

    def vec_to_params(self, vec: torch.Tensor) -> None:
        """
        Distributes a vector's values into the model's parameters.
        
        This method takes a vector, typically representing a particle in the optimization process,
        and maps its values onto the parameters of the model. This is a crucial step in evaluating
        the fitness of a candidate solution (particle) by updating the model with the particle's
        parameter values before assessing its performance.
        
        Args:
            vec (torch.Tensor): A tensor containing the parameter values for the model.
        
        Returns:
            None
        """
        if not isinstance(self.params, torch.Tensor):
            vector_to_parameters(vec, self.params)
        else:
            self.params.data = vec.reshape(self.params).data

    def build_swarm(self):
        """
        Initializes the swarm population by perturbing a base solution.
        
        The swarm is created by adding random variations to a vectorized
        representation of the initial solution. This ensures diversity in the
        search space, allowing the algorithm to explore different potential
        equation structures. The first particle in the swarm is set to the
        original solution to preserve a baseline.
        
        Args:
            None
        
        Returns:
            torch.Tensor: The initialized swarm population. Each row represents a
            particle (potential equation), and each column represents a parameter
            within the equation. The tensor is detached from the computation graph
            and requires gradients for optimization.
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

    def update_pso_params(self) -> None:
        """
        Updates the cognitive and social parameters (c1, c2) of the PSO algorithm.
        
        This adjustment refines the balance between individual exploration and social influence
        as the optimization progresses, potentially leading to improved convergence.
        
        Args:
            None
        
        Returns:
            None
        """
        self.c1 -= 2 * self.c1 / self.n_iter
        self.c2 += self.c2 / self.n_iter

    def start_velocities(self) -> torch.Tensor:
        """
        Initializes particle velocities to zero.
        
                This ensures a neutral starting point for the swarm's exploration
                of the search space, preventing any initial bias towards specific regions.
        
                Returns:
                    torch.Tensor: A tensor of zeros representing the initial velocities
                    for each particle in the swarm. The shape is (population size, vector shape).
        """
        return torch.zeros((self.pop_size, self.vec_shape))

    def gradient(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the loss with respect to the model parameters.
        
        This gradient is used to guide the optimization process, indicating the direction
        in which to adjust the parameters to reduce the loss. The calculation leverages
        automatic differentiation to efficiently compute the derivatives.
        
        Args:
            loss (torch.Tensor): The scalar loss value to differentiate.
        
        Returns:
            torch.Tensor: A flattened vector representing the gradient of the loss with
                respect to all model parameters.
        """
        dl_dparam = torch.autograd.grad(loss, self.params)

        grads = parameters_to_vector(dl_dparam)

        return grads

    def get_randoms(self) -> torch.Tensor:
        """
        Generate random values for exploration during the search process.
        
        Args:
            None
        
        Returns:
            torch.Tensor: A tensor of random values used to introduce diversity and explore the search space when updating particle positions. The shape of the tensor is (2, 1, self.vec_shape).
        """
        return torch.rand((2, 1, self.vec_shape))

    def update_p_best(self) -> None:
        """
        Updates the personal best positions of particles in the swarm.
        
        This method compares the current loss of each particle with its personal best loss so far. If a particle has found a new position with a lower loss, its personal best position and corresponding loss are updated. This ensures that each particle remembers its best-performing location, which is crucial for guiding the swarm's search for the global optimum.
        
        Args:
            None
        
        Returns:
            None
        """

        idx = torch.where(self.loss_swarm < self.f_p)

        self.p[idx] = self.swarm[idx]
        self.f_p[idx] = self.loss_swarm[idx].detach()

    def update_g_best(self) -> None:
        """
        Updates the global best position (*g-best*) of the particle swarm.
        
        The *g-best* represents the best solution found by any particle in the swarm so far. This method identifies the particle with the lowest objective function value among all particles and sets its position as the new *g-best*. This ensures that the swarm converges towards the most promising region of the search space, effectively guiding the equation discovery process towards better-fitting models.
        
        Args:
            None
        
        Returns:
            None
        """
        self.g_best = self.p[torch.argmin(self.f_p)]

    def gradient_descent(self) -> torch.Tensor:
        """
        Updates velocities of particles based on calculated gradients using Adam algorithm.
        
        The method refines the search trajectory of particles by incorporating gradient information
        into their velocities, facilitating efficient exploration of the solution space.
        
        Args:
            None
        
        Returns:
            torch.Tensor: Updated velocities for each particle, representing a refined search direction.
        """
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * self.grads_swarm
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * torch.square(
            self.grads_swarm)

        update = self.lr * self.m1 / (torch.sqrt(torch.abs(self.m2)) + self.epsilon)

        return update

    def step(self, closure=None) -> torch.Tensor:
        """
        Runs a single iteration of the Particle Swarm Optimization algorithm to refine the search for the optimal equation. It evaluates the swarm's fitness, updates particle velocities and positions based on personal and global best solutions, and adjusts PSO parameters if specified. The method ensures the swarm explores the search space effectively by adapting its movement based on individual and collective experiences, ultimately aiming to identify the equation that best describes the underlying data.
        
                Args:
                    closure (callable, optional): A function that evaluates the loss and gradients for each particle in the swarm. Defaults to None.
        
                Returns:
                    torch.Tensor: The minimum loss value achieved by the best particle in the swarm after the iteration.
        """

        self.loss_swarm, self.grads_swarm = closure()

        fix_attempt=0

        while torch.any(self.loss_swarm!=self.loss_swarm):
            self.swarm=self.swarm+0.001*torch.rand(size=self.swarm.shape)
            self.loss_swarm, self.grads_swarm = closure()
            fix_attempt+=1
            if fix_attempt>5:
                break

        if self.indicator:
            self.f_p = copy(self.loss_swarm).detach()
            self.g_best = self.p[torch.argmin(self.f_p)]
            self.indicator = False

        r1, r2 = self.get_randoms()

        self.v = self.b * self.v + (1 - self.b) * (
                self.c1 * r1 * (self.p - self.swarm) + self.c2 * r2 * (self.g_best - self.swarm))
        if self.use_grad:
            self.swarm = self.swarm + self.v - self.gradient_descent()
        else:
            self.swarm = self.swarm + self.v
        self.update_p_best()
        self.update_g_best()
        self.vec_to_params(self.g_best)
        if self.c_decrease:
            self.update_pso_params()
        min_loss = torch.min(self.f_p)

        return min_loss
