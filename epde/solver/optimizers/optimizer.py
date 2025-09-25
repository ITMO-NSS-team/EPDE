import torch
from abc import ABC
from typing import Union, Any
from epde.solver.optimizers.pso import PSO
from epde.solver.optimizers.ngd import NGD
from torch.optim.lr_scheduler import ExponentialLR


class Optimizer():
    """
    A base class for defining optimization algorithms.
    
        This class serves as a foundation for implementing various optimization techniques.
    
        Class Methods:
        - __init__: Initializes the optimizer with specified parameters.
        - step: Performs a single optimization step.
        - zero_grad: Resets the gradients of the optimized parameters.
    """

    def __init__(
            self,
            optimizer: str,
            params: dict,
            gamma: Union[float, None]=None,
            decay_every: Union[int, None]=None):
        """
        Initializes the optimizer configuration for equation discovery.
        
                Configures the optimizer type and its parameters, including optional learning rate decay settings,
                to fine-tune the search process for identifying the best equation structure.
        
                Args:
                    optimizer: The name of the optimizer to use (e.g., 'Adam', 'SGD').
                    params: A dictionary containing the optimizer's parameters (e.g., learning rate, momentum).
                    gamma: The learning rate decay factor (optional).
                    decay_every: The frequency (in steps) at which to decay the learning rate (optional).
        
                Returns:
                    None.
        
                Why:
                    This configuration is crucial for effectively training the equation discovery process.
                    By allowing flexible optimizer choices and decay schedules, the search for the optimal
                    equation structure can be more precisely controlled and potentially accelerated.
        """
        self.optimizer = optimizer
        self.params = params
        self.gamma = gamma
        self.decay_every = decay_every

    def optimizer_choice(
        self,
        mode,
        model) -> \
            Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS, PSO]:
        """
        Selects and configures the optimization algorithm for training the model.
        
                This method allows the user to choose from several optimization algorithms,
                including Adam, SGD, LBFGS, PSO, and NGD, and configures them based on
                user-specified parameters. The choice of optimizer and its configuration
                are crucial for effectively training the model to accurately represent
                the underlying differential equation.
        
                Args:
                    mode (str): Specifies the mode of operation ('NN', 'autograd', or 'mat'),
                        which determines how the model parameters are passed to the optimizer.
                    model (torch.nn.Module): The model whose parameters need to be optimized.
        
                Returns:
                    Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS, PSO]:
                        The configured optimizer instance.
        """

        if self.optimizer == 'Adam':
            torch_optim = torch.optim.Adam
        elif self.optimizer == 'SGD':
            torch_optim = torch.optim.SGD
        elif self.optimizer == 'LBFGS':
            torch_optim = torch.optim.LBFGS
        elif self.optimizer == 'PSO':
            torch_optim = PSO
        elif self.optimizer == 'NGD':
            torch_optim = NGD


        if mode in ('NN', 'autograd'):
            optimizer = torch_optim(model.parameters(), **self.params)
        elif mode == 'mat':
            optimizer = torch_optim([model.requires_grad_()], **self.params)
        
        if self.gamma is not None:
            self.scheduler = ExponentialLR(optimizer, gamma=self.gamma)

        return optimizer