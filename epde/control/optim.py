from abc import ABC
from collections import OrderedDict
from typing import List, Dict

from warnings import warn

import numpy as np
import torch

class FirstOrderOptimizerNp(ABC):
    """
    Base class for first-order optimization algorithms using NumPy.
    
        This class serves as a foundation for implementing first-order optimization
        algorithms that rely on gradient information to update parameters. It
        provides a common interface for subclasses to implement the optimization
        logic using NumPy arrays.
    
        Attributes:
            learning_rate: The learning rate for the optimizer.
            params: The parameters to be optimized.
    """

    behavior = 'None'      
    def __init__(self, parameters: np.ndarray, optimized: np.ndarray):
        """
        Initializes the abstract optimizer.
        
        This method serves as a blueprint for initializing specific optimization algorithms.
        Subclasses must override this method to implement their own initialization logic.
        It raises a NotImplementedError if called directly, ensuring that the abstract class is not instantiated.
        
        Args:
            parameters: The parameters to be optimized, represented as a NumPy array.
            optimized: The optimized values, represented as a NumPy array.
        
        Raises:
            NotImplementedError: Always raised, as this is an abstract method.
        
        Returns:
            None.
        
        Why: This abstract initializer ensures that concrete optimizer implementations define their own initialization logic,
        tailored to the specific optimization algorithm and equation discovery process.
        """
        raise NotImplementedError('Calling __init__ of an abstract optimizer')
    
    def step(self, gradient: np.ndarray):
        """
        Performs a single optimization step.
        
        This method is an abstract method and should be implemented by subclasses.
        It updates the parameters of the model based on the calculated gradient.
        This is a crucial part of the equation discovery process, as it refines the model's parameters to better match the observed data by minimizing the error between the model's predictions and the actual data.
        
        Args:
            gradient: The gradient of the loss function with respect to the model's parameters.
        
        Returns:
            None.
        
        Raises:
            NotImplementedError: Always raised, as this is an abstract method.
        """
        raise NotImplementedError('Calling step of an abstract optimizer')

class AdamOptimizerNp(FirstOrderOptimizerNp):
    """
    Implements the Adam optimization algorithm using NumPy.
    
        The Adam optimizer is a stochastic gradient descent method
        that computes adaptive learning rates for each parameter.
    
        Attributes:
            _alpha: The learning rate.
            _beta_1: The exponential decay rate for the first moment estimates.
            _beta_2: The exponential decay rate for the second moment estimates.
            _eps: A small constant for numerical stability.
    """

    behavior = 'Gradient'      
    def __init__(self, optimized: np.ndarray, parameters: np.ndarray = np.array([0.001, 0.9, 0.999, 1e-8])):
        """
        Initializes the Adam optimizer with provided parameters.
        
                The Adam optimizer is initialized with the optimized variable and a set of hyperparameters 
                that control the optimization process. These parameters include the learning rate (alpha), 
                exponential decay rates for the first and second moment estimates (beta_1 and beta_2), 
                and a small constant for numerical stability (epsilon).
        
                Args:
                    optimized (np.ndarray): The variable to be optimized.
                    parameters (np.ndarray, optional): An array containing the optimization hyperparameters:
                        - alpha (float): The learning rate.
                        - beta_1 (float): Exponential decay rate for the first moment estimates.
                        - beta_2 (float): Exponential decay rate for the second moment estimates.
                        - eps (float): A small constant for numerical stability.
                        Defaults to np.array([0.001, 0.9, 0.999, 1e-8]).
        
                Returns:
                    None
        """
        self.reset(optimized, parameters)

    def reset(self, optimized: np.ndarray, parameters: np.ndarray):
        """
        Resets the optimizer's internal state to begin a fresh optimization run.
        
                This method reinitializes the moment, second moment, and
                second moment max accumulators to zero, updates the parameters,
                and resets the time step to zero. This ensures that the optimization
                process starts from a clean slate, preventing any influence from
                previous optimization attempts.
        
                Args:
                    optimized (np.ndarray): The array to match the shape for accumulators.
                    parameters (np.ndarray): The new parameter values to be used by the optimizer.
        
                Returns:
                    None.
        
                Class Fields Initialized:
                    _moment (np.ndarray): First moment vector, initialized to zeros with the same shape as 'optimized'.
                    _second_moment (np.ndarray): Second moment vector, initialized to zeros with the same shape as 'optimized'.
                    _second_moment_max (np.ndarray):  Maximum of second moment vector, initialized to zeros with the same shape as 'optimized'.
                    parameters (np.ndarray): The parameter values to be used by the optimizer.
                    time (int): The current time step, initialized to 0.
        """
        self._moment = np.zeros_like(optimized)
        self._second_moment = np.zeros_like(optimized)
        self._second_moment_max = np.zeros_like(optimized)
        self.parameters = parameters
        self.time = 0

    def step(self, gradient: np.ndarray, optimized: np.ndarray):
        """
        Performs a single Adam optimization step to refine parameter values based on the calculated gradient.
        
                This method updates the optimized parameters, leveraging the Adam algorithm's momentum and adaptive learning rate features. This helps to efficiently navigate the search space of possible equation coefficients, ultimately aiming to discover a differential equation that accurately describes the observed data.
        
                Args:
                    gradient (np.ndarray): The gradient of the loss function with respect to the parameters. This indicates the direction of steepest ascent in the loss landscape.
                    optimized (np.ndarray): The current values of the parameters being optimized. These represent the coefficients within the candidate differential equation.
        
                Returns:
                    np.ndarray: The updated values of the optimized parameters after applying the Adam update. These updated values represent a refined set of coefficients for the candidate differential equation, potentially leading to a better fit with the observed data.
        """
        self.time += 1
        self._moment = self.parameters[1] * self._moment + (1-self.parameters[1]) * gradient
        self._second_moment = self.parameters[2] * self._second_moment +\
                              (1-self.parameters[2]) * np.power(gradient)
        moment_cor = self._moment/(1 - np.power(self.parameters[1], self.time))
        second_moment_cor = self._second_moment/(1 - np.power(self.parameters[2], self.time))
        return optimized - self.parameters[0]*moment_cor/(np.sqrt(second_moment_cor)+self.parameters[3])
    
class FirstOrderOptimizer(ABC):
    """
    Base class for first-order optimization algorithms.
    
        This class provides a foundation for implementing various first-order
        optimization algorithms. It includes abstract methods for resetting and
        performing optimization steps, which must be implemented by subclasses.
    """

    behavior = 'Gradient'    
    def __init__(self, optimized: List[torch.Tensor], parameters: list):
        """
        Initializes the abstract optimizer.
        
                This method serves as a blueprint for concrete optimizer implementations.
                It ensures that subclasses define their own initialization logic, tailored
                to the specific optimization strategy they employ for equation discovery.
                Since the base class doesn't implement a specific optimization, it raises
                a NotImplementedError to enforce proper subclassing.
        
                Args:
                    optimized: The list of optimized tensors. These tensors represent the
                        variables or parameters within the discovered equations that are
                        being adjusted to improve the fit to the data.
                    parameters: The list of parameters. These parameters define the search
                        space and influence the evolutionary process of equation discovery.
        
                Raises:
                    NotImplementedError: Always raised, as this is an abstract method and
                        must be implemented by a concrete subclass to define the
                        optimization procedure.
        
                Returns:
                    None.
        """
        raise NotImplementedError('Calling __init__ of an abstract optimizer')
    
    def reset(self, optimized: Dict[str, torch.Tensor], parameters: np.ndarray):
        """
        Resets the optimizer.
        
        This abstract method should be implemented by subclasses to reset the optimizer's internal state.
        This is crucial for re-evaluating equation candidates with a clean slate, ensuring fair comparison and preventing bias from previous optimization steps.
        It raises a NotImplementedError if called directly.
        
        Args:
            optimized (Dict[str, torch.Tensor]): A dictionary containing optimized tensors.
            parameters (np.ndarray): A NumPy array containing the parameters.
        
        Returns:
            None.
        
        Raises:
            NotImplementedError: Always raised when calling the base class method.
        """
        raise NotImplementedError('Calling reset method of an abstract optimizer')

    def step(self, gradient: Dict[str, torch.Tensor], optimized: Dict[str, torch.Tensor],
             *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Performs a single optimization step.
        
                This method is an abstract method and should be implemented by subclasses.
                It updates the optimized parameters based on the given gradients. This is a crucial step in the equation discovery process, as it refines the candidate equations based on how well they fit the observed data.
        
                Args:
                    gradient: The gradients of the parameters to be optimized.
                    optimized: The current values of the parameters being optimized.
                    *args: Variable length argument list.
                    **kwargs: Arbitrary keyword arguments.
        
                Returns:
                    The updated values of the optimized parameters.
        
                Raises:
                    NotImplementedError: If the method is called directly from the base class.
        """
        raise NotImplementedError('Calling step of an abstract optimizer')
    
class AdamOptimizer(FirstOrderOptimizer):
    """
    Implements the Adam optimization algorithm.
    
        The Adam optimizer is a stochastic gradient descent method
        that computes adaptive learning rates for each parameter.
    
        Attributes:
            _alpha: The learning rate.
            _beta_1: The exponential decay rate for the first moment estimates.
            _beta_2: The exponential decay rate for the second moment estimates.
            _eps: A small constant for numerical stability.
    """

    behavior = 'Gradient'
    def __init__(self, optimized: List[torch.Tensor], parameters: list = [0.001, 0.9, 0.999, 1e-8]):
        """
        Initializes the Adam optimizer with specified hyperparameters.
        
                This optimizer adapts learning rates for each parameter,
                enhancing the discovery of differential equations by fine-tuning
                the optimization process based on equation complexity.
        
                Args:
                    optimized (List[torch.Tensor]): List of torch.Tensors to be optimized.
                    parameters (list, optional): List of hyperparameters for Adam optimizer.
                        parameters[0] - alpha (float): The learning rate. Defaults to 0.001.
                        parameters[1] - beta_1 (float): Exponential decay rate for the first moment estimates. Defaults to 0.9.
                        parameters[2] - beta_2 (float): Exponential decay rate for the second moment estimates. Defaults to 0.999.
                        parameters[3] - eps (float): Term added to the denominator to improve numerical stability. Defaults to 1e-8.
        
                Returns:
                    None
        """
        self.reset(optimized, parameters)

    def reset(self, optimized: Dict[str, torch.Tensor], parameters: np.ndarray):
        """
        Resets the optimizer's internal state to begin a fresh optimization run.
        
                This method reinitializes the first and second moment estimates, effectively clearing the optimizer's memory of previous iterations. It also sets the initial parameters and resets the time step counter. This ensures that the optimization process starts from a clean slate, preventing any influence from prior optimization attempts.
        
                Args:
                    optimized (Dict[str, torch.Tensor]): A dictionary of optimized tensors, used to determine the shape and device of the moment estimates.
                    parameters (np.ndarray): The initial parameters to be optimized, replacing any previously held parameters.
        
                Returns:
                    None.
        
                Initializes:
                    _moment (list): A list of tensors representing the first moment estimates, initialized to zero tensors with the same shape and device as the optimized tensors.
                    _second_moment (list): A list of tensors representing the second moment estimates, initialized to zero tensors with the same shape and device as the optimized tensors.
                    parameters (np.ndarray): The parameters to be optimized, assigned from the input.
                    time (int): The current time step, initialized to 0, indicating the start of a new optimization sequence.
        """
        self._moment = [torch.zeros_like(param_subtensor) for param_subtensor in optimized.values()] 
        self._second_moment = [torch.zeros_like(param_subtensor) for param_subtensor in optimized.values()]
        self.parameters = parameters
        self.time = 0

    def step(self, gradient: Dict[str, torch.Tensor], optimized: Dict[str, torch.Tensor],
             *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Performs a single optimization step using the Adam algorithm.
        
                This method refines the equation parameters based on the calculated gradients
                and the Adam optimization parameters. It maintains and updates the first
                and second moments of the gradients, applies bias correction, and then
                updates the equation parameters. This step is crucial for iteratively improving
                the fit of the discovered equations to the observed data by adjusting the
                equation's coefficients and structure.
        
                Args:
                    gradient: A dictionary containing the gradients for each parameter.
                    optimized: A dictionary containing the parameters to be optimized.
        
                Returns:
                    A dictionary containing the updated optimized parameters after the
                    optimization step.
        """
        self.time += 1

        self._moment = [self.parameters[1] * self._moment[tensor_idx] + (1-self.parameters[1]) * grad_subtensor
                        for tensor_idx, grad_subtensor in enumerate(gradient.values())]
        
        self._second_moment = [self.parameters[2]*self._second_moment[tensor_idx] + 
                               (1-self.parameters[2])*torch.pow(grad_subtensor, 2)
                               for tensor_idx, grad_subtensor in enumerate(gradient.values())]
        
        moment_cor = [moment_tensor/(1 - self.parameters[1] ** self.time) for moment_tensor in self._moment] 
        second_moment_cor = [sm_tensor/(1 - self.parameters[2] ** self.time) for sm_tensor in self._second_moment] 
        return OrderedDict([(subtensor_key, optimized[subtensor_key] - self.parameters[0] * moment_cor[tensor_idx]/\
                             (torch.sqrt(second_moment_cor[tensor_idx]) + self.parameters[3]))
                            for tensor_idx, subtensor_key in enumerate(optimized.keys())])
    
class CoordDescentOptimizer(FirstOrderOptimizer):
    """
    The CoordDescentOptimizer class implements a coordinate descent optimization algorithm.
    
        It iteratively updates parameters by optimizing one parameter at a time,
        keeping others fixed.
    
        Attributes:
            loc (int): The location of the parameter to update.
    """

    behavior = 'Coordinate'    
    def __init__(self, optimized: List[torch.Tensor], parameters: list = [0.001,]):
        """
        Initializes the CoordDescentOptimizer with optimization parameters.
        
                The optimizer uses a coordinate descent approach, adjusting individual parameters
                to minimize the loss function. This initialization sets up the optimizer
                with the parameters that control the step size and other optimization dynamics.
        
                Args:
                    optimized (List[torch.Tensor]): A list of PyTorch tensors to be optimized.
                    parameters (list, optional): A list of optimization parameters. Defaults to [0.001].
                        - parameters[0] (float): Alpha, the step size or learning rate.
                        - parameters[1] (float): Beta_1, parameter for momentum (if applicable).
                        - parameters[2] (float): Beta_2, parameter for adaptive learning rates (if applicable).
                        - parameters[3] (float): Epsilon, a small value to prevent division by zero (if applicable).
        
                Returns:
                    None
        """
        self.reset(optimized, parameters)

    def reset(self, optimized: Dict[str, torch.Tensor], parameters: np.ndarray):
        """
        Resets the optimizer's state with new parameters and sets the time to zero. This is done to prepare the optimizer for a new phase of the evolutionary search process, ensuring a clean slate for evaluating candidate solutions.
        
                Args:
                    optimized (Dict[str, torch.Tensor]): A dictionary of optimized tensors (unused).
                    parameters (np.ndarray): The new parameter values to be used by the optimizer.
        
                Returns:
                    None.
        
                Initializes:
                    parameters (np.ndarray): The parameter values of the object.
                    time (int): The current time step, initialized to 0.
        """
        self.parameters = parameters
        self.time = 0

    def step(self, gradient: Dict[str, torch.Tensor], optimized: Dict[str, torch.Tensor], 
             *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Performs a coordinate descent step to refine a single parameter.
        
        This method updates a specific parameter by moving it along the
        negative gradient direction. This iterative refinement of individual
        parameters contributes to the overall optimization process,
        ultimately aiming to discover the underlying differential equation.
        
        Args:
            gradient: A dictionary containing the gradients for each parameter.
            optimized: A dictionary containing the current optimized parameter values.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments. Must contain 'loc' key
               specifying the location of the parameter to update.
        
        Returns:
            A dictionary containing the updated optimized parameter values.
        """
        self.time += 1
        assert 'loc' in kwargs.keys(), 'Missing location of parameter value shift in coordinate descent.'
        loc = kwargs['loc']
        if torch.isclose(gradient[loc[0]][tuple(loc[1:])], 
                         torch.tensor((0,)).to(device=gradient[loc[0]][tuple(loc[1:])].device).float()):
            warn(f'Gradient at {loc} is close to zero: {gradient[loc[0]][tuple(loc[1:])]}.')
        optimized[loc[0]][tuple(loc[1:])] = optimized[loc[0]][tuple(loc[1:])] -\
                                            self.parameters[0]*gradient[loc[0]][tuple(loc[1:])]
        return optimized
    
# TODO: implement coordinate descent
    #np.power(self.parameters[1], self.time)) # np.power(self.parameters[2], self.time)

# class LBFGS(FirstOrderOptimizer):
#     def __init__(self, optimized: List[torch.Tensor], parameters: list = []):
#         pass

#     def step(self, gradient: Dict[str, torch.Tensor], optimized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         pass

#     def update_hessian(self, gradient: Dict[str, torch.Tensor], x_vals: Dict[str, torch.Tensor]):
#         # Use self._prev_grad
#         for i in range(self._mem_size - 1):
#             alpha = 

#     def get_alpha(self):
#         return alpha