from typing import List, Union, Tuple, Callable
from abc import ABC, abstractmethod
from functools import singledispatchmethod

import numpy as np
import torch

from epde.supplementary import AutogradDeriv, FDDeriv
from epde.supplementary import BasicDeriv

class ConstrLocation():
    """
    Objects to contain the indices of the control training contraint location.
    
        Class Methods:
        - __init__
        - get_boundary_indices
        - apply
    
        Attributes:
        - domain_shape: shape of the domain, for which the control problem is solved.
        - axis: axis, along that the boundary conditions are selected.
        - loc: position along axis, where "bounindices = self.get_boundary_indices(self.domain_indixes, axis, loc)
        - device: string, matching the device, used for computation.
    """

    def __init__(self, domain_shape: Tuple[int], axis: int = None, loc: int = None, 
                 indices: List[np.ndarray] = None, device: str = 'cpu'):
        """
        Object to store indices representing a specific location within a domain where constraints are applied.
        
                This object facilitates the application of constraints at specific points or boundaries within the domain. 
                It computes and stores the flat indices corresponding to the specified location, enabling efficient access 
                and manipulation of these locations during the training process. This is needed to apply constraints only 
                on a part of the domain.
        
                Args:
                    domain_shape (`Tuple[int]`): Shape of the domain for which the control problem is solved.
        
                    axis (`int`, optional): Axis along which the boundary conditions are selected. Required for boundary constraints. 
                        Defaults to `None`, which matches the entire domain.
                    
                    loc (`int`, optional): Position along the axis where the boundary is located. Required for boundary constraints. 
                        Defaults to `None`, which matches the entire domain. For example, -1 corresponds to the end of the domain along the axis.
        
                    indices (`List[np.ndarray]`, optional): Explicit list of indices to use for the location. Defaults to `None`. If provided, axis and loc are ignored
        
                    device (`str`, optional): Device used for computation (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
        
                Returns:
                    None
        """
        self._device = device
        self._initial_shape = domain_shape
        
        self.domain_indixes = np.indices(domain_shape)
        if indices is not None:
            self.loc_indices = indices
        elif axis is not None and loc is not None:
            self.loc_indices = self.get_boundary_indices(self.domain_indixes, axis, loc)
        else:
            self.loc_indices = self.domain_indixes
        self.flat_idxs = torch.from_numpy(np.ravel_multi_index(self.loc_indices,
                                                               dims = self._initial_shape)).long().to(self._device)


    @staticmethod
    def get_boundary_indices(domain_indices: np.ndarray, axis: int, 
                             loc: Union[int, Tuple[int]]) -> np.array:
        """
        Method of extracting indices corresponding to specific boundaries of a domain. This is useful for defining constraints or boundary conditions when solving differential equations.
        
                Args:
                    domain_indices (`np.ndarray`): An array representing the indices of a grid. The subarrays contain index 
                        values 0, 1, … varying only along the corresponding axis. For further details inspect `np.indices(...)`
                        function.
                    
                    axis (`int`): Index of the axis along which the elements are taken.
        
                    loc (`int` or `tuple` of `int`): Positions along the specified axis, which are taken. Can be a tuple to 
                    accommodate for multiple elements along the axis.
        
                Returns:
                    `np.ndarray`: A NumPy array of indices, representing the specified boundary locations.
        
                Why:
                    This method is used to identify the indices that define the boundaries of the domain. These indices are crucial for applying boundary conditions when solving differential equations, ensuring that the solution adheres to the physical constraints of the problem.
        """
        return np.stack([np.take(domain_indices[idx], indices = loc, axis = axis).reshape(-1)
                         for idx in np.arange(domain_indices.shape[0])])
    
    def apply(self, tensor: torch.Tensor, flattened: bool = True, along_axis: int = None):
        """
        Applies the constraint location to extract specific values from a tensor.
        
                This method leverages pre-computed indices to efficiently gather elements from the input tensor
                along a specified axis. The extracted elements are then returned as a new tensor.
                This functionality is crucial for applying constraints defined by the evolutionary process
                to candidate solutions represented as tensors.
        
                Args:
                    tensor (`torch.Tensor`): The input tensor from which values will be extracted.
        
                    flattened (`bool`): A flag indicating whether the resulting tensor should be flattened.
                        Defaults to `True`. Currently, only flattened output is supported.
        
                    along_axis (`int`): The axis along which the constraint location should be applied.
                        Specifies the dimension from which elements are selected based on the pre-computed indices.
        
                Returns:
                    `torch.Tensor`: A new tensor containing the extracted values from the input tensor,
                    gathered according to the constraint location and flattened if `flattened` is `True`.
        
                Raises:
                    NotImplementedError: If `flattened` is set to `False`, as only flattened output is currently supported.
        """
        if flattened:
            shape = [1,] * tensor.ndim
            shape[along_axis] = -1
            return torch.take_along_dim(input = tensor, indices = self.flat_idxs.view(*shape), dim = along_axis)
        else:
            raise NotImplementedError('Currently, apply can be applied only to flattened tensors.')
            idxs = self.loc_indices # loop will be held over the first dimension
            return tensor.take()


class ControlConstraint(ABC):
    '''
    Abstract class for constraints declaration in the control optimization problems.
    '''

    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, indices: ConstrLocation,
                 device: str = 'cpu', deriv_axes: List = [None,], nn_output: int = 0, **kwargs):
        """
        Initializes a `ControlConstraint` object, defining a constraint on a specific location within the computational domain.
        
                This constraint is used to enforce desired behaviors or conditions at particular points or regions during the equation discovery process.
        
                Args:
                    val (Union[float, torch.Tensor]): The target value for the constraint. This could be a fixed value or a tensor.
                    deriv_method (BasicDeriv): The method used to calculate derivatives, influencing how the constraint is enforced.
                    indices (ConstrLocation): Specifies the location (index) within the domain where the constraint applies.
                    device (str, optional): The device ('cpu' or 'cuda') to use for computations. Defaults to 'cpu'.
                    deriv_axes (List, optional): The axes along which derivatives are computed for the constraint. Defaults to `[None]`.
                    nn_output (int, optional): The index of the neural network output to which this constraint applies. Defaults to 0.
                    **kwargs: Additional keyword arguments.
        
                Returns:
                    None
        
                Class Fields:
                    _val (Union[float, torch.Tensor]): The target value of the constraint.
                    _indices (ConstrLocation): The location (index) of the constraint.
                    _axes (List): The axes along which derivatives are computed.
                    _nn_output (int): The index of the neural network output.
                    _deriv_method (BasicDeriv): The derivative calculation method.
                    _device (str): The device used for computations.
        """
        self._val = val
        self._indices = indices
        self._axes = deriv_axes
        self._nn_output = nn_output
        self._deriv_method = deriv_method
        self._device = device

    @abstractmethod
    def __call__(self, fun_nn: Union[torch.nn.Sequential, torch.Tensor], 
                 arg_tensor: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        """
        Evaluates the constraint discrepancy. This method serves as an abstract interface for evaluating how well a candidate equation (represented by `fun_nn`) satisfies a given constraint at a specific point (`arg_tensor`).  Since constraints are problem-specific, concrete constraint classes must implement this method to define the evaluation logic.
        
                Args:
                    fun_nn: The neural network or tensor representing the function. This is the candidate equation being evaluated.
                    arg_tensor: The argument tensor. This is the point at which the constraint is being evaluated.
        
                Returns:
                    A tuple containing a boolean and a tensor. The boolean indicates whether the constraint is satisfied, and the tensor represents the discrepancy value. The discrepancy value quantifies the degree to which the constraint is violated.
        
                Raises:
                    NotImplementedError: Always raised, as this is an abstract method.
        """
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

    @abstractmethod
    def loss(self, fun_nn: torch.nn.Sequential, arg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the constraint violation for a given neural network and input.
        
        This abstract method is intended to be overridden by concrete constraint
        implementations to define how the constraint discrepancy is calculated.
        It serves as a placeholder and raises a NotImplementedError if called directly.
        This ensures that all specific constraint types provide their own evaluation logic
        during the equation discovery process.
        
        Args:
            fun_nn: The neural network function representing a candidate solution.
            arg_tensor: The input tensor at which to evaluate the constraint.
        
        Returns:
            torch.Tensor: This method always raises an error, so no meaningful
                return value exists.
        """
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

class ControlConstrEq(ControlConstraint):
    '''
    Class for equality constrints of type $c(u^(n)) = f(u) - val = 0$ .
    '''

    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, # grid: torch.Tensor, 
                 indices: ConstrLocation, device: str = 'cpu', deriv_axes: List = [None,],
                 nn_output: int = 0, tolerance: float = 1e-7, estim_func: Callable = None):
        """
        Initializes the constraint equation condition.
        
                This method sets up a condition that enforces a constraint on the discovered differential equation.
                It configures the value the equation should satisfy, the method for calculating derivatives,
                the locations where the constraint applies, and other parameters necessary for evaluating the constraint
                during the equation discovery process.
        
                Args:
                    val: The target value for the constraint equation. This could be a constant or a tensor.
                    deriv_method: The method used to compute derivatives within the constraint.
                    indices: Specifies where the constraint is applied (e.g., specific points or regions).
                    device: The device ('cpu' or 'cuda') used for computations. Defaults to 'cpu'.
                    deriv_axes: The axes along which derivatives are calculated. Defaults to [None].
                    nn_output: The index of the neural network output used in the constraint. Defaults to 0.
                    tolerance: Numerical tolerance for comparisons. Defaults to 1e-7.
                    estim_func: An optional function to estimate values within the constraint. Defaults to None.
        
                Returns:
                    None
        
                Class Fields:
                    _eps (float): The tolerance for numerical computations.
                    _estim_func (Callable): An optional estimation function.
        
                Why:
                    This initialization is crucial for defining the constraints that guide the equation discovery process.
                    By specifying the desired value, differentiation method, and location, we ensure that the discovered
                    equations adhere to known physical laws or observed behaviors.
        """
        print(f'Initializing condition with {deriv_method} method of differentiation.')
        super().__init__(val, deriv_method, indices, device, deriv_axes, nn_output) # grid,
        self._eps = tolerance
        self._estim_func = estim_func
 
    # TODO: add singledispatch in relation to "fun_nn": rework to accept np.ndarrays as arguments

    @singledispatchmethod
    def __call__(self, function, arg_tensor) -> Tuple[bool, torch.Tensor]:
        """
        Calculates the fulfillment of an equality constraint condition. It also quantifies the discrepancy between the observed constraint value and the desired value. This helps in evaluating candidate equation structures during the equation discovery process.
        
                Args:
                    function: The function representing the equality constraint.
                    arg_tensor: The input tensor to the function.
        
                Returns:
                    A tuple containing:
                    - A boolean indicating whether the constraint is satisfied.
                    - A tensor representing the discrepancy between the observed and desired values.
        
                Raises:
                    NotImplementedError: If the input `function` is not of a supported type (np.ndarray or torch.nn.Sequential).
        """
        raise NotImplementedError(f'Incorrect type of arguments passed into the call method. \
                                    Got {type(function)} instead of np.ndarrays of torch.nn.Sequentials.')
    
    @__call__.register
    def _(self, function: np.ndarray, arg_tensor):
        """
        Enforces a constraint by comparing the derivative of a function to a target value, ensuring the discovered equation adheres to specific conditions.
        
                This method calculates the derivative of a given function (representing a term in the equation) with respect to specified variables, and then compares the result to a predefined value. This comparison determines if the candidate equation structure satisfies the constraint within a given tolerance. This is crucial for guiding the search towards valid and physically meaningful equation forms.
        
                Args:
                  function: The function to differentiate (NumPy array), representing a term in the candidate equation.
                  arg_tensor: The tensor containing the arguments to the function, corresponding to the variables in the equation.
        
                Returns:
                  tuple: A tuple containing two elements:
                    - A boolean tensor indicating whether the constraint is satisfied (torch.Tensor).
                    - A tensor representing the difference between the calculated derivative and the target value (torch.Tensor). This difference quantifies the constraint violation.
        """
        if isinstance(self._deriv_method, AutogradDeriv):
            raise RuntimeError('Trying to call autograd differentiation of numpy npdarray. Use FDDeriv instead.')
        
        to_compare = self._deriv_method.take_derivative(u = function, 
                                                        args = self._indices.apply(arg_tensor, along_axis=0), # correct along_axis argument 
                                                        axes = self._axes)
        to_compare = torch.from_numpy(to_compare)

        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, 
                                              fill_value=self._val).to(self._device)
        else:
            if to_compare.shape != self._val.shape:
                try:
                    to_compare = to_compare.view(self._val.size())
                except:
                    raise TypeError(f'Incorrect shapes of constraint value tensor: expected {self._val.shape}, got {to_compare.shape}.')
            val_transformed = self._val
        if self._estim_func is not None:
            constr_enf = self._estim_func(to_compare, val_transformed)
        else:
            constr_enf = val_transformed - to_compare
            
        return (torch.isclose(constr_enf, torch.zeros_like(constr_enf).to(self._device), rtol = self._eps), 
                constr_enf) # val_transformed - to_compare

    @__call__.register
    def _(self, function: torch.nn.Sequential, arg_tensor):
        """
        Evaluates the constraint represented by a differential operator applied to a function.
        
                This method computes the constraint violation by evaluating the derivative of a given function (e.g., a neural network) with respect to specified axes and indices.
                It then compares the result to a target value, effectively checking if the constraint, derived from the governing differential equation, is satisfied.
                The method returns a tuple indicating whether the constraint is satisfied within a given tolerance and the difference between the target value and the evaluated derivative.
                This difference quantifies the constraint violation, guiding the optimization process towards solutions that better satisfy the underlying differential equation.
        
                Args:
                    function: The function to evaluate (e.g., a neural network).
                    arg_tensor: The input tensor to the function.
        
                Returns:
                    tuple: A tuple containing:
                        - A boolean tensor indicating whether the constraint is satisfied at each point.
                        - A tensor representing the difference between the target value and the function's output,
                          quantifying the constraint violation.
        """
        if isinstance(self._deriv_method, FDDeriv):
            raise RuntimeError('Trying to call finite differences to get derivatives of ANN, while ANN eval is not supported.\
                                Use Autograd instead.')
        to_compare = self._deriv_method.take_derivative(u = function, 
                                                        args = self._indices.apply(arg_tensor, along_axis=0), # correct along_axis argument 
                                                        axes = self._axes)
        

        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val).to(self._device)
        else:
            if to_compare.shape != self._val.shape:
                try:
                    to_compare = to_compare.view(self._val.size())
                except:
                    raise TypeError(f'Incorrect shapes of constraint value tensor: expected {self._val.shape}, got {to_compare.shape}.')
            val_transformed = self._val
        if self._estim_func is not None:
            constr_enf = self._estim_func(to_compare, val_transformed)
        else:
            constr_enf = val_transformed - to_compare
            
        return (torch.isclose(constr_enf, torch.zeros_like(constr_enf).to(self._device), rtol = self._eps), 
                constr_enf) # val_transformed - to_compare
        
    def loss(self, function: Union[torch.nn.Sequential, np.ndarray], arg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Return the magnitude of the constraint violation, which quantifies the discrepancy between the predicted and actual behavior based on the learned representation. This value serves as a penalty term in the overall loss function, guiding the learning process to satisfy the imposed constraints.
        
                Args:
                    function (`torch.nn.Sequential` or `np.ndarray`): The function, represented either as a neural network or a numerical array, that approximates a part of the constraint equation.
                    arg_tensor (`torch.Tensor`): The input tensor to be evaluated by the function.
        
                Returns:
                    `torch.Tensor`: The norm of the constraint discrepancy, representing the magnitude of the violation, to be incorporated into the combined loss.
        """
        _, discrepancy = self(function, arg_tensor)
        return torch.norm(discrepancy)


class ControlConstrNEq(ControlConstraint):
    '''
    Class for constrints of type $c(u, x) = f(u, x) - val `self._sign` 0$
    '''

    def __init__(self, val : Union[float, torch.Tensor], deriv_method: BasicDeriv, # grid: torch.Tensor, 
                 indices: ConstrLocation, device: str = 'cpu', sign: str = '>', deriv_axes: List = [None,], 
                 nn_output: int = 0, tolerance: float = 1e-7, estim_func: Callable = None):
        """
        Initializes the constraint object.
        
                This method sets up the constraint by storing essential parameters required for its evaluation.
                These parameters define the constraint's value, the method for calculating derivatives,
                the location of the constraint, the device for computation, the sign of the constraint,
                the axes along which derivatives are computed, the relevant neural network output,
                the tolerance for constraint satisfaction, and an optional estimation function.
                It is a crucial step in defining the constraints that guide the equation discovery process.
        
                Args:
                    val: The value associated with the condition.
                    deriv_method: The method used for calculating derivatives.
                    indices: The indices related to the condition's location.
                    device: The device to use for computations (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
                    sign: The sign of the condition (e.g., '>', '<', '='). Defaults to '>'.
                    deriv_axes: The axes along which to calculate derivatives. Defaults to [None].
                    nn_output: The index of the neural network output to use. Defaults to 0.
                    tolerance: The tolerance level for satisfying the condition. Defaults to 1e-7.
                    estim_func: An optional estimation function. Defaults to None.
        
                Returns:
                    None
        
                Class Fields:
                    _sign: The sign of the condition (e.g., '>', '<', '=').
                    _estim_func: An optional estimation function.
        """
        print(f'Initializing condition with {deriv_method} method of differentiation.')
        super().__init__(val, deriv_method, indices, device, deriv_axes, nn_output) # grid, 
        self._sign = sign
        self._estim_func = estim_func

    @singledispatchmethod
    def __call__(self, function, arg_tensor) -> Tuple[bool, torch.Tensor]:
        """
        Calculates the fulfillment of an inequality constraint condition and the discrepancy between the observed and desired constraint values. This method serves as a base for specialized implementations handling different constraint representations within the equation discovery process.
        
                Args:
                    function: The constraint function.
                    arg_tensor: The tensor of arguments to the constraint function.
        
                Returns:
                    A tuple containing:
                      - A boolean indicating whether the inequality constraint is satisfied.
                      - A torch.Tensor representing the discrepancy between the observed and desired constraint values.
        """
        raise NotImplementedError(f'Incorrect type of arguments passed into the call method. \
                                    Got {type(function)} instead of np.ndarrays of torch.nn.Sequentials.')

    @__call__.register
    def _(self, function: np.ndarray, arg_tensor) -> Tuple[bool, torch.Tensor]:
        """
        Evaluates the constraint based on the provided function and arguments.
        
                This method calculates constraint enforcement by comparing the derivative
                of a given function with a predefined value. It leverages a specified
                derivative method to compute the derivative, then compares it against
                a stored value, considering the constraint's sign ('>' or '<'). This
                comparison determines whether the constraint is satisfied.
        
                Args:
                    function (np.ndarray): The function to evaluate the derivative of.
                    arg_tensor (torch.Tensor): The argument tensor to apply the function to.
        
                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                        - A boolean tensor indicating whether the constraint is satisfied.
                        - A ReLU-activated tensor representing the constraint enforcement.
        
                WHY: This method is crucial for verifying whether the discovered equation
                satisfies the imposed constraints, ensuring the identified model aligns
                with known physical laws or domain-specific requirements.
        """
        if isinstance(self._deriv_method, AutogradDeriv):
            raise RuntimeError('Trying to call autograd differentiation of numpy npdarray. Use FDDeriv instead.')

        to_compare = self._deriv_method.take_derivative(u = function, 
                                                        args=self._indices.apply(arg_tensor, along_axis = 0), # correct along_axis argument 
                                                        axes=self._axes, component = self._nn_output)        
                
        to_compare = torch.from_numpy(to_compare)
      
        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val)
        else:
            if not to_compare.shape == self._val.shape:
                to_compare = torch.reshape(to_compare, shape=self._val.shape)
            val_transformed = self._val
        
        if self._estim_func is not None:
            constr_enf = self._estim_func(val_transformed, to_compare)
        else:
            constr_enf = val_transformed - to_compare

        if self._sign == '>':
            return torch.greater(constr_enf, torch.zeros_like(constr_enf).to(self._device)), torch.nn.functional.relu(constr_enf)
        elif self._sign == '<':
            return (torch.less(constr_enf, torch.zeros_like(constr_enf).to(self._device)), 
                    torch.nn.functional.relu(constr_enf))

    @__call__.register
    def _(self, function: torch.nn.Sequential, arg_tensor) -> Tuple[bool, torch.Tensor]:
        """
        Evaluates a constraint by comparing a derivative with a target value.
        
                This method computes the derivative of a given function with respect to
                an input tensor, and then compares it against a predefined value.
                The comparison result, indicating constraint satisfaction, and a measure
                of constraint enforcement are returned. This process helps to identify
                equation structures that accurately represent the underlying dynamics
                of the system.
        
                Args:
                    function: The function to take the derivative of.
                    arg_tensor: The input tensor with respect to which the derivative is taken.
        
                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                        - A boolean tensor indicating whether the constraint is satisfied.
                        - A tensor representing the constraint enforcement (ReLU of the difference).
        """
        to_compare = self._deriv_method.take_derivative(u = function, 
                                                        args=self._indices.apply(arg_tensor, along_axis = 0), # correct along_axis argument 
                                                        axes=self._axes, component = self._nn_output)
        
        if isinstance(self._deriv_method, FDDeriv):
            raise RuntimeError('Trying to call finite differences to get derivatives of ANN, while ANN eval is not supported.\
                                Use Autograd instead.')        
        if not isinstance(self._val, torch.Tensor):
            val_transformed = torch.full_like(input = to_compare, fill_value=self._val)
        else:
            if not to_compare.shape == self._val.shape:
                to_compare = torch.reshape(to_compare, shape=self._val.shape)
            val_transformed = self._val
        
        if self._estim_func is not None:
            constr_enf = self._estim_func(val_transformed, to_compare)
        else:
            constr_enf = val_transformed - to_compare

        if self._sign == '>':
            return torch.greater(constr_enf, torch.zeros_like(constr_enf).to(self._device)), torch.nn.functional.relu(constr_enf)
        elif self._sign == '<':
            return (torch.less(constr_enf, torch.zeros_like(constr_enf).to(self._device)), 
                    torch.nn.functional.relu(constr_enf))
        #torch.less(val_transformed, to_compare), torch.nn.functional.relu(to_compare - val_transformed)            

    def loss(self, function: Union[torch.nn.Sequential, np.ndarray], arg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the norm of the constraint discrepancy for a given function and argument tensor.
        
                This function quantifies how well a candidate equation satisfies the imposed constraints,
                contributing to the overall loss function that guides the equation discovery process.
                A lower loss indicates a better fit to the constraints.
        
                Args:
                    function (`torch.nn.Sequential` or `np.ndarray`): The function (approximated by a neural network or represented directly)
                        that is evaluated within the constraint.
                    arg_tensor (`torch.Tensor`): The tensor representing the input values at which the constraint is evaluated.
        
                Returns:
                    `torch.Tensor`: The norm of the constraint discrepancy, representing the magnitude of the violation of the constraint.
        """
        _, discrepancy = self(function, arg_tensor)
        return torch.norm(discrepancy)


class ConditionalLoss():
    '''
    Class for the loss, used in the control function opimizaton procedure. Conrains terms of the loss
        function in `self._cond` attribute.
    '''

    def __init__(self, conditions: List[Tuple[Union[float, ControlConstraint, int]]]):
        """
        Initialize the conditional loss with the terms for evaluating control strategies.
        
                This loss function incorporates conditions based on control constraints and solution quality,
                allowing the evolutionary algorithm to prioritize controls that satisfy specific criteria
                while optimizing the equation's fit to the data.
        
                Args:
                    conditions (`list` of triplet `tuple` as (`float`, `ControlConstraint`, `int`)):
                        A list of conditions, where each condition is a tuple containing:
                        - A floating-point value representing a threshold or target.
                        - A `ControlConstraint` object defining a constraint on the control.
                        - An integer indicating the type of condition (e.g., penalty for constraint violation).
        
                Returns:
                    None
        """
        self._cond = conditions

    def __call__(self, models: List[torch.nn.Sequential], args: list): # Introduce prepare control input: get torch tensors from solver & autodiff them
        '''
        Return the summed values of the loss function component 
        '''
        temp = []
        for cond in self._cond:
        """
        Calculates the total conditional loss based on provided models and arguments.
        
                This method iterates through the defined conditional loss components, computes the loss for each component using the specified model and arguments, and then returns the sum of these losses. This aggregated loss represents the overall discrepancy between the model predictions and the observed data, guiding the optimization process towards identifying the differential equation that best describes the system.
        
                Args:
                    models (List[torch.nn.Sequential]): A list of PyTorch models used in the loss calculation. Each model corresponds to a specific term in the conditional loss.
                    args (list): A list of arguments, where each element corresponds to the input required by the respective model.
        
                Returns:
                    torch.Tensor: The summed value of all conditional loss components, representing the total loss.
        """
            temp.append(cond[0] * cond[1].loss(models[cond[2]], args[cond[2]]))

        return torch.stack(temp, dim=0).sum(dim=0).sum(dim=0)
