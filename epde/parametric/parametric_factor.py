import numpy as np
from collections import OrderedDict
from functools import reduce, singledispatchmethod

from epde.structure.factor import Factor
from epde.structure.main_structures import Term
from epde.supplementary import factor_params_to_str

from copy import deepcopy
from pprint import pprint
import epde.globals as global_var



class ParametricFactor(Factor):
    """
    Represents a parametric factor with optimizable parameters.
    
        This class provides a base for factors that depend on parameters
        which can be optimized. It includes functionalities for managing
        these parameters, setting their values, and evaluating gradients.
    
        Class Methods:
        - __deepcopy__
        - set_grad_evaluator
        - grad_cache_label
        - required_params
        - __contains__
        - use_params
        - set_defined_params
        - reset_saved_state
        - eval_grad
    
        Attributes:
        - params_defined: A flag indicating whether parameters are defined (initialized to False).
        - params_to_optimize: Parameters to be optimized, initialized with the value of the `params_to_optimize` argument.
        - params_description: A dictionary containing parameter descriptions, where keys are parameter indices and values are dictionaries with 'name' and 'bounds'.
        - params_description_odict: Original parameter descriptions (copy of params_description).
        - equality_ranges: Equality ranges, initialized with the value of the `equality_ranges` argument.
        - defined_params_passed: A flag indicating whether defined parameters have been passed (initialized to False).
        - params_predefined: A dictionary to store predefined parameters (initialized as an empty dictionary).
    """

    __slot__ = ['_params', '_params_description', '_hash_val',
                'label', 'type', 'grid_set', 'grid_idx', 'is_deriv', 'deriv_code',
                'cache_linked', '_status', 'equality_ranges', '_evaluator', 'saved',
                'params_defined', 'params_to_optimize']

    def __init__(self, token_name: str, status: dict, family_type: str, params_description=None,
                 params_to_optimize=None, deriv_code=None, equality_ranges=None):
        """
        Initializes a ParametricFactor object, preparing it for equation discovery by defining its parameters.
        
                This constructor sets up the factor with its token name, status, family type, and most importantly,
                parameter descriptions and optimization settings. It also resets the saved state to ensure a clean
                slate for each new equation discovery attempt. The parameter descriptions are converted into a numerical
                index-based format for efficient handling during the evolutionary search. This setup is crucial for
                exploring the space of possible equation structures and parameter values.
        
                Args:
                    token_name (str): The name of the token representing this factor in the equation.
                    status (dict): A dictionary containing the status of the token.
                    family_type (str): The family type of the token.
                    params_description (dict): A dictionary describing the parameters, where keys are parameter names and values are dictionaries with 'name' and 'bounds'.
                    params_to_optimize (any): Parameters to be optimized during the equation discovery process.
                    deriv_code (any): Derivative code associated with this factor.
                    equality_ranges (any): Equality ranges for constraints on the parameters.
        
                Returns:
                    None
        
                Class Fields:
                    params_defined (bool): A flag indicating whether parameters are defined (initialized to False).
                    params_to_optimize (any): Parameters to be optimized, initialized with the value of the `params_to_optimize` argument.
                    params_description (dict): A dictionary containing parameter descriptions, where keys are parameter indices and values are dictionaries with 'name' and 'bounds'.
                    params_description_odict (any): Original parameter descriptions (copy of params_description).
                    equality_ranges (any): Equality ranges, initialized with the value of the `equality_ranges` argument.
                    defined_params_passed (bool): A flag indicating whether defined parameters have been passed (initialized to False).
                    params_predefined (dict): A dictionary to store predefined parameters (initialized as an empty dictionary).
        """
        self.params_defined = False
        self.params_to_optimize = params_to_optimize
        super().__init__(token_name, status, family_type, False,
                         params_description, deriv_code, equality_ranges)

        _params_description = {}
        for param_idx, param_info in enumerate(params_description.items()):
            _params_description[param_idx] = {'name': param_info[0],
                                              'bounds': param_info[1]}
        self.params_description = _params_description
        # Костыль, разобраться с лишними объектами
        self.params_description_odict = params_description
        self.equality_ranges = equality_ranges
        self.defined_params_passed = False
        self.params_predefined = {}

        self.reset_saved_state()

    def __deepcopy__(self, memo):
        """
        Creates a deep copy of the ParametricFactor object.
        
                This method is essential for preserving the integrity of equation structures
                during evolutionary operations. It ensures that when a ParametricFactor
                is modified (e.g., during mutation or crossover), the original object
                remains unchanged, and a new, independent copy is created with the
                desired modifications. This prevents unintended side effects and maintains
                the correctness of the evolutionary search process.
        
                Args:
                    self: The ParametricFactor object to be copied.
                    memo: The memo dictionary used by `deepcopy` to prevent infinite recursion.
        
                Returns:
                    A new ParametricFactor object that is a deep copy of the original.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        for k in self.__slots__:
            try:
                if not isinstance(k, list):
                    setattr(result, k, deepcopy(getattr(self, k), memo))
                else:
                    temp = []
                    for elem in getattr(self, k):
                        temp.append(deepcopy(elem, memo))
                    setattr(result, k, temp)
            except AttributeError:
                pass

        return result

    def set_grad_evaluator(self, evaluator):
        """
        Sets the gradient evaluator for parameter optimization.
        
        The gradient evaluator is crucial for efficiently adjusting the factor's parameters
        during the equation discovery process. By setting it, we enable the factor to
        compute gradients and refine its parameters to better fit the data.
        
        Args:
            evaluator: The gradient evaluator to be set.
        
        Returns:
            None.
        
        This method initializes the following class fields:
          - _grad_evaluator: The gradient evaluator used for computing gradients.
        """
        self._grad_evaluator = evaluator

    @property
    def grad_cache_label(self):
        """
        Generates a label for the gradient cache.
        
        This label is used to identify and retrieve cached gradient computations,
        improving efficiency by avoiding redundant calculations when exploring
        the parameter space of potential equation structures. The label is
        constructed from the factor parameters, with a '_grad' suffix appended
        to the first element to distinguish it as a gradient cache label.
        
        Args:
            self: The instance of the class.
        
        Returns:
            tuple: A tuple representing the label for the gradient cache.
        """
        grad_cache_label = list(factor_params_to_str(self))
        grad_cache_label[0] += '_grad'
        return tuple(grad_cache_label)

    @property
    def required_params(self):
        """
        Returns the required parameters for optimization.
        
        This method provides the necessary information for the optimization process, 
        specifically the equation's structure (hash description) and the parameters 
        that need to be tuned to best fit the data. This is crucial for the evolutionary 
        algorithm to effectively search for the optimal equation.
        
        Args:
            None
        
        Returns:
            tuple: A tuple containing:
             - hash_descr (str): The hash description representing the equation's structure.
             - params_to_optimize (list): The parameters to optimize within the equation.
        """
        return self.hash_descr, self.params_to_optimize

    def __contains__(self, element):
        """
        Checks if a given parameter is marked for optimization within the factor.
        
                This is crucial for the optimization process, ensuring that only relevant parameters are adjusted to improve the model's fit to the data.
        
                Args:
                    element: The parameter to check for inclusion in the optimization set.
        
                Returns:
                    bool: True if the parameter is to be optimized, False otherwise.
        """
        return element in self.params_to_optimize

    def use_params(self, params):
        """
        Uses the provided parameters to update the factor model.
        
                This method takes a list of optimized parameters and updates the factor model's
                internal parameters based on these values. It also handles predefined parameters, 
                ensuring the factor is correctly configured for subsequent calculations of the objective function.
                This ensures that the factor model accurately reflects the current state of the optimization process.
        
                Args:
                    params (list): A list of tuples, where each tuple contains the parameter name and its optimized value.
        
                Returns:
                    None
        """
        self.reset_saved_state()
        assert len(params) == len(self.params_to_optimize), 'The number of the passed parameters does not match declared problem'
        _params = np.ones(shape=len(self.params_description))
        for param_idx, param_info in self.params_description.items():
            if param_info['name'] not in self.params_to_optimize and param_info['name'] != 'power':
                if not self.defined_params_passed:
                    _params[param_idx] = (np.random.randint(param_info['bounds'][0], param_info['bounds'][1] + 1) if isinstance(param_info['bounds'][0], int)
                                          else np.random.uniform(param_info['bounds'][0], param_info['bounds'][1])) if param_info['bounds'][1] > param_info['bounds'][0] else param_info['bounds'][0]
                else:
                    _params[param_idx] = self.params_predefined[param_info['name']]

            elif param_info['name'] in self.params_to_optimize:
                opt_param_idx = self.params_to_optimize.index(param_info['name'])
                _params[param_idx] = params[opt_param_idx][1]
        _kw_params = {param_info['name']: _params[idx] for idx, param_info in enumerate(list(self.params_description.values()))}

        super().set_parameters(self.params_description_odict,
                               self.equality_ranges, random=False, **_kw_params)

    def set_defined_params(self, defined_params: dict):
        """
        Sets specific parameter values to be used during the equation discovery process.
        
        This ensures that certain parameters are fixed to particular values, 
        allowing the search to focus on other aspects of the equation structure. 
        This is useful when some parameters are already known or constrained by prior knowledge.
        
        Args:
            defined_params (dict): A dictionary where keys are parameter labels (strings) 
                and values are the numerical values to which these parameters should be fixed.
        
        Raises:
            ValueError: If any of the provided parameter values is None, as this indicates 
                an undefined or missing value that cannot be used.
        
        Returns:
            None: This method modifies the internal state of the ParametricFactor object.
        """
        for param_label, val in defined_params.items():
            if val is None:
                raise ValueError('Trying to set the parameter with None value')
            self.params_predefined[param_label] = val
        self.defined_params_passed = True

    def reset_saved_state(self):
        """
        Resets the internal state tracking which components of the loss function have been computed.
        
                This ensures that during the evolutionary search for the optimal equation, 
                loss and gradient calculations are performed when necessary, avoiding 
                redundant computations and ensuring that changes to parameters are properly 
                reflected in the optimization process. This is crucial for the evolutionary 
                algorithm to efficiently explore the search space of possible equations.
                
                Args:
                    self: The object instance.
                
                Returns:
                    None.
        """
        deriv_eval_dict = {label: False for label in self.params_to_optimize}
        self.saved = {'base': False,
                      'deriv': deriv_eval_dict, 'structural': False}

    def eval_grad(self, param_label: str):
        """
        Evaluates the gradient of a specified parameter. This is a crucial step in refining the equation structure by quantifying the sensitivity of the loss function with respect to changes in the parameter.
        
                Args:
                    param_label: The label of the parameter for which to evaluate the gradient.
        
                Returns:
                    The evaluated gradient of the specified parameter.
        """
        # if self.saved['deriv'][param_label]:
        #     return global_var.tensor_cache.get(self.grad_cache_label,
        #                                        structural = False)
        # else:
        value = self._grad_evaluator[param_label].apply(self)
        # self.saved['deriv'][param_label] = global_var.tensor_cache.add(self.grad_cache_label, value, structural = False)
        return value


class ParametricTerm(Term):
    """
    Represents a term in a factor graph with parametric factors.
    
        This class manages parametric and defined factors, providing methods
        for evaluation, gradient calculation, and parameter handling.
    
        Methods:
        - __deepcopy__
        - term_id
        - parse_opt_params
        - opt_params_num
        - use_params
        - evaluate
        - evaluate_grad
        - equivalent_common_term
        - __contains__
        - _
        - _
        - _
    
        Attributes:
        - parametric_factors: A dictionary storing parametric factors.
        - defined_factors: A dictionary storing defined factors.
        - all_params: A list of all parameters to optimize, extracted from the
          parametric factors.
        - pool: A multiprocessing pool for parallel computations.
        - operator: The operator used to combine the results of individual factors.
    """

    __slots__ = ['_history', 'structure', 'interelement_operator', 'saved', 'saved_as',
                 'pool', 'max_factors_in_term', 'cache_linked', 'occupied_tokens_labels',
                 'parametric_factors', 'defined_factors', 'params_to_optimize', 'all_params']

    def __init__(self, pool, parametric_factors: dict, defined_factors: dict, interelement_operator=np.multiply):
        """
        Initializes a ParametricTerm object.
        
        This method sets up a term within an equation, preparing it for evaluation
        by storing its parametric and defined factors, extracting optimizable parameters,
        and configuring the multiprocessing pool and inter-element operator.
        
        Args:
            pool: A multiprocessing pool for parallel computations.
            parametric_factors: A dictionary of parametric factors. Keys are factor names,
                values are the corresponding factor objects.
            defined_factors: A dictionary of defined factors. Keys are factor names,
                values are the corresponding factor objects.
            interelement_operator: The operator used to combine the results of
                individual factors (default: numpy.multiply).
        
        Returns:
            None.
        
        Why:
            This initialization is crucial for structuring the term, ensuring that all
            necessary components (factors, parameters, computational resources) are
            properly configured before the term is evaluated during the equation discovery process.
        """
        self.parametric_factors = parametric_factors
        self.defined_factors = defined_factors
        # print('parametric factors:', self.parametric_factors)
        self.all_params = reduce(lambda x, y: x+y, [factor.params_to_optimize for factor in self.parametric_factors.values()], [])
        # print('Params in term:', self.all_params, len(self.all_params))
        self.pool = pool
        self.operator = interelement_operator

    def __deepcopy__(self, memo):
        """
        Creates a deep copy of the ParametricTerm object.
        
                This method is essential for preserving the integrity of equation structures
                during evolutionary processes. It ensures that when a ParametricTerm is
                modified (e.g., during crossover or mutation), the original structure
                remains unchanged, preventing unintended side effects in the equation
                discovery process.
        
                Args:
                    self: The ParametricTerm object to copy.
                    memo (dict): A dictionary used to keep track of objects that have
                        already been copied, to prevent infinite recursion.
        
                Returns:
                    ParametricTerm: A deep copy of the ParametricTerm object.
        """
        # print('while copying factor:')
        # print('properties of self')
        # pprint(vars(self))

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        # clss = self.__class__
        # new_struct = clss.__new__(clss)
        # memo[id(self)] = new_struct

        # new_struct.__dict__.update(self.__dict__)

        # attrs_to_avoid_copy = []

        for k in self.__slots__:
            try:
                if not isinstance(k, list):
                    setattr(result, k, deepcopy(getattr(self, k), memo))
                else:
                    temp = []
                    for elem in getattr(self, k):
                        temp.append(deepcopy(elem, memo))
                    setattr(result, k, temp)
            except AttributeError:
                pass
        # print('properties of copy')
        # pprint(vars(result))
        return result

    @property
    def term_id(self) -> int:
        """
        Calculates a unique ID for the term based on its parametric factors.
        
                The term ID is calculated by summing the hash descriptions of all
                parametric factors associated with the term. This ID serves to
                uniquely identify the term within the equation discovery process,
                allowing for efficient comparison and manipulation of terms during
                the evolutionary search for the best equation structure.
        
                Args:
                    self: The instance of the class.
        
                Returns:
                    int: The calculated term ID.
        """
        _term_id = sum([factor.hash_descr for factor in self.parametric_factors.values()])
        # print('_term_id', _term_id)
        return _term_id

    def parse_opt_params(self, params: np.ndarray):
        """
        Parses optimization parameters for each parametric factor.
        
                This method iterates through the parametric factors, extracts the
                required parameters for each factor, and associates them with the
                corresponding values from the input `params` array. The results are
                stored in an ordered dictionary, allowing to associate optimized values with corresponding parametric factors.
                
                Args:
                    params: A NumPy array containing the optimization parameters.
                
                Returns:
                    OrderedDict: An ordered dictionary where keys are hash descriptions
                        of the parametric factors and values are lists of tuples. Each
                        tuple contains the parameter name and its corresponding value
                        from the `params` array.
        """
        params_dict = OrderedDict()
        init_idx = 0
        for factor in self.parametric_factors.values():
            hash_descr, factor_params = factor.required_params
            params_dict[hash_descr] = list(zip(factor_params, params[init_idx: init_idx + len(factor_params)]))
            init_idx += len(factor_params)
        return params_dict

    def opt_params_num(self):
        """
        Returns the number of optimizable parameters.
        
        This count is crucial for assessing the complexity of the equation and guiding the optimization process during equation discovery.
        
        Args:
            self: The ParametricTerm instance.
        
        Returns:
            int: The number of optimizable parameters in the term.
        """
        return len(self.all_params)
    # sum([len(params) for params in self.parse_opt_params().values()])

    def use_params(self, params: dict):
        """
        Applies updated parameter values to the underlying parametric factors.
        
                The method iterates through a dictionary of parameter updates, applying
                each update to the corresponding parametric factor. This ensures that
                the model reflects the latest parameter estimates obtained during the
                equation discovery process.
        
                Args:
                    params: A dictionary where keys are factor labels and values are
                        dictionaries of parameters for each factor. These parameters
                        represent the optimized coefficients and other relevant values
                        for each term in the equation.
        
                Returns:
                    None. The method modifies the internal state of the `parametric_factors`
                    attribute by updating the parameters of its constituent factors.
        """
        for factor_label, factor_params in params.items():
            # print('setting params:', params)
            self.parametric_factors[factor_label].use_params(factor_params)

    def evaluate(self):
        """
        Evaluates the joint probability distribution.
        
                This method computes the joint probability distribution by multiplying the evaluated values of all parametric and defined factors associated with this term. This represents the overall probability of a specific configuration of variables within the model.
        
                Args:
                    self: The object instance.
        
                Returns:
                    np.ndarray: A 1D numpy array representing the joint probability
                    distribution.
        """
        value = np.multiply.reduce([factor.evaluate() for factor in self.parametric_factors.values()] +
                                   [factor.evaluate() for factor in self.defined_factors.values()])  # , initial =
        return value.reshape(-1)

    def evaluate_grad(self, parameter):
        """
        Evaluates the gradient of the term with respect to a given parameter.
        
                This method computes the gradient of the term to refine the equation discovery process. It identifies the parametric factor associated with the specified parameter, evaluates the remaining factors within the term, and combines them with the gradient of the identified factor. This ensures accurate gradient calculation for optimization.
        
                Args:
                    parameter: The parameter with respect to which the gradient is evaluated.
        
                Returns:
                    The gradient of the term with respect to the given parameter.
        """
        # print(self.parametric_factors, parameter)
        param_factor_idxs = [idx for idx, factor in enumerate(self.parametric_factors.values()) if parameter in factor]
        # print(param_factor_idxs)
        assert len(param_factor_idxs) == 1, 'More than one factor in a term contains the same parameter'

        return np.multiply.reduce([factor.evaluate() for idx, factor in enumerate(self.parametric_factors.values()) if idx != param_factor_idxs[0]] +
                                  [factor.evaluate() for factor in self.defined_factors.values()]) * list(self.parametric_factors.values())[param_factor_idxs[0]].eval_grad(parameter)

    def equivalent_common_term(self):
        """
        Generates a simplified term representation by merging parametric and defined factors.
        
                This method consolidates the factors, treating constant factors separately to
                facilitate efficient equation discovery. By combining these factors into a single
                term, the search space for potential differential equations is reduced,
                improving the efficiency of the evolutionary algorithm.
        
                Args:
                    self: The instance of the ParametricTerm class.
        
                Returns:
                    tuple: A tuple containing:
                        - const_val: The value of the constant factor (or 1 if no constant factor is present).
                        - equilvalent_term: A Term object representing the combined factors.
        """
        factors_to_convert = []
        const_set = False
        for factor in self.parametric_factors.values():
            if factor.label != 'const':
                factors_to_convert.append(factor)
            else:
                const_set = True
                const_val = factor.param(name='value')

        for factor in self.defined_factors.values():
            factors_to_convert.append(factor)

        if not const_set:
            const_val = 1
        equilvalent_term = Term(pool=self.pool, passed_term=factors_to_convert,
                                max_factors_in_term=len(factors_to_convert))
        return const_val, equilvalent_term

    @singledispatchmethod
    def __contains__(self, element):
        """
        Checks if a given element is contained within the parametric term.
        
        This method serves as a placeholder and is designed to be overridden by subclasses.
        Subclasses should implement this method to define how membership is determined
        based on the specific structure and properties of the parametric term.
        For example, a subclass representing a polynomial term might check if the element
        is a valid variable or coefficient within the polynomial.
        
        Args:
            element: The element to check for membership.  The type of this element
                depends on the specific parametric term implementation.
        
        Returns:
            bool: Always raises a NotImplementedError.
        
        Raises:
            NotImplementedError: Always raised, indicating that the method
                must be implemented by a subclass to provide meaningful
                membership checking logic. This ensures that the behavior is
                tailored to the specific type of parametric term.
        """
        raise NotImplementedError(
            'Incorrect type of the requested item for the __contains__ method')

    @__contains__.register
    def _(self, element: str):
        """
        Checks if a given parameter should be optimized during the equation discovery process.
        
        This is crucial for focusing the search on relevant parameters and improving the efficiency
        of the equation discovery process. By selectively optimizing parameters, the algorithm can
        avoid overfitting and identify more parsimonious and interpretable equation structures.
        
        Args:
            element (str): The name of the parameter to check.
        
        Returns:
            bool: True if the parameter is in the set of parameters to be optimized, False otherwise.
        """
        return element in self.params_to_optimize

    @__contains__.register
    def _(self, element: ParametricFactor):
        """
        Checks if a given parametric factor is already included within the managed set of parametric factors.
        
        This check ensures that redundant factors are not added, maintaining the uniqueness and integrity of the parametric term's structure.
        
        Args:
            element (ParametricFactor): The parametric factor to check for existence.
        
        Returns:
            bool: True if the parametric factor is already present, False otherwise.
        """
        return element in self.parametric_factors.values()

    @__contains__.register
    def _(self, element: Factor):
        """
        Checks if a given factor is already defined within the parametric term's factor graph.
        
        This check is crucial for ensuring that the evolutionary process, which searches for the optimal equation structure, does not introduce redundant or conflicting factors into the model. By verifying the existence of a factor, the algorithm avoids unnecessary computations and maintains the integrity of the equation being constructed.
        
        Args:
            element: The factor to check for existence.
        
        Returns:
            bool: True if the factor is already defined, False otherwise.
        """
        return element in self.defined_factors.values()
