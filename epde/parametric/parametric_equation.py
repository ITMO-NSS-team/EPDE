import numpy as np
import scipy.optimize as optimize
import time

from pprint import pprint
from copy import deepcopy
from collections import OrderedDict
from typing import Union
from functools import reduce, singledispatchmethod

from epde.structure.main_structures import Equation



class ParametricEquation(object):
    """
    Represents a parametric equation with optimizable terms.
    
        This class manages a collection of terms in an equation, allowing for
        optimization of their parameters to fit a desired outcome. It handles
        parameter mapping, equation evaluation, and gradient calculation.
    
        Class Methods:
        - __init__
        - optimize_equations
        - parse_eq_terms
        - parse_opt_params
        - set_term_params
        - evaluate_with_params
        - evaluate_grad
        - equation
        - set_equation
        - get_term_for_param
        - _
        - _
    
        Attributes:
        - pool: The pool associated with the object.
        - terms: A list or tuple of terms.
        - total_params_count: A list containing the number of optimizable parameters
          for each term in `terms`.
        - param_term_beloning: A dictionary mapping a global parameter index to a
          tuple containing the index of the term it belongs to and its local
          index within that term.
        - rpi: The index of the right part.
        - _optimization_held: A boolean flag indicating whether optimization is held.
    """

    def __init__(self, pool, terms: Union[list, tuple], right_part_index: int = -1):
        """
        Initializes a new instance of the ParametricEquation class.
        
                This method prepares the equation structure for the search process. It takes a pool of equation components (terms), calculates the number of optimizable parameters within each term, and establishes a mapping between global parameter indices and their corresponding term and local index. This mapping is crucial for efficiently managing and optimizing the equation's parameters during the equation discovery process.
        
                Args:
                    pool: The pool associated with the object.
                    terms: A list or tuple of terms representing the equation's components.
                    right_part_index: An optional index for the right-hand side term. If negative, it's calculated as an offset from the end of the terms list. Defaults to -1.
        
                Fields:
                    pool: The pool associated with the object.
                    terms: A list or tuple of terms.
                    total_params_count: A list containing the number of optimizable parameters for each term in `terms`.
                    param_term_beloning: A dictionary mapping a global parameter index to a tuple containing the index of the term it belongs to and its local index within that term.
                    rpi: The index of the right part.
                    _optimization_held: A boolean flag indicating whether optimization is held. Initialized to False.
        
                Returns:
                    None
        """
        self.pool = pool
        self.terms = terms

        self.total_params_count = [term.opt_params_num()
                                   for term in self.terms]
        total_count = 0
        self.param_term_beloning = {}
        for term_idx, term_params_num in enumerate(self.total_params_count):
            local_count = 0
            for _ in range(term_params_num):
                self.param_term_beloning[total_count] = (term_idx, local_count)
                total_count += 1
                local_count += 1

        self.rpi = right_part_index if right_part_index >= 0 else len(
            terms) + right_part_index
        self._optimization_held = False

    def optimize_equations(self, initial_params=None, method='L-BFGS-B'):
        """
        Optimizes the equation parameters to best represent the underlying dynamics of the system.
        
                This method employs optimization algorithms to fine-tune the parameters
                within the equation terms. By minimizing the discrepancy between the
                equation's output and the observed data, it seeks to accurately
                capture the system's behavior. This is crucial for identifying the
                governing equations from data.
        
                Args:
                    initial_params (np.ndarray, optional): Initial values for the parameters.
                        If None, defaults to an array of zeros.
                    method (str, optional): The optimization method to use.
                        Currently supports 'L-BFGS-B' and 'GD' (Gradient Descent).
                        Defaults to 'L-BFGS-B'.
        
                Returns:
                    None. The method updates the internal parameters of the equation terms
                    with the optimized values, refining the equation's representation
                    of the system.
        """
        def opt_func(params, *variables):
            '''

            Into the params the parametric tokens (or better their parameters) shall be passed,
            the variables: variables[0] - the object, containing parametric equation.  

            '''
            err = np.linalg.norm(variables[0].evaluate_with_params(params))
            print('error:', err)
            return err

        def opt_fun_grad(params, *variables):
            grad = np.zeros_like(params)
            for param_idx, param_in_term_props in variables[0].param_term_beloning.items():
                grad[param_idx] = np.sum(
                    variables[0].evaluate_grad(params, param_in_term_props))
            print('gradient:', grad)
            return grad

        if initial_params is None:
            initial_params = np.zeros(np.sum(self.total_params_count))

        # print('Reaching copy moment')
        optimizational_copy = deepcopy(self)

        # print("----------------------------------------")
        # pprint(vars(optimizational_copy))
        # print("----------------------------------------")

        # print(initial_params)
        # print('Optimized function test run', opt_func(initial_params, self))
        # print('Grad function test run', opt_fun_grad(initial_params, self))

        def opt_lbd(params): return opt_func(params, optimizational_copy)
        def opt_grd(params): return opt_fun_grad(params, optimizational_copy)

        if method == 'L-BFGS-B':
            optimal_params = optimize.minimize(opt_lbd, x0=initial_params)
        elif method == 'GD':
            optimal_params = optimize.fmin_cg(opt_lbd, x0=initial_params, fprime=opt_grd)
        else:
            raise NotImplementedError(
                'Implemented methods of parameter optimization are limited to "L-BFGS-B" and gradient descent as "GD"')
        print(type(optimal_params))

        if type(optimal_params) == np.ndarray:
            self.set_term_params(optimal_params)
        elif type(optimal_params) == optimize.optimize.OptimizeResult:
            self.set_term_params(optimal_params.x)
        self._optimization_held = True
        self.equation_set = False

    def parse_eq_terms(self):
        """
        Parses the equation terms to extract weights and equivalent terms.
        
                This method iterates through the terms, extracts the weight and
                equivalent term for each, and compiles them into separate lists.
                The weight for the right-hand side term is explicitly set to 0.
                This is a crucial step in preparing the equation for further analysis
                and processing within the EPDE framework, ensuring that each term
                contributes correctly to the overall equation structure.
        
                Args:
                  self: The instance of the class.
        
                Returns:
                  tuple[list[float], list[str]]: A tuple containing two lists:
                    - weights: A list of numerical weights corresponding to each term.
                    - equation_term: A list of equivalent terms.
        """
        weights = []
        equation_term = []
        for idx, term in enumerate(self.terms):
            weight, equivalent_term = term.equivalent_common_term()
            equation_term.append(equivalent_term)

            if idx != self.rpi:
                weights.append(weight)
        weights.append(0)
        return weights, equation_term

    def parse_opt_params(self, params):
        """
        Parses and organizes optional parameters for each term in the equation.
        
                This method iterates through the terms defined in the parametric equation,
                extracting and assigning the appropriate number of optional parameters
                from the provided list to each term. This ensures that each term
                has the correct parameter set for evaluation and contributes accurately
                to the overall equation.
        
                Args:
                    params: A list of numerical parameters to be parsed and assigned to the terms.
        
                Returns:
                    OrderedDict: An ordered dictionary where keys are term IDs and
                    values are the parsed optional parameters for each term. The order
                    matches the order of terms in the equation.
        """
        params_parsed = OrderedDict()
        cur_idx = 0
        for term in self.terms:
            # print('params', params)
            params_parsed[term.term_id] = term.parse_opt_params(params[cur_idx: cur_idx + term.opt_params_num()])
            cur_idx += term.opt_params_num()
        return params_parsed

    def set_term_params(self, params):
        """
        Sets the parameters for each term in the equation model.
        
                The method iterates through the terms of the equation and applies the corresponding optimized parameters, ensuring each term contributes appropriately to the overall equation's behavior. This step is crucial for refining the equation's fit to the observed data.
        
                Args:
                    params (dict): A dictionary where keys are term IDs and values are dictionaries of optimized parameters for that term.
        
                Returns:
                    None
        """
        params_parsed = self.parse_opt_params(params)
        for term in self.terms:
            term.use_params(params_parsed[term.term_id])

    def evaluate_with_params(self, params):
        """
        Evaluates the parametric equation with the provided parameters.
        
                This method sets the parameters for each term in the equation and then
                calculates the overall value. It achieves this by summing the evaluated
                values of all terms, excluding the term at the reduction point index (`rpi`),
                and subsequently subtracting the evaluated value of the term at `rpi`.
                This approach allows to find the equation that best describes the data by
                comparing the result to the target value, effectively minimizing the error
                introduced by the equation's structure and parameters.
        
                Args:
                    params (list): A list of parameters to be assigned to each term in the equation.
        
                Returns:
                    float: The result of the evaluation, representing the difference between the sum of
                           the terms (excluding the reduction point term) and the reduction point term's value.
        """
        self.set_term_params(params)
        if self.rpi < 0:
            val1 = np.add.reduce([term.evaluate() for term_idx, term in enumerate(self.terms) 
                                 if term_idx != len(self.terms) + self.rpi])
        else:
            val1 = np.add.reduce([term.evaluate() for term_idx, term in enumerate(self.terms)
                                 if term_idx != self.rpi])
        val2 = self.terms[self.rpi].evaluate()
        return val1 - val2

    def evaluate_grad(self, params, param_in_term_props):
        """
        Evaluates the gradient of the equation with respect to a specific parameter.
        
        This is crucial for optimization algorithms used to fit the equation to the data.
        By calculating the gradient, we determine how much the equation's output changes
        with respect to a small change in the parameter's value, guiding the optimization
        process towards the best parameter values.
        
        Args:
            params: A dictionary of parameter values.
            param_in_term_props: A tuple containing the index of the term and the index of the parameter within that term.
        
        Returns:
            The gradient of the equation with respect to the specified parameter.
        """
        param_label = self.terms[param_in_term_props[0]].all_params[param_in_term_props[1]]
        param_grad_vals = 2 * self.evaluate_with_params(params) * \
            self.terms[param_in_term_props[0]].evaluate_grad(param_label).reshape(-1)
        print('------------------------------------------------------------------')
        print(
            f'grad for param {param_label} is {np.linalg.norm(param_grad_vals)}')
        print(
            f'while eq value is {np.linalg.norm(self.evaluate_with_params(params))} and grad of factor is {np.linalg.norm(self.terms[param_in_term_props[0]].evaluate_grad(param_label))}')

        return param_grad_vals

    @property
    def equation(self):
        """
        Return the equation representing the discovered differential equation.
        
                This property provides access to the equation that has been identified as the best fit for the data.
                If the optimization process is still active and the equation hasn't been explicitly set, it automatically
                constructs the equation from the current state of the optimization. This ensures that the most up-to-date
                equation is always available during the discovery process.
        
                Args:
                    self: The object instance.
        
                Returns:
                    str: The equation of the objective function.
        
                Raises:
                    AttributeError: If equation terms have not been initialized before calling,
                                    indicating that the equation discovery process hasn't started.
        """
        if self._optimization_held:
            if not self.equation_set:
                self.set_equation()
            return self._equation
        else:
            raise AttributeError(
                'Equation terms have not been initialized before calling.')

    def set_equation(self, rpi: int = -1):
        """
        Parses and sets the equation for the current object, preparing it for symbolic regression.
        
                This method orchestrates the creation of an `Equation` object by parsing the equation's terms,
                setting the index of the term to be explained (target), and initializing the weights associated with each term.
                This setup is crucial for the subsequent steps of symbolic regression, where the equation's structure and weights
                are refined to best fit the observed data.
        
                Args:
                    rpi (int): Index of the target term. If -1, the last term is used as the target.
        
                Returns:
                    None
        
                Initializes:
                    _equation (Equation): An `Equation` object representing the equation.
                     It is initialized with the pool, basic structure (terms),
                     number of terms, and maximum factors in a term.
                    equation_set (bool): A boolean flag indicating whether the equation has been set.
                     It is set to True after the equation is created and configured.
                    _equation.target_idx (int): Index of the target term in the equation.
                     It is set to `rpi` if `rpi` is non-negative, otherwise it is set to the index of the last term.
                    _equation.weights_internal (np.ndarray): Internal weights of the equation terms.
                     Initialized with the parsed weights.
                    _equation.weights_final (np.ndarray): Final weights of the equation terms.
                     Initialized with the parsed weights.
        """
        weights, terms = self.parse_eq_terms()
        self._equation = Equation(pool=self.pool, basic_structure=terms, terms_number=len(terms),
                                  max_factors_in_term=max([len(term.structure) for term in terms]))
        self._equation.target_idx = rpi if rpi >= 0 else len(self._equation.structure) - 1
        self._equation.weights_internal = weights
        self._equation.weights_final = weights
        self.equation_set = True

    @singledispatchmethod
    def get_term_for_param(self, param):
        """
        Retrieves the symbolic term associated with a given parameter.
        
        This method serves as a base implementation that raises a NotImplementedError.
        Subclasses should override this method to define how a symbolic term
        is obtained based on the provided parameter. This is essential for
        constructing equation candidates during the equation discovery process.
        
        Args:
            param: The parameter (index or label) for which to retrieve the term.
        
        Returns:
            The symbolic term associated with the parameter.
        
        Raises:
            NotImplementedError: Always raised, indicating that the
                method must be implemented by a subclass to provide
                specific term retrieval logic. This ensures that the
                equation discovery process can correctly build and evaluate
                equation candidates.
        """
        raise NotImplementedError(
            'The term must be called by parameter index or label')

    @get_term_for_param.register
    def _(self, param: str):
        """
        Finds the first term in the equation that contains the given parameter.
        This is used to locate specific parts of the equation for manipulation or analysis.
        
        Args:
            param: The string to search for within the terms of the equation.
        
        Returns:
            str: The first term in the equation that contains the parameter.
        """
        term_index = [idx for idx, term in enumerate(self.terms) if param in term][0]
        return self.terms[term_index]

    @get_term_for_param.register
    def _(self, param: int):
        """
        Retrieves a term relevant to the equation based on the provided parameter index.
        
                Args:
                    param (int): An integer representing the index of the parameter.
        
                Returns:
                    The term in the equation corresponding to the given parameter index.
        
                Why:
                    This method is used to access specific terms within the discovered equation based on the parameter index.
                    It allows the system to dynamically retrieve and manipulate equation components during the evolutionary process.
        """
        return self.terms[self.param_term_beloning[param]]
