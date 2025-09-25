"""
Contains baseline prototypes of 'Token' instance as a gen in Individual chromosome.

Classes
----------
Token
TerminalToken
ComplexToken
"""
import random
import numpy as np
from copy import deepcopy, copy
from functools import reduce
from abc import ABC, abstractmethod, abstractproperty


class Token(ABC):
    """
    A token is an entity that has some meaning in the context
        of a given task, and encapsulates information that is sufficient to work with it.
    """


    @abstractmethod
    def value(self, grid):
        """
        Return the token's contribution to the overall equation's value, based on the provided grid data.
        
        This value represents how well the token fits the observed data within the equation.
        
        Args:
            grid (ndarray): The grid data used to calculate the token's value.
        
        Returns:
            float: The calculated value of the token.
        """
        pass

    def name(self, with_params=False):
        """
        Returns a descriptive name for the token, incorporating parameter information if requested.
        
                This method aims to provide a user-friendly representation of the token,
                which is crucial for interpreting the discovered equation structures.
                The name is constructed using available parameters to enhance readability
                and understanding of the token's role within the equation.
        
                Args:
                    with_params (bool):  If True, include parameter information in the name.
        
                Returns:
                    str: A string representing the token's name.  It prioritizes a
                         parameter-based name if available, falling back to the type name
                         (with or without parameters) if necessary. This ensures a
                         meaningful representation even when parameter information is
                         incomplete.
        """
        try:
            return str(self.params[0]) + self.name_
        except:
            try:
                if with_params:
                    return type(self).__name__ + '(params=' + str(list(self.params)) + ')'
                return type(self).__name__
            except:
                return type(self).__name__

    def copy(self):
        """
        Creates a deep copy of the token.
        
        This ensures that modifications to the copied token do not affect the original,
        which is crucial for maintaining the integrity of the equation as it evolves
        during the equation discovery process.
        
        Args:
            self: The token to be copied.
        
        Returns:
            A new token object that is a deep copy of the original.
        """
        return deepcopy(self)


class TerminalToken(Token):
    """
    TerminalToken represents a fundamental building block in constructing mathematical expressions. It encapsulates a vector of numerical values, serving as a basic operand within a larger equation. Its evaluation relies solely on these pre-defined numeric parameters.
    """



    def __init__(self, number_params: int = 0, params_description: dict = None, params: np.ndarray = None,
                 cache_val: bool = True, fix_val: bool = False, fix: bool = False,
                 val: np.ndarray = None, type_: str = 'TerminalToken', optimizer: str = None, name_: str = None,
                 mandatory: float = 0, optimize_id: int = None):
        """
        Initializes a TerminalToken instance, representing a basic building block for constructing mathematical expressions.
        
                This token holds numerical parameters and a value, and it can be configured for caching, fixing, and optimization.
                The parameters define the token's behavior, and the value represents the result of applying the token's function.
                The token can be fixed to prevent changes during optimization or its value can be cached to improve performance.
                The token is a fundamental element in the expression tree, contributing to the overall equation being discovered.
        
                Args:
                    number_params (int): Number of numeric parameters describing the behavior of the token.
                    params_description (dict): The dictionary of dictionaries for describing numeric parameters of the token.
                        Must have the form like:
                        {
                            parameter_index: dict(name='name', bounds=(min_value, max_value)[, ...]),
                            ...
                        }
                    params (numpy.ndarray): Numeric parameters of the token for calculating its value.
                    cache_val (bool): If true, token value will be calculated only when its params are changed. Calculated value
                        is written to the token property 'self.val'.
                    fix_val (bool): Defined by parameter 'cache_val'. If true, token value returns 'self.val'.
                    fix (bool): If true, numeric parameters will not be changed by optimization procedures.
                    val (np.ndarray): Value of the token.
                    type_ (str): Type of the token.
                    optimizer (str): Optimizer
                    name_ (str): The name of the token that will be used for visualisation results and  some comparison operations.
                    mandatory (float): Unique id for the token. If not zero, the token must be present in the result construct.
                    optimize_id (int): Used for identifications by optimizers which token to optimize.
        
                Returns:
                    None
        """
        self._number_params = number_params
        if params_description is None:
            params_description = {}
        self.params_description = params_description
        self.check_params_description()

        if params is None:
            self.params = np.zeros(self._number_params)
        else:
            self.params = np.array(params, dtype=float)
        self.check_params()

        self.val = val
        self.fix = fix
        self.cache_val = cache_val
        self._fix_val = fix_val
        self.type = type_
        # print(f'setting type as {self.ftype}')
        self.optimizer = optimizer
        self.name_ = name_
        self.mandatory = mandatory
        self.optimize_id = optimize_id

    def copy(self):
        """
        Creates a copy of the token.
        
        This method creates a new token with the same properties as the original,
        ensuring that any modifications to the new token's parameters do not affect the original token.
        It performs a shallow copy of the token and then deep copies the 'params' attribute.
        
        Args:
            self: The TerminalToken instance to copy.
        
        Returns:
            A new TerminalToken instance that is a copy of the original.
        """
        new_copy = copy(self)
        new_copy.params = deepcopy(new_copy.params)
        return new_copy

    # Methods for work with params and its descriptions
    @property
    def params_description(self):
        """
        Returns the description of the parameters.
        This information is essential for constructing valid equation candidates.
        
        Args:
            None
        
        Returns:
            str: The description of the parameters.
        """
        return self._params_description

    @params_description.setter
    def params_description(self, params_description: dict):
        """
        Sets the description of the token's numeric parameters.
        
        This description is crucial for defining the search space of the evolutionary algorithm.
        It specifies the name and allowed bounds for each parameter, guiding the optimization process
        towards meaningful and physically plausible solutions.
        
        Args:
            params_description (dict): A dictionary where keys are parameter indices and values are dictionaries
                describing each parameter. Each parameter dictionary must have a 'name' key and a 'bounds' key,
                specifying the parameter's name and its minimum and maximum allowed values, respectively.
                Example:
                {
                    0: {'name': 'diffusion_coefficient', 'bounds': (0.0, 1.0)},
                    1: {'name': 'reaction_rate', 'bounds': (-10.0, 10.0)}
                }
        
        Returns:
            None
        """
        assert isinstance(params_description, dict)
        self._params_description = params_description

    def check_params_description(self):
        """
        Validates the structure and content of the parameters description dictionary.
        
        This method ensures that the `params_description` attribute, which defines the characteristics of each parameter
        associated with the current token, adheres to the expected format. It verifies that the description is a dictionary
        containing nested dictionaries for each parameter, and that each nested dictionary includes the required 'name' and
        'bounds' keys with valid values. This validation is crucial for the proper functioning of the equation discovery
        process, as it guarantees that the parameters are well-defined and can be effectively used in the evolutionary
        search for the best-fitting equation.
        
        Args:
            self: The instance of the TerminalToken class.
        
        Returns:
            None. Raises an AssertionError if the `params_description` is invalid.
        """
        recomendations = "\nUse methods 'params_description.setter' or 'set_descriptor' to change params_descriptions"
        assert isinstance(self._params_description, dict), "Invalid params_description structure," \
            " must be a dictionary of dictionaries" + recomendations
        assert len(self._params_description) == self._number_params, "The number of parameters does not" \
                                                                     " match the number of descriptors" + recomendations
        for key, value in self._params_description.items():
            assert isinstance(value, dict), "Invalid params_description structure, must be a dictionary of dictionaries"
            assert 'name' in value.keys(), "Key 'name' must be in the nested" \
                                           " dictionary for each parameter" + recomendations
            assert 'bounds' in value.keys(), "Key 'bounds' must be in the nested " \
                                             "dictionary for each parameter" + recomendations
            assert key < self._number_params, "The parameter index must not exceed" \
                                              " the number of parameters" + recomendations
            assert (len(value['bounds']) == 2 and
                    value['bounds'][0] <= value['bounds'][1]), "Bounds of each parameter must have" \
                " length = 2 and contain value" \
                " boundaries MIN <= MAX." + recomendations

    def set_descriptor(self, key: int, descriptor_name: str, descriptor_value):
        """
        Sets a specific descriptor for a parameter at the given index.
        
        This allows modification of parameter properties, which is crucial for 
        fine-tuning the equation discovery process by providing additional 
        information about the parameters. If the parameter or descriptor 
        does not exist, an error message is printed.
        
        Args:
            key (int): The index of the parameter to modify.
            descriptor_name (str): The name of the descriptor to set (e.g., 'type', 'value').
            descriptor_value: The value to assign to the descriptor.
        
        Returns:
            None: This method modifies the internal state of the TerminalToken object.
        """
        try:
            self._params_description[key][descriptor_name] = descriptor_value
        except KeyError:
            print('There is no parameter with such index/descriptor')

    def get_key_use_params_description(self, descriptor_name: str, descriptor_value):
        """
        Retrieves a key from the `_params_description` dictionary based on a descriptor name and value.
        
                This method is used to locate the appropriate parameter key within the token's
                description based on specific descriptor criteria. This allows the system to dynamically
                access and utilize token parameters based on their associated descriptors,
                facilitating the automated equation discovery process.
        
                Args:
                    descriptor_name: The name of the descriptor to search for within the parameter descriptions.
                    descriptor_value: The value that the specified descriptor should match.
        
                Returns:
                    The key associated with the matching descriptor in the `_params_description` dictionary.
        
                Raises:
                    KeyError: If no key is found in the `_params_description` dictionary with the specified descriptor name and value.
        """
        for key, value in self._params_description.items():
            if value[descriptor_name] == descriptor_value:
                return key
        raise KeyError()

    def get_descriptor_foreach_param(self, descriptor_name: str) -> list:
        """
        Extract descriptor values for each parameter associated with this terminal token.
        
        This method retrieves specific descriptor values for each parameter
        defined within the token's parameter descriptions. It's used to access
        parameter properties, which are later utilized in equation construction
        and evaluation.
        
        Args:
            descriptor_name (str): The name of the descriptor to retrieve.
                                     This specifies which property of each parameter
                                     is to be extracted (e.g., 'type', 'range').
        
        Returns:
            list: A list containing the descriptor value for each parameter.
                  The list maintains the parameter order, ensuring that each
                  element corresponds to the descriptor value of the
                  corresponding parameter.
        """
        ret = [None for _ in range(self._number_params)]
        for key, value in self._params_description.items():
            ret[key] = value[descriptor_name]
        return ret

    @property
    def params(self):
        """
        Returns the parameters associated with this terminal token.
        
                These parameters define the specific characteristics of the token, 
                allowing it to be adapted and optimized during the equation discovery process.
        
                Returns:
                    dict: The parameters of the terminal token.
        """
        return self._params

    @params.setter
    def params(self, params):
        """
        Sets the parameter vector for the terminal token.
        
        This method updates the internal parameter vector of the terminal token,
        allowing its behavior to be adjusted during the equation discovery process.
        The parameters are essential for the token to accurately represent a specific
        mathematical function or value within the discovered equation.
        
        Args:
            params (list or array): A list or array of parameter values. The length
                of the input array must match the expected number of parameters
                for this terminal token.
        
        Returns:
            None.
        
        Raises:
            AssertionError: If the length of the input `params` array does not
                match the expected number of parameters for this token.
        
        Class Fields Initialized/Modified:
            _params (np.ndarray): The parameter vector, initialized or updated
                from the input `params`.
            _fix_val (bool): A flag indicating whether the parameter values are
                fixed, set to False, indicating that the parameters can be
                modified during the equation discovery process.
        """
        assert len(params) == self._number_params, "Input array has incorrect size"
        self._params = np.array(params, dtype=float)
        self._fix_val = False

    def check_params(self):
        """
        Checks and adjusts the parameter values to be within their defined bounds, ensuring the token's configuration remains valid.
        
                This method verifies that the number of provided parameters matches the
                expected number and then iterates through each parameter, ensuring that
                its value falls within the specified lower and upper bounds defined in
                the `_params_description` dictionary. If a parameter exceeds its bounds,
                it is clipped to the corresponding boundary value. This is crucial for maintaining the integrity of the equation discovery process by preventing tokens from operating with out-of-range parameters, which could lead to invalid or nonsensical equation structures.
        
                Args:
                    self: The instance of the class.
        
                Returns:
                    None. The method modifies the `_params` attribute in place.
        """
        recomendations = "\nUse methods 'params.setter' or 'set_param' to change params"
        assert len(
            self._params) == self._number_params, "The number of parameters does not match the length of params array" + recomendations
        for key, value in self._params_description.items():
            if self._params[key] > value['bounds'][1]:
                self._params[key] = value['bounds'][1]
            if self._params[key] < value['bounds'][0]:
                self._params[key] = value['bounds'][0]

    def param(self, name=None, idx=None):
        """
        Accesses a parameter of the terminal token either by its name or index within the token's parameter list.
        
                This is useful for retrieving specific components of a terminal token,
                allowing for manipulation or evaluation within the evolutionary process
                of discovering differential equations. It prioritizes access by name,
                falling back to index-based access if no name is provided.
        
                Args:
                    name (str, optional): The name of the parameter to retrieve. Defaults to None.
                    idx (int, optional): The index of the parameter to retrieve. Defaults to None.
        
                Returns:
                    The parameter at the specified index or with the specified name.
                    Returns None if the parameter is not found, and prints an error message.
        """
        try:
            idx = idx if name == None else self.get_key_use_params_description('name', name)
        except KeyError:
            print('There is no parameter with this name')
        try:
            return self._params[idx]
        except IndexError:
            print('There is no parameter with this index')

    def set_param(self, param, name=None, idx=None):
        """
        Sets the value of a parameter, updating the token's internal state. This is crucial for modifying the equation being built and evaluated during the evolutionary search process. By changing parameter values, the algorithm explores different equation configurations to find the best fit to the data.
        
                Args:
                    param: The new value for the parameter.
                    name: The name of the parameter to set (optional).
                    idx: The index of the parameter to set (optional).
        
                Returns:
                    None.
        
                Raises:
                    KeyError: If a parameter with the given name does not exist.
                    IndexError: If a parameter with the given index does not exist.
        """
        try:
            idx = idx if name is None else self.get_key_use_params_description('name', name)
        except KeyError:
            raise KeyError('"{}" have no parameter with name "{}"'.format(self, name))
        try:
            self._params[idx] = param
            self._fix_val = False
        except IndexError:
            raise IndexError('"{}" have no parameter with index "{}"'.format(self, idx))

    def init_params(self):
        """
        Initializes the token's parameters with random values based on their defined bounds.
        
        This ensures each token starts with a diverse set of configurations,
        facilitating a broader exploration of the search space during the
        equation discovery process. By randomly initializing parameters within
        their allowed ranges, the algorithm can effectively explore different
        equation forms and parameter combinations.
        
        Args:
            self: The object instance.
        
        Raises:
            OverflowError: If the parameter bounds defined in `_params_description`
                are invalid (e.g., infinite or lead to numerical overflow) during
                random number generation.
        
        Returns:
            None.
        """
        try:
            for key, value in self._params_description.items():
                self.set_param(np.random.uniform(value['bounds'][0], value['bounds'][1]), idx=key)
        except OverflowError:

            raise OverflowError('Bounds have incorrect/infinite values')

    def set_val(self, val):
        """
        Sets the terminal token's value.
        
        This value is used during the equation discovery process to represent a specific constant or variable.
        
        Args:
            val: The new value to assign to the token. This could represent a numerical constant or a symbolic variable.
        
        Returns:
            None. The method updates the internal state of the TerminalToken instance.
        """
        self.val = val

    def value(self, grid):
        """
        Returns the value of this terminal token, computed based on the input grid.
        
                The value is either retrieved from the cache (if available and caching is enabled) or computed
                by evaluating the token's parameters against the grid. This ensures that each token contributes
                meaningfully to the overall equation being discovered.
        
                Args:
                    grid (np.ndarray): The grid on which to evaluate the token. This grid represents the independent
                        variable space over which the equation is defined.
        
                Returns:
                    np.ndarray: The value of the token, with the same shape as the input grid. This value represents
                        the token's contribution to the equation's solution at each point on the grid.
        """
        if not self._fix_val or self.val is None:
            # self.check_params()
            self._fix_val = self.cache_val
            self.val = self.evaluate(self.params, grid)
            # centralization
            # self.val -= np.mean(self.val)
        assert self.val.shape == grid.shape, "Value must be the same shape as grid"
        return self.val

    @staticmethod
    def evaluate(params, grid):
        """
        Calculates the contribution of this token to the overall equation's value on the given grid.
        
                This method determines the token's effect based on its parameters and the grid's values.
                It is designed to be overridden in subclasses representing specific terminal tokens
                to implement their unique evaluation logic. The result represents a component of the
                overall equation's solution, enabling the evolutionary algorithm to explore different
                equation structures.
        
                Args:
                    params (numpy.ndarray): Numeric token parameters that influence its value.
                    grid (numpy.ndarray): The grid of independent variable values over which to evaluate the token.
        
                Returns:
                    numpy.ndarray: A numerical array representing the token's evaluated value at each point on the grid.
        """
        return np.zeros(grid.shape)


class ComplexToken(TerminalToken):
    """
    ComplexToken is the Token which consists other tokens (as subtokens in property self.subtokens)
        in addition to the numeric parameters.
        Example: Product of TerminalTokens.
    """


    def __init__(self, number_params: int = 0, params_description: dict = None, params: np.ndarray = None,
                 cache_val: bool = True, fix_val: bool = False, fix: bool = False,
                 val: np.ndarray = None, type_: str = 'TerminalToken', optimizer: str = None, name_: str = None,
                 mandatory: float = 0, optimize_id: int = None,
                 subtokens: list = None):
        """
        Initializes a ComplexToken, which represents a token composed of other tokens.
        
        This class extends TerminalToken and incorporates the concept of subtokens, enabling the construction of more complex expressions.
        The ComplexToken relies on its subtokens to compute its value, allowing for a hierarchical representation of mathematical expressions.
        
        Args:
            number_params (int, optional): The number of parameters associated with this token. Defaults to 0.
            params_description (dict, optional): A dictionary describing the parameters. Defaults to None.
            params (np.ndarray, optional): An array of parameter values. Defaults to None.
            cache_val (bool, optional): Whether to cache the computed value. Defaults to True.
            fix_val (bool, optional): Whether the value should be fixed. Defaults to False.
            fix (bool, optional): Whether the token is fixed during optimization. Defaults to False.
            val (np.ndarray, optional): The initial value of the token. Defaults to None.
            type_ (str, optional): The type of the token. Defaults to 'TerminalToken'.
            optimizer (str, optional): The optimizer to use for this token. Defaults to None.
            name_ (str, optional): The name of the token. Defaults to None.
            mandatory (float, optional): A mandatory value associated with the token. Defaults to 0.
            optimize_id (int, optional): An ID used during optimization. Defaults to None.
            subtokens (list, optional): A list of tokens that this token depends on. Defaults to None.
        
        Returns:
            None
        """
        super().__init__(number_params=number_params, params_description=params_description,
                         params=params, cache_val=cache_val, fix_val=fix_val, fix=fix,
                         val=val, type_=type_, optimizer=optimizer, name_=name_, mandatory=mandatory)
        if subtokens is None:
            subtokens = []
        self.subtokens = subtokens
        self._check_mandatory()

    @property
    def subtokens(self):
        """
        Returns the subtokens of the complex token. These subtokens represent the building blocks 
        from which the complex token is constructed, enabling a hierarchical representation 
        of mathematical expressions.
        
        Args:
            None
        
        Returns:
            list: The subtokens of the token.
        
        Why: Accessing subtokens allows for the analysis and manipulation of the token's internal structure, 
        which is crucial for tasks such as equation simplification, term rewriting, and symbolic computation 
        within the equation discovery process.
        """
        return self._subtokens

    @subtokens.setter
    def subtokens(self, subtokens: list):
        """
        Sets and configures the subtokens, preparing them for equation discovery.
        
                This method initializes the 'Amplitude' parameter for each subtoken, which is crucial for the evolutionary search process.
                By setting this parameter, the framework ensures that each subtoken is properly weighted and considered during the equation discovery process.
                The method also disables a flag and performs a mandatory check to maintain the internal state and consistency of the object.
        
                Args:
                    subtokens (list): A list of subtokens to be processed. These subtokens represent the building blocks of potential equation terms.
        
                Returns:
                    None: This method modifies the internal state of the object.
        """
        for token in subtokens:
            token.set_param(1, name='Amplitude')
        self._fix_val = False
        self._subtokens = subtokens
        self._check_mandatory()

    def add_subtoken(self, token):
        """
        Adds a subtoken to the list, marking the parent token as unfixed.
        
                This is done to incorporate the new subtoken into the equation discovery process,
                allowing the evolutionary algorithm to further refine the equation structure.
                The subtoken's 'Amplitude' parameter is also initialized.
        
                Args:
                    token: The subtoken to add.
        
                Returns:
                    None.
        """
        token.set_param(1, name='Amplitude')
        self._fix_val = False
        self.subtokens.append(token)
        self._check_mandatory()

    def set_subtoken(self, token, idx):
        """
        Sets a subtoken at a specific index, enabling equation discovery.
        
                This method sets the 'Amplitude' parameter for the given token, marks the value as
                non-fixed to allow for optimization during the equation search, and assigns the token to the specified index in the `subtokens` list.
                Finally, it checks if all mandatory subtokens are present to ensure the equation structure is complete enough for evaluation.
        
                Args:
                    token: The token to be set as a subtoken.
                    idx: The index at which to set the subtoken.
        
                Returns:
                    None.
        
                Why:
                    This method is crucial for constructing candidate equation structures. By setting subtokens at specific indices, it allows the evolutionary algorithm to explore different combinations of terms and operators, ultimately leading to the discovery of the underlying differential equation.
        """
        token.set_param(1, name='Amplitude')
        self._fix_val = False
        self.subtokens[idx] = token
        self._check_mandatory()

    def del_subtoken(self, token):
        """
        Removes a subtoken from the token's list of subtokens.
        
        This operation is essential for refining the token structure during the evolutionary search for the optimal equation. By removing less relevant subtokens, the algorithm can explore simpler and more accurate equation candidates.
        
        Args:
            token: The subtoken to remove.
        
        Returns:
            None.
        """
        self.subtokens.remove(token)

    def _check_mandatory(self):
        """
        Marks the complex token as mandatory if any of its sub-tokens are mandatory.
        
        This ensures that the evolutionary algorithm prioritizes tokens containing essential components.
        
        Args:
            self: The ComplexToken instance.
        
        Returns:
            None. Modifies the `mandatory` attribute of the ComplexToken instance.
        """
        for subtoken in self.subtokens:
            if subtoken.mandatory != 0:
                self.mandatory = np.random.uniform()
                return
        self.mandatory = 0
