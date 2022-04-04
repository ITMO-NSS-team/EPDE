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
        Return value of the token in the context of the task.

        Parameters
        ----------
        grid:
            The grid on which the value is calculated.
        """
        pass

    def name(self, with_params=False):
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
        return deepcopy(self)


class TerminalToken(Token):
    """
    TerminalToken is the token that returns a value as a vector whose evaluating
    requaires only numeric parameters.

    """
    def __init__(self, number_params: int = 0, params_description: dict = None, params: np.ndarray = None,
                 cache_val: bool = True, fix_val: bool = False, fix: bool = False,
                 val: np.ndarray = None, type_: str = 'TerminalToken', optimizer: str = None, name_: str = None,
                 mandatory: float = 0, optimize_id: int = None):
        """

        Parameters
        ----------
        number_params: int
            Number of numeric parameters describing the behavior of the token.
        params_description: dict
            The dictionary of dictionaries for describing numeric parameters of the token.
            Must have the form like:
            {
                parameter_index: dict(name='name', bounds=(min_value, max_value)[, ...]),
                ...
            }
        params: numpy.ndarray
            Numeric parameters of the token for calculating its value.
        cache_val: bool
            If true, token value will be calculated only when its params are changed. Calculated value
            is written to the token property 'self.val'.
        fix_val: bool
            Defined by parameter 'cache_val'. If true, token value returns 'self.val'.
        fix: bool
            If true, numeric parameters will not be changed by optimization procedures.
        val: np.ndarray
            Value of the token.
        type_: str
            Type of the token.
        optimizer: str
            Optimizer 
        name_: str
            The name of the token that will be used for visualisation results and  some comparison operations.
        mandatory: float
            Unique id for the token. If not zero, the token must be present in the result construct.
        optimize_id: int
            Used for identifications by optimizers which token to optimize.
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
        self.optimizer = optimizer
        self.name_ = name_
        self.mandatory = mandatory
        self.optimize_id = optimize_id

    def copy(self):
        new_copy = copy(self)
        new_copy.params = deepcopy(new_copy.params)
        return new_copy

    # Methods for work with params and its descriptions
    @property
    def params_description(self):
        return self._params_description

    @params_description.setter
    def params_description(self, params_description: dict):
        """
        Params_description is dictionary of dictionaries for describing numeric parameters of the token.
            Must have the form like:
            {
                parameter_index=0: dict(name='name', bounds=(min_value, max_value)[, ...]),
                ...
            }
        Params_description must contain all fields for work in current tokens that will be checked by
        method 'self.check_params_description()'.

        Parameters
        ----------
        params_description: dict
            Dictionary with description for each parameter
        """
        assert isinstance(params_description, dict)
        self._params_description = params_description

    def check_params_description(self):
        """
        Check params_description for requirements for current token.
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
        try:
            self._params_description[key][descriptor_name] = descriptor_value
        except KeyError:
            print('There is no parameter with such index/descriptor')

    def get_key_use_params_description(self, descriptor_name: str, descriptor_value):
        for key, value in self._params_description.items():
            if value[descriptor_name] == descriptor_value:
                return key
        raise KeyError()

    def get_descriptor_foreach_param(self, descriptor_name: str) -> list:
        ret = [None for _ in range(self._number_params)]
        for key, value in self._params_description.items():
            ret[key] = value[descriptor_name]
        return ret

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        assert len(params) == self._number_params, "Input array has incorrect size"
        self._params = np.array(params, dtype=float)
        self._fix_val = False

    def check_params(self):
        recomendations = "\nUse methods 'params.setter' or 'set_param' to change params"
        assert len(
            self._params) == self._number_params, "The number of parameters does not match the length of params array" + recomendations
        for key, value in self._params_description.items():
            if self._params[key] > value['bounds'][1]:
                self._params[key] = value['bounds'][1]
            if self._params[key] < value['bounds'][0]:
                self._params[key] = value['bounds'][0]

    def param(self, name=None, idx=None):
        try:
            idx = idx if name == None else self.get_key_use_params_description('name', name)
        except KeyError:
            print('There is no parameter with this name')
        try:
            return self._params[idx]
        except IndexError:
            print('There is no parameter with this index')

    def set_param(self, param, name=None, idx=None):
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
        try:
            for key, value in self._params_description.items():
                self.set_param(np.random.uniform(value['bounds'][0], value['bounds'][1]), idx=key)
        except OverflowError:
            # tb = sys.exc_info()[2]
            raise OverflowError('Bounds have incorrect/infinite values')  # .with_traceback(tb)

    def set_val(self, val):
        self.val = val

    def value(self, grid):
        """
        Returns value of the token on the grid.
        Returns either cache result in self.val or calculated value in self.val by method self.evaluate().

        Parameters
        ----------
        grid: np.ndarray
            Grid for evaluation.

        Returns
        -------
        Value of the token.
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
        Calculating token value on the grid depending on parameters.
        Must be override/implement in each TerminalToken.
        May be not staticmethod if it is necessary.

        Parameters
        ----------
        params: numpy.ndarray
            Numeric token parameters.
        grid: numpy.ndarray
            Grid for evaluation.

        Returns
        -------
        numpy.ndarray
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

        Parameters
        ----------
        See documentation TerminalToken.__init__.__doc__

        subtokens: list
            List of other tokens which current token uses for calculating its value.
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
        return self._subtokens

    @subtokens.setter
    def subtokens(self, subtokens: list):
        for token in subtokens:
            token.set_param(1, name='Amplitude')
        self._fix_val = False
        self._subtokens = subtokens
        self._check_mandatory()

    def add_subtoken(self, token):
        token.set_param(1, name='Amplitude')
        self._fix_val = False
        self.subtokens.append(token)
        self._check_mandatory()

    def set_subtoken(self, token, idx):
        token.set_param(1, name='Amplitude')
        self._fix_val = False
        self.subtokens[idx] = token
        self._check_mandatory()

    def del_subtoken(self, token):
        self.subtokens.remove(token)

    def _check_mandatory(self):
        """
        If some subtoken in ComplexToken is mandatory then ComplexToken is mandatory too.

        Returns
        -------

        """
        for subtoken in self.subtokens:
            if subtoken.mandatory != 0:
                self.mandatory = np.random.uniform()
                return
        self.mandatory = 0

