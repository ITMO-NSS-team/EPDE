from epde.structure.factor import Factor

from epde.parametric.parametric_factor import ParametricFactor, ParametricTerm
from epde.parametric.parametric_equation import ParametricEquation


def float_convertable(obj):
    """
    Determines if a given object can be interpreted as a floating-point number.
    
    This function is crucial for ensuring data compatibility when constructing equation terms.
    It verifies whether a value can be safely converted to a float, preventing errors during
    equation evaluation and optimization.
    
    Args:
        obj: The object to be checked for float convertibility.
    
    Returns:
        bool: True if the object can be converted to a float without error, False otherwise.
    """
    try:
        float(obj)
        return True
    except (ValueError, TypeError) as e:
        return False


def parse_params_str(param_str):
    """
    Parses a comma-separated string of parameter-value pairs into a dictionary and a list of parameters to optimize.
    
        This function is crucial for configuring the search space of the evolutionary algorithm. It takes a string
        defining parameter values, identifies which parameters should be optimized, and prepares the data for
        the equation discovery process.
    
        Args:
            param_str (str): A string containing comma-separated parameters in the format "param1:value1,param2:value2,...".
                If a value is "None", the corresponding parameter will be marked for optimization.
    
        Returns:
            tuple: A tuple containing:
                - dict: A dictionary where keys are parameter names and values are their corresponding numerical values (int or float).
                - list: A list of parameter names that should be optimized (i.e., those with a value of "None" in the input string).
    """
    assert isinstance(
        param_str, str), 'Passed parameters are not in string format'
    params_split = param_str.split(',')
    params_parsed = dict()
    params_to_optimize = []
    for param in params_split:
        temp = param.split(':')
        print(temp)
        temp[0] = temp[0].replace(' ', '')
        temp[1] = temp[1].replace(' ', '')
        if temp[1] == 'None':
            params_parsed[temp[0]] = None
            params_to_optimize.append(temp[0])
        else:
            params_parsed[temp[0]] = float(
                temp[1]) if '.' in temp[1] else int(temp[1])
    return params_parsed, params_to_optimize


def parse_parametric_factor(factor_form: str, pool):
    """
    Parses a factor string, extracting its label and parameters.
    
    This function is crucial for interpreting the structure of factors,
    determining if they contain parameters that need to be optimized during
    the equation discovery process. It breaks down the factor string into
    its constituent parts, preparing it for evaluation and optimization.
    
    Args:
        factor_form: The string representation of the factor, e.g., "factor_label{param1=value1,param2=value2}".
        pool: Unused parameter.
    
    Returns:
        A tuple containing:
          - A boolean indicating whether the factor is parametric (contains parameters to optimize).
          - The label of the factor (string).
          - A dictionary of parameter names and their values (dict).
          - A list of parameter names to optimize (list).
    """
    label_str, params_str = tuple(factor_form.split('{'))
    if not '}' in params_str:
        raise ValueError(
            'Missing brackets, denoting parameters part of factor text form. Possible explanation: passing wrong argument')
    params_dict, params_to_optimize = parse_params_str(params_str.replace('}', ''))

    # print(label_str, params_str)
    contains_parametric = len(params_to_optimize) > 0
    return contains_parametric, label_str, params_dict, params_to_optimize


def parse_parametric_equation(text_form):
    '''
    Parses a parametric equation represented as a string into a structured format.
    
    The equation is split into left-hand side (LHS) and right-hand side (RHS) components,
    then further broken down into terms and factors. This structured representation
    facilitates subsequent symbolic manipulation and numerical evaluation.
    
    Args:
        text_form (str): A string representing the parametric equation, with terms separated by ' + '
                         on the LHS and factors separated by ' * ' on both sides. The LHS and RHS
                         are separated by ' = '.
    
    Returns:
        list: A list containing the parsed terms from the LHS and RHS of the equation.
              Each term is further split into its constituent factors. The last element of the list
              is the RHS of the equation.
    '''
    left, right = text_form.split(' = ')
    left = left.split(' + ')
    for idx in range(len(left)):
        left[idx] = left[idx].split(' * ')
    right = right.split(' * ')
    return left + [right,]


def construct_parametric_factor(label, param_equality, params_to_opt,
                                status=None, family_type='const',
                                params_description={'power': (1, 1)}):
    """
    Constructs a parametric factor for equation discovery.
    
    This method creates a ParametricFactor object, a fundamental building block
    in the equation discovery process. It encapsulates a potential term within
    a differential equation, defining its label, parameter constraints, and
    optimization settings. The factor is initialized with a status dictionary
    to track its relevance and characteristics during the search for the best
    equation structure. This ensures that the evolutionary algorithm can
    effectively explore the space of possible equations.
    
    Args:
        label: The label for the parametric factor.
        param_equality: Equality ranges for the parameters.
        params_to_opt: Parameters to be optimized.
        status: A dictionary to store the status of the factor (optional).
        family_type: The type of the factor family (default: 'const').
        params_description: Description of the parameters (default: {'power': (1, 1)}).
    
    Returns:
        ParametricFactor: The constructed parametric factor object.
    """
    if status is None:
        status['meaningful'] = False
        status['s_and_d_merged'] = True
        status['unique_specific_token'] = False
        status['unique_token_type'] = False
        status['requires_grid'] = False
    factor = ParametricFactor(label, status, family_type, params_description,
                              params_to_optimize=params_to_opt, deriv_code=None,
                              equality_ranges=param_equality)
    return factor


def construct_ordinary_factor(label, param_equality, status=None, family_type='const',
                              params_description={'power': (1, 1)}):
    """
    Constructs an ordinary factor object for equation discovery.
    
        This method creates a Factor object with specified properties, which represents a building block
        within a differential equation. It initializes the factor's token name, status, family type,
        parameter descriptions, derivative code, and equality ranges. If the status is not provided,
        it initializes a default status dictionary. This ensures that each factor has a consistent
        and well-defined state, which is crucial for the evolutionary search process.
    
        Args:
            label (str): The label or name of the factor, representing a term in the equation.
            param_equality (list): The equality ranges for the factor's parameters, defining constraints
                on the parameter values during the search.
            status (dict, optional): A dictionary containing status flags for the factor. Defaults to None.
            family_type (str, optional): The family type of the factor. Defaults to 'const'.
            params_description (dict, optional): A dictionary describing the parameters of the factor.
                Defaults to {'power': (1, 1)}.
    
        Returns:
            Factor: The constructed Factor object, ready to be used in the equation discovery process.
    """
    if status is None:
        status['meaningful'] = False
        status['s_and_d_merged'] = True
        status['unique_specific_token'] = False
        status['unique_token_type'] = False
        status['requires_grid'] = False
    factor = Factor(token_name=label, status=status, family_type=family_type,
                    params_description=params_description, deriv_code=None,
                    equality_ranges=param_equality)
    return factor


def optimize_parametric_form(terms: list, pool, method='L-BFGS-B', **kwargs):
    """
    Optimizes a parametric form equation within the equation discovery process.
    
        This method takes a list of terms, each containing factors, and constructs a
        `ParametricEquation`. It distinguishes between parametric and defined factors,
        using information from the provided pool to instantiate them. The resulting
        equation is then optimized to find the best fit to the data.
    
        Args:
            terms: A list of terms, where each term is a list of factors represented as strings.
            pool: A pool object containing information about factor families, including evaluators and parameter constraints.
            method (str): Optimization method to use (default: 'L-BFGS-B').
            **kwargs: Additional keyword arguments passed to the optimization routine, such as initial parameter values.
    
        Returns:
            ParametricEquation: The optimized parametric equation, ready for evaluation and analysis.
    
        WHY: This optimization step is crucial for refining the equation structure and parameter values,
        ensuring that the discovered equation accurately represents the underlying dynamics of the system.
    """
    assert all([isinstance(term_form, list) for term_form in terms])

    # factor_constructor = partial(construct_parametric_factor, param_equality = kwargs['param_equality']),

    terms_parsed = []
    for term_list in terms:
        temp_factors_param = {}
        temp_factors_defined = {}
        for factor in term_list:
            factor_is_parametric, label, params_vals, params = parse_parametric_factor(
                factor, pool)
            cur_family = pool.get_families_by_label(label)
            assert cur_family.params_set and cur_family.evaluator_set, 'Family has not been completed before the call.'
            if factor_is_parametric:
                factor = construct_parametric_factor(label=label, param_equality=cur_family.equality_ranges,
                                                     status=cur_family.status, family_type=cur_family.ftype,
                                                     params_description=cur_family.token_params, params_to_opt=params)
                factor.set_defined_params({key: value for key, value in params_vals.items() if value is not None})
                factor.evaluator = cur_family._evaluator
                factor.set_grad_evaluator(cur_family._deriv_evaluators)
                temp_factors_param[factor.hash_descr] = factor
            else:
                factor = construct_ordinary_factor(label=label, param_equality=cur_family.equality_ranges,
                                                   status=cur_family.status, family_type=cur_family.ftype,
                                                   params_description=cur_family.token_params)
                assert all([value is not None for key,
                           value in params_vals.items()])
                factor.set_parameters(params_description=cur_family.token_params,
                                      equality_ranges=cur_family.equality_ranges,
                                      random=False,
                                      **params_vals)
                factor.evaluator = cur_family._evaluator
                temp_factors_defined[factor.hash_descr] = factor

        terms_parsed.append(ParametricTerm(pool, parametric_factors=temp_factors_param,
                                           defined_factors=temp_factors_defined))

    equation = ParametricEquation(pool, terms_parsed)
    equation.optimize_equations(kwargs['initial_params'], method=method)
    return equation
