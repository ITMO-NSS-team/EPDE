from epde.structure.factor import Factor

from epde.parametric.parametric_factor import ParametricFactor, ParametricTerm
from epde.parametric.parametric_equation import ParametricEquation


def float_convertable(obj):
    try:
        float(obj)
        return True
    except (ValueError, TypeError) as e:
        return False


def parse_params_str(param_str):
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

    Example input: '0.0 * d^3u/dx2^3{power: 1} * du/dx2{power: 1} + 0.0 * d^3u/dx1^3{power: 1} +
    0.015167810810763344 * d^2u/dx1^2{power: 1} + 0.0 * d^3u/dx2^3{power: 1} + 0.0 * du/dx2{power: 1} + 
    4.261009307104081e-07 = d^2u/dx1^2{power: 1} * du/dx1{power: 1}'

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
                factor.set_evaluator(cur_family._evaluator)
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
                factor.set_evaluator(cur_family._evaluator)
                temp_factors_defined[factor.hash_descr] = factor

        terms_parsed.append(ParametricTerm(pool, parametric_factors=temp_factors_param,
                                           defined_factors=temp_factors_defined))

    equation = ParametricEquation(pool, terms_parsed)
    equation.optimize_equations(kwargs['initial_params'], method=method)
    return equation
