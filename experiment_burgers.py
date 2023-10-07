import time
import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import traceback
import logging


def find_coeff_diff(res, coefficients: dict):
    differences = []

    for pareto_front in res:
        for soeq in pareto_front:
            if soeq.obj_fun[0] < 10.:
                eq_text = soeq.vals.chromosome['u'].value.text_form
                terms_dict = out_formatting(eq_text)
                diff = coefficients_difference(terms_dict, coefficients)
                if diff != -1:
                    differences.append(diff)

    return differences


def coefficients_difference(terms_dict, coefficients):
    mae = 0.
    eq_found = 0
    for term_hash in terms_dict.keys():
        mae += abs(terms_dict.get(term_hash) - coefficients.get(term_hash))
        if coefficients.get(term_hash) == -1.0 and abs(terms_dict.get(term_hash) - coefficients.get(term_hash)) < 0.2:
            eq_found += 1

    mae /= len(terms_dict)
    if eq_found == 2:
        return mae
    else:
        return -1


def out_formatting(string):
    string = string.replace("u{power: 1.0}", "u")
    string = string.replace("d^2u/dx2^2{power: 1.0}", "d^2u/dx2^2")
    string = string.replace("d^2u/dx1^2{power: 1.0}", "d^2u/dx1^2")
    string = string.replace("du/dx1{power: 1.0}", "du/dx1")
    string = string.replace("du/dx2{power: 1.0}", "du/dx2")
    string = string.replace(" ", "")

    ls_equal = string.split('=')
    ls_left = ls_equal[0].split('+')
    ls_terms = []
    for term in ls_left:
        ls_term = term.split('*')
        ls_terms.append(ls_term)
    ls_right = ls_equal[1].split('*')

    terms_dict = {}
    for term in ls_terms:
        if len(term) == 1:
            terms_dict[1] = float(term[0])
        else:
            coeff = float(term.pop(0))
            terms_dict[hash_term(term)] = coeff

    terms_dict[hash_term(ls_right)] = -1.
    return terms_dict


def hash_term(term):
    total_term = 0
    for token in term:
        total_token = 1
        if type(token) == tuple:
            token = token[0]
        for char in token:
            total_token += ord(char)
        total_term += total_token * total_token
    return total_term


if __name__ == '__main__':

    path = "data_burg/"
    df = pd.read_csv(f'{path}burgers_sln_100.csv', header=None)
    u = df.values
    u = np.transpose(u)
    x = np.linspace(-1000, 0, 101)
    t = np.linspace(0, 1, 101)

    boundary = 10
    dimensionality = u.ndim
    grids = np.meshgrid(t, x, indexing='ij')

    ''' Parameters of the experiment '''
    write_csv = False
    print_results = True
    max_iter_number = 1
    title = 'df0'

    terms = [('du/dx1', ), ('du/dx2', 'u'), ('u',), ('du/dx2',), ('u', 'du/dx1'), ('du/dx1', 'du/dx2'),]
    hashed_ls = [hash_term(term) for term in terms]
    coefficients = dict(zip(hashed_ls, [-1., -1., 0., 0., 0., 0.]))
    coefficients[1] = 0.

    time_ls = []
    differences_ls = []
    mean_diff_ls = []
    num_found_eq = []
    i = 0
    while i < max_iter_number:
        epde_search_obj = epde_alg.EpdeSearch(use_solver=False, boundary=boundary,
                                              dimensionality=dimensionality, coordinate_tensors=grids)

        epde_search_obj.set_moeadd_params(population_size=5, training_epochs=5)
        start = time.time()
        try:
            epde_search_obj.fit(data=u, max_deriv_order=(1, 1),
                                equation_terms_max_number=3, equation_factors_max_number=2,
                                eq_sparsity_interval=(1e-08, 1e-4))
        except Exception as e:
            logging.error(traceback.format_exc())
            i -= 1
            continue
        end = time.time()
        epde_search_obj.equation_search_results(only_print=True, num=2)
        time1 = end-start

        res = epde_search_obj.equation_search_results(only_print=False, num=2)
        difference_ls = find_coeff_diff(res, coefficients)

        if len(difference_ls) != 0:
            differences_ls.append(min(difference_ls))
            mean_diff_ls += difference_ls

        num_found_eq.append(len(difference_ls))
        print('Overall time is:', time1)
        print(f'Iteration processed: {i+1}/{max_iter_number}\n')
        time_ls.append(time1)

    if write_csv:
        arr = np.array([differences_ls, time_ls, num_found_eq])
        arr = arr.T
        df = pd.DataFrame(data=arr, columns=['MAE', 'time', 'number_found_eq'])
        df.to_csv(f'data_burg/{title}.csv')
    if print_results:
        print('\nTime for every run:')
        for item in time_ls:
            print(item)

        print()
        print(f'\nAverage time, s: {sum(time_ls) / len(time_ls):.2f}')
        print(f'Average MAE per eq: {sum(mean_diff_ls) / len(mean_diff_ls):.4f}')
        print(f'Average minimum MAE per run: {sum(differences_ls) / len(differences_ls):.4f}')
        print(f'Average # of found eq per run: {sum(num_found_eq) / len(num_found_eq):.2f}')
