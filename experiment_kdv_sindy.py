import time
import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg
from kdv_init_distrib_sindy import coefficients1, coefficients2
from scipy.io import loadmat
import traceback
import logging


def find_coeff_diff(res):
    differences = []

    for pareto_front in res:
        for soeq in pareto_front:
            if soeq.obj_fun[0] < 10:
                eq_text = soeq.vals.chromosome['u'].value.text_form
                terms_dict = out_formatting(eq_text)
                diff = coefficients_difference(terms_dict)
                if diff != -1:
                    differences.append(diff)

    return differences


def coefficients_difference(terms_dict):
    mae1 = 0.
    mae2 = 0.
    eq_found = 0
    for term_hash in terms_dict.keys():
        mae1 += abs(terms_dict.get(term_hash) - coefficients1.get(term_hash))
        mae2 += abs(terms_dict.get(term_hash) - coefficients2.get(term_hash))
        if coefficients1.get(term_hash) != 0.0 and (abs(terms_dict.get(term_hash) - coefficients1.get(term_hash)) < 0.3\
                or abs(terms_dict.get(term_hash) - coefficients2.get(term_hash)) < 0.3):
            eq_found += 1

    values = list(terms_dict.values())
    not_zero_ls = [value for value in values if value != 0.0]
    mae1 /= len(not_zero_ls)
    mae2 /= len(not_zero_ls)
    mae = min(mae1, mae2)

    if eq_found == 3:
        return mae
    else:
        return -1


def out_formatting(string):
    string = string.replace("u{power: 1.0}", "u")
    string = string.replace("d^2u/dx2^2{power: 1.0}", "d^2u/dx2^2")
    string = string.replace("d^2u/dx1^2{power: 1.0}", "d^2u/dx1^2")
    string = string.replace("du/dx1{power: 1.0}", "du/dx1")
    string = string.replace("du/dx2{power: 1.0}", "du/dx2")
    string = string.replace("cos(t)sin(x){power: 1.0}", "cos(t)sin(x)")
    string = string.replace("d^3u/dx2^3{power: 1.0}", "d^3u/dx2^3")
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
    path = "data_kdv/"
    kdV = loadmat(f'{path}/kdv.mat')
    t = np.ravel(kdV['t'])
    x = np.ravel(kdV['x'])
    u = np.real(kdV['usol'])
    u = np.transpose(u)

    boundary = 0
    dimensionality = u.ndim
    grids = np.meshgrid(t, x, indexing='ij')

    ''' Parameters of the experiment '''
    write_csv = True
    print_results = True
    max_iter_number = 50
    title = 'dfs_sindy'

    time_ls = []
    differences_ls = []
    mean_diff_ls = []
    num_found_eq = []
    differences_ls_none =[]
    i = 0
    population_error = 0
    while i < max_iter_number:
        epde_search_obj = epde_alg.EpdeSearch(use_solver=False, boundary=boundary,
                                               dimensionality=dimensionality, coordinate_tensors=grids)

        epde_search_obj.set_moeadd_params(population_size=8, training_epochs=90)
        start = time.time()

        try:
            epde_search_obj.fit(data=u, max_deriv_order=(1, 3),
                                equation_terms_max_number=4, equation_factors_max_number=2,
                                eq_sparsity_interval=(1e-08, 1e-06))
        except Exception as e:
            logging.error(traceback.format_exc())
            population_error += 1
            continue
        end = time.time()
        epde_search_obj.equation_search_results(only_print=True, num=4)
        time1 = end-start

        res = epde_search_obj.equation_search_results(only_print=False, num=4)

        difference_ls = find_coeff_diff(res)
        if len(difference_ls) != 0:
            differences_ls.append(min(difference_ls))
            differences_ls_none.append(min(difference_ls))
            mean_diff_ls += difference_ls
        else:
            differences_ls_none.append(None)

        num_found_eq.append(len(difference_ls))
        print('Overall time is:', time1)
        print(f'Iteration processed: {i+1}/{max_iter_number}\n')
        i += 1
        time_ls.append(time1)

    if write_csv:
        arr = np.array([differences_ls_none, time_ls, num_found_eq])
        arr = arr.T
        df = pd.DataFrame(data=arr, columns=['MAE', 'time', 'number_found_eq'])
        df.to_csv(f'data_kdv/{title}.csv')

    if print_results:
        print('\nTime for every run, s:')
        for item in time_ls:
            print(f'{item: .4f}')

        print()
        print(f'\nAverage time, s: {sum(time_ls) / len(time_ls):.2f}')
        print(f'Average MAE per eq: {sum(mean_diff_ls) / len(mean_diff_ls):.6f}')
        print(f'Average minimum MAE per run: {sum(differences_ls) / max_iter_number:.6f}')
        print(f'Average # of found eq per run: {sum(num_found_eq) / max_iter_number:.2f}')
        print(f"Runs where eq was not found: {num_found_eq.count(0)}")
        print(f"Num of population error occurrence: {population_error}")
