import time
import numpy as np
import pandas as pd
import epde.interface.interface as epde_alg
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import traceback
import logging
import os
from pathlib import Path
import pickle


def find_coeff_diff(res, coefficients: dict):
    differences = []

    for pareto_front in res:
        for soeq in pareto_front:
            if soeq.obj_fun[0] < 10:
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
        if coefficients.get(term_hash) != 0.0 and abs(terms_dict.get(term_hash) - coefficients.get(term_hash)) < 0.1:
            eq_found += 1

    mae /= len(terms_dict)
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
    path_full = os.path.join(Path().absolute().parent, "data_burg", "burgers.mat")
    burg = loadmat(path_full)
    t = np.ravel(burg['t'])
    x = np.ravel(burg['x'])
    u_init = np.real(burg['usol'])
    u_init = np.transpose(u_init)
    dimensionality = u_init.ndim

    boundary = 10
    grids = np.meshgrid(t, x, indexing='ij')

    terms = [('u',), ('du/dx1',), ('du/dx2',), ('d^2u/dx2^2',), ('u', 'du/dx1'), ('u', 'du/dx2'), ('u', 'd^2u/dx2^2'),
             ('du/dx1', 'du/dx2'), ('du/dx1', 'd^2u/dx2^2'), ('du/dx2', 'd^2u/dx2^2')]
    hashed_ls = [hash_term(term) for term in terms]
    coefficients = dict(zip(hashed_ls, [0., -1., 0., 0.1, 0., -1., 0., 0., 0., 0.]))
    coefficients[1] = 0.

    ''' Parameters of the experiment '''
    write_csv = True
    print_results = True
    max_iter_number = 50

    draw_not_found = []
    draw_time = []
    draw_avgmae = []
    start_gl = time.time()
    # magnitudes = [0.0075, 0.015, 0.0225, 0.03]
    # magnames = ["0.0075", "0.015", "0.0225", "0.03"]

    magnitudes = [0, 0.0075, 0.015, 0.0225, 0.03]
    magnames = ["0", "0.0075", "0.015", "0.0225", "0.03"]
    mmfs = [3.2, 3.2, 3.1, 2.8, 2.8]

    for magnitude, magname, mmf in zip(magnitudes, magnames, mmfs):
        title = f'dfs{magnitude}_tuned'

        time_ls = []
        differences_ls = []
        mean_diff_ls = []
        num_found_eq = []
        differences_ls_none = []
        i = 0
        population_error = 0
        while i < max_iter_number:
            if magnitude != 0:
                u = u_init + np.random.normal(scale=magnitude * np.abs(u_init), size=u_init.shape)
            else:
                u = u_init
            epde_search_obj = epde_alg.EpdeSearch(use_solver=False, boundary=boundary,
                                                  dimensionality=dimensionality, coordinate_tensors=grids)

            epde_search_obj.set_moeadd_params(population_size=8, training_epochs=7)
            start = time.time()
            try:
                epde_search_obj.fit(data=u, max_deriv_order=(1, 2),
                                    equation_terms_max_number=3, equation_factors_max_number=2,
                                    eq_sparsity_interval=(1e-08, 1e-1), mmf=mmf)
            except Exception as e:
                logging.error(traceback.format_exc())
                population_error += 1
                continue
            end = time.time()
            epde_search_obj.equation_search_results(only_print=True, num=4)
            time1 = end-start

            res = epde_search_obj.equation_search_results(only_print=False, num=4)
            difference_ls = find_coeff_diff(res, coefficients)

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
            df.to_csv(os.path.join(Path().absolute().parent, "data_pysindy_burg", f"{title}.csv"))
        if print_results:
            print()
            print(f'\nAverage time, s: {sum(time_ls) / len(time_ls):.2f}')
            if len(mean_diff_ls) != 0:
                print(f'Average MAE per eq: {sum(mean_diff_ls) / len(mean_diff_ls):.6f}')
                print(
                    f'Average minimum MAE per run: {sum(differences_ls) / (max_iter_number - num_found_eq.count(0)):.6f}')
            else:
                print("Equation was not found in any run")
            print(f'Average # of found eq per run: {sum(num_found_eq) / max_iter_number:.2f}')
            print(f"Runs where eq was not found: {num_found_eq.count(0)}")
            print(f"Num of population error occurrence: {population_error}")

        if len(mean_diff_ls) != 0:
            draw_avgmae.append(sum(differences_ls) / (max_iter_number - num_found_eq.count(0)))
        else:
            draw_avgmae.append(0.08)
        draw_not_found.append(num_found_eq.count(0))
        draw_time.append(sum(time_ls) / len(time_ls))

    end_gl = time.time()
    print(f"Overall time: {end_gl - start_gl:.2f}, s.")
    plt.title("SymNet")
    plt.plot(magnitudes, draw_not_found, linewidth=2, markersize=9, marker='o')
    plt.ylabel("No. runs with not found eq.")
    plt.xlabel("Magnitude value")
    plt.grid()
    plt.show()

    plt.plot(magnitudes, draw_time, linewidth=2, markersize=9, marker='o')
    plt.title("SymNet")
    plt.ylabel("Time, s.")
    plt.xlabel("Magnitude value")
    plt.grid()
    plt.show()

    plt.plot(magnitudes, draw_avgmae, linewidth=2, markersize=9, marker='o')
    plt.title("SymNet")
    plt.ylabel("Average MAE")
    plt.xlabel("Magnitude value")
    plt.grid()
    plt.show()
