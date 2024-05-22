import pandas as pd
import numpy as np
from kdv_init_distrib_sindy import coefficients1, coefficients2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import re


'''

    Analyzes the output of epde, requires pickled objects of SoEq 
    
'''
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


def out_formatting(string, hashable=True):
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
            if hashable:
                terms_dict[hash_term(term)] = coeff
            else:
                terms_dict[tuple(term)] = coeff
    if hashable:
        terms_dict[hash_term(ls_right)] = -1.
    else:
        terms_dict[tuple(ls_right)] = -1.
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


def find_not_found_idxs(error):
    not_found_idxs = []
    for i in range(50):
        with open(f"{PATH}\\dfs{error}_{i}.pickle", 'rb') as f:
            eq = pickle.load(f)
        maes = find_coeff_diff(eq)
        if len(maes) != 0:
            not_found_idxs.append(i)
    return not_found_idxs


def process_term_dict(terms_dict):
    processed_dict = {}
    for key, value in terms_dict.items():
        if np.fabs(value) > 3.5e-5:
            processed_dict[key] = value
    return processed_dict


def round_zero(number):
    n = np.fabs(number)
    if n < 1:
        s = f'{n:.99f}'
        index = re.search('[1-9]', s).start()
        return s[:index+1]
    else:
        return str(np.round(n, 1))


def create_save_str(terms_dict):
    ls_str = []
    bias = None
    for key, value in terms_dict.items():
        if value == -1.0:
            right_key = " * ".join(key)
        elif key == 1:
            bias = f"{round_zero(value)}"
        else:
            ls_str.append(f"{round_zero(value)} {' * '.join(key)}")
    if bias is not None:
        ls_str.append(bias)
    left_side = " + ".join(ls_str)
    eq_text = " = ".join((left_side, right_key))
    return eq_text


def create_save_str1(terms_dict):
    ls_str = []
    bias = None
    for key, value in terms_dict.items():
        if value == -1.0:
            right_key = " * ".join(key)
        elif key == 1:
            bias = f"{value}"
        else:
            ls_str.append(f"{value} {' * '.join(key)}")
    if bias is not None:
        ls_str.append(bias)
    left_side = " + ".join(ls_str)
    eq_text = " = ".join((left_side, right_key))
    return eq_text

def analyze_eqs(idxs, error):
    save_path = os.path.join(Path().absolute(), "data_pysindy_kdv", "equations.txt")
    all_eqs = []
    df_data = []

    magnitudes = [0.023, 0.046, 0.069, 0.092]
    magnames = ["0.023", "0.046", "0.069", "0.092"]
    ls = [1,2,3,4]

    for magnitude, magname, l in zip(magnitudes, magnames, ls):
        print(l)

    for i in idxs:
        with open(f"{PATH}\\dfs{error}_{i}.pickle", 'rb') as f:
            eq = pickle.load(f)

        for soeq in eq[0]:
            eq_text = soeq.vals.chromosome['u'].value.text_form
            terms_dict = out_formatting(eq_text, hashable=False)
            terms_dict = process_term_dict(terms_dict)
            eq_text = create_save_str(terms_dict)
            # correct1 = "0.1 du/dx1 + 0.1 d^3u/dx2^3 = u * du/dx2" # 0.03669031837099942 0.03665517683285185
            # correct2 = "0.1 d^3u/dx2^3 + 0.1 du/dx1 = u * du/dx2"
            # miss1 = "0.9 du/dx2 + 0.0004 = du/dx1"     # 1.8262033132949265
            # if eq_text == correct1 or eq_text == correct2:
            #     print()
            # if eq_text == miss1:
            #     print()

            df_data.append(eq_text)
            all_eqs.append(eq_text + "\n")
    df = pd.Series(df_data)
    unique_eqs = df.value_counts()
    # unique_eqs.to_csv("a_from_A.csv")
    print()
    # with open(save_path, "w") as f:
    #     f.writelines(all_eqs)


if __name__ == '__main__':
    PATH = os.path.join(Path().absolute(), "data_pysindy_kdv", "equations")
    error = "4e-5_142"
    # not_found_idxs = find_not_found_idxs(error)
    not_found_idxs = [0, 1, 2, 4, 5, 6, 10, 11, 14, 16, 17, 18, 19, 20, 21, 23, 24, 25, 28, 29, 31, 32, 34, 35, 36, 37, 38, 39, 40, 42,
     43, 44, 45, 46, 48, 49]
    # not_found_idxs = [3, 7, 8, 9, 12, 13, 15, 22, 26, 27, 30, 33, 41, 47]
    analyze_eqs(not_found_idxs, error)
    print()
