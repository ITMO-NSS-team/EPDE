from sympy import Mul, Symbol
import numpy as np


# def get_csym_pool(tsym: list, csym: list, pool_ls: list, left_side_name: tuple[str]):
#
#     symnet_dict = dict(zip(tsym, csym))
#     csym_pool_ls = []
#     for tsym_pool_el in pool_ls:
#         csym_pool_ls.append(symnet_dict.get(tsym_pool_el, 1e-6))
#
#     left_idx = pool_ls.index(left_to_sym(left_side_name))
#     csym_pool_ls[left_idx] = left_csym(csym)
#     return csym_pool_ls

def get_csym_pool(tsym: list, csym: list, pool_ls: list, left_side_name: str):

    symnet_dict = dict(zip(tsym, csym))
    csym_pool_ls = []
    for tsym_pool_el in pool_ls:
        csym_pool_ls.append(symnet_dict.get(tsym_pool_el, 1e-6))

    left_idx = pool_ls.index(Symbol(left_side_name))
    csym_pool_ls[left_idx] = left_csym(csym)
    return csym_pool_ls


def left_csym(csym):
    if len(csym) > 1:
        return (np.fabs(csym[0]) + np.fabs(csym[1])) / 2
    else:
        return csym[0]


def left_to_sym(left_term: tuple[str]):
    term_symbolic = list(map(lambda u: Symbol(u), left_term))
    return Mul(*term_symbolic)


def cast_to_symbols(pool_names: list[tuple[str]]):

    pool_ls = []
    for name in pool_names:
        term_symbolic = list(map(lambda u: Symbol(u), name))
        pool_ls.append(Mul(*term_symbolic))
    return pool_ls


def to_symbolic(term):
    if type(term.cache_label[0]) == tuple:
        labels = []
        for label in term.cache_label:
            labels.append(str(label[0]))
        symlabels = list(map(lambda token: Symbol(token), labels))
        return Mul(*symlabels)
    else:
        return Symbol(str(term.cache_label[0]))


def get_cross_distr(custom_cross_prob, start_idx, end_idx_exclude):
    mmf = 2.4
    values = list(custom_cross_prob.values())
    csym_arr = np.fabs(np.array(values))

    if np.max(csym_arr) / np.min(csym_arr) > 2.6:
        min_max_coeff = mmf * np.min(csym_arr) - np.max(csym_arr)
        smoothing_factor = min_max_coeff / (min_max_coeff - (mmf - 1) * np.average(csym_arr))
        uniform_csym = np.array([np.sum(csym_arr) / len(csym_arr)] * len(csym_arr))

        smoothed_array = (1 - smoothing_factor) * csym_arr + smoothing_factor * uniform_csym
        inv = 1 / smoothed_array
    else:
        inv = 1 / csym_arr
    inv_norm = inv / np.sum(inv)

    return dict(zip([i for i in range(start_idx, end_idx_exclude)], inv_norm.tolist()))