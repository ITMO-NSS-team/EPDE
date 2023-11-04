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
