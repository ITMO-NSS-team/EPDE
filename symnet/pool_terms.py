from symnet.initcoefficients import get_csym_tsym
from sympy import Symbol, Mul
# import seaborn as sns
# import matplotlib.pyplot as plt
import itertools


class PoolTerms:
    def __init__(self, max_factors_in_term, families):

        self.pool_sym_dict = None
        self.pool_sym_ls = None
        self.pool_dict = None

        token_ls = []
        for family in families:
            token_ls += family.tokens
        self.term_ls = []
        for i in range(1, max_factors_in_term + 1):
            self.term_ls += list(itertools.combinations(token_ls, i))

    def set_initial_distr(self, u, derivs, shape, names, families, grids, max_deriv_order):
        if len(families) == 1:
            self.pool_dict, self.pool_sym_ls = \
                        get_csym_tsym(u, derivs, shape, names, pool_names=self.term_ls,
                                      max_deriv_order=max_deriv_order)
        else:
            additional_tokens = _prepare_additional_tokens(families, grids)
            names = _prepare_names(names, families)
            self.pool_dict, self.pool_sym_ls = \
                get_csym_tsym(u, derivs, shape, names,
                              pool_names=self.term_ls, additional_tokens=additional_tokens,
                              max_deriv_order=max_deriv_order)
        self.pool_sym_dict = dict(zip(self.pool_sym_ls, self.term_ls))


def _prepare_names(names, families):
    names_c = names.copy()
    for i in range(1, len(families)):
        names_c += families[i].tokens
    return names_c


# TODO: Обработать общий случай additional_tokens
def _prepare_additional_tokens(families, grids):
    mx_ls = []
    for i in range(1, len(families)):
        assert len(families[i]) == 1, "Can't process family consisting from more than one token"

        name = families[i].tokens[0]
        fun = families[i]._evaluator._evaluator.evaluation_functions.get(name)
        mx = fun(*grids, **{'power': families[i].token_params.get('power')[0]})
        mx_ls.append(mx)
    return mx_ls


def to_symbolic(term):
    if type(term.cache_label[0]) == tuple:
        labels = []
        for label in term.cache_label:
            labels.append(str(label[0]))
        symlabels = list(map(lambda token: Symbol(token), labels))
        return Mul(*symlabels)
    else:
        return Symbol(str(term.cache_label[0]))
