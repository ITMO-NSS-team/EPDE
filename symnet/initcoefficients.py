import symnet.expr as expr
from symnet.preproc_input import prepare_batches
from symnet.initparams import initexpr
import torch
from symnet.loss import loss
from symnet.preproc_output import *

import seaborn as sns
import matplotlib.pyplot as plt


def clean_names(left_name, names: list):
    new_names = names.copy()
    idx = None
    if len(left_name) == 1:
        lname = left_name[0]
        if lname in new_names:
            idx = new_names.index(lname)
            new_names.remove(lname)

    return new_names, idx


def train_model(input_names, x_train, y_train, sparsity):

    def closure():
        lbfgs.zero_grad()
        tloss = loss(model, y_train, x_train, block=1, sparsity=sparsity)
        tloss.backward()
        return tloss

    model = expr.poly(2, channel_num=len(input_names), channel_names=input_names)
    initexpr(model)
    lbfgs = torch.optim.LBFGS(params=model.parameters(), max_iter=2000, line_search_fn='strong_wolfe')
    model.train()
    lbfgs.step(closure)

    return model


def get_csym_tsym(u, derivs, shape, input_names, pool_names, sparsity=0.001, additional_tokens=None):
    """
    Can process only one variable! (u)
    """
    # TODO: SymNet имеет 4 todo (+, pool_terms, preproc_input)

    # TODO: что делать с left_side_name? (случ. генер.?)
    # left_side_name = ('du/dx1', )
    # TODO: если в левой части e.g. d^2u/dx2^2, то как получить в правой слагаемое d^2u/dx2^2 * u?
    left_side_name = ('d^2u/dx2^2',)

    input_names, idx = clean_names(left_side_name, input_names)
    x_train, y_train = prepare_batches(u, derivs, shape, idx, additional_tokens=additional_tokens)

    model = train_model(input_names, x_train, y_train, sparsity)

    tsym, csym = model.coeffs(calprec=16)
    pool_sym_ls = cast_to_symbols(pool_names)

    csym_pool_ls = get_csym_pool(tsym, csym, pool_sym_ls, left_side_name)
    return dict(zip(pool_sym_ls, csym_pool_ls)), pool_sym_ls
