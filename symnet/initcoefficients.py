import symnet.expr as expr
from symnet.preproc_input import prepare_batches
from symnet.prepare_left_side import init_left_term,get_left_pool
from symnet.initparams import initexpr
import torch
from symnet.loss import loss
from symnet.preproc_output import *

import seaborn as sns
import matplotlib.pyplot as plt


def clean_names(left_name, names: list):
    new_names = names.copy()
    idx = None
    # if len(left_name) == 1:
    #     lname = left_name[0]
    if left_name in new_names:
        idx = new_names.index(left_name)
        new_names.remove(left_name)

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
    last_step_loss = loss(model, y_train, x_train, block=1, sparsity=sparsity)

    return model, last_step_loss.item()


def select_model(input_names, left_pool, u, derivs, shape, sparsity, additional_tokens):
    models = []
    losses = []
    for left_side_name in left_pool:
        m_input_names, idx = clean_names(left_side_name, input_names)
        x_train, y_train = prepare_batches(u, derivs, shape, idx, additional_tokens=additional_tokens)
        model, last_loss = train_model(m_input_names, x_train, y_train, sparsity)
        losses.append(last_loss)
        models.append(model)

    idx = losses.index(min(losses))
    return models[idx], left_pool[idx]


def get_csym_tsym(u, derivs, shape, input_names, pool_names, sparsity=0.001, additional_tokens=None,
                  max_deriv_order=None):
    """
    Can process only one variable! (u)
    """
    # TODO: SymNet имеет 4 todo (+, pool_terms, preproc_input)

    # TODO: если в левой части e.g. d^2u/dx2^2, то как получить в правой слагаемое d^2u/dx2^2 * u?

    left_pool = get_left_pool(max_deriv_order)
    model, left_side_name = select_model(input_names, left_pool, u, derivs, shape, sparsity, additional_tokens)
    tsym, csym = model.coeffs(calprec=16)
    pool_sym_ls = cast_to_symbols(pool_names)

    csym_pool_ls = get_csym_pool(tsym, csym, pool_sym_ls, left_side_name)
    return dict(zip(pool_sym_ls, csym_pool_ls)), pool_sym_ls
