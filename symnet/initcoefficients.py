import symnet.expr as expr
from symnet.preproc_input import prepare_batches
from symnet.prepare_left_side import init_left_term,get_left_pool
from symnet.initparams import initexpr
import torch
from symnet.loss import loss
from symnet.preproc_output import *

import seaborn as sns
import matplotlib.pyplot as plt
from symnet.print_selection_info import ModelsInfo


def clean_names(left_name, names: list):
    new_names = names.copy()
    idx = None
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
    last_step_loss = loss(model, y_train, x_train, block=1, sparsity=0.0)

    return model, last_step_loss.item()







# def right_matrices_coef(matrices, names: list[str], csym, tsym):
#     token_matrix = {}
#     for i in range(len(names)):
#         token_matrix[Symbol(names[i])] = matrices[i]
#
#     right_side = []
#     for i in range(len(csym)):
#         total_mx = 1
#         if type(tsym[i]) == Mul:
#             if tsym[i] == Mul(Symbol("u"), Symbol("du/dx2")):
#                 u_ux_ind = i
#             lbls = tsym[i].args
#             for lbl in lbls:
#                 if type(lbl) == Symbol:
#                     total_mx *= token_matrix.get(lbl)
#                 else:
#                     for j in range(lbl.args[1]):
#                         total_mx *= token_matrix.get(lbl.args[0])
#         elif type(tsym[i]) == Symbol:
#             total_mx *= token_matrix.get(tsym[i])
#         elif type(tsym[i]) == Pow:
#             for j in range(tsym[i].args[1]):
#                 total_mx *= token_matrix.get(tsym[i].args[0])
#         total_mx *= csym[i]
#         right_side.append(total_mx)
#
#     u_ux = 1
#     for lbl in (Symbol("u"), Symbol("du/dx2")):
#         u_ux *= token_matrix.get(lbl)
#     right_u_ux = csym[u_ux_ind] * u_ux
#     diff1 = np.fabs((np.abs(csym[u_ux_ind]) - 1) * u_ux)
#     return right_side, right_u_ux, u_ux


# def select_model1(input_names, left_pool, u, derivs, shape, sparsity, additional_tokens):
#     models = []
#     losses = []
#     for left_side_name in left_pool:
#         m_input_names, idx = clean_names(left_side_name, input_names)
#         x_train, y_train = prepare_batches(u, derivs, shape, idx, additional_tokens=additional_tokens)
#         model, last_loss = train_model(m_input_names, x_train, y_train, sparsity)
#
#         tsym, csym = model.coeffs(calprec=16)
#         losses.append(last_loss)
#         models.append(model)
#
#     idx = losses.index(min(losses))
#     return models[idx], left_pool[idx]

def select_model(input_names, left_pool, u, derivs, shape, additional_tokens):
    models, losses, left_sides = [], [], []
    info = ModelsInfo()
    for left_side_name in left_pool:
        for sparsity in [0.001, 0.0000001]:
            m_input_names, idx = clean_names(left_side_name, input_names)
            x_train, y_train = prepare_batches(u, derivs, shape, idx, additional_tokens=additional_tokens)
            model, last_loss = train_model(m_input_names, x_train, y_train, sparsity)

            losses.append(last_loss)
            models.append(model)
            left_sides.append(left_side_name)

            info.selection_info(model, last_loss, sparsity, left_side_name)

    info.print_best()
    idx = losses.index(min(losses))
    return models[idx], left_sides[idx]


def save_fig(csym, add_left=True):
    distr = np.fabs(csym.copy())
    if add_left:
        distr = np.append(distr, (distr[0] + distr[1]) / 2)
    distr.sort()
    distr = distr[::-1]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_ylim(0, np.max(distr) + 0.01)
    sns.barplot(x=np.arange(len(distr)), y=distr, orient="v", ax=ax)
    plt.grid()
    # plt.show()
    plt.yticks(fontsize=50)
    plt.savefig(f'symnet_distr{len(distr)}.png', transparent=True)


def get_csym_tsym(u, derivs, shape, input_names, pool_names, additional_tokens=None,
                  max_deriv_order=None):
    """
    Can process only one variable! (u)
    """

    left_pool = get_left_pool(max_deriv_order)
    model, left_side_name = select_model(input_names, left_pool, u, derivs, shape, sparsity, additional_tokens)
    tsym, csym = model.coeffs(calprec=16)
    # save_fig(csym)
    pool_sym_ls = cast_to_symbols(pool_names)
    csym_pool_ls = get_csym_pool(tsym, csym, pool_sym_ls, left_side_name)
    # save_fig(np.array(csym_pool_ls), add_left=False)
    return dict(zip(pool_sym_ls, csym_pool_ls)), pool_sym_ls
