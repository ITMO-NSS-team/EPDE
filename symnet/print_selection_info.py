import numpy as np
from decimal import Decimal
import os
import sys

# def print_best_info(sparsity, left_side_name="du/dx1")
class ModelsInfo():
    def __init__(self):
        print("\n")
        self.sparsity_ls = []
        self.eq_type = []
        self.pos_ls = []
        self.coef_ls = []
        self.losses = []


    def selection_info(self, model, final_loss, sparsity, left_side_name="du/dx1"):
        tsym, csym = model.coeffs(calprec=16)
        left_names, right_names, coeffs = init_ideal_coeffs()
        mae, final_places = mae_and_positions(right_names, coeffs, tsym, csym, left_side_name, left_names)
        shd = calc_shd(final_places, csym, len(right_names))

        final_coefs = [csym[i] for i in final_places]
        coef_str = ['%.2e' % Decimal(c) for c in final_coefs]

        if left_side_name == "du/dx1":
            time_der = "u_t"
        else:
            time_der = "u_tt"

        print(f"Sparsity & time der.: {sparsity: .1e}, {time_der}; MAE: {mae:.3f}; idxs: {final_places}; coefs: {coef_str}; "
              f"loss: {final_loss: .2e}; shd: {shd}")

        self.sparsity_ls. append(sparsity)
        self.eq_type.append(left_side_name)
        self.pos_ls.append(final_places)
        self.coef_ls.append(coef_str)
        self.losses.append(final_loss)

    def print_best(self):
        idx = self.losses.index(min(self.losses))
        print(f"\n Lambda & t.d.: {self.sparsity_ls[idx]: .1e}, {self.eq_type[idx]}; idxs: {self.pos_ls[idx]}; "
              f"coefs: {self.coef_ls[idx]}; loss: {self.losses[idx]: .2e}")
        print("\n")


def mae_and_positions(right_side_name, coefficients, tsym, csym, left_calc, left_true):
    MAE = 0
    final_places_ls = []
    if left_calc != left_true:
        MAE += 2.
    for j in range(len(tsym)):
        in_equation = False
        for i in range(len(right_side_name)):
            if str(tsym[j]) == right_side_name[i]:
                MAE += np.fabs(coefficients[i] - csym[j])
                in_equation = True
                final_places_ls.append(j)
                break
        if not in_equation:
            MAE += np.fabs(0. - csym[j])
    MAE = MAE / (len(tsym) + 1)
    return MAE, final_places_ls

def init_ideal_coeffs():
    name = (sys.argv[0]).split("/")[-1]
    if name == 'noised_wave.py' or "experiment_wave.py":
        coeffs = [0.04]
        right_names = ['d^2u/dx2^2']
        left_names = "d^2u/dx1^2"
    elif name == 'noised_burgers_sindy.py': # 256 x 101
        coeffs = [-1., 0.1]
        right_names = ['du/dx2*u', 'd^2u/dx2^2']
        left_names = "du/dx1"
    elif name == 'noised_kdv.py':
        left_names = "du/dx1"
        coeffs = [6., 1., 1.]
        right_names = ['du/dx2*u', 'd^3u/dx2^3', 'cos(t)sin(x)']
    elif name == 'noised_kdv_sindy.py': # 512 x 201
        left_names = "du/dx1"
        coeffs = [-6., -1.]
        right_names = ['du/dx2*u', 'd^3u/dx2^3']
    elif name == "noised_burgers.py":
        coeffs = [-1.]
        right_names = ['du/dx2*u']
        left_names = "du/dx1"
    else:
        raise NameError('Wrong type of equation')
    return left_names, right_names, coeffs


def calc_shd(idxs: list, coefs: list, true_num_terms: int):
    idxs.sort()
    shd = true_num_terms - len(idxs)

    set_idxs_all = set([i for i in range(len(coefs))])
    idxs_cut = list(set_idxs_all.difference(set(idxs)))
    for i in idxs_cut:
        if np.abs(coefs[i]) > 1e-6:
            shd += 1
    for i in idxs:
        if np.abs(coefs[i]) < 1e-6:
            shd += 1
    return shd