import torch
import numpy as np


def get_left_pool(max_deriv_order):
    left_pool = ["du/dx1"]
    if max_deriv_order[0] > 1:
        for i in range(2, max_deriv_order[0]+1):
            left_pool.append(f"d^{i}u/dx1^{i}")
    return left_pool


def init_left_term(families):
    labels = families[0].tokens.copy()
    labels.remove('u')
    return (np.random.choice(labels), )
