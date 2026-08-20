"""Benchmark scenarios, transcribed from EPDE's own functional-test suite.

Each entry keeps the data loading, token pool and ``fit`` hyper-parameters of
``tests/functional/scenarios/<name>/<name>.py`` so the comparison runs
on the framework's own examples rather than on invented problems.
"""

import os

import numpy as np

from _common import DATA_DIR


# --------------------------------------------------------------------- helpers

def _noise(data, level, rng):
    if not level:
        return data
    return level * 0.01 * np.std(data) * rng.normal(size=np.shape(data)) + data


# ------------------------------------------------------------------- scenarios

def wave():
    folder = os.path.join(DATA_DIR, 'wave')
    shape = 80
    data = np.loadtxt(os.path.join(folder, 'wave_sln_80.csv'), delimiter=',').T
    t = np.linspace(0, 1, shape + 1)
    x = np.linspace(0, 1, shape + 1)
    grids = np.meshgrid(t, x, indexing='ij')
    return dict(
        name='wave',
        data=data,
        coordinate_tensors=grids,
        variable_names=['u'],
        boundary=20,
        search_kwargs=dict(use_pic=True),
        fit_kwargs=dict(max_deriv_order=(2, 3), equation_terms_max_number=5,
                        data_fun_pow=3,
                        equation_factors_max_number={'factors_num': [1, 2],
                                                     'probas': [0.65, 0.35]},
                        eq_sparsity_interval=(1e-6, 1e-4)),
        tokens=lambda: [],
        pop_size=16, epochs=5,
        ground_truth=['0.04 * d^2u/dx1^2{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}'],
    )


def burgers():
    from scipy.io import loadmat
    folder = os.path.join(DATA_DIR, 'burgers')
    burg = loadmat(os.path.join(folder, 'burgers.mat'))
    t = np.ravel(burg['t'])
    x = np.ravel(burg['x'])
    data = np.transpose(np.real(burg['usol']))
    grids = np.meshgrid(t, x, indexing='ij')
    return dict(
        name='burgers',
        data=data,
        coordinate_tensors=grids,
        variable_names=['u'],
        boundary=10,
        search_kwargs=dict(use_pic=True),
        fit_kwargs=dict(max_deriv_order=(2, 3), equation_terms_max_number=5,
                        data_fun_pow=3,
                        equation_factors_max_number={'factors_num': [1, 2],
                                                     'probas': [0.65, 0.35]},
                        eq_sparsity_interval=(1e-5, 1e2)),
        tokens=lambda: [],
        pop_size=16, epochs=5,
        ground_truth=['-1.0 * u{power: 1.0} * du/dx1{power: 1.0} '
                      '+ 0.01 * d^2u/dx1^2{power: 1.0} + 0.0 = du/dx0{power: 1.0}'],
    )


def kdv():
    from epde import CustomTokens
    from epde.evaluators import CustomEvaluator
    folder = os.path.join(DATA_DIR, 'kdv')
    data = np.loadtxt(os.path.join(folder, 'data.csv'), delimiter=',').T
    shape = 80
    t = np.linspace(0, 1, shape + 1)
    x = np.linspace(0, 1, shape + 1)
    grids = np.meshgrid(t, x, indexing='ij')

    def tokens():
        fun = {'cos(t)sin(x)': lambda *g, **kw: (np.cos(g[0]) * np.sin(g[1])) ** kw['power']}
        evaluator = CustomEvaluator(fun, eval_fun_params_labels=['power'])
        return [CustomTokens(token_type='trigonometric',
                             token_labels=['cos(t)sin(x)'],
                             evaluator=evaluator,
                             params_ranges={'power': (1, 1)},
                             params_equality_ranges=None,
                             meaningful=True, unique_token_type=False)]

    return dict(
        name='kdv',
        data=data,
        coordinate_tensors=grids,
        variable_names=['u'],
        boundary=10,
        search_kwargs=dict(use_pic=True),
        fit_kwargs=dict(max_deriv_order=(2, 3), equation_terms_max_number=10,
                        data_fun_pow=3,
                        equation_factors_max_number={'factors_num': [1, 2],
                                                     'probas': [0.65, 0.35]},
                        eq_sparsity_interval=(1e-5, 1e-2)),
        tokens=tokens,
        pop_size=16, epochs=5,
        ground_truth=['-6.0 * du/dx1{power: 1.0} * u{power: 1.0} '
                      '+ -1.0 * d^3u/dx1^3{power: 1.0} '
                      '+ 1.0 * cos(t)sin(x){power: 1.0} + 0.0 = du/dx0{power: 1.0}'],
    )


def allen_cahn():
    folder = os.path.join(DATA_DIR, 'ac')
    data = np.load(os.path.join(folder, 'ac_data.npy'))
    t = np.linspace(0.0, 1.0, 51)
    x = np.linspace(-1.0, 0.984375, 128)
    grids = np.meshgrid(t, x, indexing='ij')
    return dict(
        name='allen_cahn',
        data=data,
        coordinate_tensors=grids,
        variable_names=['u'],
        boundary=10,
        search_kwargs=dict(use_pic=True),
        fit_kwargs=dict(max_deriv_order=(2, 3), equation_terms_max_number=5,
                        data_fun_pow=3,
                        equation_factors_max_number={'factors_num': [1, 2],
                                                     'probas': [0.65, 0.35]},
                        eq_sparsity_interval=(1e-12, 1e0)),
        tokens=lambda: [],
        pop_size=16, epochs=5,
        ground_truth=['0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} '
                      '+ 5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'],
    )


def van_der_pol():
    from epde import TrigonometricTokens, GridTokens
    folder = os.path.join(DATA_DIR, 'vdp')
    data = np.load(os.path.join(folder, 'vdp_data.npy'))
    t = np.linspace(0, 1, len(data))
    return dict(
        name='van_der_pol',
        data=[data],
        coordinate_tensors=[t],
        variable_names=['u'],
        boundary=10,
        search_kwargs=dict(use_pic=True),
        fit_kwargs=dict(max_deriv_order=(2,), equation_terms_max_number=5,
                        data_fun_pow=3,
                        equation_factors_max_number={'factors_num': [1, 2],
                                                     'probas': [0.65, 0.35]},
                        eq_sparsity_interval=(1e-5, 1e0)),
        tokens=lambda: [TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8), dimensionality=0),
                        GridTokens(['x_0'], dimensionality=0, max_power=2)],
        pop_size=16, epochs=5,
        ground_truth=['-0.2 * u{power: 2.0} * du/dx0{power: 1.0} '
                      '+ 0.2 * du/dx0{power: 1.0} + -1.0 * u{power: 1.0} '
                      '+ -0.0 = d^2u/dx0^2{power: 1.0}'],
    )


def lotka_volterra():
    from epde import TrigonometricTokens, GridTokens
    folder = os.path.join(DATA_DIR, 'lv')
    t = np.load(os.path.join(folder, 't_20.npy'))
    data = np.load(os.path.join(folder, 'data_20.npy'))
    end = 150
    t = t[:end]
    u = data[:end, 0]
    v = data[:end, 1]
    return dict(
        name='lotka_volterra',
        data=[u, v],
        coordinate_tensors=[t],
        variable_names=['u', 'v'],
        boundary=15,
        search_kwargs=dict(use_pic=True),
        fit_kwargs=dict(max_deriv_order=1, equation_terms_max_number=7,
                        data_fun_pow=3,
                        equation_factors_max_number={'factors_num': [1, 2],
                                                     'probas': [0.8, 0.2]},
                        eq_sparsity_interval=(1e-8, 1e0)),
        tokens=lambda: [TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8), dimensionality=0),
                        GridTokens(['x_0'], dimensionality=0, max_power=2)],
        pop_size=16, epochs=5,
        ground_truth=[
            '0.6666666666666666 * u{power: 1.0} '
            '+ -1.3333333333333333 * u{power: 1.0} * v{power: 1.0} + 0.0 = du/dx0{power: 1.0}',
            '1.0 * u{power: 1.0} * v{power: 1.0} + -1.0 * v{power: 1.0} + 0.0 = dv/dx0{power: 1.0}'],
    )


ALL = {
    'wave': wave,
    'burgers': burgers,
    'kdv': kdv,
    'allen_cahn': allen_cahn,
    'van_der_pol': van_der_pol,
    'lotka_volterra': lotka_volterra,
}
