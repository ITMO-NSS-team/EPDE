import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import pickle
from typing import Tuple, List
import numpy as np

import copy
from epde.interface.token_family import TFPool
from epde.structure.main_structures import SoEq, Chromosome
from epde.interface.prepared_tokens import CustomTokens, PhasedSine1DTokens, ConstantToken, CustomEvaluator
from epde.interface.equation_translator import translate_equation
from epde.interface.interface import EpdeSearch

from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.sparsity import LASSOSparsity

from epde.operators.utils.operator_mappers import map_operator_between_levels
import epde.operators.common.fitness as fitness
from epde.operators.utils.template import CompoundOperator

from epde import TrigonometricTokens, GridTokens, CacheStoredTokens
import epde.globals as global_var

import scipy.io as scio


def load_pretrained_PINN(ann_filename):
    try:
        with open(ann_filename, 'rb') as data_input_file:
            data_nn = pickle.load(data_input_file)
    except FileNotFoundError:
        print('No model located, proceeding with ann approx. retraining.')
        data_nn = None
    return data_nn


def noise_data(data, noise_level):
    # add noise level to the input data
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def compare_equations(correct_symbolic: str, eq_incorrect_symbolic: str,
                      search_obj: EpdeSearch, all_vars: List[str] = ['u', ]) -> bool:
    metaparams = {('sparsity', var): {'optimizable': False, 'value': 1E-6} for var in all_vars}

    correct_eq = translate_equation(correct_symbolic, search_obj.pool, all_vars=all_vars)
    for var in all_vars:
        correct_eq.vals[var].main_var_to_explain = var
        correct_eq.vals[var].metaparameters = metaparams
        correct_eq.vals[var].weights_internal = np.ones(len(correct_eq.vals[var].structure) - 1)
        correct_eq.vals[var].weights_internal_evald = True
    print(correct_eq.text_form)

    incorrect_eq = translate_equation(eq_incorrect_symbolic, search_obj.pool,
                                      all_vars=all_vars)  # , all_vars = ['u', 'v'])
    for var in all_vars:
        incorrect_eq.vals[var].main_var_to_explain = var
        incorrect_eq.vals[var].metaparameters = metaparams
        incorrect_eq.vals[var].weights_internal = np.ones(len(incorrect_eq.vals[var].structure) - 1)
        incorrect_eq.vals[var].weights_internal_evald = True
    print(incorrect_eq.text_form)

    fit_operator.apply(correct_eq, {})
    fit_operator.apply(incorrect_eq, {})
    print([[correct_eq.vals[var].fitness_value, incorrect_eq.vals[var].fitness_value] for var in all_vars])
    print([[correct_eq.vals[var].coefficients_stability, incorrect_eq.vals[var].coefficients_stability] for var in
           all_vars])
    print([[correct_eq.vals[var].aic, incorrect_eq.vals[var].aic] for var in all_vars])

    # print([correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability for var in all_vars])
    return all([correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability for var in
                all_vars])

def create_equation_from_str(eq_str, target_var, base_pool, all_vars):
    # Отладочная информация
    print(f"\n[DEBUG] target_var = {target_var}, eq_str = {eq_str}")
    print("[DEBUG] Families in pool:")
    for fam in base_pool.families:
        var_name = getattr(fam, 'variable', None)
        tokens = getattr(fam, 'tokens', [])
        print(f"  variable={var_name}, tokens={tokens}, demands_equation={fam.status.get('demands_equation', False)}")

    # Сохраняем оригинальные состояния demands_equation для семейств других переменных
    original_states = {}
    for fam in base_pool.families:
        if hasattr(fam, 'variable') and fam.variable is not None and fam.variable != target_var:
            original_states[fam] = fam.status.get('demands_equation', False)
            fam.status['demands_equation'] = False
    try:
        soeq = translate_equation(eq_str, base_pool, all_vars=[target_var])
        eq = soeq.vals[target_var]
    except Exception as e:
        print(f"[DEBUG] Translation failed: {e}")
        raise
    finally:
        for fam, state in original_states.items():
            fam.status['demands_equation'] = state
    return eq

def compare_systems(correct_symbolic_list, incorrect_symbolic_list, search_obj, all_vars, fit_operator):
    metaparams = {('sparsity', var): {'optimizable': False, 'value': 1E-6} for var in all_vars}

    correct_eqs = {}
    for var, eq_str in zip(all_vars, correct_symbolic_list):
        eq = create_equation_from_str(eq_str, var, search_obj.pool, all_vars)
        eq.main_var_to_explain = var
        eq.metaparameters = metaparams
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        correct_eqs[var] = eq

    correct_system = SoEq(search_obj.pool, metaparams)
    correct_system.vals = Chromosome(correct_eqs, {})
    correct_system.moeadd_set = True
    print("Correct system:")
    print(correct_system.text_form)

    incorrect_eqs = {}
    for var, eq_str in zip(all_vars, incorrect_symbolic_list):
        eq = create_equation_from_str(eq_str, var, search_obj.pool, all_vars)
        eq.main_var_to_explain = var
        eq.metaparameters = metaparams
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        incorrect_eqs[var] = eq

    incorrect_system = SoEq(search_obj.pool, metaparams)
    incorrect_system.vals = Chromosome(incorrect_eqs, {})
    incorrect_system.moeadd_set = True
    print("Incorrect system:")
    print(incorrect_system.text_form)

    fit_operator.apply(correct_system, {})
    fit_operator.apply(incorrect_system, {})

    correct_stability = [correct_system.vals[var].coefficients_stability for var in all_vars]
    incorrect_stability = [incorrect_system.vals[var].coefficients_stability for var in all_vars]
    print("Correct stability:", correct_stability)
    print("Incorrect stability:", incorrect_stability)

    correct_fitness = [correct_system.vals[var].fitness_value for var in all_vars]
    incorrect_fitness = [incorrect_system.vals[var].fitness_value for var in all_vars]
    print("Correct fitness:", correct_fitness)
    print("Incorrect fitness:", incorrect_fitness)

    return all(cs < incs for cs, incs in zip(correct_stability, incorrect_stability))

def prepare_suboperators(fitness_operator: CompoundOperator, operator_params: dict) -> CompoundOperator:
    sparsity = LASSOSparsity()
    coeff_calc = LinRegBasedCoeffsEquation()

    # sparsity = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
    # coeff_calc = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')

    fitness_operator.set_suboperators({'sparsity': sparsity,
                                       'coeff_calc': coeff_calc})
    fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
    fitness_operator.params = operator_params
    fitness_operator = map_operator_between_levels(fitness_operator, 'gene level', 'chromosome level',
                                                   objective_condition=fitness_cond)
    return fitness_operator

def ns_data(filename: str):
    data = scio.loadmat('cylinder_nektar_wake.mat')
    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    t_train = 50

    x = np.unique(X_star[:, 0:1].flatten())  # N x T
    y = np.unique(X_star[:, 1:2].flatten()) # N x T
    t = t_star.flatten()  # N x T

    u = U_star[:, 0, :].T.reshape(*t.shape, *y.shape, *x.shape)[:t_train] # N x T
    v = U_star[:, 1, :].T.reshape(*t.shape, *y.shape, *x.shape)[:t_train] # N x T
    p = P_star.T.reshape(*t.shape, *y.shape, *x.shape)[:t_train]   # N x T

    grids = np.meshgrid(t[:t_train], y, x, indexing = 'ij')  # np.stack(, axis = 2) , axis = 2)
    data = [u, v, p]
    return grids, data


def ns_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    # Базовые строки для переменной u
    eq_u_correct = '0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
    eq_u_incorrect = '4.976781518840499 * u{power: 1.0} + 0.0001 * d^2u/dx1^2{power: 1.0} + -4.974425220166616 * u{power: 3.0} + 0.0 * du/dx1{power: 1.0} * d^2u/dx0^2{power: 1.0} + 0.002262543822130977 = du/dx0{power: 1.0}'

    # Для v и p – заменяем u на v/p и производные соответственно
    def replace_var(s, old, new):
        return s.replace(f'u', new).replace(f'du/dx0', f'd{new}/dx0').replace(f'd^2u/dx1^2', f'd^2{new}/dx1^2')

    eq_v_correct = replace_var(eq_u_correct, 'u', 'v')
    eq_v_incorrect = replace_var(eq_u_incorrect, 'u', 'v')
    eq_p_correct = replace_var(eq_u_correct, 'u', 'p')
    eq_p_incorrect = replace_var(eq_u_incorrect, 'u', 'p')

    correct_system = [eq_u_correct, eq_v_correct, eq_p_correct]
    incorrect_system = [eq_u_incorrect, eq_v_incorrect, eq_p_incorrect]
    grid, data = ns_data(os.path.join(foldername, 'cylinder_nektar_wake.mat'))
    # noised_data = noise_data(data, noise_level)
    # data_nn = load_pretrained_PINN(os.path.join(foldername, 'ac_ann_pretrained.pickle'))

    # print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=grid, verbose_params={'show_iter_idx': True},
                                 device='cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=data, variable_names=["u", "v", "p"], max_deriv_order=(2, 2, 2),
                                additional_tokens=[])#, data_nn=data_nn

    assert compare_systems(correct_system, incorrect_system, epde_search_obj, all_vars=['u', 'v', 'p'],
                           fit_operator=fit_operator)


def ns_discovery(foldername, noise_level):
    grid, data = ns_data(os.path.join(foldername, 'cylinder_nektar_wake.mat'))
    # noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    # dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      use_pic=True, boundary=[21, 21, 46],
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 64

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=30)

    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    # trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

    bounds = (1e-12, 1e-0)
    epde_search_obj.fit(data=data, variable_names=["u", "v", "p"], max_deriv_order=(1, 2, 2), derivs=None,
                        equation_terms_max_number=20, data_fun_pow=1,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


if __name__ == "__main__":
    import torch
    from epde.operators.utils.default_parameter_loader import EvolutionaryParams
    print(torch.cuda.is_available())
    # Operator = fitness.SolverBasedFitness # Replace by the developed PIC-based operator.
    # Operator = fitness.PIC
    Operator = fitness.L2LRFitness
    params = EvolutionaryParams()
    operator_params = params.get_default_params_for_operator('DiscrepancyBasedFitnessWithCV') #{"penalty_coeff": 0.2, "pinn_loss_mult": 1e4}
    print('operator_params ', operator_params)
    fit_operator = prepare_suboperators(Operator(list(operator_params.keys())), operator_params)

    # Paths
    directory = os.path.dirname(os.path.realpath(__file__))
    ns_folder_name = os.path.join(directory)

    # ns_test(fit_operator, ns_folder_name, 0)
    ns_discovery(ns_folder_name, 0)
