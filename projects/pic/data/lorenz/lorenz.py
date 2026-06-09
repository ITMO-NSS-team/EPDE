import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import pickle
from typing import Tuple, List
import numpy as np
import copy

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
from epde.interface.token_family import TFPool
from epde.structure.main_structures import SoEq, Chromosome

import scipy.io as scio

original_set_adapter = fitness.PIC.set_adapter
def patched_set_adapter(self, net=None):
    from epde.integrate import SolverAdapter
    compiling_params = {'mode': 'autograd', 'tol':0.01, 'lambda_bound': 100}
    optimizer_params = {}
    training_params = {'epochs': 1e3, 'info_string_every': 1e3}
    early_stopping_params = {'patience': 4, 'no_improvement_patience': 250}
    self.adapter = SolverAdapter(net=net, use_cache=False, device='cpu')
    self.adapter.set_compiling_params(**compiling_params)
    self.adapter.set_optimizer_params(**optimizer_params)
    self.adapter.set_early_stopping_params(**early_stopping_params)
    self.adapter.set_training_params(**training_params)
fitness.PIC.set_adapter = patched_set_adapter

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
    print(correct_eq.text_form)

    incorrect_eq = translate_equation(eq_incorrect_symbolic, search_obj.pool,
                                      all_vars=all_vars)  # , all_vars = ['u', 'v'])
    for var in all_vars:
        incorrect_eq.vals[var].main_var_to_explain = var
        incorrect_eq.vals[var].metaparameters = metaparams
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

def prepare_suboperators(fitness_operator: CompoundOperator, operator_params: dict) -> CompoundOperator:
    sparsity = LASSOSparsity()
    coeff_calc = LinRegBasedCoeffsEquation()

    # Поднимаем подоператоры на уровень хромосомы для работы с SoEq
    sparsity = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
    coeff_calc = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')

    fitness_operator.set_suboperators({'sparsity': sparsity, 'coeff_calc': coeff_calc})
    fitness_operator.params = operator_params

    # Маппинг самого fitness_operator
    if 'chromosome level' not in fitness_operator._tags:
        fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
        fitness_operator = map_operator_between_levels(fitness_operator, 'gene level', 'chromosome level',
                                                       objective_condition=fitness_cond)
    return fitness_operator

def create_equation_from_str(eq_str, target_var, base_pool, all_vars):
    families_copy = [copy.deepcopy(fam) for fam in base_pool.families]
    for fam in families_copy:
        if hasattr(fam, 'variable') and fam.variable is not None:
            if fam.variable != target_var:
                fam.status['demands_equation'] = False
    temp_pool = TFPool(families_copy)
    soeq = translate_equation(eq_str, temp_pool, all_vars=[target_var])
    # Извлекаем уравнение по ключу (имени переменной)
    eq = soeq.vals[target_var]
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

def lorenz_test(fit_operator, noise_level=0):
    t = np.load(os.path.join(os.path.dirname(__file__), 't.npy'))
    data = np.load(os.path.join(os.path.dirname(__file__), 'lorenz.npy'))
    end = 1000
    t = t[:end]
    x = data[:end, 0]
    y = data[:end, 1]
    z = data[:end, 2]

    correct_eqs = [
        '10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}',
        '28.0 * u{power: 1.0} + -1.0 * u{power: 1.0} * w{power: 1.0} + -1.0 * v{power: 1.0} + 0.0 = dv/dx0{power: 1.0}',
        '1.0 * u{power: 1.0} * v{power: 1.0} + -2.6666666666666665 * w{power: 1.0} + 0.0 = dw/dx0{power: 1.0}'
    ]
    incorrect_eqs = [
        '10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} + 0.1 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}',
        '28.0 * u{power: 1.0} + -1.0 * u{power: 1.0} * w{power: 1.0} + -1.0 * v{power: 1.0} + 0.1 * v{power: 1.0} + 0.0 = dv/dx0{power: 1.0}',
        '1.0 * u{power: 1.0} * v{power: 1.0} + -2.6666666666666665 * w{power: 1.0} + 0.1 * w{power: 1.0} + 0.0 = dw/dx0{power: 1.0}'
    ]

    epde_search_obj = EpdeSearch(
        use_solver=False,
        multiobjective_mode=True,
        use_pic=True,
        boundary=(100,),
        coordinate_tensors=[t],
        verbose_params={'show_iter_idx': True},
        device='cpu'
    )
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    epde_search_obj.create_pool(
        data=[x, y, z],
        variable_names=['u', 'v', 'w'],
        max_deriv_order=1,
        additional_tokens=[]
    )

    assert compare_systems(correct_eqs, incorrect_eqs, epde_search_obj, all_vars=['u', 'v', 'w'], fit_operator=fit_operator)


def lorenz_discovery(noise_level):
    t_file = os.path.join(os.path.dirname( __file__ ), 't.npy')
    t = np.load(t_file)
    data_file = os.path.join(os.path.dirname(__file__), 'lorenz.npy')
    data = np.load(data_file)

    end = 1000
    t = t[:end]
    x = data[:end, 0]
    y = data[:end, 1]
    z = data[:end, 2]

    dimensionality = x.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True, boundary=(100),
                                 coordinate_tensors=[t, ], verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 16
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=5)

    factors_max_number = {'factors_num': [1, 2], 'probas' : [0.8, 0.2]}

    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj.fit(data=[x, y, z], variable_names=['u', 'v', 'w'], max_deriv_order=(1,),
                        equation_terms_max_number=5, data_fun_pow=1, additional_tokens=[trig_tokens, ],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-8, 1e-0))  #

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

if __name__ == "__main__":
    import torch
    from epde.operators.utils.default_parameter_loader import EvolutionaryParams
    print(torch.cuda.is_available())
    global_var.solution_guess_nn = None
    # Operator = fitness.SolverBasedFitness # Replace by the developed PIC-based operator.
    Operator = fitness.PIC
    #Operator = fitness.L2LRFitness
    params = EvolutionaryParams()
    operator_params = params.get_default_params_for_operator('PIC')#'DiscrepancyBasedFitnessWithCV') #{"penalty_coeff": 0.2, "pinn_loss_mult": 1e4}
    #operator_params = params.get_default_params_for_operator('DiscrepancyBasedFitnessWithCV') #{"penalty_coeff": 0.2, "pinn_loss_mult": 1e4}
    # Operator = fitness.DeepXDEBasedFitness
    # params = EvolutionaryParams()
    #
    # try:
    #     operator_params = params.get_default_params_for_operator('DeepXDEBasedFitness')
    # except Exception as e:
    #     print(f"Предупреждение: не удалось загрузить параметры для DeepXDEBasedFitness: {e}")
    #     print("Использую ручную конфигурацию.")
    #     operator_params = {
    #         "deepxde_config": {
    #             "net": [50, 50, 50],
    #             "activation": "tanh",
    #             "optimizer": "adam",
    #             "lr": 1e-3,
    #             "num_domain": 1000,
    #             "num_boundary": 200,
    #             "num_initial": 200,
    #             "epochs": 2000
    #         },
    #         "penalty_coeff": 0.2,
    #         "error_metric": "rmse"
    #     }
    # print('operator_params ', operator_params)
    fit_operator = prepare_suboperators(Operator(list(operator_params.keys())), operator_params)

    lorenz_discovery(0)
    # lorenz_test(fit_operator, noise_level=0)


    def get_pic_network_summary(operator):
        if operator.adapter is None or operator.adapter.net is None:
            return None
        net = operator.adapter.net
        total_params = sum(p.numel() for p in net.parameters())
        layers = [str(layer) for layer in net.layers] if hasattr(net, 'layers') else []
        return {'total_parameters': total_params, 'layers': layers}


    pic_info = get_pic_network_summary(fit_operator)
    print("PIC network summary:", pic_info)

