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

from epde import TrigonometricTokens, GridTokens, CacheStoredTokens
import epde.globals as global_var
from epde.interface.token_family import TFPool
from epde.structure.main_structures import SoEq, Chromosome

import scipy.io as scio

import epde.operators.common.fitness as fitness

# original_set_adapter = fitness.PIC.set_adapter
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
# fitness.PIC.set_adapter = patched_set_adapter

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

    popsize = 48
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=5)

    factors_max_number = {'factors_num': [1, 2], 'probas' : [0.8, 0.2]}

    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj.fit(data=[x, y, z], variable_names=['u', 'v', 'w'], max_deriv_order=(2,),
                        equation_terms_max_number=5, data_fun_pow=3, additional_tokens=[trig_tokens, ],
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

    lorenz_discovery(0)


    def get_pic_network_summary(operator):
        if operator.adapter is None or operator.adapter.net is None:
            return None
        net = operator.adapter.net
        total_params = sum(p.numel() for p in net.parameters())
        layers = [str(layer) for layer in net.layers] if hasattr(net, 'layers') else []
        return {'total_parameters': total_params, 'layers': layers}

