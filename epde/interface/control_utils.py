import numpy as np
import torch

from typing import List
from abc import ABC, abstractmethod

from epde.interface.interface import EpdeMultisample, EpdeSearch, ExperimentCombiner
 
# Add logic of transforming control function as a fixed equation token into the  neural network 
def get_control_nn(n_indep: int, n_dep: int, n_control: int):
    hidden_neurons = 128
    layers = [torch.nn.Linear(n_indep + n_dep, hidden_neurons),
              torch.nn.Tanh(),
              torch.nn.Linear(hidden_neurons, hidden_neurons),
              torch.nn.Tanh(),
              torch.nn.Linear(hidden_neurons, hidden_neurons),
              torch.nn.Tanh(),
              torch.nn.Linear(hidden_neurons, n_control)]
    return torch.nn.Sequential(*layers)    

class ControlConstraint(ABC):
    def __init__(self, val):
        self._val = val

    @abstractmethod
    def __call__(self, fun_val) -> None:
        raise NotImplementedError('Trying to call abstract constraint discrepancy evaluation.')

    @abstractmethod

class ControlConstrEq(ABC):
    def __init__(self, val):
        self._val = val

    def __call__(self, fun_val) -> None:
        return np.linalg.norm
        


class ControlLoss():
    def __init__(self, u_constraints : List):
        pass

    def 


class ControlExp():
    def __init__(self):
        pass # TODO: parameters? boundary conditions? 

    def train_equation(self):
        # raise NotImplementedError() # TODO: combine input samples, train equations

        res_combiner = ExperimentCombiner(optimal_equations)
        return res_combiner.create_best(self._pool)   

    def train_pinn(self, epochs: int = 1e4):
        t = 0
        stop_training = False

        # Make preparations for L-BFGS method use
        while t < epochs and not stop_training: # training of control function
            