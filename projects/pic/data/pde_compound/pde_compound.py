import os
import pickle
import sys
from typing import List, Tuple

import numpy as np
import scipy.io as scio
import torch

# EPDE imports
import epde.globals as global_var
from epde import CacheStoredTokens, GridTokens, TrigonometricTokens
from epde.interface.equation_translator import translate_equation
from epde.interface.interface import EpdeSearch
from epde.interface.prepared_tokens import (ConstantToken, CustomEvaluator,
                                            CustomTokens, PhasedSine1DTokens)
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.fitness import L2LRFitness
from epde.operators.common.sparsity import LASSOSparsity
from epde.operators.utils.default_parameter_loader import EvolutionaryParams
from epde.operators.utils.operator_mappers import map_operator_between_levels
from epde.operators.utils.template import CompoundOperator

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class PDEComparator:
    """Class for comparing PDE equations and their fitness."""

    def __init__(self, fitness_operator: CompoundOperator):
        self.fitness_operator = fitness_operator

    def compare(self, correct_symbolic: str, incorrect_symbolic: str,
                search_obj: EpdeSearch, all_vars: List[str] = ['u']) -> bool:
        """
        Compare two equations based on their fitness values and coefficient stability.

        Args:
            correct_symbolic: Symbolic form of the correct equation
            incorrect_symbolic: Symbolic form of the incorrect equation
            search_obj: EpdeSearch object containing the token pool
            all_vars: List of variable names to compare

        Returns:
            bool: True if correct equation has better stability for all variables
        """
        metaparams = {('sparsity', var): {'optimizable': False, 'value': 1E-6} for var in all_vars}

        correct_eq = self._prepare_equation(correct_symbolic, search_obj, all_vars, metaparams)
        incorrect_eq = self._prepare_equation(incorrect_symbolic, search_obj, all_vars, metaparams)

        self._evaluate_equations(correct_eq, incorrect_eq)
        self._print_comparison(correct_eq, incorrect_eq, all_vars)

        return all(correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability
                   for var in all_vars)

    def _prepare_equation(self, eq_symbolic: str, search_obj: EpdeSearch,
                          all_vars: List[str], metaparams: dict):
        """Translate symbolic equation and set its parameters."""
        equation = translate_equation(eq_symbolic, search_obj.pool, all_vars=all_vars)
        for var in all_vars:
            equation.vals[var].main_var_to_explain = var
            equation.vals[var].metaparameters = metaparams
        print(equation.text_form)
        return equation

    def _evaluate_equations(self, *equations):
        """Apply fitness operator to all provided equations."""
        for eq in equations:
            self.fitness_operator.apply(eq, {})

    def _print_comparison(self, correct_eq, incorrect_eq, all_vars):
        """Print comparison metrics for the equations."""
        print([[correct_eq.vals[var].fitness_value, incorrect_eq.vals[var].fitness_value]
               for var in all_vars])
        print([[correct_eq.vals[var].coefficients_stability, incorrect_eq.vals[var].coefficients_stability]
               for var in all_vars])
        print([[correct_eq.vals[var].aic, incorrect_eq.vals[var].aic] for var in all_vars])


class DataHandler:
    """Handles data loading and preprocessing operations."""

    @staticmethod
    def load_pretrained_pinn(ann_filename: str):
        """Load pretrained PINN model from file."""
        try:
            with open(ann_filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print('No model located, proceeding with ANN approximation retraining.')
            return None

    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to the input data."""
        noise_amplitude = noise_level * 0.01 * np.std(data)
        return noise_amplitude * np.random.normal(size=data.shape) + data

    @staticmethod
    def load_pc_data(filename: str, shape: int = 80) -> Tuple[List[np.ndarray], np.ndarray]:
        """Load and prepare PC data with corresponding grid."""
        data = np.load(filename)
        nx, nt = 100, 251
        x = np.linspace(1, 2, nx)
        t = np.linspace(0, 0.5, nt)
        grids = np.meshgrid(t, x, indexing='ij')
        return grids, data


class OperatorFactory:
    """Factory for creating and configuring fitness operators."""

    @staticmethod
    def create_fitness_operator(operator_class, params: dict) -> CompoundOperator:
        """Create and configure a fitness operator."""
        fitness_operator = operator_class(list(params.keys()))
        return OperatorFactory._prepare_suboperators(fitness_operator, params)

    @staticmethod
    def _prepare_suboperators(fitness_operator: CompoundOperator, operator_params: dict) -> CompoundOperator:
        """Configure suboperators for the fitness operator."""
        sparsity = LASSOSparsity()
        coeff_calc = LinRegBasedCoeffsEquation()

        fitness_operator.set_suboperators({
            'sparsity': sparsity,
            'coeff_calc': coeff_calc
        })

        fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
        fitness_operator.params = operator_params

        return map_operator_between_levels(
            fitness_operator,
            'gene level',
            'chromosome level',
            objective_condition=fitness_cond
        )


class PCExperiment:
    """Class for conducting PC (PDE Compound) experiments."""

    def __init__(self, foldername: str, fitness_operator: CompoundOperator = None):
        self.foldername = foldername
        self.fitness_operator = fitness_operator
        self.comparator = PDEComparator(fitness_operator) if fitness_operator else None

    def run_test(self, noise_level: int = 0) -> bool:
        """Run the PC test scenario comparing known correct and incorrect equations."""
        correct_eq = '1.0 * du/dx1{power: 2.0} + 1.0 * d^2u/dx1^2{power: 1.0} * u{power: 1.0} + \
                      0.0 = du/dx0{power: 1.0}'
        incorrect_eq = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'

        grid, data = DataHandler.load_pc_data(os.path.join(self.foldername, 'PDE_compound.npy'))
        noised_data = DataHandler.add_noise(data, noise_level)
        data_nn = DataHandler.load_pretrained_pinn(os.path.join(self.foldername, 'kdv_0_ann.pickle'))

        print('Shapes:', data.shape, grid[0].shape)
        dimensionality = 1

        trig_tokens = TrigonometricTokens(
            freq=(1 - 1e-8, 1 + 1e-8),
            dimensionality=dimensionality
        )

        search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=(10, 25),
            coordinate_tensors=[grid[0], grid[1]],
            verbose_params={'show_iter_idx': True},
            device='cuda'
        )

        search_obj.set_preprocessor(
            default_preprocessor_type='FD',
            preprocessor_kwargs={}
        )

        search_obj.create_pool(
            data=noised_data,
            variable_names=['u'],
            max_deriv_order=(2, 3),
            additional_tokens=[trig_tokens]
        )

        return self.comparator.compare(correct_eq, incorrect_eq, search_obj)

    def run_discovery(self, noise_level: int = 0) -> EpdeSearch:
        """Run PC equation discovery experiment."""
        grid, data = DataHandler.load_pc_data(os.path.join(self.foldername, 'PDE_compound.npy'))
        noised_data = DataHandler.add_noise(data, noise_level)
        dimensionality = data.ndim - 1

        search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=20,
            coordinate_tensors=grid,
            device='cuda'
        )

        search_obj.set_preprocessor(
            default_preprocessor_type='FD',
            preprocessor_kwargs={}
        )

        popsize = 20
        search_obj.set_moeadd_params(
            population_size=popsize,
            training_epochs=12
        )

        # Prepare custom tokens
        custom_grid_tokens = CacheStoredTokens(
            token_type='grid',
            token_labels=['t', 'x'],
            token_tensors={'t': grid[0], 'x': grid[1]},
            params_ranges={'power': (1, 1)},
            params_equality_ranges=None
        )

        custom_trig_eval_fun = {
            'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']
        }
        custom_trig_evaluator = CustomEvaluator(custom_trig_eval_fun, eval_fun_params_labels=['power'])

        custom_trig_tokens = CustomTokens(
            token_type='trigonometric',
            token_labels=['cos(t)sin(x)'],
            evaluator=custom_trig_evaluator,
            params_ranges={'power': (1, 1)},
            params_equality_ranges={},
            meaningful=True,
            unique_token_type=True
        )

        trig_tokens = TrigonometricTokens(
            dimensionality=dimensionality,
            freq=(0.999, 1.001)
        )

        factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}
        bounds = (1e-8, 1e-0)

        search_obj.fit(
            data=noised_data,
            variable_names=['u'],
            max_deriv_order=(2, 3),
            derivs=None,
            equation_terms_max_number=5,
            data_fun_pow=1,
            additional_tokens=[],
            deriv_fun_pow=2,
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=bounds,
            fourier_layers=False
        )

        search_obj.equations(only_print=True, num=1)
        search_obj.visualize_solutions()

        return search_obj


def main():
    """Main execution function."""
    print("CUDA available:", torch.cuda.is_available())

    # Initialize fitness operator
    params = EvolutionaryParams()
    operator_params = params.get_default_params_for_operator('DiscrepancyBasedFitnessWithCV')
    print('Operator params:', operator_params)

    fitness_operator = OperatorFactory.create_fitness_operator(L2LRFitness, operator_params)

    # Set up paths
    directory = os.path.dirname(os.path.realpath(__file__))
    pc_folder_name = os.path.join(directory)

    # Run experiments
    experiment = PCExperiment(pc_folder_name, fitness_operator)
    # experiment.run_test(0)  # Uncomment to run comparison test
    experiment.run_discovery(0)  # Run discovery experiment


if __name__ == "__main__":
    main()