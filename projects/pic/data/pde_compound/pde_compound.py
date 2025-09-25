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
    """
    Class for comparing PDE equations and their fitness.
    """


    def __init__(self, fitness_operator: CompoundOperator):
        """
        Initializes the FitnessProportionateSelection object.
        
        This class implements fitness proportionate selection, also known as roulette wheel selection.
        It selects individuals from a population based on their fitness, where the probability of
        selection is proportional to their fitness value. This ensures that equation structures with better fitness (i.e., those that better describe the data) are more likely to be selected for the next generation of the evolutionary process.
        
        Args:
            fitness_operator: The operator used to calculate the fitness of individuals.
        
        Returns:
            None
        """
        self.fitness_operator = fitness_operator

    def compare(self, correct_symbolic: str, incorrect_symbolic: str,
                search_obj: EpdeSearch, all_vars: List[str] = ['u']) -> bool:
        """
        Compares two symbolic equations to determine which better represents the underlying dynamics based on coefficient stability.
        
                This comparison helps refine the search for the most accurate equation by favoring those with more stable coefficients across different variables.
        
                Args:
                    correct_symbolic: Symbolic form of the equation presumed to be more accurate.
                    incorrect_symbolic: Symbolic form of the equation presumed to be less accurate.
                    search_obj: EpdeSearch object containing the token pool and search parameters.
                    all_vars: List of variable names to compare coefficient stability. Defaults to ['u'].
        
                Returns:
                    bool: True if the 'correct' equation exhibits better coefficient stability than the 'incorrect' equation for all specified variables, indicating a potentially better representation of the underlying dynamics.
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
        """
        Transforms a symbolic equation into an executable form and configures its variables for the equation discovery process. This involves translating the equation string into a computational representation and associating each variable with relevant search parameters.
        
                Args:
                    eq_symbolic (str): The symbolic representation of the equation.
                    search_obj (EpdeSearch): The EPDE search object containing the token pool and search configurations.
                    all_vars (List[str]): A list of all variable names present in the equation.
                    metaparams (dict): A dictionary of metaparameters to be assigned to each variable.
        
                Returns:
                    Equation: The translated equation object with configured variables.
        
                Why:
                    This step is crucial for converting the symbolic equation into a format that can be evaluated and optimized within the EPDE framework. By setting the metaparameters for each variable, the search algorithm can effectively explore the solution space and identify the best-fitting equation.
        """
        equation = translate_equation(eq_symbolic, search_obj.pool, all_vars=all_vars)
        for var in all_vars:
            equation.vals[var].main_var_to_explain = var
            equation.vals[var].metaparameters = metaparams
        print(equation.text_form)
        return equation

    def _evaluate_equations(self, *equations):
        """
        Evaluate the fitness of the provided equations using the configured fitness operator. This step is crucial for assessing how well each equation represents the underlying dynamics of the system, guiding the evolutionary search towards better models.
        
                Args:
                    *equations: One or more equation objects to be evaluated.
        
                Returns:
                    None
        """
        for eq in equations:
            self.fitness_operator.apply(eq, {})

    def _print_comparison(self, correct_eq, incorrect_eq, all_vars):
        """
        Prints comparison of fitness value, coefficients stability and AIC for each variable in correct and incorrect equations.
                This is useful for understanding how well the discovered equation matches the data compared to other candidates.
        
                Args:
                    correct_eq (Equation): The correct equation object.
                    incorrect_eq (Equation): The incorrect equation object.
                    all_vars (list): A list of all variables.
        
                Returns:
                    None
        """
        print([[correct_eq.vals[var].fitness_value, incorrect_eq.vals[var].fitness_value]
               for var in all_vars])
        print([[correct_eq.vals[var].coefficients_stability, incorrect_eq.vals[var].coefficients_stability]
               for var in all_vars])
        print([[correct_eq.vals[var].aic, incorrect_eq.vals[var].aic] for var in all_vars])


class DataHandler:
    """
    Handles data loading and preprocessing operations.
    """


    @staticmethod
    def load_pretrained_pinn(ann_filename: str):
        """
        Loads a pre-trained Physics-Informed Neural Network (PINN) model from a specified file.
        
        This allows to reuse previously trained models, avoiding the need to retrain them from scratch, which saves computational resources and time when exploring different equation structures or refining existing models.
        
        Args:
            ann_filename (str): The path to the file containing the serialized PINN model.
        
        Returns:
            The loaded PINN model if the file exists; otherwise, returns None and prints a message indicating that the model will be retrained.
        """
        try:
            with open(ann_filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print('No model located, proceeding with ANN approximation retraining.')
            return None

    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add Gaussian noise to the input data to simulate real-world measurement errors or to augment the dataset for more robust equation discovery.
        
        Args:
            data (np.ndarray): The input data to which noise will be added.
            noise_level (float): The standard deviation of the Gaussian noise, expressed as a percentage of the data's standard deviation.
        
        Returns:
            np.ndarray: The data with added Gaussian noise.
        """
        noise_amplitude = noise_level * 0.01 * np.std(data)
        return noise_amplitude * np.random.normal(size=data.shape) + data

    @staticmethod
    def load_pc_data(filename: str, shape: int = 80) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load PC data from a file, create a corresponding grid, and return both for usage in equation discovery.
        
                Args:
                    filename (str): The path to the .npy file containing the PC data.
                    shape (int): The shape of the grid to be created (default: 80).
        
                Returns:
                    Tuple[List[np.ndarray], np.ndarray]: A tuple containing the grid and the loaded PC data. The grid is a list of numpy arrays representing the coordinate axes.
        """
        data = np.load(filename)
        nx, nt = 100, 251
        x = np.linspace(1, 2, nx)
        t = np.linspace(0, 0.5, nt)
        grids = np.meshgrid(t, x, indexing='ij')
        return grids, data


class OperatorFactory:
    """
    Factory for creating and configuring fitness operators.
    """


    @staticmethod
    def create_fitness_operator(operator_class, params: dict) -> CompoundOperator:
        """
        Create and configure a fitness operator.
        
        This operator is a crucial part of the equation discovery process, responsible for evaluating the fitness of candidate equations based on how well they describe the observed data.
        
        Args:
            operator_class: The class of the fitness operator to create.
            params (dict): A dictionary of parameters to configure the operator. These parameters define the specific settings and options used during the fitness evaluation process.
        
        Returns:
            CompoundOperator: A configured fitness operator, ready to be used in the evolutionary search for differential equations.
        """
        fitness_operator = operator_class(list(params.keys()))
        return OperatorFactory._prepare_suboperators(fitness_operator, params)

    @staticmethod
    def _prepare_suboperators(fitness_operator: CompoundOperator, operator_params: dict) -> CompoundOperator:
        """
        Configures the fitness operator with necessary sub-operators and applies level mapping.
        
        The fitness operator requires specific sub-operators for calculating sparsity and coefficients. This method sets these sub-operators and then maps the fitness operator between the gene and chromosome levels. This mapping ensures that the fitness evaluation is correctly applied at the appropriate level of the evolutionary process, contributing to the overall goal of discovering accurate differential equation models.
        
        Args:
            fitness_operator (CompoundOperator): The fitness operator to configure.
            operator_params (dict): Parameters for the fitness operator.
        
        Returns:
            CompoundOperator: The configured fitness operator after level mapping.
        """
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
    """
    Class for conducting PC (PDE Compound) experiments.
    """


    def __init__(self, foldername: str, fitness_operator: CompoundOperator = None):
        """
        Initializes a PCExperiment instance, preparing the environment for equation discovery within a specified folder.
        
                The initialization process sets up the necessary components for conducting experiments,
                including the fitness evaluation mechanism used to assess candidate equations.
        
                Args:
                    foldername (str): The name of the folder where experiment data and results will be stored.
                    fitness_operator (CompoundOperator, optional): The operator used to evaluate the fitness of candidate equations.
                        Defaults to None. If None, no fitness evaluation is performed during the experiment setup.
        
                Returns:
                    None
        
                Class Fields:
                    foldername (str): The name of the folder associated with this experiment.
                    fitness_operator (CompoundOperator): The fitness operator used for evaluating equations.
                    comparator (PDEComparator): The comparator object, initialized based on the fitness operator, used to compare different equations.
        """
        self.foldername = foldername
        self.fitness_operator = fitness_operator
        self.comparator = PDEComparator(fitness_operator) if fitness_operator else None

    def run_test(self, noise_level: int = 0) -> bool:
        """
        Compares the performance of the equation discovery process with a known correct equation and a deliberately incorrect equation. This comparison helps assess the robustness and accuracy of the equation discovery algorithm.
        
                Args:
                    noise_level (int, optional): The level of noise added to the data. Defaults to 0.
        
                Returns:
                    bool: True if the correct equation is identified as superior to the incorrect one, False otherwise.
        """
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
            boundary=10,
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
        """
        Discovers a differential equation from data using a combination of finite difference preprocessing, custom token definitions, and multi-objective evolutionary optimization.
        
                This method sets up and executes the equation discovery process. It involves loading data, adding noise, defining custom tokens representing known relationships, and configuring the evolutionary search to find the equation that best fits the data. The method leverages finite difference approximations for derivative calculations and custom tokens to represent specific functional forms, enhancing the search's ability to identify relevant equation structures. The evolutionary algorithm explores the space of possible equations, balancing model complexity and accuracy to identify the governing dynamics.
        
                Args:
                    noise_level (int, optional): The level of noise to add to the data. Defaults to 0.
        
                Returns:
                    EpdeSearch: The EpdeSearch object containing the results of the equation discovery process.
        """
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

        popsize = 12
        search_obj.set_moeadd_params(
            population_size=popsize,
            training_epochs=30
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
        bounds = (1e-9, 1e-2)

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
    """
    Initializes the environment, configures the fitness operator, and executes the equation discovery experiment.
    
    This function sets up the necessary components for discovering differential equations, including checking CUDA availability, initializing the fitness operator with specified parameters, and configuring the experiment environment. It then runs the discovery process to identify potential equation candidates.
    
    Args:
        None
    
    Returns:
        None
    """
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