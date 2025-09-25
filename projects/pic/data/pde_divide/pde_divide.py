import os
import pickle
import sys
from typing import List, Tuple

import numpy as np
import torch
from epde import CacheStoredTokens, GridTokens, TrigonometricTokens
from epde.interface.equation_translator import translate_equation
from epde.interface.interface import EpdeSearch
from epde.interface.prepared_tokens import CustomEvaluator, CustomTokens
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.fitness import L2LRFitness
from epde.operators.common.sparsity import LASSOSparsity
from epde.operators.utils.default_parameter_loader import EvolutionaryParams
from epde.operators.utils.operator_mappers import map_operator_between_levels
from epde.operators.utils.template import CompoundOperator

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class PDEAnalysis:
    """
    Base class for PDE analysis operations.
    """


    def __init__(self, foldername: str, fitness_operator: CompoundOperator = None):
        """
        Initializes the analysis environment for a given dataset.
        
                This setup prepares the necessary components for comparing and evaluating potential differential equation models against the data stored in the specified folder. The fitness operator defines how well a model fits the data, and the comparator uses this operator to rank different models.
        
                Args:
                    foldername (str): The name of the folder containing the dataset to be analyzed.
                    fitness_operator (CompoundOperator, optional): The operator used to evaluate the fitness of candidate equations. Defaults to None.
        
                Returns:
                    None
        
                Class Fields:
                    foldername (str): The name of the folder containing the dataset.
                    fitness_operator (CompoundOperator): The fitness operator used for equation evaluation.
                    comparator (PDEComparator):  An object responsible for comparing and ranking candidate equations based on their fitness.
        """
        self.foldername = foldername
        self.fitness_operator = fitness_operator
        self.comparator = PDEComparator(fitness_operator) if fitness_operator else None

    @staticmethod
    def load_data(filename: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load data from a file and generate corresponding space-time grid.
        
                The data is assumed to represent a solution of a differential equation
                on a 2D spatial-temporal domain. The method prepares the data and
                creates a grid representing the domain, which is essential for
                further analysis and equation discovery.
        
                Args:
                    filename (str): The name of the file containing the data ('.npy' format).
        
                Returns:
                    Tuple[List[np.ndarray], np.ndarray]: A tuple containing:
                        - grids (List[np.ndarray]): A list of two numpy arrays representing the spatial and temporal grids.
                        - data (np.ndarray): The loaded data from the file.
        """
        data = np.load(filename)
        nx, nt = 100, 251
        x = np.linspace(1, 2, nx)
        t = np.linspace(0, 0.5, nt)
        grids = np.meshgrid(t, x, indexing='ij')
        return grids, data

    @staticmethod
    def create_custom_tokens(grid: List[np.ndarray]) -> Tuple[CacheStoredTokens, CustomTokens]:
        """
        Creates the grid and trigonometric tokens required for symbolic regression. These tokens define the basis functions used to represent candidate equation terms. The grid tokens represent the independent variables, while the trigonometric tokens introduce non-polynomial terms, enriching the search space for potential solutions.
        
                Args:
                    grid (List[np.ndarray]): A list containing the grid points for the independent variables (e.g., time and space).
        
                Returns:
                    Tuple[CacheStoredTokens, CustomTokens]: A tuple containing the grid tokens and the trigonometric tokens.
        """
        grid_tokens = CacheStoredTokens(
            token_type='grid',
            token_labels=['t', 'x'],
            token_tensors={'t': grid[0], 'x': grid[1]},
            params_ranges={'power': (1, 1)},
            params_equality_ranges=None
        )

        trig_eval_fun = {
            'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']
        }
        trig_evaluator = CustomEvaluator(trig_eval_fun, eval_fun_params_labels=['power'])

        trig_tokens = CustomTokens(
            token_type='trigonometric',
            token_labels=['cos(t)sin(x)'],
            evaluator=trig_evaluator,
            params_ranges={'power': (1, 1)},
            params_equality_ranges={},
            meaningful=True,
            unique_token_type=True
        )

        return grid_tokens, trig_tokens


class PDEDivideExperiment(PDEAnalysis):
    """
    Class for conducting PDE divide experiments.
    """


    def run_test(self, noise_level: int = 0) -> bool:
        """
        Compares the performance of the equation discovery process on a known correct equation against an incorrect one.
        
                This test evaluates the ability of the EPDE framework to distinguish between a true governing equation and a false one.
                It assesses whether the search algorithm correctly identifies the known equation within the solution space.
        
                Args:
                    noise_level (int): The level of noise to add to the data. Defaults to 0.
        
                Returns:
                    bool: True if the correct equation is preferred over the incorrect one, False otherwise.
        """
        correct_eq = (
            '1.0 * du/dx1{power: 1.0} + 0.25 * d^2u/dx1^2{power: 1.0} * x{power: 1.0} + '
            '0.0 = du/dx0{power: 1.0} * x{power: 1.0}'
        )
        incorrect_eq = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'

        grid, data = self.load_data(os.path.join(self.foldername, 'PDE_divide.npy'))
        noised_data = DataHandler.add_noise(data, noise_level)

        print('Shapes:', data.shape, grid[0].shape)
        dimensionality = 1

        search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=10,
            coordinate_tensors=(grid[0], grid[1]),
            verbose_params={'show_iter_idx': True},
            device='cuda'
        )

        grid_tokens, trig_tokens = self.create_custom_tokens(grid)
        standard_trig_tokens = TrigonometricTokens(
            freq=(1 - 1e-8, 1 + 1e-8),
            dimensionality=dimensionality
        )

        search_obj.set_preprocessor(
            default_preprocessor_type='FD',
            preprocessor_kwargs={}
        )

        search_obj.create_pool(
            data=noised_data,
            variable_names=['u'],
            max_deriv_order=(2, 3),
            additional_tokens=[standard_trig_tokens, grid_tokens]
        )

        return self.comparator.compare(correct_eq, incorrect_eq, search_obj)

    def run_discovery(self, noise_level: int = 0) -> EpdeSearch:
        """
        Discovers a differential equation that describes the provided dataset by exploring combinations of mathematical terms and evaluating their fit to the data. This method leverages evolutionary algorithms to efficiently search the space of possible equations.
        
                Args:
                    noise_level (int, optional): The level of noise to add to the data. Defaults to 0.
        
                Returns:
                    EpdeSearch: The `EpdeSearch` object containing the results of the equation discovery process. The object stores the discovered equations and related information.
        
                Why:
                    This method automates the process of finding a differential equation that best represents the underlying dynamics of the provided data. By using evolutionary algorithms, it efficiently explores a wide range of possible equation structures, identifying the ones that accurately capture the relationships within the data.
        """
        grid, data = self.load_data(os.path.join(self.foldername, 'PDE_divide.npy'))
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

        popsize = 8
        search_obj.set_moeadd_params(
            population_size=popsize,
            training_epochs=50
        )

        grid_tokens, custom_trig_tokens = self.create_custom_tokens(grid)
        standard_trig_tokens = TrigonometricTokens(
            dimensionality=dimensionality,
            freq=(0.999, 1.001)
        )
        standard_grid_tokens = GridTokens(
            ['x_0', 'x_1'],
            dimensionality=dimensionality,
            max_power=2
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
            additional_tokens=[grid_tokens],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=bounds,
            fourier_layers=False
        )

        search_obj.equations(only_print=True, num=1)
        search_obj.visualize_solutions()

        return search_obj


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


class PDEComparator:
    """
    Class for comparing PDE equations and their fitness.
    """


    def __init__(self, fitness_operator: CompoundOperator):
        """
        Initializes the FitnessProportionateSelection object.
        
        This class implements fitness proportionate selection, where the
        probability of selecting an individual is proportional to its fitness.
        This initialization is crucial for the evolutionary process, ensuring
        that individuals (potential equation structures) are selected for
        reproduction based on how well they fit the observed data.
        
        Args:
            fitness_operator (CompoundOperator): The operator used to calculate
                the fitness of an individual, quantifying how well a candidate
                equation matches the data.
        
        Returns:
            None.
        
        Class Fields:
            fitness_operator (CompoundOperator): The operator used to calculate
                the fitness of an individual.
        """
        self.fitness_operator = fitness_operator

    def compare(self, correct_symbolic: str, incorrect_symbolic: str,
                search_obj: EpdeSearch, all_vars: List[str] = ['u']) -> bool:
        """
        Compares two symbolic equations to determine which better represents the underlying dynamics of the system.
        
                This comparison is based on the stability of their coefficients after numerical evaluation.
                The equation with more stable coefficients is considered to be a better representation.
        
                Args:
                    correct_symbolic (str): Symbolic representation of the equation considered to be more correct.
                    incorrect_symbolic (str): Symbolic representation of the equation considered to be less correct.
                    search_obj (EpdeSearch): The search object containing the data and settings for equation discovery.
                    all_vars (List[str], optional): List of dependent variables. Defaults to ['u'].
        
                Returns:
                    bool: True if the 'correct' equation exhibits more stable coefficients than the 'incorrect' one across all variables, False otherwise.
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
            *equations: One or more equations to be evaluated.
        
        Returns:
            None. The fitness is applied to the equations in place.
        """
        for eq in equations:
            self.fitness_operator.apply(eq, {})

    def _print_comparison(self, correct_eq, incorrect_eq, all_vars):
        """
        Prints comparison metrics of the discovered and the ground truth equations.
                It compares fitness values, coefficient stability, and AIC scores for each variable in both equations.
                This helps to evaluate how well the discovered equation approximates the ground truth equation based on multiple criteria.
        
                Args:
                    correct_eq (Equation): The ground truth equation.
                    incorrect_eq (Equation): The discovered equation.
                    all_vars (list): A list of all variables in the equations.
        
                Returns:
                    None
        """
        print([[correct_eq.vals[var].fitness_value, incorrect_eq.vals[var].fitness_value]
               for var in all_vars])
        print([[correct_eq.vals[var].coefficients_stability, incorrect_eq.vals[var].coefficients_stability]
               for var in all_vars])
        print([[correct_eq.vals[var].aic, incorrect_eq.vals[var].aic] for var in all_vars])


class OperatorFactory:
    """
    Factory for creating and configuring fitness operators.
    """


    @staticmethod
    def create_fitness_operator(operator_class, params: dict) -> CompoundOperator:
        """
        Create and configure a fitness operator.
        
        This operator is a crucial part of the equation discovery process, responsible for evaluating the fitness of candidate equations based on the provided data.
        
        Args:
            operator_class: The class of the fitness operator to create.
            params (dict): A dictionary of parameters to configure the operator. These parameters define the specific settings and configurations used during the fitness evaluation.
        
        Returns:
            CompoundOperator: A configured fitness operator, ready to be used in the evolutionary search process. The returned operator encapsulates the logic for assessing how well a given equation fits the observed data.
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


def main():
    """
    Initializes the environment, configures the fitness operator, and executes the equation discovery process.
    
    This function sets up the necessary components for discovering differential equations, including checking CUDA availability, initializing the fitness operator with specified parameters, and configuring the experiment environment. It then orchestrates the execution of the equation discovery process.
    
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
    pd_folder_name = os.path.join(directory)

    # Run experiments
    experiment = PDEDivideExperiment(pd_folder_name, fitness_operator)
    # experiment.run_test(0)  # Uncomment to run comparison test
    experiment.run_discovery(0)  # Run discovery experiment


if __name__ == "__main__":
    main()