import logging
import re
import itertools
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import epde
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EquationProcessor:
    """Class for processing and organizing equations discovered by EPDE."""

    def __init__(self):
        self.regex = re.compile(r', freq:\s\d\S\d+')

    def dict_update(self, d_main: Dict, term: str, coeff: float, k: int) -> Dict:
        """Updates dictionaries for equations, handling permutations and duplicates."""
        str_t = '_r' if '_r' in term else ''
        arr_term = re.sub('_r', '', term).split(' * ')
        perm_set = list(itertools.permutations(range(len(arr_term))))
        structure_added = False
        for p_i in perm_set:
            temp = " * ".join([arr_term[i] for i in p_i]) + str_t
            if temp in d_main:
                if k >= len(d_main[temp]):
                    d_main[temp] += [0.0] * (k - len(d_main[temp]) + 1)
                d_main[temp][k] += coeff
                structure_added = True
                break
        if not structure_added:
            d_main[term] = [0.0] * k + [coeff]
        return d_main

    def equation_table(self, k: int, equation: any, dict_main: Dict, dict_right: Dict) -> Tuple[Dict, Dict]:
        """Creates equation tables from a single EPDE equation object."""
        equation_s = equation.structure
        equation_c = np.array(equation.weights_final)
        text_form_eq = self.regex.sub('', equation.text_form)
        flag = False
        for t_eq in equation_s:
            term = self.regex.sub('', t_eq.name)
            if f'= {term}' in text_form_eq:
                if not flag:
                    dict_right = self.dict_update(dict_right, term, 1.0, k)
                    flag = True
                continue
            coeff_found = False
            for t_idx, c in enumerate(equation_c):
                if f'{c:.6f} * {term}' in text_form_eq or f'{c:+.6f} * {term}' in text_form_eq:
                    dict_main = self.dict_update(dict_main, term, c, k)
                    equation_c = np.delete(equation_c, t_idx)
                    coeff_found = True
                    break
            if not coeff_found and 'C' in term and 'C' in t_eq.name:
                dict_main = self.dict_update(dict_main, 'C', equation_c[0], k)
                equation_c = np.delete(equation_c, 0)
        return dict_main, dict_right

    def object_table(self, res: List[List], variable_names: List[str]) -> Tuple[List[Dict], int]:
        """Creates a structured object for equation tables."""
        table_main = [{'LHS': {}, 'RHS': {}} for _ in range(len(variable_names))]
        k = 0
        for list_SoEq in res:
            for SoEq in list_SoEq:
                for n, value in enumerate(variable_names):
                    gene = SoEq.vals.chromosome.get(value)
                    if gene and hasattr(gene, 'value') and gene.value is not None:
                        table_main[n]['LHS'], table_main[n]['RHS'] = self.equation_table(
                            k, gene.value, table_main[n]['LHS'], table_main[n]['RHS'])
                k += 1
        return table_main, k

    def preprocessing_table(self, variable_names: List[str], table_main: List[Dict], k: int) -> pd.DataFrame:
        """Prepares a DataFrame from the processed equation tables."""
        data_frame_total = pd.DataFrame()
        for n, var_name in enumerate(variable_names):
            lhs_dict = table_main[n]['LHS']
            rhs_dict = table_main[n]['RHS']
            combined_dict = {f'LHS: {key}': value for key, value in lhs_dict.items()}
            combined_dict.update({f'RHS: {key}': value for key, value in rhs_dict.items()})
            max_len = k + 1
            for key, value in combined_dict.items():
                if len(value) < max_len:
                    combined_dict[key] = value + [0.0] * (max_len - len(value))
            df_temp = pd.DataFrame(combined_dict)
            df_temp.columns = [f'{col}_{var_name}' for col in df_temp.columns]
            data_frame_total = pd.concat([data_frame_total, df_temp], axis=1)
        return data_frame_total


def epde_multisample_discovery(t_ax, variables, diff_method: str = 'poly', find_simple_eqs: bool = False):
    """Executes the EPDE multisample discovery process."""
    samples = [[t_ax[i], [var[i] for var in variables]] for i in range(len(t_ax))]
    epde_search_obj = epde.EpdeMultisample(data_samples=samples, use_solver=False, boundary=0)
    if diff_method == 'ANN':
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', preprocessor_kwargs={'epochs_max': 50})
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                         preprocessor_kwargs={'use_smoothing': True, 'sigma': 1,
                                                              'polynomial_window': 7, 'poly_order': 4})
    # epde_search_obj.set_moeadd_params(population_size=15, training_epochs=50)
    variable_names = [f'u{i}' for i in range(len(variables))]

    if not find_simple_eqs:
        epde_search_obj.set_moeadd_params(population_size=15, training_epochs=50)
        logger.info("Searching for complex equations")
        factors_max_number = {'factors_num': [1, 2], 'probas': [0.5, 0.5]}
        epde_search_obj.fit(
            samples=samples, variable_names=variable_names,
            max_deriv_order=(2,), equation_terms_max_number=5,
            data_fun_pow=2, deriv_fun_pow=2,
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=(1e-5, 1e-3)
        )
    else:
        epde_search_obj.set_moeadd_params(population_size=15, training_epochs=5)
        logger.info("Searching for linear equations")
        epde_search_obj.fit(
            samples=samples, variable_names=variable_names,
            max_deriv_order=(1,), equation_terms_max_number=5,
            data_fun_pow=1, deriv_fun_pow=1,
            equation_factors_max_number={'factors_num': [1], 'probas': [1.0]},
            eq_sparsity_interval=(1e-8, 1e-7)
        )
    return epde_search_obj


def run_discovery_and_save(t_axis, x_vars, is_simple_search: bool, output_dir: Path):
    """
    Runs the EPDE discovery for a specific search type and saves the results.
    """
    if is_simple_search:
        logger.info("\n--- Running Simple Equations Search ---")
        csv_filename = 'discovered_equations_simple.csv'
    else:
        logger.info("\n--- Running Complex Equations Search ---")
        csv_filename = 'discovered_equations_complex.csv'

    output_path = output_dir / csv_filename

    epde_result = epde_multisample_discovery(
        t_axis, x_vars, diff_method='poly', find_simple_eqs=is_simple_search
    )

    results = epde_result.equations()
    num_vars = len(x_vars)
    variable_names = [f'u{i}' for i in range(num_vars)]

    if results:
        logger.info("Equations found. Processing for saving.")
        processor = EquationProcessor()
        table_main, k = processor.object_table(results, variable_names)

        if k == 0:
            equations_df = processor.preprocessing_table(variable_names, table_main, k)

            try:
                equations_df.to_csv(output_path, sep=',', encoding='utf-8', index=False)
                logger.info(f"Equations successfully saved to: {output_path}")
            except IOError as e:
                logger.error(f"Failed to write to file {output_path}. Check permissions. Error: {e}")
        else:
            logger.warning("Processing resulted in an empty table. No file created.")
    else:
        logger.warning("No equations were found. The output file will not be created.")


if __name__ == "__main__":
    file_path = Path('data')
    output_dir = file_path / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = np.load(file_path / 'trajectories-2')
        data = data[:, :21, :]
        print(f"Loaded data with shape: {data.shape}")

        num_trajectories = 5
        num_timesteps = data.shape[1]
        num_vars = data.shape[2]

        trajectories = [data[i] for i in range(num_trajectories)]
        t_axis = [[np.linspace(0, 1, num_timesteps)] for _ in range(num_trajectories)]
        x_vars = [[traj[:, i] for traj in trajectories] for i in range(num_vars)]

        # Запуск
        run_discovery_and_save(t_axis, x_vars, is_simple_search=True, output_dir=output_dir)


    except FileNotFoundError:
        logger.error(f"Data file not found at: {file_path / 'trajectories-2'}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")