import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epde.evaluators import CustomEvaluator
from epde.interface.prepared_tokens import CustomTokens
import epde.interface.interface as epde_alg
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.validation_rules import has_no_self_cycled_nodes, has_no_cycle, has_one_root, \
    has_no_isolated_components, has_no_isolated_nodes
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, EvoGraphOptimiser
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.graph import OptNode, OptGraph
from fedot.core.optimisers.optimizer import GraphGenerationParams

from examples.epde_fedot.Enaluator import generate_tokens, generate_initial_assumption, \
    _has_no_single_input_operations, _has_sum_as_root_node, _has_no_double_operations, PdeEvaluator

OPERATIONS_LIST = ['mul', 'sum']


def form_csv():
    np_data = pd.read_csv('heat_data_1000_1000.csv', header=None).to_numpy()
    timestep = np.linspace(0, 30, 1001)
    col_names = np.linspace(0.0005, 0.008, 1001)
    df = pd.DataFrame(data=np_data, columns=col_names)
    df.insert(0, 'timestep', timestep)
    df.to_csv('heat_data_pd.csv', index=False)


def visualize_data():
    df = pd.read_csv('heat_data_pd.csv')
    for i, col in enumerate(df.columns):
        if col != 'timestep':
            if i%20 == 0:
                plt.plot(df['timestep'], df[col], label=col)
    plt.xlabel("Timestep")
    plt.ylabel("Temperature, K")
    plt.show()


def genereate_epde_objects_pool(data_path):
    df = pd.read_csv(data_path)
    grid_steps_x = np.array(df.columns)[1:].astype(np.float)
    grid_steps_y = np.array(df['timestep'])
    grid = np.meshgrid(grid_steps_y, grid_steps_x, indexing='ij')
    df = df.drop('timestep', axis=1)
    data = df.to_numpy()

    dimensionality = data.ndim
    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power'])
    custom_inverse_function_evaluator = CustomEvaluator(evaluation_functions=custom_inverse_eval_fun,
                                                        eval_fun_params_labels=['dim', 'power'],
                                                        use_factors_grids=True)
    inverse_function_params_ranges = {'power': (1, 2),
                                      'dim': (0, dimensionality - 1)}
    custom_inverse_function_tokens = CustomTokens(token_type='inverse',
                                                  token_labels=['1/x_[dim]'],
                                                  evaluator=custom_inverse_function_evaluator,
                                                  params_ranges=inverse_function_params_ranges,
                                                  params_equality_ranges=None)
    epde_search_obj = epde_alg.epde_search(dimensionality=dimensionality)
    epde_search_obj.create_pool(data=data,
                                max_deriv_order=(1, 2),
                                boundary=dimensionality,
                                additional_tokens=[custom_inverse_function_tokens],
                                method='poly',
                                coordinate_tensors=grid)
    return epde_search_obj.pool


def start_searching():
    # visualize_data()
    timeout = datetime.timedelta(minutes=10)
    pool = genereate_epde_objects_pool(data_path='heat_data_pd.csv')
    tokens = generate_tokens(pool)
    initial_population = generate_initial_assumption(tokens)
    rules = [has_no_self_cycled_nodes,
             has_no_isolated_components,
             has_no_cycle,
             has_no_isolated_nodes,
             _has_no_single_input_operations,
             _has_sum_as_root_node,
             _has_no_double_operations,
             has_one_root]

    requirements = PipelineComposerRequirements(
        primary=tokens,
        secondary=OPERATIONS_LIST, max_arity=5,
        max_depth=3, pop_size=10, num_of_generations=50,
        crossover_prob=0.8, mutation_prob=0.9, timeout=timeout)

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[MutationTypesEnum.simple,
                        MutationTypesEnum.local_growth],
        crossover_types=[CrossoverTypesEnum.subtree],
        regularization_type=RegularizationTypesEnum.none)

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=OptGraph, base_node_class=OptNode),
        rules_for_constraint=rules)

    optimizer = EvoGraphOptimiser(
        graph_generation_params=graph_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_graph=initial_population)

    optimized_network = optimizer.optimise(PdeEvaluator(pool))
    optimized_network.show()


def check_ideal_score():
    ideal_graph = OptGraph()
    node1 = OptNode(content={'name': 'du/dx1'})
    node2 = OptNode(content={'name': 'd^2u/dx2^2'})

    node4 = OptNode(content={'name': 'du/dx2'})
    node5 = OptNode(content={'name': '1/x_[dim]'})

    node3 = OptNode(content={'name': 'mul'}, nodes_from=[node4, node5])

    node_root = OptNode(content={'name': 'sum'}, nodes_from=[node1, node2, node3])

    ideal_graph.add_node(node_root)
    ideal_graph.show()


    pool = genereate_epde_objects_pool(data_path='heat_data_pd.csv')

    pde_ev = PdeEvaluator(pool)
    print(pde_ev(ideal_graph))


if __name__ == '__main__':
    visualize_data()