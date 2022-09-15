import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epde.evaluators import CustomEvaluator
from epde.interface.prepared_tokens import CustomTokens
import epde.interface.interface as epde_alg
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.validation_rules import has_no_self_cycled_nodes, has_no_cycle
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters, EvoGraphOptimiser
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.graph import OptNode, OptGraph
from fedot.core.optimisers.optimizer import GraphGenerationParams
from sklearn.linear_model import LinearRegression

OPERATIONS_LIST = ['mul', 'sum']

def form_csv():
    np_data = np.genfromtxt('Data_32_points_.dat', delimiter=' ')
    timestep = np_data[:, 0]
    np_data = np.delete(np_data, 0, axis=1)
    col_names = np.linspace(0.5, 16, 32)
    df = pd.DataFrame(data=np_data, columns=col_names)
    df.insert(0, 'timestep', timestep)
    df.to_csv('wire_heat_data.csv', index=False)


def visualize_data():
    df = pd.read_csv('wire_heat_data.csv')
    for col in df.columns:
        if col != 'timestep':
            plt.plot(df['timestep'], df[col], label=col)
            plt.legend(ncol=3)
    plt.xlabel("Timestep")
    plt.ylabel("Temperature, K")
    plt.show()


def operations_node_has_correct_children(node: OptNode):
    """
    Recursive search incorrect input (children) in nodes from root
    If in operation with multiple input enters single input it causes an error
    """
    children = [operations_node_has_correct_children(children_node) for children_node in node.nodes_from]
    if node.content['name'] in OPERATIONS_LIST and len(children) < 2:
        raise ValueError(f'Operation {node.content["name"]} has incorrect children')
    if len(children) < 1:
        return node
    return children


def _has_no_single_input_operations(graph):
    """
    Validation rule for detecting single inputs in operations for multiple input
    """
    _ = operations_node_has_correct_children(graph.root_node)
    return True


def genereate_epde_objects_pool(data_path, data_max_rows):
    df = pd.read_csv(data_path)
    df = df[:data_max_rows]
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


def generate_tokens(pool):
    return [token for family in pool.families for token in family.tokens]


def generate_graph(tokens, min_size=1, max_size=5):
    """
    Generates graph structure with simple rules, which contains all tokens and random operations between them

    :param tokens: list with epde token objects
    :param min_size: min arity for graph
    :param max_size: max arity for graph
    :return: graph
    """
    tokens = tokens.copy()
    token_nodes = [OptNode(content={'name': token}) for token in tokens]
    np.random.shuffle(token_nodes)
    nodes = []

    i = 0
    while i < len(token_nodes):
        parents_indx = random.randrange(min_size, max_size) + i
        op = random.choice(OPERATIONS_LIST)
        nodes.append(OptNode(content={'name': op}, nodes_from=token_nodes[i:parents_indx]))
        i = parents_indx

    root = OptNode(content={'name': 'sum'}, nodes_from=nodes)
    graph = OptGraph(nodes=[root, *nodes, *token_nodes])

    return graph


def generate_initial_assumption(tokens, size=10):
    """
    Generates list with graphs - initial assumption to start composing

    :param tokens: list with epde token objects
    :param size: size of initial population
    :return: list with initial population
    """
    population = []
    while len(population) < size:
        try:
            graph = generate_graph(tokens)
            _ = _has_no_single_input_operations(graph)
            population.append(graph)
        except ValueError:
            pass
    return population


def simplify_node(node: OptNode):
    children = [simplify_node(children_node) for children_node in node.nodes_from]
    new_children = []
    if node.content['name'] in OPERATIONS_LIST:
        for child in children:
            if node.content['name'] == child.content['name']:
                new_children = [*new_children, *child.nodes_from]
            else:
                new_children = [*new_children, *child]
            return new_children
    else:
        new_children.append(node)
        return new_children
    return children

def simplify_graph(graph):
    simplify_node(graph.root_node)


class PdeEvaluator:
    def __init__(self, pool, coef_threshold=0.01):
        self.pool = pool
        self.coef_threshold = coef_threshold

    def __call__(self, graph: OptGraph):
        graph.show()
        #simplified_root = simplify_node(graph.root_node)
        simplified_root = self.simplify(graph.root_node)
        old_root = graph.root_node
        graph.delete_subtree(old_root)
        graph.add_node(simplified_root)
        graph.show()
        terms = self.evaluate(graph.root_node)
        if len(terms) < 2:
            return [float('inf'), float('inf')]
        y = -terms[-1]
        X = np.asarray(terms[:-1])
        X = np.transpose(X)
        lr = LinearRegression()
        lr.fit(X, y)
        indices_to_del = [i for i in range(0, len(lr.coef_)) if abs(lr.coef_[i]) < self.coef_threshold]
        to_del = [simplified_root.nodes_from[i] for i in indices_to_del]
        for node in to_del:
            graph.delete_subtree(node)
        #graph.show()
        if X.shape[1] == len(indices_to_del):
            return [float('inf'), float('inf')]
        X = np.delete(X, indices_to_del, axis=1)
        lr.fit(X, y)
        f = np.sum((lr.predict(X) - y) ** 2)
        return [f, graph.length]

    def _evaluate_token(self, node: OptNode):
        if 'token' not in node.content:
            for family in self.pool.families:
                if node.content['name'] in family.tokens:
                    factor_obj = family.create(node.content['name'])
                    node.content['token'] = factor_obj[1]
        token_values = node.content['token'].evaluate()
        flatten_token_values = token_values.flatten()
        return [flatten_token_values]

    def _evaluate_operation(self, node: OptNode):
        operation = node.content['name']
        result = []
        for node in node.nodes_from:
            if operation == 'sum':
                evaluated_node = self.evaluate(node)
                if type(evaluated_node) is not list:
                    evaluated_node = [evaluated_node]
                result = [*result, *evaluated_node]
            if operation == 'mul':
                if len(result) == 0:
                    result = self.evaluate(node)
                else:
                    result = [ev1 * ev2 for ev1 in result for ev2 in self.evaluate(node)]
        return result

    def evaluate(self, opt_node: OptNode):
        if opt_node.content['name'] in OPERATIONS_LIST:
            return self._evaluate_operation(opt_node)
        else:
            return self._evaluate_token(opt_node)


    def simplify(self, opt_node: OptNode):
        if opt_node.content['name'] != 'mul' and opt_node.content['name'] != 'sum':
            return opt_node
        simplified_children = [self.simplify(children_node) for children_node in opt_node.nodes_from]
        opt_node = OptNode(content=opt_node.content.copy(), nodes_from=[])
        if len(simplified_children) == 1:
            return simplified_children[0]

        if opt_node.content['name'] == 'sum':
            for child in simplified_children:
                if child is not None:
                    if child.content['name'] == 'sum':
                        opt_node.nodes_from += child.nodes_from
                    else:
                        opt_node.nodes_from.append(child)
        elif opt_node.content['name'] == 'mul':
            opt_node = None
            for child in simplified_children:
                if child is not None:
                    if opt_node is None:
                        opt_node = child
                    else:
                        def multiply_nodes(node1: OptNode, node2: OptNode):
                            if node1.content['name'] == 'mul':
                                if node2.content['name'] == 'mul':
                                    return OptNode(content={'name': 'mul'},
                                                   nodes_from=[*node1.nodes_from, *node2.nodes_from])
                                elif node2.content['name'] == 'sum':
                                    return OptNode(content={'name': 'sum'},
                                                   nodes_from=[multiply_nodes(node1, n2) for n2 in node2.nodes_from])
                                else:
                                    return OptNode(content={'name': 'mul'}, nodes_from=[*node1.nodes_from, node2])
                            elif node1.content['name'] == 'sum':
                                if node2.content['name'] == 'mul':
                                    return multiply_nodes(node2, node1)
                                elif node2.content['name'] == 'sum':
                                    return OptNode(content={'name': 'sum'},
                                                   nodes_from=[multiply_nodes(n1, n2) for n1 in node1.nodes_from for n2 in
                                                               node2.nodes_from])
                                else:
                                    return OptNode(content={'name': 'sum'},
                                                   nodes_from=[multiply_nodes(n1, node2) for n1 in node1.nodes_from])
                            else:
                                if node2.content['name'] == 'mul' or node2.content['name'] == 'sum':
                                    return multiply_nodes(node2, node1)
                                else:
                                    return OptNode(content={'name': 'mul'}, nodes_from=[node1, node2])
                        opt_node = multiply_nodes(opt_node, child)
        return opt_node


def start_searching():
    #visualize_data()
    timeout = datetime.timedelta(minutes=10)
    pool = genereate_epde_objects_pool(data_path='wire_heat_data.csv',
                                       data_max_rows=400)
    tokens = generate_tokens(pool)
    initial_population = generate_initial_assumption(tokens)
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_single_input_operations]

    requirements = PipelineComposerRequirements(
        primary=tokens,
        secondary=OPERATIONS_LIST, max_arity=10,
        max_depth=3, pop_size=5, num_of_generations=20,
        crossover_prob=0.8, mutation_prob=0.9, timeout=timeout)

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[MutationTypesEnum.simple,
                        MutationTypesEnum.local_growth,
                        MutationTypesEnum.single_add,
                        MutationTypesEnum.single_drop],
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


if __name__ == '__main__':
    start_searching()
