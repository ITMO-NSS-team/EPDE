import datetime
import random
import math

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
from sklearn.linear_model import LinearRegression

OPERATIONS_LIST = ['mul', 'sum']

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


def _has_sum_as_root_node(graph):
    """
    Validation rule for selecting only graphs with sum node as root
    """
    if graph.root_node.content['name'] != 'sum':
        raise ValueError(f'Root node is {graph.root_node.content["name"]}, not sum!')
    return True


def _has_no_double_operations(graph):
    for node in graph.nodes:
        if node.content['name'] in OPERATIONS_LIST:
            for child_node in node.nodes_from:
                if node.content['name'] == child_node.content['name']:
                    raise ValueError(f'Two same operations in subsequence')
    return True

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


def simplify_graph(graph):
    """
    Function for removing double nodes of operations
    """
    for node in graph.nodes:
        if node.content['name'] in OPERATIONS_LIST:
            for child_node in node.nodes_from:
                if node.content['name'] == child_node.content['name']:
                    new_children = child_node.nodes_from
                    graph.delete_subtree(child_node)
                    node.nodes_from = [*node.nodes_from, *new_children]
                    graph.nodes = [*graph.nodes, *new_children]
                    graph = simplify_graph(graph)
    if isinstance(graph.root_node, list):
        root_node = graph.root_node[0]
        graph = OptGraph()
        graph.add_node(root_node)
    return graph


def calculate_frobenius_norm(matrix):
    """
    Function for calculation the significance if element
    """
    return np.linalg.norm(matrix)


class PdeEvaluator:
    def __init__(self, pool, coef_threshold=0.01):
        self.pool = pool
        self.coef_threshold = coef_threshold

    def __call__(self, graph: OptGraph):
        graph = simplify_graph(graph)
        terms = self.evaluate_graph(graph)
        if len(terms) < 2:
            return [float('inf'), float('inf')]
        y_index, y = self._select_definition_right_part(terms)
        X = np.transpose(np.asarray(terms[:y_index] + terms[y_index + 1:]))
        lr = LinearRegression()
        lr.fit(X, y)

        # временно удаляем из графа узел, который вынесен в правую часть
        right_node = graph.root_node.nodes_from[y_index]
        graph.delete_subtree(right_node)
        # удаляем узлы вклад которых незначителен
        indices_to_del = [i for i in range(0, len(lr.coef_)) if abs(lr.coef_[i]) < self.coef_threshold]
        to_del = [graph.root_node.nodes_from[i] for i in indices_to_del]
        for node in to_del:
            graph.delete_subtree(node)
        # проверяем, что с двух сторон разные токены
        if not self._check_unique_definition_parts(graph, right_node):
            return [float('inf'), float('inf')]
        # проверяем, что с одной стороны нет одинаковых токенов
        if not self._check_unique_elements_for_parts(graph):
            return [float('inf'), float('inf')]
        # проверяем, не удалим ли мы всю выборку
        if X.shape[1] == len(indices_to_del):
            return [float('inf'), float('inf')]
        # удаляем столбцы таблицы, которые незначительны
        X = np.delete(X, indices_to_del, axis=1)
        # обучаем коэффициенты заново
        lr.fit(X, y)
        # считаем rmse
        _lambda = 1000
        f = math.sqrt(np.sum((lr.predict(X) - y) ** 2)) + _lambda*(np.linalg.norm(lr.coef_, ord=1))
        print('___________________________')
        print(f'{graph.root_node.nodes_from} = {right_node.content["name"]}')
        for i in range(len(graph.root_node.nodes_from)):
            graph.root_node.nodes_from[i].content['params'] = {'coeff': round(lr.coef_[i], 3)}
            print(f'{round(lr.coef_[i], 3)}*({graph.root_node.nodes_from[i]})')
        print(f'Raw score: {math.sqrt(np.sum((lr.predict(X) - y) ** 2))}')
        print(f'Score {f}')
        graph.root_node.nodes_from = [*graph.root_node.nodes_from, right_node]
        graph.add_node(right_node)
        # graph.show()
        return [f, graph.length]

    def _check_unique_definition_parts(self, graph, right_part):
        for node in graph.root_node.nodes_from:
            if node.content['name'] not in OPERATIONS_LIST:
                if node.content['name'] == right_part.content["name"]:
                    return False
            if node.content['name'] in OPERATIONS_LIST and right_part.content["name"] in OPERATIONS_LIST:
                left_part_nodes = [n.content['name'] for n in node.nodes_from]
                right_part_nodes = [n.content['name'] for n in right_part.nodes_from]
                intersection = set(left_part_nodes) & set(right_part_nodes)
                if len(intersection) > 0:
                    return False
        return True

    def _check_unique_elements_for_parts(self, graph):
        names = [node.content['name'] for node in graph.root_node.nodes_from]
        names_without_operations = [name for name in names if name not in OPERATIONS_LIST]
        if len(set(names_without_operations)) != len(names_without_operations):
            return False
        return True

    def _select_definition_right_part(self, parts):
        """
        Function to select the index of right part and right part matrix

        """
        norm_values = [calculate_frobenius_norm(part) for part in parts]
        max_value_index = norm_values.index(max(norm_values))
        return max_value_index, parts[max_value_index]

    def _evaluate_subtree(self, graph):
        """
        Function to return the matrix which is the result of subtree operations and tokens
        """
        components_nodes = graph.root_node.nodes_from
        tokens_calculated_list = []
        for node in components_nodes:
            if node.content['name'] in OPERATIONS_LIST:
                subtree_graph = OptGraph()
                subtree_graph.add_node(node)
                tokens_calculated_list.append(self._evaluate_subtree(subtree_graph))
            else:
                tokens_calculated_list.append(self._evaluate_token(node.content['name']))
        if graph.root_node.content['name'] == 'sum':
            result = np.sum(tokens_calculated_list, axis=0)
        if graph.root_node.content['name'] == 'mul':
            result = math.prod(tokens_calculated_list)
        return result

    def evaluate_graph(self, graph):
        """
        Function to return main components of definition as list of matrices
        """
        components_nodes = graph.root_node.nodes_from
        calculated_tokens = []
        for node in components_nodes:
            if node.content['name'] not in OPERATIONS_LIST:
                calculated_tokens.append(self._evaluate_token(node.content['name']))
            else:
                subtree_graph = OptGraph()
                subtree_graph.add_node(node)
                calculated_tokens.append(self._evaluate_subtree(subtree_graph))
        return calculated_tokens

    def _evaluate_token(self, name: str):
        """
        Function to return matrices by its token name
        """
        token = None
        for family in self.pool.families:
            if name in family.tokens:
                factor_obj = family.create(name)
                token = factor_obj[1]
        if token is None:
            raise ValueError(f"{name} not found")
        token_values = token.evaluate()
        flatten_token_values = token_values.flatten()
        return flatten_token_values