import datetime
import random

import numpy
import numpy as np
from epde.evaluators import CustomEvaluator

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import GraphOptimiser, GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum, EvoGraphOptimiser
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from sklearn.linear_model import LinearRegression
import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens
from epde.interface.prepared_tokens import CustomTokens

random.seed(12)
np.random.seed(12)


def generate_pool():
    t_max = 400

    file = np.loadtxt('Data_32_points_.dat',
                      delimiter=' ', usecols=range(33))

    x = np.linspace(0.5, 16, 32)
    t = file[:t_max, 0]
    grids = np.meshgrid(t, x, indexing='ij')
    u = file[:t_max, 1:]

    boundary = 2
    dimensionality = u.ndim

    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power'])
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun,
                                               eval_fun_params_labels=['dim', 'power'], use_factors_grids=True)

    inv_fun_params_ranges = {'power': (1, 2), 'dim': (0, dimensionality - 1)}

    custom_inv_fun_tokens = CustomTokens(token_type='inverse',
                                         # Выбираем название для семейства токенов - обратных функций.
                                         token_labels=['1/x_[dim]', ],
                                         # Задаём названия токенов семейства в формате python-list'a.
                                         # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
                                         evaluator=custom_inv_fun_evaluator,
                                         # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                         params_ranges=inv_fun_params_ranges,
                                         # Используем заявленные диапазоны параметров
                                         params_equality_ranges=None)  # Используем None, т.к. значения по умолчанию
    # (равенство при лишь полном совпадении дискретных параметров)
    # нас устраивает.

    trig_tokens = TrigonometricTokens()
    epde_search_obj = epde_alg.epde_search(dimensionality=dimensionality)
    epde_search_obj.create_pool(data=u, max_deriv_order=(1, 2), boundary=boundary,
                                additional_tokens=[custom_inv_fun_tokens], method='poly',
                                #method_kwargs={'epochs_max': 20},
                                coordinate_tensors=grids)
    return epde_search_obj.pool


class PdeFitness:
    def __init__(self, pool, coef_threshold=0.01):
        self.pool = pool
        self.coef_threshold = coef_threshold

    def __call__(self, graph: OptGraph):

        graph.show()
        new_root = self.simplify(graph.root_node)
        self.check(new_root)
        old_root = graph.root_node
        self.check(old_root)
        graph.delete_subtree(old_root)
        graph.add_node(new_root)
        graph.show()
        terms = self.evaluate(graph.root_node)
        if len(terms) < 2:
            return [float('inf'), float('inf')]
        lr = LinearRegression()
        y = -terms[-1]
        X = numpy.asarray(terms[:-1])
        X = numpy.transpose(X)
        lr.fit(X, y)
        indices_to_del = [i for i in range(0, len(lr.coef_)) if abs(lr.coef_[i]) < self.coef_threshold]
        to_del = [new_root.nodes_from[i] for i in indices_to_del]
        for node in to_del:
            graph.delete_subtree(node)
        graph.show()
        self.check(graph.root_node)
        if X.shape[1] == len(indices_to_del):
            return [float('inf'), float('inf')]
        X = numpy.delete(X, indices_to_del, axis=1)
        lr.fit(X, y)
        f = np.sum((lr.predict(X) - y) ** 2)
        #print(f)
        return [f, graph.length]

    def check(self, node: OptNode):
        if node.content['name'] == 'sum' or node.content['name'] == 'mul':
            for ch in node.nodes_from:
                self.check(ch)
        else:
            if len(node.nodes_from) != 0:
                print("wtf")

    def evaluate(self, opt_node: OptNode):
        if opt_node.content['name'] == 'sum':
            res = []
            for node in opt_node.nodes_from:
                ev = self.evaluate(node)
                if type(ev) is list:
                    res = [*res, *ev]
                else:
                    res.append(ev)
            return res
        elif opt_node.content['name'] == 'mul':
            res = []
            for node in opt_node.nodes_from:
                if len(res) == 0:
                    res = self.evaluate(node)
                else:
                    res = [ev1 * ev2 for ev1 in res for ev2 in self.evaluate(node)]
            return res
        else:
            if 'token' not in opt_node.content:
                for family in self.pool.families:
                    if opt_node.content['name'] in family.tokens:
                        opt_node.content['token'] = family.create(opt_node.content['name'])[1]
            return [opt_node.content['token'].evaluate().flatten()]

    def simplify(self, opt_node: OptNode):

        if opt_node.content['name'] != 'mul' and opt_node.content['name'] != 'sum':
            return opt_node

        simplified_children = [self.simplify(children_node) for children_node in opt_node.nodes_from]
        opt_node = OptNode(content=opt_node.content.copy(), nodes_from=[])
        if len(simplified_children) == 1:
            return simplified_children[0]

        if opt_node.content['name'] == 'sum':
            for child in simplified_children:
                if child.content['name'] == 'sum':
                    opt_node.nodes_from += child.nodes_from
                else:
                    opt_node.nodes_from.append(child)
        elif opt_node.content['name'] == 'mul':
            opt_node = None
            for child in simplified_children:
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


def generate_tokens(pool):
    return [token for family in pool.families for token in family.tokens]


def generate_graph(tokens, min_size=1, max_size=4):
    tokens = tokens.copy()
    token_nodes = [OptNode(content={'name': token}) for token in tokens]
    numpy.random.shuffle(token_nodes)
    nodes = []

    i = 0
    while i < len(token_nodes):
        r = random.randrange(min_size, max_size) + i
        op = random.choice(['sum', 'mul'])
        nodes.append(OptNode(content={'name': op}, nodes_from=token_nodes[i:r]))
        i = r

    root = OptNode(content={'name': 'sum'}, nodes_from=nodes)
    return OptGraph(nodes=[root, *nodes, *token_nodes])


def main(timeout: datetime.timedelta = None):
    if not timeout:
        timeout = datetime.timedelta(minutes=10)

    pool = generate_pool()

    tokens = generate_tokens(pool)
    rules = [has_no_self_cycled_nodes, has_no_cycle]

    initial = [generate_graph(tokens) for _ in range(10)]

    requirements = PipelineComposerRequirements(
        primary=tokens,
        secondary=['sum', 'mul'], max_arity=10,
        max_depth=3, pop_size=5, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.9, timeout=timeout)

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[MutationTypesEnum.simple, MutationTypesEnum.local_growth, ],
        crossover_types=[CrossoverTypesEnum.subtree],
        regularization_type=RegularizationTypesEnum.none)

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=OptGraph, base_node_class=OptNode),
        rules_for_constraint=rules)

    optimizer = EvoGraphOptimiser(
        graph_generation_params=graph_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_graph=initial,
        log=default_log(logger_name='logger', verbose_level=1))

    optimized_network = optimizer.optimise(PdeFitness(pool))

    optimized_network.show()


if __name__ == '__main__':
    main()