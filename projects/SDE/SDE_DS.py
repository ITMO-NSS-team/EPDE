import os
import sys
import datetime
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

from typing import List
import statsmodels.api as sm
from epde import EpdeSearch
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pandas as pd
import torch
from epde.interface.prepared_tokens import TrigonometricTokens, CustomTokens, CustomEvaluator, PreparedTokens
from sklearn import preprocessing
from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.device import solver_device, check_device, device_type
from tedeous.callbacks import early_stopping
from tedeous.data import Domain, Conditions
# from mpl_interactions import ioff, panhandler, zoom_factory
from scipy import stats
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')


#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#current_directory = os.getcwd()
#print("Текущий рабочий каталог:", current_directory)

def preprocess_data(inputs: np.ndarray):
    '''
    Calculate trend, standard deviation and mean value of the input time series, passed as `np.ndarray`.
    '''
    inputs = preprocessing.normalize([inputs]).reshape((21,))
    noise = sm.tsa.tsatools.detrend(inputs, order=2, axis=1)
    trend = inputs - noise
    std = np.std(inputs)
    mean = np.var(inputs)
    return inputs, trend, std, mean

def multiple_monte_carlo_sampling(mean, std, sample_size, samples_num = 15) -> List[np.ndarray]:
    multiple_samples = []
    for N in range(samples_num):
        samples = np.random.normal(mean, std, sample_size)
        multiple_samples.append(samples)
    return multiple_samples

class MonteExp():
    def __init__(self, sample: np.ndarray):
        self._sample = sample
        self._stat = {'mean': np.mean(sample), 'std': np.std(sample)}

    def discover_equation(self, epde_search_obj: EpdeSearch, additional_tokens: List, max_factors: int):
        epde_search_obj.fit(data = [self._sample,], variable_names = ['u', ], max_deriv_order = 2,
                            equation_terms_max_number = 4, data_fun_pow = 1, additional_tokens = additional_tokens,
                            equation_factors_max_number=max_factors, eq_sparsity_interval=(1e-10, 1))
        return epde_search_obj.equations(only_print=False, only_str=False)[0][0].vals['u'].weights_final

def main():

    sample = np.array([197, 191, 189, 181, 175, 172, 173, 176, 174, 159, 153, 159, 154, 146, 139, 136, 126, 117, 110, 120, 119])
    inputs, trend, std, mean = preprocess_data(sample)
    normalized_arr, trend, std, mean = preprocess_data(sample)

    N_samples = multiple_monte_carlo_sampling(mean = mean, std = std, sample_size = 21)
    print(N_samples)

    summ_result = trend + N_samples
    summ_result = np.array(summ_result)


    experiments = [MonteExp(summ_result[i].reshape([21,])) for i in range(15)]


    dimensionality = 0
    t = np.arange(21)
    t = np.linspace(0, 1, len(t))
    print(t.shape)
    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], kwargs['power'])
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, eval_fun_params_labels=['dim', 'power']) # use_factors_grids=False
    grid_params_ranges = {'power': (1, 2), 'dim': (0, dimensionality)}

    epde_search_obj = EpdeSearch(use_solver=True, dimensionality=dimensionality, boundary=1,
                                 coordinate_tensors=[t, ])
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                     preprocessor_kwargs={'use_smoothing': True})
    popsize = 5
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=100)
    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq=(0, np.pi / 2))
    custom_grid_tokens = CustomTokens(token_type='grid',
                                      token_labels=['1/x_[dim]', ],
                                      evaluator=custom_inv_fun_evaluator,
                                      params_ranges= grid_params_ranges, #{'power' : (1, 1)},              #grid_params_ranges
                                      params_equality_ranges=None)
    factors_max_number = {'factors_num': [1, 2], 'probas': [0.85, 0.15]}


    vals = [exp.discover_equation(epde_search_obj, additional_tokens = [custom_grid_tokens,], max_factors=2) for exp in experiments]


    col_names = ['0', '1', '2', '3']
    df = pd.DataFrame(vals, columns = col_names)
    df.reset_index(drop=True, inplace=True)
    print(df)

    column_stats = df.aggregate(['mean', 'var'])
    print(column_stats)

    # Комбинируем статистику по запускам в словарь
    stat = {key: (column_stats.loc['mean', key], column_stats.loc['var', key]) for key in col_names}


    def monte(mean, variance, size):
        return np.random.normal(mean, variance, size)

    monte_vals = {col_name: (torch.tensor(monte(stats[0], stats[1], size = 50), requires_grad=True),
                             torch.tensor(stats[0], requires_grad=True))
                  for col_name, stats in stat.items()}


    solver_device('cpu')
    grid_res = 20

    domain = Domain()
    domain.variable('t', [0, 1], grid_res)
    boundaries = Conditions()
    boundaries.dirichlet({'t': 0}, value=normalized_arr[0])
    boundaries.dirichlet({'t': 0.5}, value=normalized_arr[10])


    monte_eqs = [{'monte1*du/dx': {'coeff': mean.item(), 'du/dx': [0], 'pow': 1, 'var': 0},
                  'monte2*u':     {'coeff': mean.item(), 'u': [None],  'pow': 1, 'var': 0},} for i in range(16)]


    # Также задаём eqs через list comprehension и потом с каждым соотносим нужную структуру уравнения
    eqs = [Equation() for i in range(16)]
    for eq_idx, eq in enumerate(eqs):
        eq.add(monte_eqs[eq_idx])

    h = 0.001
    lambda_bound = 100

    def get_ann() -> torch.nn.Sequential:
        return torch.nn.Sequential(
                                   torch.nn.Linear(1, 100),
                                   torch.nn.Tanh(),
                                   torch.nn.Linear(100, 100),
                                   torch.nn.Tanh(),
                                   torch.nn.Linear(100, 100),
                                   torch.nn.Tanh(),
                                   torch.nn.Linear(100, 100),
                                   torch.nn.Tanh(),
                                   torch.nn.Linear(100, 1)
                                   )

    anns = [get_ann() for eq in eqs]
    img_dir = os.path.join(os.path.dirname(__file__), 'eq_img')
    c_cache = cache.Cache(cache_verbose=False, model_randomize_parameter=1e-6)
    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         loss_window=10,
                                         no_improvement_patience=100,
                                         patience=5,
                                         randomize_parameter=1e-10)
    cb_plots = plot.Plots(save_every=None, print_every=None, img_dir=img_dir)
    optimizer = Optimizer('Adam', {'lr': 1e-2})


    start = time.time()
    for eq_idx, equation in enumerate(eqs):
        model = Model(anns[eq_idx], domain, equation, boundaries)
        model.compile('NN', lambda_operator=1, lambda_bound=lambda_bound, h=h)
        model.train(optimizer, 1e-4, save_model=False, callbacks=[cb_es, c_cache, cb_plots])

    end = time.time()
    print('Time taken 10= ', end - start)


    grid = domain.build('NN').cpu()
    grid = check_device(grid)
    print(f'grid_shape {grid.shape}')

    solutions = []
    for net_idx, net in enumerate(anns):
        anns[net_idx] = net.to(device=device_type())
        #output = net(grid)
        #solutions.append(output.detach().cpu())
        solutions.append(anns[net_idx](grid))#.detach().numpy().reshape(-1))

    sols_stacked = torch.stack(solutions, dim=0).mean(dim=0)
    avg_sol = sols_stacked.detach().numpy().reshape(-1)
    print(f'sols_stacked.shape {sols_stacked.shape}, t.shape {t.shape}, avg_sol.shape {avg_sol.shape}')
    #avg_sol = torch.stack(solutions[:21], dim=0).mean(dim=0)
    #avg_sol = avg_sol.numpy().reshape(-1)
    window_size = 3
    std_dev_average = np.zeros_like(avg_sol)  # стандартное отклонение
    for i in range(len(avg_sol)):
        std_dev_average[i] = np.std(avg_sol[i - window_size:i])#/2.: i + window_size/2.])
    co_interval = 2.576 * std_dev_average

    plt.figure(figsize=(12, 6))
    plt.plot(t, avg_sol, label='Average solution', color='slateblue', marker='.')
    #plt.fill_between(t, avg_sol - co_interval, avg_sol + co_interval, color='lightskyblue', alpha=0.3,
    #                label='CI 95 percent')
    #plt.plot(t, normalized_arr, label='Real Data')
    plt.ylim(0, 0.350)
    plt.xlim(0, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('Family of ODEs solved by TEDEouS')
    plt.show()

if __name__ == '__main__':
        main()