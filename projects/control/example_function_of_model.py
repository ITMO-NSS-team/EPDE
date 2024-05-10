import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.callbacks import adaptive_lambda, cache, early_stopping, plot
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

solver_device('cpu')

a = 4


def u(x, a):
  tanarg = torch.full_like(input = x, fill_value = 0.5)
  return 2*torch.atan(torch.exp(x)*torch.tan(tanarg))

def u_xx(x, a):
  return (torch.pi * a) ** 2 * torch.sin(torch.pi * a * x)

t0 = 0
tmax = 1
Nt = 99

domain = Domain()

domain.variable('t', [t0, tmax], Nt, dtype='float32')

boundaries = Conditions()

boundaries.dirichlet({'t': 0}, value=1)
#boundaries.dirichlet({'t': 1}, value=0)

grid = domain.variable_dict['t'].reshape(-1,1)

# equation: d2u/dx2 = -16*pi^2*sin(4*pi*x)

equation = Equation()

poisson = {
    'd2u/dx2':
        {
        'coeff': 1,
        'term': [0],
        'pow': 1,
        },

    'sin(u)':
        {
        'coeff': -1,
        'term': [[None],],
        'pow': [np.sin,],
        }
}

equation.add(poisson)

net = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

model = Model(net, domain, equation, boundaries)

model.compile('autograd', lambda_operator=1, lambda_bound=17)

# img_dir = os.path.join(os.path.dirname( __file__ ), 'function_of_model_img')

cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = early_stopping.EarlyStopping(eps=1e-5,
                                     loss_window=100,
                                     no_improvement_patience=1000,
                                     patience=5,
                                     info_string_every=1000,
                                     randomize_parameter=1e-5)

# cb_plots = plot.Plots(save_every=1000, print_every=None, img_dir=None)

#cb_lambda = adaptive_lambda.AdaptiveLambda()

optimizer = Optimizer('Adam', {'lr': 1e-3}, gamma=0.9, decay_every=1000)
# cb_plots
model.train(optimizer, 1e5, save_model=True, callbacks=[cb_cache, cb_es]) 

plt.plot(grid.detach().numpy(), u(grid,a).detach().numpy(), label='Exact')
plt.plot(grid.detach().numpy(), net(grid).detach().numpy(), '--', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.show()