import numpy as np
from scipy.io import loadmat
import time
import pandas as pd
import pysindy as ps
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)


np.random.seed(100)
integrator_keywords = {'rtol': 1e-12, 'method': 'LSODA', 'atol': 1e-12}

path_full = os.path.join(Path().absolute().parent, "data_burg", "burgers.mat")
burg = loadmat(path_full)
t = np.ravel(burg['t'])
x = np.ravel(burg['x'])
u = np.real(burg['usol'])

dt = t[1] - t[0]
dx = x[1] - x[0]
u = u.reshape(len(x), len(t), 1)

library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]

''' Parameters of the experiment '''
iter_number = 1
print_results = False
write_csv = False


a = True
b = False
time_ls = []
for i in range(iter_number):
    start = time.time()
    pde_lib = ps.PDELibrary(library_functions=library_functions,
                            function_names=library_function_names,
                            derivative_order=3, spatial_grid=x,
                            include_bias=b, is_uniform=a, include_interaction=True)

    optimizer = ps.STLSQ(threshold=2, alpha=1e-5, normalize_columns=True)

    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, t=dt)
    end = time.time()
    time1 = end-start
    time_ls.append(time1)
    model.print()

if print_results:
    print('Run time')
    for t in time_ls:
        print(t, 's')

if write_csv:
    arr = np.array([time_ls])
    arr = arr.T
    df = pd.DataFrame(data=arr, columns=['time'])
    df.to_csv(os.path.join(Path().absolute().parent, "data_burg", "time_pys_burg.csv"))
