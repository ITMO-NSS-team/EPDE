import numpy as np
from scipy.io import loadmat

import pandas as pd
import pysindy as ps
import time
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)


np.random.seed(100)
integrator_keywords = {'rtol': 1e-12, 'method': 'LSODA', 'atol': 1e-12}

path_full = os.path.join(Path().absolute().parent, "data_kdv", "kdv.mat")
kdV = loadmat(path_full)
t = np.ravel(kdV['t'])
x = np.ravel(kdV['x'])
u = np.real(kdV['usol'])

dt = t[1] - t[0]
dx = x[1] - x[0]

u = u.reshape(len(x), len(t), 1)

library_functions = [lambda x: x, lambda x: x * x]#, lambda x: np.cos(x)*np.cos(x)]#, lambda x: 1/x]
library_function_names = [lambda x: x, lambda x: x + x]#, lambda x: 'cos(' +x+ ')'+'sin(' +x+ ')']#, lambda x: '1/'+x]

''' Parameters of the experiment '''
iter_number = 1
print_results = True
write_csv = False


time_ls = []
for i in range(iter_number):
    start = time.time()

    pde_lib = ps.PDELibrary(library_functions=library_functions,
                            function_names=library_function_names,
                            derivative_order=3, spatial_grid=x,
                            include_bias=True, is_uniform=True, include_interaction=True)
    feature_library = ps.feature_library.PolynomialLibrary(degree=3)
    optimizer = ps.SR3(threshold=7, max_iter=10000, tol=1e-15, nu=1e2,
                       thresholder='l0', normalize_columns=True)
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
    df.to_csv(os.path.join(Path().absolute().parent, "data_kdv", f"time_pys_kdv.csv"))
