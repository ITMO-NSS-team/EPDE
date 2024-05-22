import numpy as np
from scipy.io import loadmat
import time
import pandas as pd
import pysindy as ps
import warnings
import os
from pathlib import Path
import matplotlib.pyplot as plt
from pysindy_calc_mae import calc_difference
import pickle
warnings.filterwarnings("ignore", category=UserWarning)


np.random.seed(100)
integrator_keywords = {'rtol': 1e-12, 'method': 'LSODA', 'atol': 1e-12}

path_full = os.path.join(Path().absolute().parent, "data_burg", "burgers.mat")
burg = loadmat(path_full)
t = np.ravel(burg['t'])
x = np.ravel(burg['x'])
u_init = np.real(burg['usol'])
u_init = u_init.reshape(len(x), len(t), 1)

dt = t[1] - t[0]
dx = x[1] - x[0]

library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]
true_coef = [0.1, -1.001]
true_names = ["x0_11", "x0x0_1"]

''' Parameters of the experiment '''
iter_number = 50
write_csv = True
print_results = True
eq_type = "data_pysindy_burg"

draw_not_found = []
draw_time = []
draw_mae = []
magnitudes = [0, 0.0075,  0.015,  0.0225, 0.03]
magnames = ["0", "0.0075",  "0.015",  "0.0225", "0.03"]

for magnitude, magname in zip(magnitudes, magnames):
    title = f"dfp{magname}"
    time_ls = []
    mae_ls = []
    found_ls = []
    for i in range(iter_number):
        if magnitude == 0:
            u = u_init
        else:
            u = u_init + np.random.normal(scale=magnitude * np.abs(u_init), size=u_init.shape)
        start = time.time()
        pde_lib = ps.PDELibrary(library_functions=library_functions,
                                function_names=library_function_names,
                                derivative_order=3, spatial_grid=x,
                                include_bias=False, is_uniform=True, include_interaction=True)

        optimizer = ps.STLSQ(threshold=2, alpha=1e-5, normalize_columns=True)

        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        model.fit(u, t=dt)
        end = time.time()
        time1 = end-start
        time_ls.append(time1)
        model.print()

        eq = model.equations(17)

        # path_exp = os.path.join(Path().absolute().parent, eq_type, "equations", f"{title}_{i}.pickle")
        # with open(path_exp, "wb") as f:
        #     pickle.dump(eq, f)

        mae, found_flag = calc_difference(eq[0], true_coef, true_names)
        mae_ls.append(mae)

        found_ls.append(found_flag)

    arr = np.array([mae_ls, time_ls, found_ls])
    arr = arr.T
    df = pd.DataFrame(data=arr, columns=["MAE", 'time', "found"])

    draw_time.append(sum(time_ls) / iter_number)
    draw_mae.append(df.MAE.mean())
    draw_not_found.append(iter_number - sum(found_ls))
    if write_csv:
        arr = np.array([mae_ls, time_ls, found_ls])
        arr = arr.T
        df = pd.DataFrame(data=arr, columns=["MAE", 'time', "found"])
        df.to_csv(os.path.join(Path().absolute().parent, "data_pysindy_burg", f"{title}.csv"))
    if print_results:
        print(f"Average time, s: {sum(time_ls) / iter_number:.2f}")
        print(f"Average min MAE: {df.MAE.mean():.2f}")
        print(f"Average # of found eqs: {sum(found_ls) / iter_number:.2f}")

plt.title("Pysindy")
plt.plot(magnitudes, draw_not_found, linewidth=2, markersize=9, marker='o')
plt.ylabel("No. runs with not found eq.")
plt.xlabel("Magnitude value")
plt.grid()
plt.show()

plt.plot(magnitudes, draw_time, linewidth=2, markersize=9, marker='o')
plt.title("Pysindy")
plt.ylabel("Time, s.")
plt.xlabel("Magnitude value")
plt.grid()
plt.show()

plt.plot(magnitudes, draw_mae, linewidth=2, markersize=9, marker='o')
plt.title("Pysindy")
plt.ylabel("Average MAE")
plt.xlabel("Magnitude value")
plt.grid()
plt.show()

