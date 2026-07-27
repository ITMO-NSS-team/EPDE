"""EPDE discovery of Lotka-Volterra from the REAL Hudson's Bay lynx-hare records.

Data: hudson-bay-lynx-hare.csv (1900-1920, 21 annual points). u = hare (prey),
v = lynx (predator). The OLS signal-check (real_lynxhare_check.py) confirmed all four
LV signs are canonical (R^2=0.93/0.88), so this is a genuine target.

21 points is below EPDE's practical floor for derivative estimation, so by default the
series is densified with a cubic spline (the populations vary smoothly year-to-year) to
~210 points. Set densify=1 to run on the raw 21 points instead. Autonomous system -> no
trig/grid tokens; LV's u*v product is assembled from the u and v base tokens via
factors_num=[1,2].
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from scipy.interpolate import CubicSpline

from epde.interface.interface import EpdeSearch

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, 'hudson-bay-lynx-hare.csv')


def load_lynxhare(densify=10):
    rows = []
    with open(CSV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line[0].isalpha():
                continue
            rows.append([float(x) for x in line.split(',')])
    arr = np.array(rows)
    year, lynx, hare = arr[:, 0], arr[:, 1], arr[:, 2]
    t = (year - year[0]).astype(np.float64)
    H, V = hare.astype(np.float64), lynx.astype(np.float64)        # prey, predator
    if densify and densify > 1:
        tt = np.linspace(t[0], t[-1], (t.size - 1) * densify + 1)
        H = CubicSpline(t, H)(tt)
        V = CubicSpline(t, V)(tt)
        t = tt
    return t, H, V


def lv_discovery(densify=10, poly_window=21, poly_order=4, boundary=10,
                 pop=32, epochs=18, terms=8, data_fun_pow=1):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t, u, v = load_lynxhare(densify=densify)
    dt = float(np.median(np.diff(t)))
    print(f"[lv real] pts={t.size} dt={dt:.3f}yr densify={densify} "
          f"u(hare)[{u.min():.1f},{u.max():.1f}] v(lynx)[{v.min():.1f},{v.max():.1f}]")

    epde = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True,
                      boundary=boundary, coordinate_tensors=(t,),
                      verbose_params={'show_iter_idx': True}, device=device)
    epde.set_preprocessor(default_preprocessor_type='poly',
                          preprocessor_kwargs={'polynomial_window': poly_window,
                                               'poly_order': poly_order})

    epde.set_moeadd_params(population_size=pop, training_epochs=epochs)
    factors_max_number = {'factors_num': [1, 2], 'probas': [0.7, 0.3]}

    # data_fun_pow=1: LV has NO squared terms -- u*v comes from combining the u and v
    # base tokens (factors_num=2), so allowing power-2 only hands the search a u^2 trap
    # (the prey eq overfit u^2 at data_fun_pow=2).
    epde.fit(data=[u, v], variable_names=['u', 'v'],
             max_deriv_order=(1,), equation_terms_max_number=terms,
             data_fun_pow=data_fun_pow, additional_tokens=[],
             equation_factors_max_number=factors_max_number,
             eq_sparsity_interval=(1e-8, 1e-0))

    print("\n================ discovered system (real lynx-hare LV) ================")
    epde.equations(only_print=True, num=1)
    return epde


if __name__ == "__main__":
    import torch
    print('cuda', torch.cuda.is_available())
    lv_discovery()
