"""What does the frequency ridge do? Compute the per-term vcoef instability
score NC_j/gamma0_j^2 for AC true terms [u_xx, u^3, u] + spurious u^2, with the
ridge ON (freq_coef=1) vs OFF (freq_coef=0, only the 1e-30->1e-6 floor), at 0%
and 0.5% noise. Same pool (=> same resolved modes) per noise level, so the only
thing toggled is the ridge.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import numpy as np
from epde import globals as gv
import epde.globals as global_var
from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation
from epde.operators.common.stability import VaryingCoefSetup

gv.set_gram_config('vcoef')

UXX = 'd^2u/dx1^2{power: 1.0}'
U1, U2, U3 = 'u{power: 1.0}', 'u{power: 2.0}', 'u{power: 3.0}'
UT = 'du/dx0{power: 1.0}'
TERMS = [UXX, U3, U1, U2]          # 3 true + 1 spurious (u^2)
SHORT = ['u_xx', 'u^3', 'u', 'u^2(spur)']


def build(noise_frac, seed=0):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = np.linspace(0., 1., 51)
    x = np.linspace(-1., 0.984375, 128)
    data = np.load(os.path.join(os.path.dirname(__file__), 'ac_data.npy'))
    if noise_frac > 0:
        rng = np.random.default_rng(seed)
        data = data + noise_frac * np.std(data) * rng.standard_normal(data.shape)
    grid = np.meshgrid(t, x, indexing='ij')
    boundary = tuple(max(1, n // 10) for n in data.shape)
    search = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True,
                        boundary=boundary, coordinate_tensors=grid, device=device)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    search.create_pool(data=[data], variable_names=['u'], max_deriv_order=(2, 4),
                       additional_tokens=[], data_fun_pow=3, deriv_fun_pow=2)
    gv.vc_modes_cache.clear()      # resolve modes from THIS (noisy) field
    return search


def design(search):
    lhs = " + ".join(f"1.0 * {t}" for t in TERMS)
    eq = translate_equation(f"{lhs} = {UT}", search.pool, all_vars=['u']).vals['u']
    tgt = eq.target_idx
    fidx = [i for i in range(len(eq.structure)) if i != tgt]
    F = np.stack([np.asarray(eq.structure[i].evaluate(False), float).reshape(-1) for i in fidx], 1)
    y = np.asarray(eq.structure[tgt].evaluate(False), float).reshape(-1)
    return F, y


def main():
    for noise in (0.0, 0.005, 0.02, 0.05):
        search = build(noise)
        w = (global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask]
             .reshape(-1).astype(float))
        gs = tuple(int(n) for n in global_var.grid_cache.inner_shape)
        F, y = design(search)
        modes = VaryingCoefSetup._resolve_modes(None, gs, 'u', None)
        print(f"\n===== noise = {noise*100:.1f}%   resolved modes K = {modes} =====")
        print(f"  {'term':<12}{'ridge ON (fc=1)':>18}{'ridge OFF (fc=0)':>18}")
        on = VaryingCoefSetup(F, y, w, gs, main_var='u', fit_intercept=False,
                              freq_coef=1.0).score(None)
        off = VaryingCoefSetup(F, y, w, gs, main_var='u', fit_intercept=False,
                               freq_coef=0.0).score(None)
        for s, a, b in zip(SHORT, on, off):
            print(f"  {s:<12}{a:>18.5g}{b:>18.5g}")
        print(f"  {'SUM':<12}{float(np.sum(on)):>18.5g}{float(np.sum(off)):>18.5g}")
        # separation = spurious / worst-true
        sep_on = on[3] / max(on[:3].max(), 1e-30)
        sep_off = off[3] / max(off[:3].max(), 1e-30)
        print(f"  separation  u^2 / max(true) :  ON={sep_on:.1f}x   OFF={sep_off:.1f}x")


if __name__ == "__main__":
    import torch
    print('cuda available:', torch.cuda.is_available())
    main()
