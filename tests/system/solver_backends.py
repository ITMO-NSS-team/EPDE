"""End-to-end gate for both PDE solver backends.

    python tests/system/solver_backends.py [backend] [epochs] [device]
    python tests/system/solver_backends.py            # all four combinations

Both backends were dead against the post-multisample API and are exercised by
nothing in the fast unit suite -- ``tests/unit/test_solver_path.py`` pins the
individual defects, but only a real solve proves the path end to end. Budget
roughly 45 s per DeepXDE run and 95 s per autograd run.

The target is the textbook ODE ``u'' = -u``; a healthy run recovers
``-0.99866 * u + 0.0 = d^2u/dx0^2`` on every backend/device combination.
"""
import itertools
import sys
import traceback

import numpy as np

import epde

#: A deliberately tiny net: this is a wiring gate, not a convergence study.
DEEPXDE_CONFIG = {'net': [16, 16], 'activation': 'tanh', 'optimizer': 'adam',
                  'lr': 1e-3, 'num_domain': 200, 'num_boundary': 20,
                  'num_initial': 20, 'epochs': 50}


def run(backend: str, epochs: int = 2, device: str = 'cpu') -> str:
    np.random.seed(0)
    t = np.linspace(0, 4 * np.pi, 100)
    u = np.sin(t) + 1.3 * np.cos(t)

    config = {'deepxde_config': DEEPXDE_CONFIG} if backend == 'deepxde' else {}
    search = epde.EpdeSearch(use_solver=True, solver_backend=backend,
                             device=device, multiobjective_mode=True,
                             verbose_params={'show_iter_idx': False}, **config)
    search.set_preprocessor(
        default_preprocessor_type='poly',
        preprocessor_kwargs={'use_smoothing': False, 'sigma': 1,
                             'polynomial_window': 3, 'poly_order': 3})
    search.set_moeadd_params(population_size=4, training_epochs=epochs)
    _, domain = search.createDomain(t, boundary_width=10, ID=0)
    _, trajectory = search.createTrajectory({'u': u}, domain, cache_id=0)
    search.fit(data=[trajectory], max_deriv_order=(2,), data_fun_pow=1,
               equation_terms_max_number=3, equation_factors_max_number=1)
    return ' '.join(search.equations(only_print=False)[0][0].text_form.split())


def main(argv):
    if len(argv) > 1:
        cases = [(argv[1], int(argv[2]) if len(argv) > 2 else 2,
                  argv[3] if len(argv) > 3 else 'cpu')]
    else:
        import torch
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        cases = [(backend, 2, device) for backend, device
                 in itertools.product(('deepxde', 'autograd'), devices)]

    failures = 0
    for backend, epochs, device in cases:
        try:
            best = run(backend, epochs, device)
        except Exception:
            failures += 1
            print('FAIL[%s/%s]' % (backend, device))
            traceback.print_exc()
            continue
        print('OK[%s/%s] %s' % (backend, device, best[:110]))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
