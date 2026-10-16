"""A small real evolutionary run shared by the notebook and interactive lesson."""
import contextlib
import copy
import io
import time

from . import env


def run_optimizer_demo():
    """Return observed epoch fronts for a 121-point exact damped oscillator.

    Uses the existing default objectives, sparsity and evolutionary strategy.
    Only the population and term budget are reduced. The callback observes
    candidates and never modifies the population or requests an early stop.
    """
    env.pin_blas_threads(1)
    env.seed_everything(7)
    import numpy as np
    from epde.optimizers.moeadd.moeadd import MOEADDOptimizer
    from epde.operators.common.objectives import ideal_point
    from .config import load_config, resolve_for_problem
    from .problem import Problem, mesh
    from .runner import build_search

    t = np.linspace(0, 6, 121)
    decay = np.exp(-0.25 * t)
    demo = Problem('optimizer_demo', 'Damped oscillator', 'ode', 'synthetic',
                   mesh(t), {'u': decay * np.sin(t)}, ('t',))
    demo.derivs = [np.column_stack((
        decay * (np.cos(t) - 0.25 * np.sin(t)),
        decay * (-0.9375 * np.sin(t) - 0.5 * np.cos(t))))]
    demo.deriv_orders = [2]
    cfg = load_config('ode', overrides={'search': {
        'preprocessing': {'max_deriv_order': [2]},
        'search_space': {'data_fun_pow': 2, 'deriv_fun_pow': 1,
                         'equation_terms_max_number': 4,
                         'equation_factors_max_number': 1},
        'evolution': {'population_size': 4, 'training_epochs': 3,
                      'neighbors_number': 2}}})
    search = None
    history = []

    def observe_front(snapshot, epoch_idx):
        history.append({'epoch': int(epoch_idx) + 1, 'front': copy.deepcopy(snapshot)})
        return False

    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            search, trajectory, families = build_search(demo, resolve_for_problem(cfg, demo))
            search.create_pool(data=[trajectory], additional_tokens=families)
            params = dict(search.optimizer_init_params)
            params['population_instruct'] = {
                'pool': search.pool, 'terms_number': 4, 'max_factors_in_term': 1,
                'sparsity_interval': (1., 1.), 'second_objective': 'instability'}
            params['best_sol_vals'] = ideal_point(('discrepancy', 'instability'))
            optimizer = MOEADDOptimizer(**params)
            optimizer.set_strategy(search.director)
            optimizer.optimize(epochs=3, early_stopping_callback=observe_front)
        if not history:
            raise RuntimeError('The optimizer did not produce observable epoch fronts.')
        rows = []
        for entry in history:
            values = np.asarray([s['obj_fun'] for s in entry['front']], dtype=float)
            if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
                raise RuntimeError('The observed front contains invalid objective values.')
            rows.append({'Epoch': entry['epoch'], 'Front size': len(values),
                         'Smallest discrepancy': float(values[:, 0].min()),
                         'Smallest instability': float(values[:, 1].min())})
        equations = [line.strip() for solution in history[-1]['front']
                     for line in solution['text_form'].splitlines()
                     if '=' in line and not line.strip().startswith('{')]
        return {'history': history, 'rows': rows, 'equations': equations,
                'seconds': time.perf_counter() - started}
    finally:
        if search is not None:
            search.close()
