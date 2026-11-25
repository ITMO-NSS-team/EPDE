"""Replay coupled NSGA-II on two time windows of the PIC cylinder wake."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def active_support(equation):
    from epde_bench.reconstruction import parse_equation

    predictors, intercept, target, _ = parse_equation(equation)
    terms = {tuple(sorted(factors)) for coefficient, factors in predictors if coefficient != 0}
    terms.add(tuple(sorted(target)))
    if intercept != 0:
        terms.add(())
    return frozenset(terms)


def registered_supports(truth):
    x_momentum, y_momentum, continuity = map(active_support, truth)
    registry = {continuity: 'continuity'}
    # In a convective product, continuity allows u_x to replace -v_y and vice versa.
    # These signatures track active terms, independent of the coefficient or pivot.
    (first,), (second,) = continuity
    exchange = {first: second, second: first}
    for support, label in ((x_momentum, 'x_momentum'), (y_momentum, 'y_momentum')):
        registry[support] = label
        rewritten = frozenset(tuple(sorted(exchange.get(factor, factor) for factor in term))
                              if len(term) > 1 else term for term in support)
        registry[rewritten] = label
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=125, help='Number of complete NSGA-II epochs.')
    parser.add_argument('--log', type=Path, help='Write native search progress, including each epoch.')
    parser.add_argument('--no-beam', action='store_true', help='Reproduce the earlier beam-disabled control.')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive.')
    if os.environ.get('PYTHONHASHSEED') != '0':
        parser.error('Set PYTHONHASHSEED=0 before launching the worker.')
    os.environ['EPDE_LOOP_STATS'] = '1'
    sys.path.insert(0, str(ROOT / 'projects/pic/experiments/complementary_windows_2026-11-24'))
    from cpu_limit import cap_worker
    cap_worker()
    sys.path.insert(0, str(ROOT / 'projects/pic'))
    sys.path.insert(0, str(args.package_root.resolve()))
    import numpy as np
    import torch
    import epde
    from epde import _loop_stats
    from epde.interface.search_config import resolve_plain_params
    from epde_bench.datasets import load_ns
    from epde_bench.metrics import canonical_tokens

    assert Path(epde.__file__).resolve().parent.parent == args.package_root.resolve()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    problem = load_ns(subset='full')
    t, y, x = problem.grids
    samples = []
    for start in (0, 100):
        ts = slice(start, start + 50)
        ys, xs = slice(5, 45, 2), slice(0, 72, 2)
        coords = {'t': t[ts, 0, 0], 'y': y[0, ys, 0], 'x': x[0, 0, xs]}
        fields = {k: a[ts, ys, xs].copy() for k, a in problem.data.items()}
        samples.append(epde.TrajectoryData(coords, fields, trajectory_id=f'wake_frames_{start}_{start+49}'))

    moves = {'plain_target': True, 'target_axes': None}
    if args.no_beam:
        moves['fill_probability'] = 0.
    config = dict(engine='nsga2', niching='auto', system_rps_mode='plain', system_rps_params=moves,
                  multisample_mode='joint', multisample_weighting='samples', use_solver=False, device='cpu',
                  evaluation_form='weak', default_preprocessor_type='FD', boundary_width=[2, 4, 7],
                  data_fun_pow=1, deriv_fun_pow=1, equation_terms_max_number=7, equation_factors_max_number=2,
                  population_size=8, training_epochs=args.epochs, early_stopping_patience=None,
                  verbose_params={'show_iter_idx': args.log is not None})
    _loop_stats.reset()
    wall, cpu = time.perf_counter(), time.process_time()
    with open(args.log or os.devnull, 'w', encoding='utf-8', buffering=1) as log, redirect_stdout(log), epde.EpdeMultisample(**config) as search:
        params = resolve_plain_params(search._config)
        beam = {key: value for key, value in params.items() if key.startswith('fill_')}
        search.fit(samples=samples, max_deriv_order=(1, 2, 2))
        cpu_seconds, wall_seconds = time.process_time()-cpu, time.perf_counter()-wall
        initial_population = search.optimizer._mu
        completed_epochs = _loop_stats.timers_snapshot()['NSGA.epoch']['entries']
        if completed_epochs != args.epochs:
            raise RuntimeError(f'Search completed {completed_epochs} of {args.epochs} requested epochs')
        result = search.result
    truth = {label: canonical_tokens([eq])[0] for label, eq in
             zip(('x_momentum', 'y_momentum', 'continuity'), problem.truth)}
    registered = registered_supports(problem.truth)
    systems = [system for level in result.levels for system in level]
    vectors = np.asarray([system['objectives'] for system in systems])
    assert vectors.ndim == 2 and vectors.shape[1] == 6 and np.isfinite(vectors).all()
    front = []
    for system, row in zip(systems, vectors):
        if np.any(np.all(vectors <= row, axis=1) & np.any(vectors < row, axis=1)):
            continue
        equations = {e['variable']: e['text'] for e in system['equations']}
        supports = canonical_tokens(list(equations.values()))
        matches = [label for label, support in truth.items() if support in supports]
        laws = {registered.get(active_support(eq)) for eq in equations.values()} - {None}
        front.append(dict(equations=equations, objectives=system['objectives'], reference_supports=matches,
                          registered_laws=sorted(laws)))
    assert front
    counts = {name: _loop_stats._stats_for('SoEqRPS.plain.' + name)['entries']
              for name in ('fit', 'fit_requests', 'fit_reused')}
    if counts['fit'] and not counts['fit_requests']:
        # Older releases only instrumented actual fitness evaluations.
        counts['fit_requests'] = counts['fit_reused'] = None
    counts['beam_fills'] = _loop_stats._stats_for('PlainMutation.fill')['entries']
    prefix = 'PlainMutation.fill_fallback.'
    counts['beam_fallbacks'] = {name[len(prefix):]: _loop_stats._stats_for(name)['entries']
                                for name in _loop_stats._stats if name.startswith(prefix)}
    record = dict(seed=args.seed, config=config, beam=beam, measurements=result.measurements, counts=counts,
                  initial_population=initial_population, completed_epochs=completed_epochs,
                  returned_systems=len(systems),
                  cpu_seconds=cpu_seconds, wall_seconds=wall_seconds, front=front)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    print(f'{args.package_root.name}: {counts}, {cpu_seconds:.2f} CPU seconds, {len(front)} front systems')


if __name__ == '__main__':
    main()
