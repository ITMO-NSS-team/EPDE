"""The command line shared by the per-data-set scripts in ``scripts/``.

Each of those scripts is a docstring plus ``main([...names...])``, so the
logic lives in one place and every script has the same options.
"""

import argparse
import sys

from . import env

env.pin_blas_threads(1)


def main(dataset_names, argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=f'EPDE on {", ".join(dataset_names)}. Settings: projects/pic/configs/<name>.yaml.')
    if len(dataset_names) > 1:
        ap.add_argument('--dataset', choices=dataset_names, default=dataset_names[0])
    ap.add_argument('--noise', type=float, default=0.0, help='percent of the std of the data')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', default='default', help='a key of configs/variants.yaml')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override a config value, e.g. search.evolution.training_epochs=10')
    ap.add_argument('--out', help='also write the JSON record here')
    ap.add_argument('--check', action='store_true',
                    help='only check how well the known law holds on the data (no search)')
    ap.add_argument('--plot', action='store_true', help='only plot the data')
    args = ap.parse_args(argv)
    name = getattr(args, 'dataset', dataset_names[0])

    if args.plot:
        import matplotlib.pyplot as plt
        from .datasets import load
        from .plotting import plot_problem
        from .config import load_config, parse_set
        cfg = load_config(name, args.variant, parse_set(args.set))
        problem = load(name, **cfg.get('loader', {}))
        print(problem.summary())
        plot_problem(problem, problem.noisy(args.noise, args.seed))
        plt.show()
        return 0

    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
    import bench
    cmd = ['check' if args.check else 'run', name, '--noise', str(args.noise), '--seed', str(args.seed), '--variant', args.variant]
    for item in args.set:
        cmd += ['--set', item]
    if args.out and not args.check:
        cmd += ['--out', args.out]
    return bench.main(cmd)
