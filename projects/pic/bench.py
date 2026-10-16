#!/usr/bin/env python
"""Command line for the projects/pic benchmark. Same commands on Windows,
Linux and macOS; run from anywhere (paths are resolved from this file).

    python projects/pic/bench.py list [--suite core]
    python projects/pic/bench.py info ac
    python projects/pic/bench.py check ac --noise 0,1,5
    python projects/pic/bench.py run ac --noise 1 --seed 0 [--variant poly] [--set search.evolution.training_epochs=10]
    python projects/pic/bench.py campaign --suite core --variants default,poly --noise 0,1,5 --seeds 0-2 --workers 8 --name my_campaign
    python projects/pic/bench.py report results/my_campaign

See projects/pic/README.md for the full description.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from epde_bench import env  # noqa: E402  (must pin threads before numpy loads)

env.pin_blas_threads(1)

# Prefer this checkout over another editable EPDE installed in the environment.
sys.path.insert(0, str(HERE.parent.parent))


def _int_list(text: str):
    """'0-4' -> [0..4]; '0,2,5' -> [0, 2, 5]."""
    out = []
    for part in text.split(','):
        if '-' in part.strip()[1:]:
            lo, hi = part.split('-', 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _float_list(text: str):
    return [float(x) for x in text.split(',')]


def cmd_list(args):
    from epde_bench.datasets import REGISTRY, names
    rows = [REGISTRY[n] for n in names(args.suite)]
    print(f'{"name":18} {"suite":9} {"class":11} {"source":10} title')
    for spec in rows:
        print(f'{spec.name:18} {spec.suite:9} {spec.kind:11} {spec.source:10} {spec.title}')


def cmd_info(args):
    from epde_bench.config import load_config, resolve_for_problem
    from epde_bench.datasets import load
    cfg = load_config(args.dataset, args.variant)
    problem = load(args.dataset, **cfg.get('loader', {}))
    print(problem.summary())
    if cfg['method'] == 'epde':
        import yaml
        print('\nEpdeSearch config (resolved for this data set):')
        print(yaml.safe_dump(resolve_for_problem(cfg, problem), sort_keys=False))


def cmd_run(args):
    from epde_bench.config import parse_set
    from epde_bench.runner import run_one
    record = run_one(args.dataset, args.variant, args.noise, args.seed,
                     overrides=parse_set(args.set), config_path=args.config)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1, default=str), encoding='utf-8')
    _print_record(record)
    return 0 if record['status'] in ('ok', 'unsupported') else 1


def _print_record(record):
    head = (f"{record['dataset']} | {record['variant']} | noise {record['noise']}% | "
            f"seed {record['seed']} | {record['status']}")
    print('\n' + head)
    if record['status'] != 'ok':
        print(record.get('reason') or record.get('traceback') or record.get('error'))
        return
    m = record['metrics']
    print(f"fit {record['fit_seconds']:.1f} s, Pareto front: {m['front_size']} solutions")
    for i, system in enumerate(record['front']):
        mark = ' <- compromise pick' if i == m.get('selected_index') else ''
        print(f'  [{i}]{mark}')
        for eq in system:
            print(f'      {eq}')
    if 'success_front' in m:
        print(f"truth on the front: {m['success_front']} (min Hamming {m['hamming_min']}); "
              f"compromise pick correct: {m['success_selected']} (Hamming {m['hamming_selected']})")
        if m.get('coef_error') is not None:
            print(f"mean relative coefficient error of the matching solution: {m['coef_error']:.3%}")
    else:
        print('truth unknown: no structural score')


def cmd_check(args):
    from epde_bench.truthcheck import check, print_check
    from epde_bench.config import parse_set
    problem, report = check(args.dataset, args.noise, args.seed, variant=args.variant,
                            overrides=parse_set(args.set), config_path=args.config)
    print_check(problem, report)


def cmd_campaign(args):
    from epde_bench.campaign import run_campaign
    return run_campaign(args)


def cmd_report(args):
    from epde_bench.report import make_report
    make_report(Path(args.campaign), formats=args.formats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('list', help='list the data sets')
    p.add_argument('--suite', default='all', help="core | extended | real | other | all (join with '+')")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser('info', help='describe a data set and its resolved config')
    p.add_argument('dataset')
    p.add_argument('--variant', default='default')
    p.set_defaults(func=cmd_info)

    p = sub.add_parser('run', help='one discovery run, printed and optionally saved as JSON')
    p.add_argument('dataset')
    p.add_argument('--variant', default='default')
    p.add_argument('--noise', type=float, default=0.0, help='percent of the std of the data')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='override a config value, e.g. search.evolution.training_epochs=10')
    p.add_argument('--config', help='use this YAML instead of configs/<dataset>.yaml')
    p.add_argument('--out', help='write the JSON record here')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('check', help='signal check: does the known law hold on the (noisy) data?')
    p.add_argument('dataset')
    p.add_argument('--noise', type=_float_list, default=[0.0, 1.0, 5.0])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--variant', default='default')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    p.add_argument('--config', help='use this YAML instead of configs/<dataset>.yaml')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('campaign', help='many runs in parallel processes, resumable')
    p.add_argument('--suite', default='core')
    p.add_argument('--datasets', help='comma-separated names (instead of --suite)')
    p.add_argument('--variants', default='default')
    p.add_argument('--noise', type=_float_list, default=[0.0])
    p.add_argument('--seeds', type=_int_list, default=[0])
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--timeout', type=float, default=3 * 3600, help='seconds per run')
    p.add_argument('--name', required=True, help='results/<name>/ (or EPDE_BENCH_RESULTS/<name>)')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=cmd_campaign)

    p = sub.add_parser('report', help='tables and figures from a campaign directory')
    p.add_argument('campaign')
    p.add_argument('--formats', default='png', help='figure formats, e.g. png,pdf')
    p.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
