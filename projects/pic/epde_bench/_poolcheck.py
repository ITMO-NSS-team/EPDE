"""Build search + domain + trajectory + token pool (no evolution) for every
data set, each in its own process: catches config errors in seconds.

    python -m epde_bench._poolcheck [names...]
"""

import subprocess
import sys

from . import env

env.pin_blas_threads(1)


def one(name: str) -> None:
    from .config import load_config, resolve_for_problem
    from .datasets import load
    from .runner import build_search
    cfg = load_config(name)
    problem = load(name, **cfg.get('loader', {}))
    search_cfg = resolve_for_problem(cfg, problem)
    search, trajectory, families = build_search(problem, search_cfg)
    search.create_pool(data=[trajectory], additional_tokens=families)
    print('POOL OK', name, [f.ftype for f in search.pool.families])


def main(names):
    from .datasets import names as all_names
    failures = 0
    selected = names or all_names('all')
    for name in selected:
        if not names and name == 'darcy':
            from .paths import DATA_DIR
            if not (DATA_DIR / 'darcy/darcy_1.0.npy').exists() or not (DATA_DIR / 'darcy/darcy_nu_1.0.npy').exists():
                print('SKIP darcy: data files are absent', flush=True)
                continue
        res = subprocess.run([sys.executable, '-m', 'epde_bench._poolcheck', '--one', name],
                             capture_output=True, text=True)
        ok = [l for l in res.stdout.splitlines() if l.startswith('POOL OK')]
        if res.returncode != 0 or not ok:
            failures += 1
        print(ok[0] if ok and res.returncode == 0 else f'FAIL {name}: ' + (res.stderr.strip().splitlines() or ['?'])[-1][:200],
              flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--one':
        one(sys.argv[2])
    else:
        sys.exit(main(sys.argv[1:]))
