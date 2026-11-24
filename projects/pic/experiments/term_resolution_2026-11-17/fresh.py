"""Run the existing three-bank benchmark with fresh noise and an isolated output root."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'nuisance_direction_2026-11-17'
OUT = ROOT/'fresh'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'


def run(job):
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location('archived_three_bank_run', SOURCE/'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ROOT == SOURCE
    # The previous routine uses ROOT for its output files and frozen design.
    # Its called numerical routines and provenance functions retain their own roots.
    module.ROOT = OUT
    return module.run('kdv_cossin', *job)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    for directory in ('frozen', 'results'):
        (OUT/directory).mkdir(exist_ok=True)
    design = (SOURCE/'design_screen.json').read_bytes()
    (OUT/'design_screen.json').write_bytes(design)
    print('Frozen design', hashlib.sha256(design).hexdigest(), flush=True)
    jobs = [(profile, seed) for profile in ('iid','positive','negative') for seed in (4701,4702)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for item in pool.map(run,jobs):
            print(item,flush=True)
