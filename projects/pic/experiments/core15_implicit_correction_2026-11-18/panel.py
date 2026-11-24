"""Shared panel setup and preparation of the fresh held-out records."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "implicit_objective_2026-11-18"
sys.path.insert(0,str(PRIOR))
import records

SYSTEMS = tuple(records.SYSTEMS)
REPLAY = (6111,6112,6113)
FRESH = tuple(range(73001,73007))


def record_path(system,seed):
    directory = PRIOR if seed in REPLAY else ROOT
    return directory/"records"/f"{system}_{seed}.npz"


def prepare(job):
    # The unchanged builder writes to its module's ROOT/records directory.
    # This assignment is local to the worker running this new experiment.
    records.ROOT = ROOT
    return records.build(job)


if __name__=="__main__":
    (ROOT/"records").mkdir(exist_ok=True)
    jobs = [(s,n) for n in FRESH for s in SYSTEMS if not record_path(s,n).with_suffix(".json").exists()]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for f in as_completed([pool.submit(prepare,j) for j in jobs]):
            print(f.result(),flush=True)
