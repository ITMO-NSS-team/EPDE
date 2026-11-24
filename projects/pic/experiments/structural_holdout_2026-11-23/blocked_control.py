"""Single contiguous search with the same mutation fallback as multi-slab discovery."""
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
from unittest.mock import patch
from experiment import ROOT, live, trial
from epde.operators.common import fill_prune


def dest(name,seed):
    return ROOT/'blocked_controls'/name/str(seed)


def run(job):
    name,seed=job
    path=dest(name,seed)
    if (path/'archive.json.gz').exists():
        return
    path.mkdir(parents=True,exist_ok=True)
    with (path/'search.log').open('w') as log:
        child=subprocess.run([sys.executable,'-u',__file__,name,str(seed)],stdout=log,stderr=subprocess.STDOUT,
                             env=dict(os.environ,PYTHONHASHSEED='0'))
    if child.returncode:
        raise RuntimeError((path/'search.log').read_text(errors='replace')[-2800:])


if __name__=='__main__':
    if len(sys.argv)==3:
        name,seed=sys.argv[1],int(sys.argv[2])
        def fallback():
            raise fill_prune._Fallback('trajectories')
        with (patch.object(trial,'folder',return_value=dest(name,seed)),
              patch.object(fill_prune,'_trajectory_key',fallback)):
            trial.worker(name,seed,'native')
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run,[(n,s) for s in live.SEEDS for n in live.DATASETS]))
