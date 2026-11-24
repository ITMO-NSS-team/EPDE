"""Launch paired searches with bounded concurrency and isolated random state."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def launch(job):
    system, seed, arm = job
    tag = f'{system}_{seed}_{arm}'
    if (ROOT/'results'/f'{tag}.json').is_file():
        return 0
    with (ROOT/'logs'/f'{tag}.log').open('w') as f:
        proc = subprocess.run([sys.executable,'-u',str(ROOT/'search.py'),system,str(seed),arm],
                              env=dict(os.environ,PYTHONHASHSEED='0'),stdout=f,stderr=subprocess.STDOUT)
    print(tag,proc.returncode,flush=True)
    return proc.returncode


if __name__=='__main__':
    (ROOT/'logs').mkdir(exist_ok=True)
    jobs = [(s,n,a) for n in (916201,916202,916203) for s in ('pde_compound','wave','vdp','ac','ode')
            for a in ('wape','wls','tls')]
    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(launch,jobs))
    sys.exit(int(any(codes)))
