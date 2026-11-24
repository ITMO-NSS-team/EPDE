"""Launch paired native searches on disjoint time intervals and on the whole record."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'wave_archive_2026-11-19'))
from capture import Archive

SYSTEMS = ('wave','pde_compound')
SEEDS = tuple(range(921101,921107))
PARTS = ('early','late','full')
SPATIAL = ('left','right')


def measurements(system,seed):
    from search import U, np
    axes, clean = U.load(system)
    field = clean['u']
    rng = np.random.default_rng([seed,list(U.TRUTH).index(system)])
    noisy = field + .05*field.std()*rng.standard_normal(field.shape)
    n = len(axes[0])//2
    assert len(axes[0])==2*n+1
    parts = {part:([axes[0][s],*axes[1:]], {'u':noisy[s]})
             for part,s in [('early',slice(0,n)),('late',slice(-n,None)),('full',slice(None))]}
    if system=='wave':
        half = len(axes[1])//2
        for part,s in [('left',slice(0,half)),('right',slice(-half,None))]:
            parts[part] = ([axes[0],axes[1][s]],{'u':noisy[:,s]})
    return parts


def folder(system,part):
    return ROOT/'results'/system/part


def launch(job):
    system, seed, part = job
    dest = folder(system,part)
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/f'{system}_{seed}_wape.json').is_file() and (dest/f'archive_{seed}.json.gz').is_file():
        return 0
    with (dest/f'search_{seed}.log').open('w') as log:
        task = subprocess.run([sys.executable,'-u',__file__,system,str(seed),part],stdout=log,stderr=subprocess.STDOUT,
                              env=dict(os.environ,PYTHONHASHSEED='0'))
    print(system,seed,part,task.returncode,flush=True)
    return task.returncode


if __name__=='__main__':
    if len(sys.argv)==4:
        from search import run_job
        system, seed, part = sys.argv[1:]
        assert os.environ['PYTHONHASHSEED']=='0'
        dest = folder(system,part)
        run_job(system,int(seed),'wape',Archive(dest),dest,measurements(system,int(seed))[part])
    else:
        jobs = [(s,n,p) for n in SEEDS for p in PARTS for s in SYSTEMS]
        if sys.argv[1:]==['--space']:
            jobs = [('wave',n,p) for n in SEEDS for p in SPATIAL]
        with ThreadPoolExecutor(max_workers=6) as pool:
            raise SystemExit(int(any(list(pool.map(launch,jobs)))))
