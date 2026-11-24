"""Rerun native discovery from scratch with one of four raw slabs withheld."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from experiment import ROOT, live, np, trial


def slabs(axes):
    axis=max(range(len(axes)),key=lambda j:len(axes[j]))
    size=len(axes[axis])//4
    parts=[]
    for fold in range(4):
        cuts=tuple(slice(fold*size,(fold+1)*size) if j==axis else slice(None) for j in range(len(axes)))
        parts.append((cuts,[a[s] for a,s in zip(axes,cuts)]))
    return axis,parts


def fold_path(name,seed,fold):
    return ROOT/'blocked_results'/name/str(seed)/str(fold)


def discover(name,seed,fold,destination=None):
    dest=destination or fold_path(name,seed,fold)
    if (dest/'fold.json').is_file():
        return
    dest.mkdir(exist_ok=True,parents=True)
    problem=live.load_data(name)
    axis,parts=slabs(live.R.axes_of(problem))
    training=[i for i in range(4) if i!=fold]
    raw=problem.noisy(5,seed)
    def input_split(axes):
        return axis,parts[training[0]],parts[fold]
    native_fit=live.U.EpdeSearch.fit
    def fit(search, *, data, **kwargs):
        trajectories=list(data)
        for key,index in enumerate(training[1:],start=1):
            cuts,axes=parts[index]
            _,dom=search.createDomain(tuple(np.meshgrid(*axes,indexing='ij')),ID=key)
            _,traj=search.createTrajectory({k:v[cuts] for k,v in raw.items()},dom,cache_id=key)
            trajectories.append(traj)
        assert len(trajectories)==3
        return native_fit(search,data=trajectories,**kwargs)
    status='completed'
    try:
        with (patch.object(live,'split',input_split), patch.object(live,'BUDGET',750),
              patch.object(live.U.EpdeSearch,'fit',fit), patch.object(trial,'folder',return_value=dest)):
            trial.worker(name,seed,'native')
    except ValueError as err:
        if 'no feasible weak-form window' not in str(err):
            raise
        status=str(err)
    meta=dict(held=fold,training=training,axis=axis,status=status,
              sample_hashes=[{k:live.R.digest(v[cut]) for k,v in raw.items()} for cut,_ in parts])
    (dest/'fold.json').write_text(json.dumps(meta))


def launch(job):
    name,seed=job
    if all((fold_path(name,seed,f)/'fold.json').exists() for f in range(4)):
        return
    dest=fold_path(name,seed,0).parent
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'search.log').open('w') as log:
        result=subprocess.run([sys.executable,'-u',__file__,'--worker',name,str(seed)],
                              stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED='0'))
    if result.returncode:
        raise RuntimeError((dest/'search.log').read_text(errors='replace')[-3500:])
    print(name,seed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',nargs=2)
    parser.add_argument('--datasets',nargs='+',default=live.DATASETS)
    parser.add_argument('--seeds',nargs='+',type=int,default=live.SEEDS)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.worker:
        name,seed=args.worker
        for f in range(4):
            discover(name,int(seed),f)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch,[(name,s) for s in args.seeds for name in args.datasets]))
