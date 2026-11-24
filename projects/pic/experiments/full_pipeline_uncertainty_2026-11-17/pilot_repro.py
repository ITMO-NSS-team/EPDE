from bridge import *
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(job):
    seed,rep=job;f=read(ROOT/'factorial_frozen'/f'ks_positive_seed{seed}.json');p=R.datasets.load('ks');pid,rho=S.PROFILES['positive'];rng=np.random.default_rng(np.random.SeedSequence([982101,pid,seed]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};pilot,info=Pilot.reconstruct(raw,R.axes_of(p),f['model']);key=f'seed{seed}_rep{rep}';path=ROOT/'pilot_replays'/(key+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**pilot);hh={v:R.digest(y) for v,y in pilot.items()};out=dict(seed=seed,replay=rep,pilot_sha256=hh,info=info,matches_original=hh==f['pilot_sha256']['correlated'],array_sha256=sha(path));dump(path.with_suffix('.json'),out);return out

def main():
    rows=[]
    # One task per fresh process, so each reconstruction has a cold library/cache.
    with ProcessPoolExecutor(max_workers=4,max_tasks_per_child=1) as pool:
        for f in as_completed([pool.submit(one,(s,r)) for s in (45,46,47) for r in range(4)]):row=f.result();rows.append(row);print(row['seed'],row['replay'],row['matches_original'],flush=True)
    comparisons=[]
    for seed in (45,46,47):
        rr=[r for r in rows if r['seed']==seed];canonical=next((r for r in rr if r['matches_original']),rr[0]);base=ROOT/'pilot_replays'/f"seed{seed}_rep{canonical['replay']}.npz"
        with np.load(base) as z:a={k:z[k] for k in z.files}
        for r in rr:
            path=ROOT/'pilot_replays'/f"seed{seed}_rep{r['replay']}.npz"
            with np.load(path) as z:
                comparisons.append(dict(seed=seed,replay=r['replay'],canonical=str(base),matches_original=canonical['matches_original'],max_abs_difference=max(float(np.max(np.abs(z[v]-a[v]))) for v in a),max_relative_difference=max(float(np.max(np.abs(z[v]-a[v]))/np.max(np.abs(a[v]))) for v in a)))
    dump(ROOT/'pilot_reproducibility.json',dict(rows=rows,comparisons=comparisons,source_sha256=sources('pilot_repro.py'),input_sha256={str(p):sha(p) for p in (ROOT/'factorial_frozen').glob('*.json')}));print(comparisons,flush=True)
if __name__=='__main__':main()
