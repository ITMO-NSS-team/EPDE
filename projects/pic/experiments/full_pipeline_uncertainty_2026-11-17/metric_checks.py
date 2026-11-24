from bridge import *
from full_pipeline import Experiment,MC,score,source_files
from scipy.linalg import eigvalsh
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

def one(name):
    start=time.monotonic();e=Experiment(name,50,'iid');rows=[]
    for rho in (None,.4,-.4):
        model=e.model if rho is None else {v:[m[0],[rho]*len(m[1]),m[2]] for v,m in e.model.items()};mc=MC if rho is None else 4;batched=e.metric(e.pilot,model,mc=mc);base=e.matrices(e.pilot,correct=False);parts={b:[] for b in e.banks}
        for j in range(mc):
            rng=np.random.default_rng(np.random.SeedSequence([984105,e.system_id,e.pid,j]));fields={v:y+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in e.pilot.items()}
            for b,m in e.matrices(fields,model).items():parts[b].append(m-base[b])
        for b,pp in parts.items():
            z=np.stack(pp);assert np.all(z[:,:,e.deterministic]==0);direct=np.einsum('nri,nrj->ij',z,z)/mc;scale=np.max(np.abs(direct));error=float(np.max(np.abs(direct-batched[b]))/scale);np.testing.assert_allclose(direct,batched[b],rtol=5e-9,atol=5e-12*scale);w,info=S.metric(batched[b],e.active);assert w is not None,info;assert np.all(batched[b][e.deterministic]==0);rows.append(dict(profile=rho,bank=b,relative_error=error,rank=info['rank']))
    return dict(system=name,status='PASS',rows=rows,seconds=time.monotonic()-start)

def main():
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,n) for n in SYSTEMS]):r=f.result();rows.append(r);print(r['system'],'PASS',round(r['seconds'],1),flush=True)
    dump(ROOT/'metric_checks.json',dict(status='PASS',systems=15,matrix_pairs=90,max_relative_error=max(x['relative_error'] for r in rows for x in r['rows']),rows=rows,source_sha256={**source_files(),**sources('metric_checks.py')}))
if __name__=='__main__':main()
