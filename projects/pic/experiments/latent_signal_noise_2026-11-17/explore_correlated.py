from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
import correlated_fit as F
np=F.np
sys.path.insert(0,str(ROOT.parent/'spectral_noise_2026-11-17'))
import coverage as C
S,R=C.S,C.R
from concurrent.futures import ProcessPoolExecutor,as_completed
SYSTEMS=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']

def one(job):
    name,seed,profile=job;e=C.Study(name,seed,profile,'iid');key=f'{name}_{profile}_seed{seed}';raw={}
    prior=C.ROOT/'results'/f'{name}_{profile}_spectral_seed{seed}.json';old=json.loads(prior.read_text());hh={v:R.digest(y) for v,y in e.raw.items()};assert hh==old['raw_sha256']
    for v,y in e.raw.items():m,info=F.estimate(y);raw[v]=dict(model=m,info=info)
    sources={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'latent_fit.py',ROOT/'correlated_fit.py',ROOT/'protocol.txt',ROOT/'correlated_curvature_protocol.txt')};fp=ROOT/'correlated_frozen'/f'{key}.json'
    R.dump(fp,dict(raw_fits=raw,raw_sha256=hh,source_sha256=sources,input_sha256={str(prior):S.E.sha(prior)}))
    rows=[]
    for mode,fields in [('raw',e.raw),('noise_only',{v:e.raw[v]-e.p.data[v] for v in e.raw})]:
        for v,y in fields.items():
            model,info=(raw[v]['model'],raw[v]['info']) if mode=='raw' else F.estimate(y);true=.05*float(np.std(e.p.data[v]))
            rows.append(dict(system=name,seed=seed,profile=profile,mode=mode,variable=v,status=info['status'],model=model,info=info,sigma_ratio=model[0]/true if model else None,true_rho=e.rho))
    R.dump(ROOT/'correlated_explore'/f'{key}.json',dict(rows=rows,frozen_sha256=S.E.sha(fp),source_sha256=sources))
    return dict(record=key,statuses=[r['status'] for r in rows])

def main():
    assert json.loads((ROOT/'checks_correlated.json').read_text())['status']=='PASS'
    jobs=[(n,s,'iid') for n in SYSTEMS for s in (33,34,35)]+[(n,s,p) for n in ('ac','kdv_cossin','ks','wave') for s in (33,34,35) for p in ('positive','negative')]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for f in as_completed([pool.submit(one,j) for j in jobs]):print(f.result(),flush=True)
    rows=[r for p in (ROOT/'correlated_explore').glob('*.json') for r in json.loads(p.read_text())['rows']];summary=[]
    for profile in ('iid','positive','negative'):
        for mode in ('raw','noise_only'):
            rr=[r for r in rows if r['profile']==profile and r['mode']==mode];good=[r for r in rr if r['model']]
            summary.append(dict(profile=profile,mode=mode,variables=len(rr),incomplete=sum(r['status']!='ok' for r in rr),
                sigma_ratio_quantiles=np.quantile([r['sigma_ratio'] for r in good],[0,.1,.5,.9,1]),rho_error_quantiles=np.quantile([r-rw['true_rho'] for rw in good for r in rw['model'][1]],[0,.1,.5,.9,1])))
    R.dump(ROOT/'correlated_summary.json',dict(status='COMPLETE',records=len(jobs),summary=summary,source_sha256={str(Path(__file__)):S.E.sha(__file__)}))
    for row in summary:print(row)

if __name__=='__main__':main()
