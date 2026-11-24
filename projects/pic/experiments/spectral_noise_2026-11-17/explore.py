from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
import noise_fit as F
np=F.np
sys.path.insert(0,str(ROOT.parent/'joint_moment_regions_2026-11-17'))
import study as S
from concurrent.futures import ProcessPoolExecutor,as_completed
SYSTEMS=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']

def one(job):
    name,seed,profile=job;e=S.Study(name,seed,profile,'ar');key=f'{name}_{profile}_seed{seed}';raw={}
    for design in ('strips','corner'):
        raw[design]={}
        for v,y in e.raw.items():
            m,info=F.estimate(y,design);raw[design][v]=dict(model=m,info=info)
    sources={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'noise_fit.py',ROOT/'protocol.txt')}
    fp=ROOT/'explore_frozen'/f'{key}.json';S.R.dump(fp,dict(raw_fits=raw,raw_hashes={v:S.R.digest(y) for v,y in e.raw.items()},source_sha256=sources))
    rows=[]
    for mode,field in [('raw',e.raw),('noise_only',{v:e.raw[v]-e.p.data[v] for v in e.raw})]:
        for design in ('strips','corner'):
            for v,y in field.items():
                m,info=(raw[design][v]['model'],raw[design][v]['info']) if mode=='raw' else F.estimate(y,design)
                truth=.05*float(np.std(e.p.data[v]));rows.append(dict(system=name,seed=seed,profile=profile,mode=mode,design=design,variable=v,
                    status=info['status'],model=m,info=info,sigma_ratio=m[0]/truth if m else None,true_rho=e.rho))
    S.R.dump(ROOT/'explore'/f'{key}.json',dict(rows=rows,old_ar=e.ar,frozen_sha256=S.E.sha(fp),source_sha256=sources))
    return dict(record=key,rows=len(rows))

def main():
    assert json.loads((ROOT/'checks.json').read_text())['status']=='PASS'
    jobs=[(n,s,'iid') for n in SYSTEMS for s in (30,31,32)]+[(n,s,p) for n in ('ac','kdv_cossin','ks','wave') for s in (30,31,32) for p in ('positive','negative')]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for future in as_completed([pool.submit(one,j) for j in jobs]):print(future.result(),flush=True)
    rows=[r for p in (ROOT/'explore').glob('*.json') for r in json.loads(p.read_text())['rows']];summary=[]
    for profile in ('iid','positive','negative'):
        for mode in ('raw','noise_only'):
            for design in ('strips','corner'):
                rr=[r for r in rows if r['profile']==profile and r['mode']==mode and r['design']==design];valid=[r for r in rr if r['model']]
                out=dict(profile=profile,mode=mode,design=design,variables=len(rr),non_ok=sum(r['status']!='ok' for r in rr),sigma_ratio_quantiles=np.quantile([r['sigma_ratio'] for r in valid],[0,.1,.5,.9,1]),
                    rho_error_quantiles=np.quantile([x-r['true_rho'] for r in valid for x in r['model'][1]],[0,.1,.5,.9,1]))
                summary.append(out)
    S.R.dump(ROOT/'explore_summary.json',dict(status='COMPLETE',records=len(jobs),summary=summary,source_sha256={str(Path(__file__)):S.E.sha(__file__)}))
    for r in summary:print(r,flush=True)

if __name__=='__main__':main()
