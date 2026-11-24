from pathlib import Path
import sys,json,time,importlib.util
ROOT=Path(__file__).resolve().parent
import fit as F
np=F.np
spec=importlib.util.spec_from_file_location('ar2_previous_coverage',F.PREVIOUS/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R=C.S,C.R
from concurrent.futures import ProcessPoolExecutor,as_completed
SYSTEMS=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']

def sources():return {str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'fit.py',ROOT/'protocol.txt',F.PREVIOUS/'correlated_fit.py',F.PREVIOUS/'latent_fit.py',F.PREVIOUS/'coverage.py')}

def one(job):
    name,profile=job;seed=36;start=time.monotonic();e=C.Study(name,seed,profile,'known_rho');key=f'{name}_{profile}_seed{seed}'
    parent=F.PREVIOUS/'results'/f'{name}_{profile}_curvature_seed{seed}.json';old=json.loads(parent.read_text());hh={v:R.digest(y) for v,y in e.raw.items()};assert hh==old['raw_sha256'];raw={}
    for v,y in e.raw.items():m,info=F.estimate(y,warm=old['fit_info'][v]);raw[v]=dict(model=m,info=info)
    fp=ROOT/'screen_frozen'/f'{key}.json';ss=sources();inputs={str(parent):S.E.sha(parent),**e.inputs};R.dump(fp,dict(raw_fits=raw,raw_sha256=hh,source_sha256=ss,input_sha256=inputs))
    rows=[]
    for mode,fields in [('raw',e.raw),('noise_only',{v:e.raw[v]-e.p.data[v] for v in e.raw})]:
        for v,y in fields.items():
            model,info=(raw[v]['model'],raw[v]['info']) if mode=='raw' else F.estimate(y);true=.05*float(np.std(e.p.data[v]));warm=info.get('warm',{})
            rows.append(dict(system=name,seed=seed,profile=profile,variable=v,mode=mode,model=model,info=info,status=info['status'],sigma_ratio=model[0]/true if model else None,
                true_rho=e.rho,old_model=old['model'][v] if mode=='raw' else None,objective_improvement=warm['objective']-info['objective'] if 'objective' in warm and 'objective' in info else None))
    R.dump(ROOT/'screen'/f'{key}.json',dict(rows=rows,seconds=time.monotonic()-start,frozen_sha256=S.E.sha(fp),source_sha256=ss,input_sha256=inputs))
    return dict(record=key,statuses=[r['status'] for r in rows],seconds=round(time.monotonic()-start,1))

def main():
    assert json.loads((ROOT/'checks.json').read_text())['status']=='PASS'
    jobs=[(n,'iid') for n in SYSTEMS]+[(n,p) for n in ('ac','kdv_cossin','ks','wave') for p in ('positive','negative')]
    jobs.sort(key=lambda j:(j[0] not in ('ks','ns','pde_compound'),j));results=[]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for f in as_completed([pool.submit(one,j) for j in jobs]):row=f.result();results.append(row);print(row,flush=True)
    rows=[r for p in (ROOT/'screen').glob('*.json') for r in json.loads(p.read_text())['rows']];summary=[]
    for profile in ('iid','positive','negative'):
        for mode in ('raw','noise_only'):
            rr=[r for r in rows if r['profile']==profile and r['mode']==mode];good=[r for r in rr if r['model']]
            summary.append(dict(profile=profile,mode=mode,variables=len(rr),incomplete=sum(r['status']!='ok' for r in rr),sigma_ratio_quantiles=np.quantile([r['sigma_ratio'] for r in good],[0,.1,.5,.9,1]),
                rho_error_quantiles=np.quantile([rho-r['true_rho'] for r in good for rho in r['model'][1]],[0,.1,.5,.9,1])))
    R.dump(ROOT/'screen_summary.json',dict(status='COMPLETE',records=len(jobs),summary=summary,processes=results,source_sha256=sources()))
    for r in summary:print(r,flush=True)

if __name__=='__main__':main()
