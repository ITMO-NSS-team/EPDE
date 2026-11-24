from compat_context import *
import compat_fit as H
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache

RHOS=(-.8,-.4,0.,.4,.8,.95)
PROFILES=('iid','positive','negative')
SIGNAL=[('primary',n,p,s) for n,ss in [('ode',range(500,564)),('vdp',range(500,516)),('duffing',range(500,508))] for s in ss for p in PROFILES]+[('stress','ode','positive',s) for s in range(700,1724)]

def sources():return src('compat_fit.py','compat_study.py','protocol.txt')
def compact(f):return {k:v for k,v in f.items() if k not in ('gram','beta')}
def key_of(cohort,name,p,s):return f'{cohort}_{name}_{p}_seed{s}'
def controlled(phase,n,k,j):
    rng=np.random.default_rng(np.random.SeedSequence([990501,phase,n,k,j]));return R.WF.ar_noise(rng,(n,),1.,[RHOS[k]])

def control_chunk(phase,n,k,start):
    rows=[]
    for j in range(start,start+16):
        y=controlled(phase,n,k,j);reg=G.Regression(y,64);fit=reg.fit();limit=H.boundary_limit(y);nll0=reg.df*(fit['objective']+1);value=H.likelihood(reg,1.,RHOS[k]);rows.append(dict(index=j,fit=compact(fit),nll0=nll0,truth_nll=value,lr=value-nll0,limit=limit,limit_lr=limit['nll']-nll0,field_sha256=R.digest(y)))
    folder='control_calibration' if phase==0 else 'control_validation';key=f'n{n}_rho{k}_start{start:03d}';dump(ROOT/folder/(key+'.json'),dict(status='COMPLETE',phase=phase,n=n,rho_index=k,rho=RHOS[k],start=start,rows=rows,source_sha256=sources()));return key

def freeze_calibration():
    rows=[read(p) for p in sorted((ROOT/'control_calibration').glob('*.json'))];assert len(rows)==96;cal={};configs=[]
    for n in (320,1001):
        cuts=[]
        for k,rho in enumerate(RHOS):
            rr=[x for d in rows if d['n']==n and d['rho_index']==k for x in d['rows']];assert len(rr)==128;cut=float(sorted(r['lr'] for r in rr)[125]);cuts.append(cut);configs.append(dict(n=n,rho=rho,cutoff=cut,selected_warnings=sum(r['fit']['status']!='ok' for r in rr)))
        cal[str(n)]=max(cuts)
    dump(ROOT/'calibration_frozen.json',dict(status='FROZEN',cutoffs=cal,configurations=configs,source_sha256=sources(),input_sha256={str(p):sha(p) for p in sorted((ROOT/'control_calibration').glob('*.json'))}));print('CALIBRATION FROZEN',cal,flush=True)

def observe(job):
    cohort,name,p,s=job;e=Point.case(name,p,s);y=e.raw['u'];model,info,mean=G.estimate(y);guard,gi=H.select(y,info);rich=info['full_fits']['64'];richmodel=[rich['sigma'],[rich['rho']],[2]];limit=H.boundary_limit(y);key=key_of(*job);nll0=(len(y)-64)*(rich['objective']+1);info['full_fits']={p:compact(f) for p,f in info['full_fits'].items()};path=ROOT/'signal_frozen'/(key+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,raw=y,cv_mean=mean)
    dump(path.with_suffix('.json'),dict(status='FROZEN',key=key,cohort=cohort,name=name,profile=p,seed=s,n=len(y),models=dict(cv=model,compatible=guard,rich=richmodel),cv_info=info,compatible_info=gi,rich_status=rich['status'],nll0=nll0,limit=limit,limit_lr=limit['nll']-nll0,raw_sha256=R.digest(y),array_sha256=sha(path),source_sha256=sources(),input_sha256=e.inputs));return key

def evaluate(key):
    path=ROOT/'signal_frozen'/(key+'.json');d=read(path);e=Point.case(d['name'],d['profile'],d['seed']);y=e.raw['u'];assert R.digest(y)==d['raw_sha256'];sigma=.05*np.std(e.p.data['u']);reg=G.Regression(y,64);truth_nll=H.likelihood(reg,sigma,e.rho);lr=truth_nll-d['nll0'];cal=read(ROOT/'calibration_frozen.json');cuts=dict(nominal=H.CUTOFF,grid_calibrated=cal['cutoffs'][str(len(y))]);errs={m:dict(sigma_ratio=model[0]/sigma,rho_error=model[1][0]-e.rho,near_unit=abs(model[1][0])>.95) for m,model in d['models'].items()};warnings=dict(cv=d['cv_info']['status'],compatible=d['compatible_info']['status'],rich=d['rich_status']);limitwarning=d['limit_lr']<-1e-5
    out=dict(status='COMPLETE',key=key,cohort=d['cohort'],name=d['name'],profile=d['profile'],seed=d['seed'],errors=errs,selected_p=dict(cv=d['cv_info']['selected_p'],compatible=d['compatible_info']['selected_p'],rich=64),selected_status=warnings,true_sigma=sigma,true_rho=e.rho,truth_nll=truth_nll,lr=lr,limit_lr=d['limit_lr'],limit_below_stored_optimum=limitwarning,coverage={m:lr<=c for m,c in cuts.items()},unbounded_certified={m:d['limit_lr']<c-1e-6 for m,c in cuts.items()},limit_near_cutoff={m:abs(d['limit_lr']-c)<=1e-6 for m,c in cuts.items()},cutoffs=cuts,source_sha256=sources(),input_sha256={str(path):sha(path),str(ROOT/'calibration_frozen.json'):sha(ROOT/'calibration_frozen.json')});dump(ROOT/'signal_results'/(key+'.json'),out);return key

def main():
    assert read(ROOT/'checks.json')['status']=='PASS';dump(ROOT/'launch.json',dict(status='FROZEN',signal_jobs=SIGNAL,control_rhos=RHOS,source_sha256=sources()))
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(control_chunk,0,n,k,start) for n in (320,1001) for k in range(6) for start in range(0,128,16)]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%24==0:print('control calibration',i,'of96',flush=True)
        freeze_calibration()
        futures=[pool.submit(control_chunk,1,n,k,start) for n in (320,1001) for k in range(6) for start in range(0,64,16)]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%24==0:print('control validation',i,'of48',flush=True)
        keys=[]
        for f in as_completed([pool.submit(observe,j) for j in SIGNAL]):
            keys.append(f.result())
            if len(keys)%64==0:print('fresh signal fits',len(keys),'of1288',flush=True)
        dump(ROOT/'signal_all_frozen.json',dict(status='FROZEN',source_sha256=sources(),input_sha256={str(ROOT/'signal_frozen'/(key+'.json')):sha(ROOT/'signal_frozen'/(key+'.json')) for key in keys}))
        for f in as_completed([pool.submit(evaluate,key) for key in keys]):f.result()
    dump(ROOT/'process.json',dict(status='COMPLETE',signals=len(keys),controlled_calibration=1536,controlled_validation=768,source_sha256=sources()));print('STUDY COMPLETE',len(keys),flush=True)

if __name__=='__main__':main()
