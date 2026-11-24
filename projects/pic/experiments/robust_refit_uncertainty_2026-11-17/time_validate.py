from common import *
import time_fit as F
import local6_fit as L
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,time

SPECIAL={('ac','positive'),('pde_compound','iid'),('ns','iid'),('ks','positive')}

def sources():return hashes('time_validate.py','time_fit.py','local6_fit.py','state_score.py','repair_fit.py','time_validation_protocol.txt','time_protocol.txt','common.py')

def one(job):
    name,profile,seed=job;start=time.monotonic();p=R.datasets.load(name);pid,rho=S.PROFILES[profile];key=f'{name}_{profile}_seed{seed}'
    rng=np.random.default_rng(np.random.SeedSequence([982101,pid,seed]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()}
    fit={v:F.estimate(y) for v,y in raw.items()};local={v:L.estimate(y) for v,y in raw.items()};boot=[];pilot_hash=None
    if (name,profile) in SPECIAL:
        pilot,pi=R.reconstruct(raw,R.axes_of(p),np.arange(len(R.axes_of(p)[0])));pilot_hash={v:R.digest(y) for v,y in pilot.items()};rng=np.random.default_rng(np.random.SeedSequence([982102,pid,seed]))
        for j in range(16):
            fields={v:y+R.WF.ar_noise(rng,y.shape,*fit[v][0][:2]) for v,y in pilot.items()}
            boot.append(dict(draw=j,field_sha256={v:R.digest(y) for v,y in fields.items()},time={v:F.estimate(y) for v,y in fields.items()},local6={v:L.estimate(y) for v,y in fields.items()}))
    fp=ROOT/'time_validation_frozen'/(key+'.json');dump(fp,dict(system=name,profile=profile,seed=seed,raw_sha256={v:R.digest(y) for v,y in raw.items()},time=fit,local6=local,bootstrap=boot,pilot_sha256=pilot_hash,source_sha256=sources()));fhash=sha(fp)
    rows=[]
    for v,y in raw.items():
        for method,result in [('time',fit[v]),('local6',local[v]),('time_noise_only',F.estimate(y-p.data[v]))]:
            model,info=result;rows.append(dict(variable=v,method=method,model=model,info=info,sigma_ratio=model[0]/(.05*np.std(p.data[v])),true_rho=rho))
    out=dict(system=name,profile=profile,seed=seed,status='COMPLETE',rows=rows,frozen_sha256=fhash,source_sha256=sources(),seconds=time.monotonic()-start)
    dump(ROOT/'time_validation'/(key+'.json'),out);return dict(system=name,profile=profile,seed=seed,seconds=out['seconds'],warnings=sum(r['info']['status']!='ok' for r in rows),bootstrap=len(boot))

def noise(job):
    n,rho=job;pid={0.:0,.4:1,-.4:2}[rho];source=PREVIOUS/'noise_trials_score.json';prior={(r['n'],r['true_rho'],r['draw']):r for r in read(source)['rows']};rows=[]
    for b in range(64):
        rng=np.random.default_rng(np.random.SeedSequence([960105,n,pid,b]));y=R.WF.ar_noise(rng,(n,),.2,[rho]);assert R.digest(y)==prior[n,rho,b]['field_sha256'];m,i=F.estimate(y);rows.append(dict(n=n,true_rho=rho,draw=b,model=m,info=i,field_sha256=R.digest(y)))
    out=dict(status='COMPLETE',rows=rows,n=n,true_rho=rho,source_sha256=sources(),input_sha256={str(source):sha(source)})
    dump(ROOT/'time_noise'/f'n{n}_profile{pid}.json',out);return dict(n=n,rho=rho,warnings=sum(r['info']['status']!='ok' for r in rows))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fresh','noise']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();assert read(ROOT/'time_checks.json')['status']=='PASS'
    jobs=[(n,p,s) for n,p in JOBS for s in (45,46,47)] if args.mode=='fresh' else [(n,r) for n in (301,320,1001) for r in (0.,.4,-.4)]
    # Long bootstrap tasks first, so the remaining jobs fill spare workers.
    if args.mode=='fresh':jobs.sort(key=lambda x:(x[:2] not in SPECIAL,x))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one if args.mode=='fresh' else noise,j) for j in jobs]):r=f.result();rows.append(r);print(r,flush=True)
    dump(ROOT/f'time_{args.mode}_processes.json',dict(status='COMPLETE',rows=rows,source_sha256=sources()))
if __name__=='__main__':main()
