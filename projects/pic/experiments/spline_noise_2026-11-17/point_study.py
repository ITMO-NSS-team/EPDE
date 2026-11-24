from spline_context import *
import spline_fit as G
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache

PROFILES=('iid','positive','negative')
JOBS=[(name,p,s) for name,seeds in [('ode',range(300,364)),('vdp',range(300,316)),('duffing',range(300,308))] for s in seeds for p in PROFILES]
def src():return sources('spline_fit.py','point_study.py','protocol.txt')
def key_of(name,p,s):return f'{name}_{p}_seed{s}'


def case(name,profile,seed):
    e,_,_=C.primary(name,'iid',51);e.profile=profile;e.seed=seed;e.pid,e.rho=S.PROFILES[profile];e.raw=e.true_field(0,True);return e


def observe(name,profile,seed):
    start=time.monotonic();e=case(name,profile,seed);assert list(e.raw)==['u'];y=e.raw['u'];old,oldinfo=Fit.estimate(y);model,info,mean=G.estimate(y);key=key_of(name,profile,seed);path=ROOT/'point_frozen'/(key+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,raw=y,mean=mean)
    dump(path.with_suffix('.json'),dict(status='FROZEN',key=key,name=name,profile=profile,seed=seed,old_model=old,old_info=oldinfo,model=model,info=info,raw_sha256=R.digest(y),clean_sha256=R.digest(e.p.data['u']),array_sha256=sha(path),seconds=time.monotonic()-start,source_sha256=src(),input_sha256=e.inputs));return key


def evaluate(key):
    path=ROOT/'point_frozen'/(key+'.json');d=read(path);e=case(d['name'],d['profile'],d['seed']);truth=.05*np.std(e.p.data['u']);r=e.rho;assert sha(path.with_suffix('.npz'))==d['array_sha256'];models={'guarded':d['old_model'],'spline':d['model']};models.update({f'p{k}':[v['sigma'],[v['rho']],[2]] for k,v in d['info']['full_fits'].items()});errs={k:dict(sigma_ratio=m[0]/truth,rho_error=m[1][0]-r,near_unit=abs(m[1][0])>.95) for k,m in models.items()}
    with np.load(path.with_suffix('.npz')) as a:mean_error=float(np.sqrt(np.mean((a['mean']-e.p.data['u'])**2))/truth)
    out=dict(status='COMPLETE',key=key,name=d['name'],profile=d['profile'],seed=d['seed'],selected_p=d['info']['selected_p'],errors=errs,selected_status=d['info']['status'],old_status=d['old_info']['status'],mean_rmse_over_noise=mean_error,source_sha256=src(),input_sha256={str(path):sha(path)});dump(ROOT/'point_results'/(key+'.json'),out);return key


def main():
    assert read(ROOT/'checks.json')['status']=='PASS';dump(ROOT/'point_launch.json',dict(status='FROZEN',jobs=JOBS,source_sha256=src()));keys=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(observe,*j) for j in JOBS]):
            keys.append(f.result())
            if len(keys)%24==0:print('point fits',len(keys),'of264',keys[-1],flush=True)
        dump(ROOT/'point_all_frozen.json',dict(status='FROZEN',input_sha256={str(ROOT/'point_frozen'/(k+'.json')):sha(ROOT/'point_frozen'/(k+'.json')) for k in keys},source_sha256=src()))
        for f in as_completed([pool.submit(evaluate,key) for key in keys]):f.result()
    dump(ROOT/'point_process.json',dict(status='COMPLETE',cases=len(keys),source_sha256=src()));print('POINT STUDY COMPLETE',len(keys),flush=True)


if __name__=='__main__':main()
