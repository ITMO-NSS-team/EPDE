from context import *
import profile_guard as G
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

def one(cohort,profile,seed):
    start=time.monotonic();e,_,_=C.primary('ode','iid',51);e.seed=seed;e.profile=profile;e.pid,e.rho=S.PROFILES[profile];y=e.true_field(0,True)['u'];key=f'{cohort}_{profile}_seed{seed}';inputs={}
    if cohort=='archived':
        p=ROOT/'survey'/f'ode_{profile}_seed{seed}.json';d=read(p);assert R.digest(y)==d['raw_sha256']['u'];old=d['model']['u'],d['fit_info']['u'];inputs[str(p)]=sha(p)
    else:old=Repair.estimate(y)
    m,i=G.estimate(y,old);meta=dict(cohort=cohort,profile=profile,seed=seed,model=m,info=i,old_model=old[0],old_info=old[1],raw_sha256=R.digest(y),source_sha256=sources('profile_guard.py','run_guard.py','guard_protocol.txt'),input_sha256=inputs)
    fp=ROOT/'guard_frozen'/(key+'.json');dump(fp,dict(**meta,status='FROZEN'));fh=sha(fp);truth=.05*np.std(e.p.data['u'])
    out=dict(**meta,status='COMPLETE',sigma_ratio=m[0]/truth,old_sigma_ratio=old[0][0]/truth,rho_error=m[1][0]-e.rho,old_rho_error=old[0][1][0]-e.rho,innovation_sigma_ratio=m[0]*np.sqrt(1-m[1][0]**2)/(truth*np.sqrt(1-e.rho**2)),frozen_sha256=fh,seconds=time.monotonic()-start)
    dump(ROOT/'guard_results'/(key+'.json'),out);print('guard',cohort,profile,seed,'sigma',round(out['old_sigma_ratio'],3),'->',round(out['sigma_ratio'],3),'gain',round(i['guard_improvement']*i['contrast_count'],3),'status',i['status'],flush=True);return dict(key=key,status=i['status'])

def main():
    # Fresh cases first; archived outcomes cannot alter the frozen guard.
    jobs=[('fresh',p,s) for p in ('positive','iid','negative') for s in range(132,164 if p=='positive' else 140)]+[('archived',p,s) for p in ('iid','positive','negative') for s in range(100,132)];rows=[]
    dump(ROOT/'guard_launch.json',dict(status='FROZEN',jobs=jobs,source_sha256=sources('profile_guard.py','run_guard.py','guard_protocol.txt')))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,*j) for j in jobs]):rows.append(f.result())
    dump(ROOT/'guard_process.json',dict(status='COMPLETE',rows=rows,source_sha256=sources('profile_guard.py','run_guard.py','guard_protocol.txt')))

if __name__=='__main__':main()
