from bridge import *
from full_pipeline_saved import Experiment
from exact_metric_pair import paired
import compressed_time as F
from scipy.linalg import qr
from copy import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

def one(name):
    e=Experiment(name,50,'iid');rng=np.random.default_rng(984906);pilots={'anchor':e.pilot,'perturbed':{v:y+.01*rng.normal(size=y.shape) for v,y in e.pilot.items()}};errors=[];timings=[];fit_deltas=[];profiles=[]
    for rho in (None,.4,-.4):
        model=e.model if rho is None else {v:[m[0],[rho]*len(m[1]),m[2]] for v,m in e.model.items()};start=time.monotonic();exact={tag:e.metric(pil,model,mc=4) for tag,pil in pilots.items()};oldtime=time.monotonic()-start;start=time.monotonic();fast=paired(e,pilots,model,mc=4);newtime=time.monotonic()-start;timings.append(dict(rho=rho,old=oldtime,new=newtime))
        for tag in pilots:
            for b in e.banks:
                scale=np.max(np.abs(exact[tag][b]));error=float(np.max(np.abs(exact[tag][b]-fast[tag][b]))/scale);np.testing.assert_allclose(fast[tag][b],exact[tag][b],rtol=2e-8,atol=2e-11*scale);errors.append(error)
    for v,y in e.raw.items():
        cc=B.A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;scale=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/scale];data=cc.fields[0]
        if data.shape[1]<=2*data.shape[0]:continue
        _,rr=qr(data.T,mode='economic',check_finite=False);comp=copy(cc);comp.fields=[rr.T*np.sqrt(data.shape[0]/data.shape[1])];comp.count=data.shape[0]**2
        for rho in (0.,.4,-.4):
            for k in (1e-3,.1,3.):
                args=([rho],[k],[.9],[-.5],B.A.covariance_band);old=B.A.profile_score(cc,*args);new=B.A.profile_score(comp,*args);np.testing.assert_allclose(old[0],new[0],rtol=1e-10,atol=1e-10);np.testing.assert_allclose(old[1],new[1],rtol=1e-8,atol=1e-9);profiles.append(dict(objective_error=abs(old[0]-new[0]),gradient_error=float(np.max(np.abs(old[1]-new[1])))))
        start=time.monotonic();m,i=F.estimate(y);fit_time=time.monotonic()-start;old=e.model[v];sigma=abs(m[0]/old[0]-1);rho=float(np.max(np.abs(np.array(m[1])-old[1])));assert sigma<1e-4 and rho<1e-4,(name,v,sigma,rho);assert i['status']=='ok';fit_deltas.append(dict(variable=v,sigma_relative_difference=sigma,max_rho_difference=rho,new_seconds=fit_time))
    return dict(system=name,status='PASS',matrix_pairs=len(errors),max_covariance_relative_error=max(errors),timings=timings,profile_checks=profiles,fit_deltas=fit_deltas)

def main():
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,n) for n in SYSTEMS]):r=f.result();rows.append(r);print(r['system'],'PASS',r['fit_deltas'],flush=True)
    dump(ROOT/'acceleration_checks.json',dict(status='PASS',systems=15,matrix_pairs=sum(r['matrix_pairs'] for r in rows),max_covariance_relative_error=max(r['max_covariance_relative_error'] for r in rows),rows=rows,source_sha256=sources('acceleration_checks.py','compressed_time.py','exact_metric_pair.py','exact_acceleration_protocol.txt')))
if __name__=='__main__':main()
