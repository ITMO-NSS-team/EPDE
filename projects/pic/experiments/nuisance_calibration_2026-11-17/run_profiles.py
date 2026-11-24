from calibration_context import *
import profile_region as Q
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.linalg import cholesky

PROFILES=('iid','positive','negative')
ANCHORS=dict(iid='ode_iid_seed60',positive='ode_positive_seed57',negative='ode_negative_seed60')
NOMINAL=float(-2*np.log(.025))


def sources():return src('protocol.txt','profile_region.py','run_profiles.py')
def key_of(cohort,profile,seed):return f'{cohort}_{profile}_seed{seed}'


def generate(cohort,profile,seed):
    if cohort=='ode':
        e,_,_=C.primary('ode','iid',51);e.profile=profile;e.seed=seed;e.pid,e.rho=S.PROFILES[profile];y=e.true_field(0,True)['u'];sigma=.05*np.std(e.p.data['u']);return y,np.array([np.log(sigma),np.arctanh(e.rho)]),dict(kind='fixed_ODE_signal',raw_sha256=R.digest(y))
    anchor=read(PREVIOUS/'anchors'/(ANCHORS[profile]+'.json'));t=anchor['fit_info']['u']['theta'];k,p1,p2=np.exp(t[1]),t[2],t[3];pid,rho=S.PROFILES[profile];domain=989201 if cohort=='calibration' else 989202;rng=np.random.default_rng(np.random.SeedSequence([domain,pid,seed]));noise=R.WF.ar_noise(rng,(320,),1.,[rho]);curvature=np.zeros(318);curvature[:2]=np.sqrt(k)*cholesky(np.array([[1.,p1],[p1,1.]]),lower=True)@rng.normal(size=2);innov=rng.normal(size=316)*np.sqrt(k*(1-p1*p1)*(1-p2*p2));a1=p1*(1-p2)
    for j in range(2,318):curvature[j]=a1*curvature[j-1]+p2*curvature[j-2]+innov[j-2]
    latent=np.r_[0.,0.,np.cumsum(np.cumsum(curvature))];y=noise+latent;np.testing.assert_allclose(np.diff(latent,n=2),curvature,rtol=1e-7,atol=1e-10)
    return y,np.array([0.,np.arctanh(rho)]),dict(kind='Gaussian_AR2_curvature',sigma=1.,rho=rho,kappa=k,p1=p1,p2=p2,raw_sha256=R.digest(y),curvature_sha256=R.digest(curvature),input_sha256={str(PREVIOUS/'anchors'/(ANCHORS[profile]+'.json')):sha(PREVIOUS/'anchors'/(ANCHORS[profile]+'.json'))})


def one(cohort,profile,seed):
    start=time.monotonic();y,truth,design=generate(cohort,profile,seed);model,info=Fit.estimate(y);region=Q.region_fit(y,model,info);evaluation=Q.evaluate(y,model,info,region,truth);key=key_of(cohort,profile,seed)
    path=ROOT/cohort/(key+'.json');out=dict(status='COMPLETE',cohort=cohort,profile=profile,seed=seed,key=key,model=model,fit_info=info,region=region,evaluation=evaluation,true_noise=truth,design=design,seconds=time.monotonic()-start,source_sha256=sources())
    if cohort!='calibration':
        cuts=read(ROOT/'calibration_cutoffs.json');out['cutoff_sha256']=sha(ROOT/'calibration_cutoffs.json');out['coverage']={kind:{method:(evaluation[method]<=cut if evaluation[method] is not None else None) for method,cut in cc.items()} for kind,cc in [('nominal',dict(lr=NOMINAL,wald=NOMINAL)),('model_calibrated',cuts['cuts'][profile])]}
    dump(path,out);return dict(key=key,seconds=out['seconds'],fit_status=info['status'],lr_status=evaluation['lr_status'],wald_status=evaluation['wald_status'])


def freeze():
    cuts={};inputs={};counts={}
    for p in PROFILES:
        paths=[ROOT/'calibration'/(key_of('calibration',p,s)+'.json') for s in range(300,428)];rows=[read(path) for path in paths];cuts[p]={};counts[p]={}
        for method in ('lr','wald'):
            values=[r['evaluation'][method] if r['evaluation'][method] is not None else float('inf') for r in rows];cuts[p][method]=float(sorted(values)[125]);counts[p][method]=sum(not np.isfinite(v) for v in values)
        inputs.update({str(path):sha(path) for path in paths})
    dump(ROOT/'calibration_cutoffs.json',dict(status='FROZEN',cuts=cuts,unresolved=counts,rank=126,n=128,nominal=NOMINAL,source_sha256=sources(),input_sha256=inputs));print('CALIBRATION FROZEN',cuts,'unresolved',counts,flush=True)


def main():
    assert read(PREVIOUS/'completion_audit.json')['status']=='PASS';jobs=[('calibration',p,s) for s in range(300,428) for p in PROFILES];allrows=[]
    dump(ROOT/'launch.json',dict(status='FROZEN',calibration=jobs,validation=[(c,p,s) for c,seeds in [('model_validation',range(500,564)),('ode',range(200,264))] for s in seeds for p in PROFILES],workers=8,source_sha256=sources()))
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(one,*j) for j in jobs]):
            allrows.append(f.result())
            if len(allrows)%24==0:print('calibration',len(allrows),'of384',allrows[-1],flush=True)
        freeze();jobs=[(c,p,base+j) for j in range(64) for c,base in [('model_validation',500),('ode',200)] for p in PROFILES]
        for f in as_completed([pool.submit(one,*j) for j in jobs]):
            allrows.append(f.result())
            if len(allrows)%24==0:print('validation',len(allrows)-384,'of384',allrows[-1],flush=True)
    dump(ROOT/'process.json',dict(status='COMPLETE',rows=allrows,source_sha256=sources()))


if __name__=='__main__':main()
