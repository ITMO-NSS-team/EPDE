from common import *
import correlated_pilot as P
from audit_time import temporal
from audit_followups import links
from scipy.linalg import solve

def main():
    rng=np.random.default_rng(983006);mathchecks=[]
    for time_axis in (True,False):
        axis=np.linspace(0,1,41);n=len(axis)
        if time_axis:d,u,_=R.spline_basis(tuple(axis),tuple(range(n)))
        else:d=(np.arange(n)/n)**4;u=P.dct(np.eye(n),axis=0,norm='ortho').T
        k=(u*d)@u.T;y=np.sin(5*axis)[:,None]+.03*rng.normal(size=(n,3))
        for rho in (0.,.4,-.4):
            dd,back,forward=P.axis_basis(axis,rho,time_axis);cov=rho**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
            for lam in (.01,.3,2.):
                direct=solve(np.eye(n)+lam*cov@k,y,assume_a='gen');spectral=back@((1/(1+lam*dd))[:,None]*(forward@y));error=float(np.max(np.abs(direct-spectral)));np.testing.assert_allclose(direct,spectral,rtol=2e-7,atol=2e-8);mathchecks.append(error)
    axes=[np.linspace(0,1,51),np.linspace(0,1,32)];yy=np.sin(5*axes[0])[:,None]*np.cos(3*axes[1])[None,:]+.03*rng.normal(size=(51,32));zero={'u':[1.,[0.,0.],[]]};a,ai=R.reconstruct({'u':yy},axes,np.arange(51));b,bi=P.reconstruct({'u':yy},axes,zero);np.testing.assert_allclose(a['u'],b['u'],rtol=2e-7,atol=2e-8);assert all(np.isfinite(row['lam']) and row['lam']>0 for row in bi['u'])
    records=[];hashes_count=0;fits=0
    for seed in (45,46,47):
        name=f'ac_positive_seed{seed}.json';out=read(ROOT/'pilot_screen'/name);fpath=ROOT/'pilot_screen_frozen'/name;f=read(fpath);assert sha(fpath)==out['frozen_sha256'];hashes_count+=links(out)+links(f)+1;parent=read(ROOT/'time_validation_frozen'/name);p=R.datasets.load('ac');pid,rho=S.PROFILES['positive'];rng=np.random.default_rng(np.random.SeedSequence([982101,pid,seed]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};models={v:mm[0] for v,mm in parent['time'].items()};pilot,info=P.reconstruct(raw,R.axes_of(p),models);assert {v:R.digest(y) for v,y in pilot.items()}==f['pilot_sha256'];assert info==f['pilot_info'];rng=np.random.default_rng(np.random.SeedSequence([982102,pid,seed]))
        for j,row in enumerate(f['rows']):
            fields={v:y+R.WF.ar_noise(rng,y.shape,*models[v][:2]) for v,y in pilot.items()};assert {v:R.digest(y) for v,y in fields.items()}==row['field_sha256']
            for v,y in fields.items():temporal(y,*row['fit'][v]);fits+=1
            np.testing.assert_allclose(out['ratios_new'][j],row['fit']['u'][0][0]/models['u'][0],rtol=1e-14)
        records.append(dict(seed=seed,mean_sigma_ratio_old=float(np.mean(out['ratios_old'])),mean_sigma_ratio_new=float(np.mean(out['ratios_new'])),statuses=out['statuses']))
    dump(ROOT/'audit_pilot_screen.json',dict(status='PASS',records=records,normal_equation_checks=len(mathchecks),max_normal_equation_error=max(mathchecks),finite_smoothing_iid_equivalence=True,bootstrap_score_replays=fits,hash_links=hashes_count,source_sha256=hashes('audit_pilot_screen.py','audit_time.py','correlated_pilot.py')));print(records,flush=True)
if __name__=='__main__':main()
