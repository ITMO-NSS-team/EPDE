from common import *
import state_score as F
import repair_fit as P
from scipy.optimize import minimize
import time
def main():
    assert read(ROOT/'state_checks.json')['status']=='PASS';rows=[];start=time.monotonic()
    for path in sorted((ROOT/'repair').glob('*.json')):
        prior=read(path);p=R.datasets.load(prior['system']);pid,rho=S.PROFILES[prior['profile']];rng=np.random.default_rng(np.random.SeedSequence([970101,pid,prior['seed']]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};assert {v:R.digest(y) for v,y in raw.items()}==prior['raw_sha256'];out=[]
        for row in prior['rows']:
            y=raw[row['variable']];dim=y.ndim;cc=A.Contrasts(y);scale=np.sqrt(sum(np.sum(d*d) for d in cc.fields)/cc.count);cc.fields=[d/scale for d in cc.fields]
            lo=np.r_[np.full(dim,-P.BOUND),np.full(dim,-40.),np.full(2*dim,-P.BOUND)];hi=np.r_[np.full(dim,P.BOUND),np.full(dim,20.),np.full(2*dim,P.BOUND)]
            best=min(row['info']['fits'],key=lambda f:f['value']);u=np.array(best['transformed'])
            def fun(u):
                t,jac=P.inverse(u,dim);value,g,q=F.profile(cc,t[:dim],np.exp(t[dim:2*dim]),t[2*dim:3*dim],t[3*dim:]);return value,g*jac
            initial_value,initial_g=fun(u);initial_pg=float(np.max(np.abs(u-np.clip(u-initial_g,lo,hi))))
            optimize=initial_pg>2e-5 or row['info']['status']!='ok';message='stationary reevaluation';nit=0
            if optimize:
                opt=minimize(fun,u,jac=True,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(ftol=1e-15,gtol=2e-7,maxiter=500,maxls=40));u=opt.x;message=str(opt.message);nit=int(opt.nit)
            value,g=fun(u);pg=float(np.max(np.abs(u-np.clip(u-g,lo,hi))));t,jac=P.inverse(u,dim);v,gg,q=F.profile(cc,t[:dim],np.exp(t[dim:2*dim]),t[2*dim:3*dim],t[3*dim:]);model=[float(scale*np.sqrt(q/cc.count)),t[:dim].tolist(),[2]*dim]
            status='ok' if pg<=2e-5 else 'unresolved_optimization'
            if np.any(np.abs(t[:dim])>1-1e-5) or np.any(t[dim:2*dim]>19.99):status='boundary'
            result=dict(record=path.stem,variable=row['variable'],status=status,model=model,transformed=u,theta=t,gradient=g,objective=value,initial_objective=initial_value,initial_projected_gradient=initial_pg,projected_gradient=pg,banded_objective=best['value'],banded_minus_state_initial=best['value']-initial_value,optimized=optimize,iterations=nit,message=message)
            rows.append(result);out.append(result);print(path.stem,row['variable'],status,'pg',pg,'iter',nit,'elapsed',round(time.monotonic()-start,1),flush=True)
        dump(ROOT/'state_polish'/path.name,dict(rows=out,raw_sha256=prior['raw_sha256'],source_sha256=hashes('polish_state.py','state_score.py','repair_fit.py','common.py','state_protocol.txt'),input_sha256={str(path):sha(path)}))
    dump(ROOT/'state_polish_summary.json',dict(status='COMPLETE',rows=rows,source_sha256=hashes('polish_state.py')))
if __name__=='__main__':main()
