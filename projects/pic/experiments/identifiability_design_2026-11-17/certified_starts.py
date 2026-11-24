from research_context import *
from scipy.optimize import minimize
import time

def estimate(y,original=None):
    old_model,old_info=F.estimate(y) if original is None else original
    cc=F.A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;scale=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/scale]
    lo=np.array([-F.P.BOUND,-40.,-F.P.BOUND,-F.P.BOUND]);hi=np.array([F.P.BOUND,20.,F.P.BOUND,F.P.BOUND]);bounds=list(zip(lo,hi))
    def objective(u):
        t,jac=F.P.inverse(u,1)
        try:
            value,g,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);return value,g*jac
        except (np.linalg.LinAlgError,FloatingPointError):return 1e30,np.zeros_like(u)
    starts=[('original_final',old_info['transformed'])]+[(f'band_start_{i}',d['transformed']) for i,d in enumerate(old_info['fits'])];rows=[]
    for label,start in starts:
        u=np.array(start);value,g=objective(u);before=value;pg=float(max(abs(u-np.clip(u-g,lo,hi))));polished=pg>2e-5
        if polished:
            opt=minimize(objective,u,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50));u=opt.x;value,g=objective(u);pg=float(max(abs(u-np.clip(u-g,lo,hi))))
        rows.append(dict(start=label,transformed=u,objective=value,initial_exact_objective=before,projected_gradient=pg,gradient=g,polished=polished,status='ok' if pg<=2e-5 and value<1e29 else 'unresolved_optimization'))
    best=min(rows,key=lambda r:r['objective']);u=np.array(best['transformed']);t,jac=F.P.inverse(u,1);value,g,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);sigma=float(scale*np.sqrt(q/cc.count))
    rhos=[float(t[0])]+list(old_model[1][1:]);status=best['status']
    if max(abs(r) for r in rhos)>1-1e-5 or t[1]>19.99:status='boundary'
    info=dict(status=status,sigma=sigma,rhos=rhos,objective=value,old_objective=old_info['objective'],improvement=old_info['objective']-value,
              projected_gradient=best['projected_gradient'],transformed=u,theta=t,selected_start=best['start'],starts=rows,contrast_count=cc.count,shape=y.shape)
    return [sigma,rhos,old_model[2]],info

def main():
    out=[]
    for key in read(ROOT/'noise_preparation.json')['objects']:
        p=ROOT/'noise_inputs'/(key+'.json');d=read(p)
        with np.load(p.with_suffix('.npz')) as z:y=z['field']
        st=time.monotonic();model,info=estimate(y,(d['model'],d['fit_info']));row=dict(key=key,old_model=d['model'],model=model,info=info,seconds=time.monotonic()-st)
        dump(ROOT/'repair_characterization'/(key+'.json'),dict(**row,source_sha256=sources('certified_starts.py','repair_protocol.txt'),input_sha256={str(p):sha(p)}));out.append(row)
        print(key,model,'gain',ccgain(info),'pg',info['projected_gradient'],round(row['seconds'],1),flush=True)
    dump(ROOT/'repair_characterization.json',dict(status='COMPLETE',rows=out,source_sha256=sources('certified_starts.py','repair_protocol.txt')))

def ccgain(info):return info['contrast_count']*info['improvement']
if __name__=='__main__':main()
