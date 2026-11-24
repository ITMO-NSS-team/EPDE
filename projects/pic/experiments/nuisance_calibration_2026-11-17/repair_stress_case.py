"""Posthoc diagnostic only: release a better restricted solution.

The start was found during truth-based validation; it is NOT an operational
replacement or a retroactive change to any frozen primary result.
"""
from run_profiles import *
from scipy.optimize import minimize
import audit_new as A
import native_score as Native


def main():
    stress=read(ROOT/'profile_stress.json');rows=[]
    for r in stress['rows']:
        if r['new_raw_lr']>=-1e-5:continue
        path=ROOT/r['source_cohort']/(r['key']+'.json');d=read(path);y,theta,design=generate('ode',d['profile'],d['seed']);cc=F.A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;sc=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/sc];start=np.r_[theta[1],r['selected']['nuisance']];lo=np.array([-F.P.BOUND,-40.,-F.P.BOUND,-F.P.BOUND]);hi=np.array([F.P.BOUND,20.,F.P.BOUND,F.P.BOUND])
        def fn(u):
            t,jac=F.P.inverse(u,1);v,g,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);return v,g*jac
        fit=minimize(fn,start,jac=True,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50));u=fit.x;v,g,q=Fit.reference(cc,u);pg=float(max(abs(u-np.clip(u-g,lo,hi))));model=[float(sc*np.sqrt(q/cc.count)),[float(np.tanh(u[0]))],[2]];info=dict(objective=v,transformed=u,projected_gradient=pg,status='ok' if pg<=2e-5 else 'unresolved');nll,grad=Q.likelihood(F.A.Contrasts(y).fields[0],np.r_[np.log(model[0]),u],True);new_lr=2*cc.count*(r['selected_original_value']-nll);gain=cc.count*(d['fit_info']['objective']-v);assert gain>0;Native.restore();A.exactfit(y,model,info);Native.install();rows.append(dict(key=d['key'],old_model=d['model'],posthoc_model=model,likelihood_gain=gain,posthoc_lr=new_lr,info=info,restriction_source='true_noise_pair_during_validation_not_operational',input_sha256={str(path):sha(path),str(ROOT/'profile_stress.json'):sha(ROOT/'profile_stress.json')}))
    dump(ROOT/'posthoc_repair.json',dict(status='COMPLETE',rows=rows,source_sha256=src('repair_stress_case.py','profile_region.py','profile_stress_protocol.txt')));print('POSTHOC',[(r['key'],r['likelihood_gain'],r['old_model'],r['posthoc_model'],r['posthoc_lr']) for r in rows],flush=True)


if __name__=='__main__':main()
