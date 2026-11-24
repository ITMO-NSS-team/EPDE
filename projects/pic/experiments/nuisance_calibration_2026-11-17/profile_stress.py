from run_profiles import *
from itertools import product
from scipy.optimize import minimize


def one(path):
    path=Path(path);d=read(path);y,theta,design=generate('ode',d['profile'],d['seed']);data=F.A.Contrasts(y).fields[0];bounds=[(-40.,20.),(-F.P.BOUND,F.P.BOUND),(-F.P.BOUND,F.P.BOUND)];lo=np.array([b[0] for b in bounds]);hi=np.array([b[1] for b in bounds]);rows=[dict(start='incumbent',nuisance=d['evaluation']['profile']['nuisance'],value=d['evaluation']['profile']['value'])]
    def fn(u):
        try:v,g=Q.likelihood(data,np.r_[theta,u]);return v,g[2:]
        except (FloatingPointError,np.linalg.LinAlgError):return 1e30,np.zeros(3)
    for lk,p1,p2 in product((-15.,-11.,-7.,-3.,0.),(-.5,0.,.9,.99),(-.99,-.5,0.,.5)):
        start=np.array([lk,np.arctanh(p1),np.arctanh(p2)]);fit=minimize(fn,start,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50));v,g=fn(fit.x);rows.append(dict(start=[lk,p1,p2],nuisance=fit.x,value=v,projected_gradient=float(max(abs(fit.x-np.clip(fit.x-g,lo,hi))))))
    best=min(rows,key=lambda r:r['value']);u=np.array(best['nuisance']);value,g=Q.likelihood(data,np.r_[theta,u],True);pg=float(max(abs(u-np.clip(u-g[2:],lo,hi))));gain=2*data.size*(d['evaluation']['profile']['value']-value);assert gain>=-1e-7;raw_lr=2*data.size*(value-d['region']['negative_log_likelihood_per_contrast']);cut=read(ROOT/'calibration_cutoffs.json')['cuts'][d['profile']]['lr'];out=dict(status='COMPLETE',key=path.stem,source_cohort=path.parent.name,profile=d['profile'],seed=d['seed'],selected=best,selected_original_value=value,selected_original_gradient=g,selected_projected_gradient=pg,lr_improvement=gain,old_raw_lr=d['evaluation']['raw_lr'],new_raw_lr=raw_lr,nominal_membership_changed=(raw_lr<=NOMINAL)!=d['coverage']['nominal']['lr'],calibrated_membership_changed=(raw_lr<=cut)!=d['coverage']['model_calibrated']['lr'],starts=rows,source_sha256=src('profile_stress.py','profile_stress_protocol.txt','profile_region.py'),input_sha256={str(path):sha(path)})
    dump(ROOT/'profile_stress'/(path.parent.name+'_'+path.name),out);return out


def main():
    paths=list((ROOT/'ode').glob('ode_positive*.json'))+list((ROOT/'paired_profiles').glob('*.json'));assert len(paths)==88
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=[]
        for f in as_completed([pool.submit(one,str(p)) for p in paths]):
            rows.append(f.result())
            if len(rows)%16==0:print('profile stress',len(rows),'of88',flush=True)
    out=dict(status='COMPLETE',rows=rows,max_lr_improvement=max(r['lr_improvement'] for r in rows),nominal_membership_changes=sum(r['nominal_membership_changed'] for r in rows),calibrated_membership_changes=sum(r['calibrated_membership_changed'] for r in rows),source_sha256=src('profile_stress.py','profile_stress_protocol.txt','profile_region.py'));dump(ROOT/'profile_stress.json',out);print('PROFILE STRESS',out['max_lr_improvement'],out['nominal_membership_changes'],out['calibrated_membership_changes'],flush=True)


if __name__=='__main__':main()
