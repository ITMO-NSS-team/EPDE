from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parent
import correlated_fit as F
np=F.np
from scipy.optimize import minimize
sys.path.insert(0,str(ROOT.parent/'spectral_noise_2026-11-17'))
spec=importlib.util.spec_from_file_location('old_profile_coverage',ROOT.parent/'spectral_noise_2026-11-17'/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R=C.S,C.R

def main():
    rows=[];inputs={}
    for name,seed,profile in [('ac',s,'negative') for s in (33,34,35)]+[('pde_compound',s,'iid') for s in (33,34,35)]+[('ns',33,'iid')]:
        path=ROOT/'correlated_explore'/f'{name}_{profile}_seed{seed}.json';d=json.loads(path.read_text());inputs[str(path)]=S.E.sha(path);e=C.Study(name,seed,profile,'iid')
        for old in [r for r in d['rows'] if r['mode']=='raw']:
            y=e.raw[old['variable']];dim=y.ndim;cc=F.Contrasts(y);scale=np.sqrt(sum(float(np.sum(x*x)) for x in cc.fields)/cc.count);cc.fields=[x/scale for x in cc.fields]
            def obj(theta):
                try:q,ld=cc.evaluate(theta[:dim],np.exp(theta[dim:2*dim]),theta[2*dim:]);return float(np.log(q/cc.count)+ld/cc.count)
                except (np.linalg.LinAlgError,FloatingPointError):return 1e50
            def pack(opt):
                return dict(success=bool(opt.success),value=float(opt.fun),theta=opt.x.tolist(),iterations=int(opt.nit),message=str(opt.message))
            fixedrho=np.full(dim,e.rho);bounds=[(-40.,20.)]*dim+[(-1+1e-6,1-1e-6)]*dim
            bestold=min((f for f in old['info']['fits'] if f['success']),key=lambda f:f['value']);starts=[np.array(bestold['theta'][dim:]),np.r_[np.full(dim,-3.),np.full(dim,.99)],np.r_[np.full(dim,-11.),np.zeros(dim)]]
            opts=dict(ftol=1e-12,gtol=2e-6,maxiter=300,maxls=40,finite_diff_rel_step=1e-4)
            fixed=[pack(minimize(lambda u:obj(np.r_[fixedrho,u]),x,method='L-BFGS-B',jac='3-point',bounds=bounds,options=opts)) for x in starts]
            good=[f for f in fixed if f['success'] and f['value']<1e40];row=dict(system=name,seed=seed,profile=profile,variable=old['variable'],original=old['info'],fixed_rho_fits=fixed)
            if good:
                best=min(good,key=lambda f:f['value']);theta=np.r_[fixedrho,best['theta']];free=pack(minimize(obj,theta,method='L-BFGS-B',jac='3-point',bounds=[(-1+1e-6,1-1e-6)]*dim+bounds,options=opts))
                q,_=cc.evaluate(fixedrho,np.exp(theta[dim:2*dim]),theta[2*dim:]);row.update(fixed_objective=best['value'],fixed_sigma=float(scale*np.sqrt(q/cc.count)),free_restart=free,
                    fixed_minus_original=best['value']-old['info']['objective'],restart_minus_original=free['value']-old['info']['objective'],contrast_count=cc.count)
            rows.append(row);print(name,seed,old['variable'],row.get('fixed_minus_original'),row.get('restart_minus_original'),flush=True)
    sources={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'profile_diagnostic_protocol.txt',ROOT/'correlated_fit.py',ROOT/'latent_fit.py')}
    R.dump(ROOT/'profile_diagnostic.json',dict(status='COMPLETE',rows=rows,input_sha256=inputs,source_sha256=sources))

if __name__=='__main__':main()
