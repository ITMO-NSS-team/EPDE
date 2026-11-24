"""Conditional uncertainty and independent pilot-bootstrap refits for forced KdV."""
import run_outer as S
np,E,R,ROOT=S.np,S.E,S.R,S.ROOT
import json,csv,argparse,time
from collections import Counter


def linearized(point,draws,c,coef):
    x=point[:,c['xi']];e=draws-draws.mean(axis=0)
    residual=e[:,:,c['yi']]-np.einsum('brj,j->br',e[:,:,c['xi']],coef)
    cov=residual.T@residual/(len(draws)-1);ev,u=np.linalg.eigh((cov+cov.T)/2)
    if ev[0]<=E.RCOND*ev[-1]:return dict(status='singular_residual_covariance')
    white=(u/np.sqrt(ev)).T;wx=white@x;scale=np.linalg.norm(wx,axis=0)
    if np.any(scale==0):return dict(status='rank_deficient_design')
    _,singular,vh=np.linalg.svd(wx/scale,full_matrices=False)
    if singular[-1]**2<=E.RCOND*singular[0]**2:return dict(status='rank_deficient_design')
    inv=(vh.T/(singular**2))@vh
    cc=inv/scale[:,None]/scale[None,:];sd=np.sqrt(np.diag(cc))
    influence=np.linalg.solve(wx.T@wx,wx.T)@white
    empirical=np.cov((residual@influence.T).T,ddof=1)
    np.testing.assert_allclose(cc,empirical,rtol=2e-8,atol=1e-11)
    direction=vh[-1]/scale;direction/=np.max(np.abs(direction))
    return dict(status='ok',coefficient_covariance=cc,conditional_sd=sd,
                correlation=cc/sd[:,None]/sd[None,:],normalized_singular_values=singular,
                normalized_design_eigen_ratio=float((singular[-1]/singular[0])**2),
                weakest_coefficient_direction=direction,
                residual_covariance_eigen_ratio=float(ev[0]/ev[-1]))


def fit(point,draws,c,method):
    return E.ordinary(point,c) if method=='ols' else E.irls(point,draws,c)


def run(seed):
    begin=time.monotonic();path=ROOT/'results'/f'kdv_cossin_seed{seed}.json'
    d=json.loads(path.read_text());c=next(c for c in d['candidates'] if c['id']=='e0:reference')
    arrpath=ROOT/'training_arrays'/path.with_suffix('.npz').name
    oldpath=S.N.ROOT/'results'/'kdv_cossin_seed3.json';old=json.loads(oldpath.read_text())
    rows=[]
    with np.load(arrpath) as z:
        for g,fold in enumerate(d['folds']):
            for bank in ('local','broad','mixed'):
                point,pilot,draws,independent=[z[f'g{g}_{bank}_{key}'] for key in ('point','pilot','draws','independent')]
                clean=next(r for r in old['folds'][g]['rows'] if r['bank']==bank and r['method']=='native_clean' and r['candidate']==c['id'])
                for method in ('ols','irls'):
                    observed=next(r for r in fold['rows'] if r['bank']==bank and r['method']==method and r['candidate']==c['id'])
                    coef,info=fit(pilot,draws,c,method);samples=[]
                    for i,matrix in enumerate(independent):
                        bc,bi=fit(matrix,draws,c,method)
                        samples.append(dict(draw=i,status=bi['status'],coefficients=bc,iterations=bi.get('iterations')))
                    row=dict(training=g,bank=bank,method=method,observed_status=observed['fit']['status'],
                             observed_coefficients=observed['coefficients'],pilot_status=info['status'],
                             pilot_coefficients=coef,native_clean_ols_coefficients=clean['coefficients'],
                             bootstrap_statuses=dict(Counter(r['status'] for r in samples)),samples=samples)
                    if observed['coefficients'] is not None:
                        observed_coef=np.asarray(observed['coefficients']);row['error']=observed_coef-c['exact']
                        if method=='irls':
                            lin=linearized(point,draws,c,observed_coef);row['linearized']=lin
                            if lin['status']=='ok':row['error_in_conditional_sd']=row['error']/lin['conditional_sd']
                    good=np.asarray([r['coefficients'] for r in samples if r['status']=='ok'])
                    row['bootstrap_successes']=len(good)
                    if len(good)>=2:
                        row.update(bootstrap_mean=good.mean(axis=0),bootstrap_sd=good.std(axis=0,ddof=1),
                                   bootstrap_quantiles=np.quantile(good,[.025,.5,.975],axis=0))
                        if coef is not None:row['bootstrap_drift_from_pilot']=good.mean(axis=0)-coef
                    rows.append(row)
                print('sensitivity',seed,'fold',g,bank,'done',round(time.monotonic()-begin,1),'s',flush=True)
    out=dict(system='kdv_cossin',seed=seed,candidate=c,coefficient_labels=[d['labels'][i] for i in c['xi']],
             conditional_only=True,independent_bootstrap_draws=64,rows=rows,
             input_sha256={str(p):S.sha(p) for p in (path,arrpath,oldpath)},
             source_sha256={**S.sources(),str(__file__):S.sha(__file__)})
    R.dump(ROOT/'sensitivity'/f'seed{seed}.json',out)


def checks():
    rng=np.random.default_rng(915);x=np.column_stack([np.ones(8),np.linspace(-1,1,8)])
    p=np.column_stack([x,1.3*x[:,0]-.6*x[:,1]])
    draws=np.broadcast_to(p,(512,*p.shape)).copy();draws[:,:,2]+=rng.normal(size=(512,8))*.2
    c=dict(xi=[0,1],yi=2);a=linearized(p,draws,c,np.array([1.3,-.6]));assert a['status']=='ok'
    scale=np.array([.1,30.,2.]);b=linearized(p*scale,draws*scale,c,np.array([1.3,-.6])*scale[2]/scale[:2])
    units=scale[:2]/scale[2]
    np.testing.assert_allclose(np.array(b['coefficient_covariance'])*units[:,None]*units[None,:],a['coefficient_covariance'],rtol=2e-12)
    R.dump(ROOT/'sensitivity_checks.json',dict(linearized_draw_covariance_identity=True,feature_units_invariant=True,
                                              source_sha256=S.sha(__file__)))


if __name__=='__main__':
    if S.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',default=[6,7,8]);ap.add_argument('--checks-only',action='store_true');a=ap.parse_args()
    checks()
    if not a.checks_only:
        for seed in a.seeds:run(seed)
