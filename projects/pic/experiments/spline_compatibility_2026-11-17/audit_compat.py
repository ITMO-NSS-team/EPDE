from compat_study import *
from scipy.linalg import svd,cholesky,cho_solve
import traceback

def close(x,y):np.testing.assert_allclose(x,y,rtol=3e-8,atol=2e-8)

@lru_cache(None)
def dense_cov(n,rho):
    q=H.contrasts(n);r=rho**abs(np.arange(n)[:,None]-np.arange(n)[None,:]);c=q.T@r@q;l=cholesky((c+c.T)/2,lower=True);return l,float(2*np.log(np.diag(l)).sum())

def dense_truth(y,sigma,rho):
    z=H.contrasts(len(y)).T@y;l,ld=dense_cov(len(y),rho);return len(z)*np.log(sigma*sigma)+float(z@cho_solve((l,True),z))/(sigma*sigma)+ld

@lru_cache(None)
def difference_geometry(n):
    # Alternative limit calculation: first differences have covariance 2*eta I.
    # The differentiated spline design has rank p-1; no dense K is used here.
    u,s,v=svd(np.diff(G.basis(n,64),axis=0),full_matrices=False);assert s[-1]<1e-12*s[0];assert s[-2]>1e-8*s[0];ld=(n-64)*np.log(2)-np.log(n)+2*np.log(s[:-1]).sum();return u[:,:-1],float(ld)

def limit_check(y,saved):
    u,ld=difference_geometry(len(y));dy=np.diff(y);res=dy-u@(u.T@dy);q=float(res@res)/2;eta=q/(len(y)-64);nll=(len(y)-64)*(np.log(eta)+1)+ld;close(eta,saved['eta']);close(nll,saved['nll'])

def selected_check(y,fit):
    reg=G.Regression(y,fit['p']);v,d=reg.evaluate(fit['transformed_rho'],True,True);close(v,fit['objective']);close(d['sigma'],fit['sigma']);close(d['rho'],fit['rho']);assert fit['status']=='boundary' if abs(fit['rho'])>1-1e-5 else fit['status'] in ('ok','unresolved_arithmetic')

def control_one(path):
    path=Path(path);d=read(path);warnings=[]
    for row in d['rows']:
        y=controlled(d['phase'],d['n'],d['rho_index'],row['index']);assert R.digest(y)==row['field_sha256'];selected_check(y,row['fit']);limit_check(y,row['limit']);close(dense_truth(y,1.,d['rho']),row['truth_nll']);close(row['truth_nll']-row['nll0'],row['lr']);fine=G.Regression(y,64).fit(np.linspace(-G.BOUND,G.BOUND,129));improvement=(row['fit']['objective']-fine['objective'])*(len(y)-64)
        if improvement>1e-5:warnings.append(dict(index=row['index'],improvement=improvement,fine=compact(fine)))
        if row['index']%16==0:close(H.dense_likelihood(y,row['fit']['sigma'],row['fit']['rho']),row['nll0'])
    return dict(status='PASS',record=str(path),cases=len(d['rows']),fine_grid_improvements=warnings,dense_truth_checks=len(d['rows']),independent_limit_checks=len(d['rows']))

def signal_one(key):
    path=ROOT/'signal_frozen'/(key+'.json');d=read(path);out=read(ROOT/'signal_results'/(key+'.json'));e=Point.case(d['name'],d['profile'],d['seed']);y=e.raw['u'];assert R.digest(y)==d['raw_sha256'];assert sha(path.with_suffix('.npz'))==d['array_sha256'];m,info,mean=G.estimate(y)
    with np.load(path.with_suffix('.npz')) as a:assert np.array_equal(y,a['raw']);close(mean,a['cv_mean'])
    for p,f in d['cv_info']['full_fits'].items():
        selected_check(y,f)
        for field in ('objective','sigma','rho'):close(info['full_fits'][p][field],f[field])
    assert info['selected_p']==d['cv_info']['selected_p'];close([r['score'] for r in info['selection']],[r['score'] for r in d['cv_info']['selection']]);guard,gi=H.select(y,info);assert gi['selected_p']==d['compatible_info']['selected_p'];assert gi['status']==d['compatible_info']['status'];close([r['lr'] for r in gi['rows']],[r['lr'] for r in d['compatible_info']['rows']]);close(guard[0],d['models']['compatible'][0]);close(guard[1],d['models']['compatible'][1]);limit_check(y,d['limit']);sigma=.05*np.std(e.p.data['u']);close(dense_truth(y,sigma,e.rho),out['truth_nll']);close(out['truth_nll']-d['nll0'],out['lr']);fine=G.Regression(y,64).fit(np.linspace(-G.BOUND,G.BOUND,129));improvement=(info['full_fits']['64']['objective']-fine['objective'])*(len(y)-64)
    for method,cut in out['cutoffs'].items():assert out['coverage'][method]==(out['lr']<=cut);assert out['unbounded_certified'][method]==(out['limit_lr']<cut-1e-6)
    for method,model in d['models'].items():close(model[0]/sigma,out['errors'][method]['sigma_ratio']);close(model[1][0]-e.rho,out['errors'][method]['rho_error'])
    dense_selected=0
    if d['seed']%16==0 or gi['selected_p']!=info['selected_p'] or gi['status']!='ok':
        for p in {64,info['selected_p'],gi['selected_p']}:
            f=info['full_fits'][str(p)];v=H.dense_likelihood(y,f['sigma'],f['rho']);lr=next(r['lr'] for r in gi['rows'] if r['p']==p);close(v-d['nll0'],lr);dense_selected+=1
    return dict(status='PASS',key=key,selection_replays=1,dense_truth_checks=1,dense_selected_checks=dense_selected,independent_limit_checks=1,fine_grid_improvements=[dict(improvement=improvement,fine=compact(fine))] if improvement>1e-5 else [])

def safe(kind,key):
    try:r=control_one(key) if kind=='control' else signal_one(key)
    except Exception:r=dict(status='FAIL',record=str(key),error=traceback.format_exc())
    name=Path(key).parent.name+'_'+Path(key).stem if kind=='control' else key;r['source_sha256']=src('audit_compat.py','compat_study.py','compat_fit.py');dump(ROOT/(kind+'_audits')/(name+'.json'),r);return r

def main():
    assert read(ROOT/'process.json')['status']=='COMPLETE';jobs=[('control',str(p)) for folder in ('control_calibration','control_validation') for p in sorted((ROOT/folder).glob('*.json'))]+[('signal',p.stem) for p in sorted((ROOT/'signal_frozen').glob('*.json'))];rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(safe,*j) for j in jobs]):
            r=f.result();rows.append(r)
            if len(rows)%64==0 or r['status']!='PASS':print('audit',len(rows),'of',len(jobs),r['status'],r.get('error',''),flush=True)
    out=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=src('audit_compat.py'));dump(ROOT/'audit.json',out);assert out['status']=='PASS';print('AUDIT PASS',len(rows),flush=True)

if __name__=='__main__':main()
