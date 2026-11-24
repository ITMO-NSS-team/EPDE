from adequacy_study import *
from check_adequacy import dense_f
from functools import lru_cache
from scipy.linalg import qr,cholesky,cho_solve
import traceback

def close(x,y):np.testing.assert_allclose(x,y,rtol=5e-8,atol=2e-8)

@lru_cache(None)
def contrasts(n,p):return qr(G.basis(n,p),mode='full')[0][:,p:]

@lru_cache(None)
def dense_cov(n,p,rho):
    q=contrasts(n,p);r=rho**abs(np.arange(n)[:,None]-np.arange(n)[None,:]);c=q.T@r@q;l=cholesky((c+c.T)/2,lower=True);return l,float(2*np.log(np.diag(l)).sum())

@lru_cache(None)
def limit_cov(n,p):
    q=contrasts(n,p);k=-q.T@abs(np.arange(n)[:,None]-np.arange(n)[None,:])@q;l=cholesky((k+k.T)/2,lower=True);return l,float(2*np.log(np.diag(l)).sum())

def one(key):
    path=ROOT/'frozen'/(key+'.json');d=read(path);out=read(ROOT/'results'/(key+'.json'));y,clean,sigma,rho,inputs=generate(d['job']);assert R.digest(y)==d['raw_sha256'];assert sha(path.with_suffix('.npz'))==d['array_sha256'];model,info,mean=G.estimate(y)
    with np.load(path.with_suffix('.npz')) as z:assert np.array_equal(y,z['raw']);close(mean,z['cv_mean'])
    assert info['selected_p']==d['cv_info']['selected_p'];close([r['score'] for r in info['selection']],[r['score'] for r in d['cv_info']['selection']])
    for p in G.SIZES:
        for field in ('objective','sigma','rho'):close(info['full_fits'][str(p)][field],d['cv_info']['full_fits'][str(p)][field])
    own,oc=A.analyze(y,info);adj,ac=B.analyze(y,info);dense_checks=0
    for kind,rows in [('own',own),('adj',adj)]:
        for got,saved in zip(rows,d[kind+'_screens']):
            for field in ('small_quadratic','large_quadratic','f_statistic','pvalue','limit_lr'):close(got[field],saved[field])
            assert got['fit_status']==saved['fit_status']
        for label,sel in d['selections'].items():
            if not label.startswith(kind+'_'):continue
            allowed=[r['p'] for r in rows if r['fit_status']=='ok' and r['pvalue']>=sel['alpha'] and r['limit_lr']>H.CUTOFF+1e-6];assert allowed==sel['allowed'];cv={r['p']:r['score'] for r in info['selection']};chosen=min(allowed,key=lambda p:(cv[p],p)) if allowed and sel['ordering']=='cv' else min(allowed) if allowed else None;assert chosen==sel['selected_p'];assert (sel['status']=='ok')==(chosen is not None)
    dense_required=d['job']['seed']%16==0 or any(sel['selected_p'] is None for sel in d['selections'].values())
    if dense_required:
        for kind,rows in [('own',own),('adj',adj)]:
            for s in rows:
                rr=info['full_fits'][str(s['p'])]['rho'] if kind=='own' else s['screen_rho'];ref=dense_f(y,s['p'],rr)
                for a,b in [(ref['q0'],s['small_quadratic']),(ref['q1'],s['large_quadratic']),(ref['stat'],s['f_statistic']),(ref['pvalue'],s['pvalue'])]:np.testing.assert_allclose(a,b,rtol=5e-7,atol=2e-8)
                dense_checks+=1
    used=sorted({r['selected_p'] for r in d['selections'].values() if r['selected_p'] is not None});values={};warnings=[]
    for p in used:
        fit=info['full_fits'][str(p)];q=contrasts(len(y),p);z=q.T@y;l,ld=dense_cov(len(y),p,rho);nll=(len(y)-p)*np.log(sigma*sigma)+float(z@cho_solve((l,True),z))/(sigma*sigma)+ld;kl,kld=limit_cov(len(y),p);eta=float(z@cho_solve((kl,True),z))/(len(y)-p);lim=(len(y)-p)*(np.log(eta)+1)+kld;orig=next(r for r in own if r['p']==p);close(lim,orig['limit']['nll']);base=(len(y)-p)*(fit['objective']+1);values[p]=dict(truth_nll=nll,nll0=base,lr=nll-base,limit_lr=lim-base)
        fine=G.Regression(y,p).fit(np.linspace(-G.BOUND,G.BOUND,129));improve=(fit['objective']-fine['objective'])*(len(y)-p)
        if improve>1e-5:warnings.append(dict(p=p,improvement=improve,fine=compact(fine)))
    for method,sel in d['selections'].items():
        row=out['methods'][method];p=sel['selected_p'];assert row['status']==sel['status'];assert row['selected_p']==p
        if p is None:assert row['coverage'] is None and row['sigma_ratio'] is None;continue
        fit=info['full_fits'][str(p)]
        for k,v in values[p].items():close(v,row[k])
        close(fit['sigma']/sigma,row['sigma_ratio']);close(fit['rho']-rho,row['rho_error']);assert row['coverage']==(row['lr']<=H.CUTOFF);assert row['unbounded_certified']==(row['limit_lr']<H.CUTOFF-1e-6)
        if method.startswith(('own_','adj_')):assert not row['unbounded_certified']
    return dict(status='PASS',key=key,selection_replays=1,dense_f_checks=dense_checks,dense_likelihood_checks=len(used),independent_limit_checks=len(used),fine_grid_checks=len(used),fine_grid_improvements=warnings)

def safe(key):
    try:r=one(key)
    except Exception:r=dict(status='FAIL',key=key,error=traceback.format_exc())
    r['source_sha256']=src('audit_adequacy.py','check_adequacy.py','adequacy_study.py');dump(ROOT/'audits'/(key+'.json'),r);return r

def main():
    assert read(ROOT/'process.json')['status']=='COMPLETE';keys=sorted(p.stem for p in (ROOT/'frozen').glob('*.json'));assert len(keys)==1832;rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(safe,k) for k in keys]):
            r=f.result();rows.append(r)
            if len(rows)%64==0 or r['status']!='PASS':print('audit',len(rows),'of1832',r['status'],r.get('key'),r.get('error',''),flush=True)
    out=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=src('audit_adequacy.py'));dump(ROOT/'audit.json',out);assert out['status']=='PASS';print('AUDIT PASS',len(rows),flush=True)

if __name__=='__main__':main()
