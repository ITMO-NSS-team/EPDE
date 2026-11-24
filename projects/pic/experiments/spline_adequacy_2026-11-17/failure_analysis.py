"""Truth-based diagnostics after the main study; never used in fitting."""
from adequacy_study import *
from scipy.linalg import qr

def main():
    assert read(ROOT/'audit.json')['status']=='PASS';rows=[read(p) for p in (ROOT/'results').glob('stress_*.json')];groups={}
    for method in ('cv','own_cv_a01','adj_cv_a01','adj_smallest_a01','adj_cv_a05'):
        groups[method]={}
        for p in G.SIZES:
            a=[r for r in rows if r['methods'][method]['selected_p']==p]
            if not a:continue
            values=[r['methods'][method] for r in a];pvalues=[]
            for r in a:
                if not method.startswith(('own_','adj_')):continue
                d=read(ROOT/'frozen'/(r['key']+'.json'));kind='own' if method.startswith('own') else 'adj';pvalues.append(next(s['pvalue'] for s in d[kind+'_screens'] if s['p']==p))
            groups[method][str(p)]=dict(n=len(a),covered=sum(r['coverage'] for r in values),median_sigma_ratio=float(np.median([r['sigma_ratio'] for r in values])),median_rho_error=float(np.median([r['rho_error'] for r in values])),adequacy_pvalue_range=[min(pvalues),float(np.median(pvalues)),max(pvalues)] if pvalues else None)
    e=Point.case('ode','positive',2000);clean=e.p.data['u'];sigma=.05*np.std(clean);rho=.4;geometry=[]
    for p in (24,32,48,64):
        reg=G.Regression(clean,p);v,d=reg.evaluate(np.arctanh(rho),True,True);w=G.whiten(clean,np.arange(len(clean)),rho);u=qr(G.whiten(A.union_basis(len(clean),p),np.arange(len(clean)),rho),mode='economic')[0];res=w-u@(u.T@w);small=d['quadratic'];large=float(res@res);geometry.append(dict(p=p,signal_residual_noncentrality=small/(sigma*sigma),additional_mean_noncentrality=(small-large)/(sigma*sigma),rich_signal_noncentrality=large/(sigma*sigma),expected_variance_ratio_at_true_rho=1+small/(reg.df*sigma*sigma)))
    out=dict(status='PASS',role='Posthoc diagnosis, no estimator/cutoff changes.',groups=groups,signal_geometry=geometry,source_sha256=src('failure_analysis.py','adequacy_study.py'),input_sha256={str(ROOT/'audit.json'):sha(ROOT/'audit.json')});dump(ROOT/'failure_analysis.json',out);print('FAILURE ANALYSIS',groups['adj_cv_a01'],geometry,flush=True)

if __name__=='__main__':main()
