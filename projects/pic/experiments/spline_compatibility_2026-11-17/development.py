from compat_context import *
from scipy.stats import chi2

def evaluate(y,info):
    reg=G.Regression(y,64);base=info['full_fits']['64'];nll0=reg.df*(base['objective']+1);rows=[]
    for p in G.SIZES:
        f=info['full_fits'][str(p)];v,d=reg.evaluate(f['transformed_rho'],True,True);nll=reg.df*np.log(f['sigma']**2)+d['quadratic']/f['sigma']**2+d['logdet_covariance']+d['logdet_mean_gram'];lr=nll-nll0;cv=next(r['score'] for r in info['selection'] if r['p']==p);rows.append(dict(p=p,lr=lr,cv=cv,sigma=f['sigma'],rho=f['rho'],status=f['status']))
    chosen=min([r for r in rows if r['lr']<=chi2.ppf(.975,2)],key=lambda r:(r['cv'],r['p']));return dict(old_p=info['selected_p'],selected=chosen,rows=rows)

def main():
    rows=[]
    for key in ('ode_positive_seed354','ode_positive_seed334','ode_positive_seed302'):
        d=read(PREVIOUS/'point_frozen'/(key+'.json'))
        with np.load(PREVIOUS/'point_frozen'/(key+'.npz')) as a:y=a['raw']
        rows.append(dict(key=key,**evaluate(y,d['info'])))
    key='ode_positive_seed402_spline';d=read(PREVIOUS/'bootstrap_anchors'/(key+'.json'));f=read(PREVIOUS/'bootstrap_frozen'/(key+'.json'))
    with np.load(PREVIOUS/'bootstrap_anchors'/(key+'.npz')) as a:pilot=a['pilot']
    e=Point.case('ode',d['profile'],d['seed']);rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,e.seed]))
    for j in range(77):
        y=pilot+R.WF.ar_noise(rng,pilot.shape,*d['model'][:2])
        if j not in (46,76):continue
        assert R.digest(y)==f['refits'][j]['field_sha256'];rows.append(dict(key=key+f'_draw{j}',**evaluate(y,f['refits'][j]['info'])))
    dump(ROOT/'development.json',dict(status='COMPLETE',role='Archived failure development only; not validation.',rows=rows,source_sha256=src('development.py')));print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
