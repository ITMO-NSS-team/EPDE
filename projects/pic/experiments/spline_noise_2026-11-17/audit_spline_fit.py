from spline_bootstrap import *
import audit_new as A
import native_score as Native
from check_spline import dense
from scipy.linalg import cholesky,solve_triangular
from scipy.optimize import linprog
import traceback,argparse
Native.restore()


def splinecheck(y,model,info,dense_selected=False):
    # Re-run selection using observations only; selected objective is checked
    # by orthogonal error contrasts and dense covariance in the point cohort.
    m,again,mean=G.estimate(y);A.close(m[0],model[0]);A.close(m[1],model[1]);assert again['selected_p']==info['selected_p'];assert again['status']==info['status']
    for aa,bb in zip(again['selection'],info['selection']):
        assert aa['p']==bb['p'];A.close(aa['score'],bb['score'])
        for x,z in zip(aa['folds'],bb['folds']):
            for k in ('score','sigma','rho'):A.close(x[k],z[k])
            assert x['status']==z['status']
    for p,row in info['full_fits'].items():
        reg=G.Regression(y,int(p));val,det=reg.evaluate(row['transformed_rho'],True,True)
        for a,b in [(val,row['objective']),(det['sigma'],row['sigma']),(det['beta'],row['beta']),(det['gram'],row['gram'])]:A.close(a,b)
    if dense_selected:
        row=info['full_fits'][str(info['selected_p'])];v,s=dense(G.Regression(y,info['selected_p']),row['transformed_rho']);A.close(v,row['objective']);A.close(s,model[0])
    return mean


def point_one(key):
    path=ROOT/'point_frozen'/(key+'.json');d=read(path);result=read(ROOT/'point_results'/(key+'.json'));e=case(d['name'],d['profile'],d['seed']);y=e.raw['u'];assert R.digest(y)==d['raw_sha256'];assert sha(path.with_suffix('.npz'))==d['array_sha256'];A.exactfit(y,d['old_model'],d['old_info']);mean=splinecheck(y,d['model'],d['info'],True)
    with np.load(path.with_suffix('.npz')) as z:A.close(z['mean'],mean);assert np.array_equal(y,z['raw'])
    truth=.05*np.std(e.p.data['u'])
    models={'guarded':d['old_model'],'spline':d['model']};models.update({f'p{k}':[r['sigma'],[r['rho']],[2]] for k,r in d['info']['full_fits'].items()})
    for name,m in models.items():A.close(m[0]/truth,result['errors'][name]['sigma_ratio']);A.close(m[1][0]-e.rho,result['errors'][name]['rho_error'])
    assert result['selected_p']==d['info']['selected_p'];A.close(np.sqrt(np.mean((mean-e.p.data['u'])**2))/truth,result['mean_rmse_over_noise'])
    return dict(status='PASS',key=key,original_fit_replays=1,spline_selection_replays=1,dense_selected_checks=1)


def bootstrap_one(key):
    d,y,pilot,sid=load(key);r=read(ROOT/'bootstrap_frozen'/(key+'.json'));out=read(ROOT/'bootstrap_results'/(key+'.json'));yy,truth,_=generate('ode',d['profile'],d['seed']);assert np.array_equal(y,yy);assert R.digest(y)==d['raw_sha256'];assert R.digest(pilot)==d['pilot_sha256'];method=d['method']
    check=A.exactfit if method=='guarded' else splinecheck
    check(y,d['model'],d['fit_info']);e=case('ode',d['profile'],d['seed']);pil,_=Pilot.reconstruct({'u':y},e.axes,{'u':d['model']});A.close(pil['u'],pilot);pid,rho=S.PROFILES[d['profile']];rng=np.random.default_rng(np.random.SeedSequence([984102,sid,pid,d['seed']]));samples=[]
    for j,fit in enumerate(r['refits']):
        field=pilot+R.WF.ar_noise(rng,pilot.shape,*d['model'][:2]);assert R.digest(field)==fit['field_sha256'];assert j==fit['draw'];check(field,fit['model'],fit['info']);samples.append([np.log(fit['model'][0]),np.arctanh(fit['model'][1][0])])
    samples=np.array(samples);A.close(samples,r['samples']);mean=samples.mean(axis=0);centered=samples-mean;cov=centered.T@centered/127;A.close(cov,r['covariance']);theta=np.array([np.log(d['model'][0]),np.arctanh(d['model'][1][0])]);cen=2*theta-mean;A.close(cen,r['basic_center']);l=cholesky(cov,lower=True);dist=np.sum(solve_triangular(l,centered.T,lower=True)**2,axis=0);A.close(sorted(dist)[125],r['empirical_cutoff'])
    for mode,c,cut in [('uncorrected',theta,NOMINAL),('basic_nominal',cen,NOMINAL),('basic_empirical',cen,r['empirical_cutoff'])]:
        if r['status']!='FROZEN':assert out['coverage'][mode] is None;continue
        q=float(np.sum(solve_triangular(l,truth-c,lower=True)**2));A.close(q,out['regions'][mode]['score']);assert out['coverage'][mode]==(q<=cut);rad=np.sqrt(cut*np.diag(cov));A.close(out['regions'][mode]['sigma_bounds'],np.exp([c[0]-rad[0],c[0]+rad[0]]));A.close(out['regions'][mode]['rho_bounds'],np.tanh([c[1]-rad[1],c[1]+rad[1]]))
    xy=np.vstack([theta,2*theta-samples]);lp=linprog(np.zeros(len(xy)),A_eq=np.vstack([xy.T,np.ones(len(xy))]),b_eq=np.r_[truth,1.],bounds=(0,None),method='highs');assert bool(lp.success)==out['coverage']['finite_hull']
    return dict(status='PASS',key=key,original_fit_replays=129 if method=='guarded' else 0,spline_selection_replays=129 if method=='spline' else 0,field_hashes=128)


def safe(kind,key):
    try:r=point_one(key) if kind=='point' else bootstrap_one(key)
    except Exception:r=dict(status='FAIL',key=key,error=traceback.format_exc())
    r['source_sha256']=src('audit_spline_fit.py','check_spline.py','spline_bootstrap.py');dump(ROOT/(kind+'_audits')/(key+'.json'),r);return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['point','bootstrap']);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args();kind=args.kind;assert read(ROOT/(kind+'_process.json'))['status']=='COMPLETE';keys=sorted(p.stem for p in (ROOT/(kind+'_frozen')).glob('*.json'));expected=264 if kind=='point' else 24;assert len(keys)==expected;rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(safe,kind,key) for key in keys]):
            r=f.result();rows.append(r)
            if len(rows)%12==0 or r['status']!='PASS':print(kind,'audit',len(rows),'of',expected,r['key'],r['status'],r.get('error',''),flush=True)
    out=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=src('audit_spline_fit.py'));dump(ROOT/(kind+'_audit.json'),out);assert out['status']=='PASS';print(kind,'AUDIT PASS',flush=True)


if __name__=='__main__':main()
