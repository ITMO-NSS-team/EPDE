from bootstrap_regions import *
import audit_new as A
import native_score as Native
from scipy.optimize import linprog
from scipy.linalg import cholesky,solve_triangular
import traceback
Native.restore()


def one(key):
    d,y,pilot,sid=load(key);r=read(ROOT/'bootstrap_frozen'/(key+'.json'));out=read(ROOT/'bootstrap_results'/(key+'.json'));yy,truth,_=generate('ode',d['profile'],d['seed']);assert np.array_equal(y,yy);assert R.digest(y)==d['raw_sha256'];assert R.digest(pilot)==d['pilot_sha256'];A.exactfit(y,d['model'],d['fit_info'])
    e,_,_=C.primary('ode','iid',51);pil,_=Pilot.reconstruct({'u':y},e.axes,{'u':d['model']});A.close(pil['u'],pilot);pid,rho=S.PROFILES[d['profile']];rng=np.random.default_rng(np.random.SeedSequence([984102,sid,pid,d['seed']]));samples=[]
    for j,fit in enumerate(r['refits']):
        field=pilot+R.WF.ar_noise(rng,pilot.shape,*d['model'][:2]);assert R.digest(field)==fit['field_sha256'];assert j==fit['draw'];A.exactfit(field,fit['model'],fit['info']);samples.append([np.log(fit['model'][0]),np.arctanh(fit['model'][1][0])])
    samples=np.array(samples);A.close(samples,r['samples']);mean=samples.mean(axis=0);centered=samples-mean;cov=centered.T@centered/127;A.close(cov,r['covariance']);theta=np.array([np.log(d['model'][0]),np.arctanh(d['model'][1][0])]);cen=2*theta-mean;A.close(cen,r['basic_center']);l=cholesky(cov,lower=True);dist=np.sum(solve_triangular(l,centered.T,lower=True)**2,axis=0);A.close(sorted(dist)[125],r['empirical_cutoff'])
    for mode,c,cut in [('uncorrected',theta,NOMINAL),('basic_nominal',cen,NOMINAL),('basic_empirical',cen,r['empirical_cutoff'])]:
        if r['status']!='FROZEN':assert out['coverage'][mode] is None;continue
        q=float(np.sum(solve_triangular(l,truth-c,lower=True)**2));A.close(q,out['regions'][mode]['score']);assert out['coverage'][mode]==(q<=cut);rad=np.sqrt(cut*np.diag(cov));A.close(out['regions'][mode]['sigma_bounds'],np.exp([c[0]-rad[0],c[0]+rad[0]]));A.close(out['regions'][mode]['rho_bounds'],np.tanh([c[1]-rad[1],c[1]+rad[1]]))
    xy=np.vstack([theta,2*theta-samples]);lp=linprog(np.zeros(len(xy)),A_eq=np.vstack([xy.T,np.ones(len(xy))]),b_eq=np.r_[truth,1.],bounds=(0,None),method='highs');assert bool(lp.success)==out['coverage']['finite_hull']
    return dict(status='PASS',key=key,original_fit_replays=129,field_hashes=128,source_sha256=bsrc(),input_sha256={str(ROOT/'bootstrap_results'/(key+'.json')):sha(ROOT/'bootstrap_results'/(key+'.json'))})


def safe(key):
    try:r=one(key)
    except Exception:r=dict(status='FAIL',key=key,error=traceback.format_exc())
    dump(ROOT/'bootstrap_audits'/(key+'.json'),r);return r


def main():
    assert read(ROOT/'bootstrap_process.json')['status']=='COMPLETE'
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=[]
        for f in as_completed([pool.submit(safe,bkey(*j)) for j in JOBS]):r=f.result();rows.append(r);print('bootstrap audit',r['key'],r['status'],r.get('error',''),flush=True)
    out=dict(status='PASS' if len(rows)==24 and all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=src('audit_bootstrap.py','bootstrap_regions.py','bootstrap_protocol.txt'));dump(ROOT/'bootstrap_audit.json',out);assert out['status']=='PASS'


if __name__=='__main__':main()
