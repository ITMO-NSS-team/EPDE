from target_bootstrap import *
import audit_adequacy as V
from scipy.linalg import cho_solve
import traceback

def replay(y,info,full=False):
    for p,fit in info['cv_info']['full_fits'].items():
        v,d=G.Regression(y,int(p)).evaluate(fit['transformed_rho'],True,True);V.close(v,fit['objective']);V.close(d['sigma'],fit['sigma'])
    screens,choices=B.analyze(y,info['cv_info']);got=choices['cv_a01'];saved=info['selection'];assert got['selected_p']==saved['selected_p'] and got['status']==saved['status'];assert got['allowed']==saved['allowed']
    for a,b in zip(screens,info['screens']):
        for k in ('pvalue','f_statistic','limit_lr'):V.close(a[k],b[k])
    if full:
        model,again=estimate(y);assert again['selection']['selected_p']==saved['selected_p'];assert again['selection']['status']==saved['status'];V.close([r['score'] for r in again['cv_info']['selection']],[r['score'] for r in info['cv_info']['selection']])

def dense_score(y,theta,info):
    p=info['selection']['selected_p']
    if p is None:return None
    q=V.contrasts(len(y),p);z=q.T@y;l,ld=V.dense_cov(len(y),p,theta[1][0]);nll=(len(y)-p)*np.log(theta[0]**2)+float(z@cho_solve((l,True),z))/(theta[0]**2)+ld;fit=info['cv_info']['full_fits'][str(p)];return float(nll-(len(y)-p)*(fit['objective']+1))

def one(key):
    d,a,sid,pid=load(key);fr=read(ROOT/'target_frozen'/(key+'.json'));out=read(ROOT/'target_results'/(key+'.json'));y,clean,sigma,rho,inputs=field(d['job']);assert np.array_equal(y,a['raw']);replay(y,d['info'],True);rows=[];scores=[];full=1
    if d['model'] is not None:
        e=Point.case('ode',d['job']['profile'],d['job']['seed']);pilot,pinfo=Pilot.reconstruct({'u':y},e.axes,{'u':d['model']});V.close(pilot['u'],a['pilot']);rng=np.random.default_rng(np.random.SeedSequence([991201,sid,pid,d['job']['seed']]))
        for start in range(0,128,8):rows.extend(read(ROOT/'target_chunks'/f'{key}_start{start:03d}.json')['rows'])
        assert len(rows)==128
        for j,r in enumerate(rows):
            yy=a['pilot']+R.WF.ar_noise(rng,a['pilot'].shape,*d['model'][:2]);assert R.digest(yy)==r['field_sha256'];assert j==r['index'];do_full=j in (0,127) or r['model'] is None;replay(yy,r['info'],do_full);full+=do_full;v=dense_score(yy,d['model'],r['info'])
            if v is None:assert r['score'] is None and r['score_kind']=='infinite_abstention';scores.append(float('inf'))
            else:V.close(v,r['score']);assert r['score_kind']=='finite';scores.append(v)
        cut=sorted(scores)[125]
        if np.isfinite(cut):assert fr['cutoff_kind']=='finite';V.close(cut,fr['cutoff'])
        else:assert fr['cutoff_kind']=='infinite' and fr['cutoff'] is None
        assert fr['abstentions']==sum(not np.isfinite(v) for v in scores);assert fr['negative_statistics']==[j for j,v in enumerate(scores) if v<-1e-5];observed=dense_score(y,[sigma,[rho],[2]],d['info']);V.close(observed,out['lr']);p=d['info']['selection']['selected_p'];q=V.contrasts(len(y),p);z=q.T@y;l,ld=V.limit_cov(len(y),p);eta=float(z@cho_solve((l,True),z))/(len(y)-p);fit=d['info']['cv_info']['full_fits'][str(p)];tail=(len(y)-p)*(np.log(eta)-fit['objective'])+ld;V.close(tail,out['limit_lr'])
        if fr['status']=='FROZEN':
            for m,c in [('nominal',H.CUTOFF),('calibrated',cut)]:assert out['coverage'][m]==(observed<=c);assert out['unbounded_certified'][m]==(tail<c-1e-6)
        else:assert all(v is None for v in out['coverage'].values())
    else:assert fr['draws']==0 and fr['status']=='unresolved_observed_selection';assert all(v is None for v in out['coverage'].values())
    return dict(status='PASS',key=key,draws=len(rows),full_selection_replays=full,fit_parameter_replays=6*(len(rows)+1),source_sha256=bsrc(),input_sha256={str(ROOT/'target_results'/(key+'.json')):sha(ROOT/'target_results'/(key+'.json'))})

def safe(key):
    try:r=one(key)
    except Exception:r=dict(status='FAIL',key=key,error=traceback.format_exc())
    dump(ROOT/'target_audits'/(key+'.json'),r);return r

def main():
    assert read(ROOT/'target_process.json')['status']=='COMPLETE';keys=read(ROOT/'target_selection.json')['selected'];rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(safe,key) for key in keys]):r=f.result();rows.append(r);print('bootstrap audit',len(rows),'of'+str(len(keys)),r['key'],r['status'],r.get('error',''),flush=True)
    out=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=src('audit_target.py','audit_adequacy.py','target_bootstrap.py'));dump(ROOT/'target_audit.json',out);assert out['status']=='PASS';print('BOOTSTRAP AUDIT PASS',flush=True)

if __name__=='__main__':main()
