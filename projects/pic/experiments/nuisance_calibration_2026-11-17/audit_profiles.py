from run_profiles import *
import audit_new as A
import native_score as Native
import traceback


def one(path):
    path=Path(path);d=read(path);y,truth,design=generate(d['cohort'],d['profile'],d['seed']);assert design==d['design'];np.testing.assert_array_equal(truth,d['true_noise']);m=d['model'];i=d['fit_info'];reg=d['region'];ev=d['evaluation'];data=F.A.Contrasts(y).fields[0]
    Native.restore();A.exactfit(y,m,i)
    z=np.array(reg['parameters']);v,g=Q.likelihood(data,z,True);A.close(v,reg['negative_log_likelihood_per_contrast']);A.close(g,reg['gradient']);A.close(z[:2],reg['theta']);A.close(z[0],np.log(m[0]));A.close(z[1:],i['transformed'])
    null=ev['profile'];nv,ng=Q.likelihood(data,np.r_[truth,null['nuisance']],True);A.close(nv,null['value']);A.close(ng,null['gradient']);u=np.array(null['nuisance']);lo=np.array([-40.,-F.P.BOUND,-F.P.BOUND]);hi=np.array([20.,F.P.BOUND,F.P.BOUND]);pg=max(abs(u-np.clip(u-ng[2:],lo,hi)));A.close(pg,null['projected_gradient']);assert (pg<=2e-5)==(null['status']=='ok')
    raw=2*data.size*(nv-v);A.close(raw,ev['raw_lr']);expected_status=null['status']
    if i['status']!='ok':expected_status='unresolved_unrestricted_fit'
    if raw < -1e-5:expected_status='restricted_better_than_unrestricted'
    assert expected_status==ev['lr_status'];assert (ev['lr'] is None)==(expected_status!='ok')
    if ev['lr'] is not None:A.close(max(raw,0.),ev['lr'])
    h=np.array(reg['hessian']);A.close(np.linalg.eigvalsh(h),reg['hessian_eigenvalues']);hessian_replay=False
    if d['seed']%16==0 or reg['status']!='ok':
        step=1e-4;hh=[]
        for j in range(5):
            dz=np.zeros(5);dz[j]=step;hh.append((Q.likelihood(data,z+dz,True)[1]-Q.likelihood(data,z-dz,True)[1])/(2*step))
        hh=np.array(hh).T*data.size;hh=(hh+hh.T)/2;np.testing.assert_allclose(hh,h,rtol=3e-5,atol=3e-7*max(abs(h).max(),1.));hessian_replay=True
    if ev['wald'] is not None:
        schur=h[:2,:2]-h[:2,2:]@np.linalg.solve(h[2:,2:],h[2:,:2]);delta=truth-z[:2];A.close(float(delta@schur@delta),ev['wald']);A.close(np.linalg.inv(schur),reg['noise_covariance'])
    if d['cohort']!='calibration':
        cuts=read(ROOT/'calibration_cutoffs.json');assert sha(ROOT/'calibration_cutoffs.json')==d['cutoff_sha256']
        for kind,cc in [('nominal',dict(lr=NOMINAL,wald=NOMINAL)),('model_calibrated',cuts['cuts'][d['profile']])]:
            for method,cut in cc.items():assert d['coverage'][kind][method]==(ev[method]<=cut if ev[method] is not None else None)
    return dict(status='PASS',key=d['key'],hessian_replay=hessian_replay,original_likelihood_replays=2,input_sha256={str(path):sha(path)})


def safe(path):
    try:r=one(path)
    except Exception:r=dict(status='FAIL',key=Path(path).stem,error=traceback.format_exc())
    dump(ROOT/'audit_records'/(r['key']+'.json'),r);return r


def main():
    assert read(ROOT/'process.json')['status']=='COMPLETE';paths=[p for c in ('calibration','model_validation','ode') for p in (ROOT/c).glob('*.json')];assert len(paths)==768;rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(safe,str(p)) for p in paths]):
            r=f.result();rows.append(r)
            if r['status']!='PASS':print('AUDIT FAIL',r,flush=True)
            elif len(rows)%96==0:print('audited',len(rows),'of768',flush=True)
    cuts=read(ROOT/'calibration_cutoffs.json')
    for profile in PROFILES:
        cal=[read(p) for p in (ROOT/'calibration').glob('*.json') if read(p)['profile']==profile];assert len(cal)==128
        for method in ('lr','wald'):
            values=[d['evaluation'][method] if d['evaluation'][method] is not None else float('inf') for d in cal];assert sorted(values)[125]==cuts['cuts'][profile][method]
    out=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=src('audit_profiles.py','profile_region.py','run_profiles.py','protocol.txt'));dump(ROOT/'audit.json',out);assert out['status']=='PASS'


if __name__=='__main__':main()
