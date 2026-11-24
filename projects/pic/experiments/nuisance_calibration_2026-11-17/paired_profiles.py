from bootstrap_regions import *
import audit_profiles as Audit


def one(key):
    d,y,pilot,sid=load(key);yy,truth,design=generate('ode',d['profile'],d['seed']);assert np.array_equal(y,yy);reg=Q.region_fit(y,d['model'],d['fit_info']);ev=Q.evaluate(y,d['model'],d['fit_info'],reg,truth);cuts=read(ROOT/'calibration_cutoffs.json');path=ROOT/'paired_profiles'/(key+'.json');cov={kind:{m:(ev[m]<=cut if ev[m] is not None else None) for m,cut in cc.items()} for kind,cc in [('nominal',dict(lr=NOMINAL,wald=NOMINAL)),('model_calibrated',cuts['cuts'][d['profile']])]}
    out=dict(status='COMPLETE',key=key,cohort='ode',profile=d['profile'],seed=d['seed'],design=design,true_noise=truth,model=d['model'],fit_info=d['fit_info'],region=reg,evaluation=ev,coverage=cov,cutoff_sha256=sha(ROOT/'calibration_cutoffs.json'),source_sha256=src('paired_profiles.py','paired_comparison_protocol.txt','profile_region.py'),input_sha256={str(ROOT/'bootstrap_anchors'/(key+'.json')):sha(ROOT/'bootstrap_anchors'/(key+'.json'))});dump(path,out);check=Audit.one(path);dump(ROOT/'paired_audits'/(key+'.json'),check);return check


def main():
    assert read(ROOT/'bootstrap_process.json')['status']=='COMPLETE'
    with ProcessPoolExecutor(max_workers=8) as pool:rows=[f.result() for f in as_completed([pool.submit(one,bkey(*j)) for j in JOBS])]
    dump(ROOT/'paired_process.json',dict(status='COMPLETE',rows=rows,source_sha256=src('paired_profiles.py','paired_comparison_protocol.txt','profile_region.py')));print('PAIRED PROFILES COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
