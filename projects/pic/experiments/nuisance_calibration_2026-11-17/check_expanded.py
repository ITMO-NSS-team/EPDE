from calibration_context import *
import check_exact_starts as Check
import expanded_fit as Expanded
from concurrent.futures import ProcessPoolExecutor,as_completed
Check.Exact=Expanded

def one(path):
    # Reproduce the comparison without overwriting the failed earlier attempt.
    p=Path(path);d=read(p);e,_,_=C.primary('ode','iid',51);e.seed=d['seed'];e.pid,e.rho=S.PROFILES[d['profile']];y=e.true_field(0,True)['u'];assert R.digest(y)==d['raw_sha256'];start=time.monotonic();m,i=Expanded.estimate(y)
    delta=318*(i['objective']-d['info']['objective']);sr=abs(m[0]/d['model'][0]-1);rr=abs(m[1][0]-d['model'][1][0]);classification='equivalent' if sr<=1e-4 and rr<=1e-4 and abs(delta)<=1e-4 else ('improved_objective' if delta < -1e-4 else 'material_change')
    out=dict(record=p.name,status='PASS' if classification in ('equivalent','improved_objective') and i['status']=='ok' else 'FAIL',classification=classification,old_model=d['model'],model=m,info=i,likelihood_difference=delta,sigma_relative_difference=sr,rho_difference=rr,seconds=time.monotonic()-start,source_sha256=src('exact_start_fit.py','expanded_fit.py','check_expanded.py'),input_sha256={str(p):sha(p)})
    dump(ROOT/'expanded_checks'/p.name,out);return out

def main():
    folder=PREVIOUS.parent/'nuisance_direction_2026-11-17'/'guard_results';files=list(folder.glob('*.json'));assert len(files)==144
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=[]
        for f in as_completed([pool.submit(one,str(p)) for p in files]):
            r=f.result();rows.append(r)
            if r['status']!='PASS' or len(rows)%24==0:print('checks',len(rows),r['record'],r['classification'],r['status'],r['likelihood_difference'],flush=True)
    out=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=src('exact_start_fit.py','expanded_fit.py','check_expanded.py'));dump(ROOT/'expanded_checks.json',out);print('EXPANDED CHECKS',out['status'],len(rows),flush=True)

if __name__=='__main__':main()
