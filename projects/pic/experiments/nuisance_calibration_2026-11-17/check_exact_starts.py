from calibration_context import *
import exact_start_fit as Exact
import audit_new as A
from concurrent.futures import ProcessPoolExecutor,as_completed


def one(path):
    p=Path(path);d=read(p);e,_,_=C.primary('ode','iid',51);e.seed=d['seed'];e.pid,e.rho=S.PROFILES[d['profile']];y=e.true_field(0,True)['u'];assert R.digest(y)==d['raw_sha256']
    start=time.monotonic();m,i=Exact.estimate(y);sec=time.monotonic()-start
    # guarded_fit already certifies the selected fit with the original Python score.
    delta=318*(i['objective']-d['info']['objective']);sr=abs(m[0]/d['model'][0]-1);rr=abs(m[1][0]-d['model'][1][0])
    classification='equivalent' if sr<=1e-4 and rr<=1e-4 and abs(delta)<=1e-4 else ('improved_objective' if delta < -1e-4 else 'material_change')
    out=dict(record=p.name,status='PASS' if classification in ('equivalent','improved_objective') and i['status']=='ok' else 'FAIL',classification=classification,old_model=d['model'],model=m,info=i,likelihood_difference=delta,sigma_relative_difference=sr,rho_difference=rr,seconds=sec,source_sha256=src('exact_start_fit.py','check_exact_starts.py'),input_sha256={str(p):sha(p)})
    dump(ROOT/'exact_start_checks'/p.name,out);return out


def main():
    folder=PREVIOUS.parent/'nuisance_direction_2026-11-17'/'guard_results';files=list(folder.glob('fresh*.json'))+[folder/f'archived_{p}_seed{s}.json' for p,s in [('positive',102),('positive',103),('positive',121),('iid',112),('iid',128),('negative',101)]]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=[]
        for f in as_completed([pool.submit(one,str(p)) for p in files]):r=f.result();rows.append(r);print(r['record'],r['classification'],r['status'],round(r['seconds'],3),r['likelihood_difference'],flush=True)
    out=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=src('exact_start_fit.py','check_exact_starts.py'));dump(ROOT/'exact_start_checks.json',out);print('CHECKS',out['status'],len(rows),flush=True)


if __name__=='__main__':main()
