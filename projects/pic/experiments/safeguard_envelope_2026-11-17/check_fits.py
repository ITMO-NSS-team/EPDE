from study_context import *
import guarded_fit as Fit
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

def one(path):
    p=Path(path);d=read(p);e,_,_=C.primary('ode','iid',51);e.seed=d['seed'];e.pid,e.rho=S.PROFILES[d['profile']];y=e.true_field(0,True)['u'];assert R.digest(y)==d['raw_sha256'];start=time.monotonic();m,i=Fit.estimate(y)
    result=dict(record=p.name,old_model=d['model'],model=m,info=i,sigma_relative_error=abs(m[0]/d['model'][0]-1),rho_error=abs(m[1][0]-d['model'][1][0]),likelihood_difference=318*(i['objective']-d['info']['objective']),seconds=time.monotonic()-start,input_sha256={str(p):sha(p)})
    result['status']='PASS' if result['sigma_relative_error']<=1e-4 and result['rho_error']<=1e-4 and abs(result['likelihood_difference'])<=1e-4 and i['status']==d['info']['status'] else 'FAIL'
    dump(ROOT/'fit_checks'/p.name,result);print('fit check',p.name,result['status'],round(result['seconds'],2),flush=True);return result

def main():
    files=list((PREVIOUS/'guard_results').glob('fresh*.json'))+[PREVIOUS/'guard_results'/f'archived_{p}_seed{s}.json' for p,s in [('positive',102),('positive',103),('positive',121),('iid',112),('iid',128),('negative',101)]];rows=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(one,str(p)) for p in files]):rows.append(f.result())
    out=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',rows=rows,source_sha256=sources('guarded_fit.py','native_score.c','native_score.py','native_score.dll','check_fits.py'));dump(ROOT/'native_fit_checks.json',out);print('FIT CHECKS',out['status'],len(rows),flush=True)

if __name__=='__main__':main()
