from bridge import *
from scipy.linalg import eigvalsh
import csv

def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==69;rows=[];parameters=[]
    for path in files:
        d=read(path);meta={k:d[k] for k in ('system','profile','seed')};active=d['active']
        with np.load(path.with_suffix('.npz')) as z:
            for bank in ('local','broad'):
                eig=[]
                for old,new in zip(z['fixed_pilot_'+bank+'_covariances'],z['refit_pilot_'+bank+'_covariances']):
                    old=old[np.ix_(active,active)];new=new[np.ix_(active,active)];scale=np.sqrt(np.diag(old));old=old/scale[:,None]/scale[None,:];new=new/scale[:,None]/scale[None,:];eig.append(eigvalsh((new+new.T)/2,(old+old.T)/2))
                eig=np.stack(eig);change=np.max(np.abs(eig-1),axis=1);cf=d['cuts']['fixed_pilot'][bank];cr=d['cuts']['refit_pilot'][bank];rows.append(dict(**meta,bank=bank,median_max_covariance_change=float(np.median(change)),max_covariance_change=float(np.max(change)),min_covariance_ratio=float(np.min(eig)),max_covariance_ratio=float(np.max(eig)),cutoff_ratio=cr/cf,observed_score_over_fixed=d['actual_scores'][bank]/cf,observed_score_over_refit=d['actual_scores'][bank]/cr))
        for v,base in d['model'].items():
            sig=np.array([r['model'][v][0]/base[0] for r in d['refits']]);rho=np.array([max(abs(np.array(r['model'][v][1])-base[1])) for r in d['refits']]);parameters.append(dict(**meta,variable=v,median_sigma_ratio=float(np.median(sig)),min_sigma_ratio=float(sig.min()),max_sigma_ratio=float(sig.max()),median_rho_difference=float(np.median(rho)),max_rho_difference=float(rho.max()),near_unit=sum(max(abs(np.array(r['model'][v][1])))>.95 for r in d['refits'])))
    for name,rr in [('pilot_metric_effect.csv',rows),('bootstrap_parameters.csv',parameters)]:
        with (ROOT/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
    dump(ROOT/'pilot_effect.json',dict(status='COMPLETE',datasets=len(files),metric_rows=rows,parameter_rows=parameters,source_sha256=sources('pilot_effect.py'),input_sha256={str(p):sha(p) for p in files}));print('COMPLETE',len(files),flush=True)
if __name__=='__main__':main()
