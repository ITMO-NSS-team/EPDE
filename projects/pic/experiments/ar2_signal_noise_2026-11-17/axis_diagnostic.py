from pathlib import Path
import json,importlib.util,argparse
ROOT=Path(__file__).resolve().parent
import axis_fit as F
np=F.np
spec=importlib.util.spec_from_file_location('axis_previous',F.PREVIOUS/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R=C.S,C.R
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(job):
    name,v,axis=job;e=C.Study(name,36,'iid','iid');parent=F.PREVIOUS/'results'/f'{name}_iid_curvature_seed36.json';old=json.loads(parent.read_text());hh={v:R.digest(y) for v,y in e.raw.items()};assert hh==old['raw_sha256']
    info=old['fit_info'][v];dim=e.raw[v].ndim;best=min((f for f in info['fits'] if f['success'] and f['value']<1e40),key=lambda f:f['value']);theta=best['theta'];warm=dict(status='global_ar1_axis_initialization',fits=[dict(success=True,value=0.,theta=[theta[axis+j*dim] for j in range(3)])])
    model,fit=F.estimate(e.raw[v],axis,warm);fp=ROOT/'axis_frozen'/f'{name}_{v}_axis{axis}.json';sources={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'axis_fit.py',ROOT/'score_core.py',ROOT/'axis_diagnostic_protocol.txt')}
    R.dump(fp,dict(system=name,variable=v,axis=axis,model=model,info=fit,raw_sha256=hh,source_sha256=sources,input_sha256={str(parent):S.E.sha(parent)}))
    d=np.moveaxis(np.diff(e.p.data[v],n=2,axis=axis),axis,0).reshape(e.raw[v].shape[axis]-2,-1);energy=float(np.sum(d*d));flat=np.sort((d*d).ravel());n_top=max(1,int(np.ceil(len(flat)*.01)));true=.05*float(np.std(e.p.data[v]));global_fit=json.loads((ROOT/'score_screen'/f'{name}_iid_seed36.json').read_text());g=next(r for r in global_fit['rows'] if r['mode']=='raw' and r['variable']==v)
    row=dict(system=name,variable=v,axis=axis,status=fit['status'],model=model,sigma_ratio=model[0]/true if model else None,global_model=g['model'],global_sigma_ratio=g['sigma_ratio'],
        clean_curvature_mean_energy_fraction=float(np.sum(np.mean(d,axis=0)**2)*len(d)/energy),clean_curvature_top1percent_energy=float(flat[-n_top:].sum()/energy),
        clean_curvature_to_expected_noise_energy=energy/(d.size*6*true*true),clean_curvature_lag_correlations=[float(np.sum(d[lag:]*d[:-lag])/np.sqrt(np.sum(d[lag:]**2)*np.sum(d[:-lag]**2))) for lag in range(1,5)],frozen_sha256=S.E.sha(fp))
    return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);args=parser.parse_args();jobs=[]
    for name in ('pde_compound','ks','ns'):
        for v,y in R.datasets.load(name).data.items():jobs.extend((name,v,a) for a in range(y.ndim))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one,j) for j in jobs]):r=f.result();rows.append(r);print(r,flush=True)
    R.dump(ROOT/'axis_diagnostic.json',dict(status='COMPLETE',records=len(jobs),workers=args.workers,rows=rows,source_sha256={str(Path(__file__)):S.E.sha(__file__)}))

if __name__=='__main__':main()
