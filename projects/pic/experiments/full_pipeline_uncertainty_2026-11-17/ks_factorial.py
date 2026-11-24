from bridge import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import time

def one(seed):
    start=time.monotonic();key=f'ks_positive_seed{seed}';parent=PREVIOUS/'time_validation_frozen'/(key+'.json');old=read(parent);p=R.datasets.load('ks');pid,rho=S.PROFILES['positive'];rng=np.random.default_rng(np.random.SeedSequence([982101,pid,seed]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};assert {v:R.digest(y) for v,y in raw.items()}==old['raw_sha256'];model={v:r[0] for v,r in old['time'].items()};axes=R.axes_of(p)
    pilots={};pilots['ordinary'],_=R.reconstruct(raw,axes,np.arange(len(axes[0])));pilots['correlated'],info=Pilot.reconstruct(raw,axes,model);assert {v:R.digest(y) for v,y in pilots['ordinary'].items()}==old['pilot_sha256'];rng=np.random.default_rng(np.random.SeedSequence([982102,pid,seed]));rows=[]
    for j in range(16):
        noise={v:R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in raw.items()};fields={tag:{v:y+noise[v] for v,y in pil.items()} for tag,pil in pilots.items()};assert {v:R.digest(y) for v,y in fields['ordinary'].items()}==old['bootstrap'][j]['field_sha256']
        for pilot in ('ordinary','correlated'):
            for method,F in [('time',Temporal),('local6',Local)]:
                fits=old['bootstrap'][j][method] if pilot=='ordinary' else {v:F.estimate(y) for v,y in fields[pilot].items()}
                rows.append(dict(draw=j,pilot=pilot,method=method,fits=fits,field_sha256={v:R.digest(y) for v,y in fields[pilot].items()},noise_sha256={v:R.digest(y) for v,y in noise.items()}))
        print(key,'draw',j+1,flush=True)
    src=sources('ks_factorial.py','protocol.txt');fp=ROOT/'factorial_frozen'/(key+'.json');dump(fp,dict(seed=seed,model=model,rows=rows,pilot_info=info,pilot_sha256={tag:{v:R.digest(y) for v,y in pil.items()} for tag,pil in pilots.items()},source_sha256=src,input_sha256={str(parent):sha(parent)}))
    summary=[]
    for pilot in pilots:
        for method in ('time','local6'):
            rr=[r for r in rows if (r['pilot'],r['method'])==(pilot,method)];fits=[(v,m,i) for r in rr for v,(m,i) in r['fits'].items()];ratios=np.array([m[0]/model[v][0] for v,m,i in fits]);err=np.array([max(abs(np.array(m[1])-model[v][1])) for v,m,i in fits]);summary.append(dict(pilot=pilot,method=method,fits=len(fits),sigma_ratio_quantiles=np.quantile(ratios,[0,.5,1]),rho_error_quantiles=np.quantile(err,[0,.5,1]),near_unit=sum(max(abs(np.array(m[1])))>.95 for v,m,i in fits),warnings=sum(i['status']!='ok' for v,m,i in fits)))
    diagnostics={tag:{v:float(np.sqrt(np.mean((y-p.data[v])**2))/model[v][0]) for v,y in pil.items()} for tag,pil in pilots.items()};out=dict(status='COMPLETE',seed=seed,summary=summary,pilot_rmse_relative_noise=diagnostics,seconds=time.monotonic()-start,frozen_sha256=sha(fp),source_sha256=src);dump(ROOT/'factorial'/(key+'.json'),out);return out

def main():
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,s) for s in (45,46,47)]):r=f.result();rows.append(r);print(r['seed'],r['summary'],flush=True)
    dump(ROOT/'factorial_summary.json',dict(status='COMPLETE',records=3,rows=rows,source_sha256=sources('ks_factorial.py','protocol.txt')))
if __name__=='__main__':main()
