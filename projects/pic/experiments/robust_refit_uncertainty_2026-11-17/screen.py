from common import *
import robust_fit as F
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,time
def one(job):
    name,profile=job;start=time.monotonic();p=R.datasets.load(name);pid,rho=S.PROFILES[profile];rng=np.random.default_rng(np.random.SeedSequence([960101,pid,36]))
    raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()}
    key=f'{name}_{profile}_seed36';parent=PREVIOUS/'score_frozen'/(key+'.json');old=read(parent);hh={v:R.digest(y) for v,y in raw.items()};assert hh==old['raw_sha256']
    fits={v:F.estimate(y) for v,y in raw.items()};sources=hashes('screen.py','robust_fit.py','common.py','protocol.txt');fp=ROOT/'screen_frozen'/(key+'.json')
    dump(fp,dict(raw_fits=fits,raw_sha256=hh,source_sha256=sources,input_sha256={str(parent):sha(parent)}));rows=[]
    for mode,fields in [('raw',raw),('noise_only',{v:raw[v]-p.data[v] for v in raw})]:
        for v,y in fields.items():
            m,info=fits[v] if mode=='raw' else F.estimate(y)
            rows.append(dict(system=name,profile=profile,variable=v,mode=mode,model=m,info=info,sigma_ratio=m[0]/(.05*np.std(p.data[v])),true_rho=rho))
    dump(ROOT/'screen'/(key+'.json'),dict(rows=rows,source_sha256=sources,frozen_sha256=sha(fp),seconds=time.monotonic()-start))
    return [(r['system'],r['profile'],r['variable'],r['mode'],round(r['sigma_ratio'],4),np.round(r['model'][1],4).tolist(),r['info']['status']) for r in rows]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();assert read(ROOT/'checks.json')['status']=='PASS'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one,j) for j in JOBS]):print(f.result(),flush=True)
    rows=[r for p in (ROOT/'screen').glob('*.json') for r in read(p)['rows']];dump(ROOT/'screen_summary.json',dict(status='COMPLETE',records=23,rows=rows,source_sha256=hashes('screen.py')))
if __name__=='__main__':main()
