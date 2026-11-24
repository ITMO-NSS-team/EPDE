from common import *
import repair_fit as F
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,time
def one(path):
    start=time.monotonic();old=read(path);p=R.datasets.load(old['system']);pid,rho=S.PROFILES[old['profile']];rng=np.random.default_rng(np.random.SeedSequence([970101,pid,old['seed']]))
    raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};assert {v:R.digest(y) for v,y in raw.items()}==old['raw_sha256']
    rows=[]
    for v,y in raw.items():
        m,info=F.estimate(y,old['fit_info'][v]);rows.append(dict(variable=v,model=m,info=info))
    dump(ROOT/'repair'/path.name,dict(system=old['system'],seed=old['seed'],profile=old['profile'],rows=rows,raw_sha256=old['raw_sha256'],source_sha256=hashes('run_repair.py','repair_fit.py','common.py','protocol.txt'),input_sha256={str(path):sha(path)},seconds=time.monotonic()-start))
    return (path.stem,[(r['variable'],r['info']['status'],r['info']['improvement'],r['info']['projected_gradient']) for r in rows],round(time.monotonic()-start,1))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);args=ap.parse_args();files=list((PREVIOUS/'results').glob('*_ar2_*.json'));assert len(files)==69
    files.sort(key=lambda p:('ks_' not in p.name,p.name))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one,p) for p in files]):print(f.result(),flush=True)
    rows=[dict(record=p.stem,**r) for p in (ROOT/'repair').glob('*.json') for r in read(p)['rows']];dump(ROOT/'repair_summary.json',dict(status='COMPLETE',records=69,rows=rows,source_sha256=hashes('run_repair.py')))
if __name__=='__main__':main()
