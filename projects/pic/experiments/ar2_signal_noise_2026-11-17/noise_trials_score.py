from pathlib import Path
import json,importlib.util,argparse,time
ROOT=Path(__file__).resolve().parent
import score_fit as F
np=F.np
spec=importlib.util.spec_from_file_location('score_trial_coverage',F.PREVIOUS/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R=C.S,C.R
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(job):
    index,old=job;n,rho,b=old['n'],old['true_rho'],old['draw'];pid={0.:0,.4:1,-.4:2}[rho];rng=np.random.default_rng(np.random.SeedSequence([960105,n,pid,b]));y=R.WF.ar_noise(rng,(n,),.2,[rho]);assert R.digest(y)==old['field_sha256']
    m,info=F.estimate(y,warm=old['info']);return dict(index=index,n=n,true_rho=rho,draw=b,model=m,info=info,field_sha256=R.digest(y))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=7);args=parser.parse_args();start=time.monotonic();parent=F.PREVIOUS/'noise_trials_correlated.json';prior=json.loads(parent.read_text());rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one,(i,r)) for i,r in enumerate(prior['rows'])]):
            rows.append(f.result())
            if len(rows)%32==0:print('DONE',len(rows),round(time.monotonic()-start,1),flush=True)
    rows.sort(key=lambda r:r['index']);summary=[]
    for n in (301,320,1001):
        for rho in (0.,.4,-.4):
            rr=[r for r in rows if r['n']==n and r['true_rho']==rho];good=[r['model'] for r in rr if r['model']]
            summary.append(dict(n=n,true_rho=rho,draws=len(rr),incomplete=sum(r['info']['status']!='ok' for r in rr),available=len(good),sigma_ratio_quantiles=np.quantile([m[0]/.2 for m in good],[0,.1,.5,.9,1]),rho_rmse=float(np.sqrt(np.mean([(m[1][0]-rho)**2 for m in good])))))
    R.dump(ROOT/'noise_trials_score.json',dict(status='COMPLETE',rows=rows,summary=summary,workers=args.workers,seconds=time.monotonic()-start,
        source_sha256={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'fit.py',ROOT/'score_fit.py',ROOT/'score_core.py',ROOT/'score_protocol.txt',ROOT/'protocol.txt',F.PREVIOUS/'correlated_fit.py',F.PREVIOUS/'latent_fit.py')},input_sha256={str(parent):S.E.sha(parent)}))
    for r in summary:print(r,flush=True)

if __name__=='__main__':main()
