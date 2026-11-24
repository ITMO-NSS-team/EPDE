from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
import correlated_fit as F
np=F.np
sys.path.insert(0,str(ROOT.parent/'spectral_noise_2026-11-17'))
import coverage as C
S,R=C.S,C.R

def main():
    prior=json.loads((ROOT/'noise_trials.json').read_text());rows=[]
    for old in prior['rows']:
        n,rho,b=old['n'],old['true_rho'],old['draw'];pid={0.:0,.4:1,-.4:2}[rho]
        rng=np.random.default_rng(np.random.SeedSequence([960105,n,pid,b]));y=R.WF.ar_noise(rng,(n,),.2,[rho]);assert R.digest(y)==old['field_sha256']
        m,info=F.estimate(y);rows.append(dict(n=n,true_rho=rho,draw=b,field_sha256=R.digest(y),model=m,info=info))
        if b==63:print(n,rho,'DONE',flush=True)
    summary=[]
    for n in (301,320,1001):
        for rho in (0.,.4,-.4):
            rr=[r for r in rows if r['n']==n and r['true_rho']==rho];good=[r['model'] for r in rr if r['model'] is not None]
            summary.append(dict(n=n,true_rho=rho,draws=len(rr),incomplete=sum(r['info']['status']!='ok' for r in rr),
                sigma_ratio_quantiles=np.quantile([m[0]/.2 for m in good],[0,.1,.5,.9,1]),rho_rmse=float(np.sqrt(np.mean([(m[1][0]-rho)**2 for m in good])))))
    R.dump(ROOT/'noise_trials_correlated.json',dict(status='COMPLETE',rows=rows,summary=summary,
        source_sha256={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'latent_fit.py',ROOT/'correlated_fit.py',ROOT/'correlated_curvature_protocol.txt')},
        input_sha256={str(ROOT/'noise_trials.json'):S.E.sha(ROOT/'noise_trials.json')}))
    for r in summary:print(r,flush=True)

if __name__=='__main__':main()
