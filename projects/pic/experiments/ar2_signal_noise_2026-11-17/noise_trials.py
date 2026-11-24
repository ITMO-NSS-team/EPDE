from pathlib import Path
import sys,json,importlib.util,time
ROOT=Path(__file__).resolve().parent
import fit as F
np=F.np
spec=importlib.util.spec_from_file_location('ar2_trial_coverage',F.PREVIOUS/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R=C.S,C.R

def main():
    parent=F.PREVIOUS/'noise_trials_correlated.json';prior=json.loads(parent.read_text());rows=[];start=time.monotonic()
    for old in prior['rows']:
        n,rho,b=old['n'],old['true_rho'],old['draw'];pid={0.:0,.4:1,-.4:2}[rho];rng=np.random.default_rng(np.random.SeedSequence([960105,n,pid,b]));y=R.WF.ar_noise(rng,(n,),.2,[rho]);assert R.digest(y)==old['field_sha256']
        model,info=F.estimate(y,warm=old['info']);rows.append(dict(n=n,true_rho=rho,draw=b,model=model,info=info,field_sha256=R.digest(y)))
        if b==63:print(n,rho,'DONE',round(time.monotonic()-start,1),flush=True)
    summary=[]
    for n in (301,320,1001):
        for rho in (0.,.4,-.4):
            rr=[r for r in rows if r['n']==n and r['true_rho']==rho];good=[r['model'] for r in rr if r['model']]
            summary.append(dict(n=n,true_rho=rho,draws=len(rr),incomplete=sum(r['info']['status']!='ok' for r in rr),available=len(good),
                sigma_ratio_quantiles=np.quantile([m[0]/.2 for m in good],[0,.1,.5,.9,1]),rho_rmse=float(np.sqrt(np.mean([(m[1][0]-rho)**2 for m in good])))))
    R.dump(ROOT/'noise_trials.json',dict(status='COMPLETE',rows=rows,summary=summary,seconds=time.monotonic()-start,
        source_sha256={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'fit.py',ROOT/'protocol.txt',F.PREVIOUS/'correlated_fit.py',F.PREVIOUS/'latent_fit.py')},input_sha256={str(parent):S.E.sha(parent)}))
    for r in summary:print(r,flush=True)

if __name__=='__main__':main()
