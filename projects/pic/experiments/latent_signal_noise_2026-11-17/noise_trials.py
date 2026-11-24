from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parent
import latent_fit as F
np=F.np
sys.path.insert(0,str(ROOT.parent/'spectral_noise_2026-11-17'))
import stable_fit as Old
import coverage as C
S,R=C.S,C.R
spec=importlib.util.spec_from_file_location('latent_trial_oracle',C.ROOT/'identification.py');I=importlib.util.module_from_spec(spec);spec.loader.exec_module(I)

def main():
    rows=[]
    for n in (301,320,1001):
        for pid,rho in ((0,0.),(1,.4),(2,-.4)):
            for b in range(64):
                rng=np.random.default_rng(np.random.SeedSequence([960105,n,pid,b]));y=R.WF.ar_noise(rng,(n,),.2,[rho]);m,info=F.estimate(y);old,oi=Old.estimate(y,'strips');sig,cor=I.oracle(y)
                rows.append(dict(n=n,true_rho=rho,draw=b,field_sha256=R.digest(y),latent_model=m,latent_info=info,spectral_model=old,spectral_status=oi['status'],oracle_sigma=sig,oracle_rho=cor))
            print(n,rho,'DONE',flush=True)
    summary=[]
    for n in (301,320,1001):
        for rho in (0.,.4,-.4):
            rr=[r for r in rows if r['n']==n and r['true_rho']==rho]
            for method in ('latent','spectral','oracle'):
                mm=[r[method+'_model'] for r in rr] if method!='oracle' else [[r['oracle_sigma'],[r['oracle_rho']]] for r in rr]
                good=[m for m in mm if m is not None];sg=np.array([m[0]/.2 for m in good]);rh=np.array([m[1][0] for m in good]);statuses=[r['latent_info']['status'] if method=='latent' else r.get('spectral_status') if method=='spectral' else 'ok' for r in rr]
                summary.append(dict(n=n,true_rho=rho,method=method,incomplete=sum(s!='ok' for s in statuses),draws=len(rr),sigma_ratio_quantiles=np.quantile(sg,[0,.1,.5,.9,1]),rho_rmse=float(np.sqrt(np.mean((rh-rho)**2)))))
    R.dump(ROOT/'noise_trials.json',dict(status='COMPLETE',rows=rows,summary=summary,source_sha256={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'latent_fit.py',ROOT/'protocol.txt',C.ROOT/'noise_fit.py',C.ROOT/'stable_fit.py',C.ROOT/'identification.py')}))
    for r in summary:print(r,flush=True)

if __name__=='__main__':main()
