from common import *
import local6_fit as F
import time
def main():
    source=PREVIOUS/'noise_trials_score.json';prior=read(source);rows=[];start=time.monotonic()
    for old in prior['rows']:
        n,rho,b=old['n'],old['true_rho'],old['draw'];pid={0.:0,.4:1,-.4:2}[rho];rng=np.random.default_rng(np.random.SeedSequence([960105,n,pid,b]));y=R.WF.ar_noise(rng,(n,),.2,[rho]);assert R.digest(y)==old['field_sha256']
        m,i=F.estimate(y);rows.append(dict(n=n,true_rho=rho,draw=b,model=m,info=i,field_sha256=R.digest(y)))
        if len(rows)%64==0:print('DONE',len(rows),round(time.monotonic()-start,1),flush=True)
    summary=[]
    for n in (301,320,1001):
        for rho in (0.,.4,-.4):
            rr=[r for r in rows if (r['n'],r['true_rho'])==(n,rho)]
            summary.append(dict(n=n,true_rho=rho,draws=64,warnings=sum(r['info']['status']!='ok' for r in rr),sigma_ratio_quantiles=np.quantile([r['model'][0]/.2 for r in rr],[0,.1,.5,.9,1]),rho_rmse=np.sqrt(np.mean([(r['model'][1][0]-rho)**2 for r in rr]))))
    dump(ROOT/'noise_trials.json',dict(status='COMPLETE',rows=rows,summary=summary,source_sha256=hashes('noise_trials.py','local6_fit.py','common.py'),input_sha256={str(source):sha(source)}));print(summary,flush=True)
if __name__=='__main__':main()
