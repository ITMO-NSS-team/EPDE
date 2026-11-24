from common import *
import robust_fit as F3
import local6_fit as F6
import repair_fit as P
from collections import Counter
def links(d):
    count=0
    for key in ('source_sha256','input_sha256','generated_sha256'):
        for p,h in d.get(key,{}).items():assert sha(p)==h,(key,p);count+=1
    return count
def checkfit(F,y,m,info):
    cc=F.Contrasts(y);best=min(info['fits'],key=lambda f:f['value']);value,g=cc.evaluate(np.array(best['theta']))
    np.testing.assert_allclose(value,info['objective'],rtol=2e-12,atol=2e-12);np.testing.assert_allclose(g,best['gradient'],rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(m[0],cc.scale*np.exp(best['theta'][0]),rtol=2e-12);np.testing.assert_allclose(m[1],np.tanh(best['theta'][1:]),atol=2e-12)
def main():
    count=0;replay=0
    for folder,ff,F in [('screen','screen_frozen',F3),('screen6','screen6_frozen',F6)]:
        files=list((ROOT/folder).glob('*.json'));assert len(files)==23
        for path in files:
            d=read(path);count+=links(d);frozen=ROOT/ff/path.name;assert sha(frozen)==d['frozen_sha256'];f=read(frozen);count+=links(f)
            r=d['rows'][0];p=R.datasets.load(r['system']);pid,rho=S.PROFILES[r['profile']];rng=np.random.default_rng(np.random.SeedSequence([960101,pid,36]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};assert {v:R.digest(y) for v,y in raw.items()}==f['raw_sha256']
            for r in d['rows']:
                v=r['variable'];y=raw[v] if r['mode']=='raw' else raw[v]-p.data[v];checkfit(F,y,r['model'],r['info']);np.testing.assert_allclose(r['sigma_ratio'],r['model'][0]/(.05*np.std(p.data[v])),rtol=1e-14);replay+=1
    repair=[]
    for path in sorted((ROOT/'repair').glob('*.json')):
        d=read(path);count+=links(d);p=R.datasets.load(d['system']);pid,rho=S.PROFILES[d['profile']];rng=np.random.default_rng(np.random.SeedSequence([970101,pid,d['seed']]));raw={v:y+R.WF.ar_noise(rng,y.shape,.05*np.std(y),[rho]*y.ndim) for v,y in p.data.items()};assert {v:R.digest(y) for v,y in raw.items()}==d['raw_sha256']
        for row in d['rows']:
            y=raw[row['variable']];info=row['info'];cc=A.Contrasts(y);scale=np.sqrt(sum(np.sum(d*d) for d in cc.fields)/cc.count);cc.fields=[d/scale for d in cc.fields];dim=y.ndim;best=min(info['fits'],key=lambda f:f['value']);t,jac=P.inverse(best['transformed'],dim)
            value,g=A.profile_score(cc,t[:dim],np.exp(t[dim:2*dim]),t[2*dim:3*dim],t[3*dim:],A.covariance_band);np.testing.assert_allclose(value,info['objective'],atol=1e-11,rtol=1e-11);np.testing.assert_allclose(g*jac,best['gradient'],atol=2e-8,rtol=2e-7)
            q,ld=cc.evaluate(t[:dim],np.exp(t[dim:2*dim]),t[2*dim:3*dim],t[3*dim:]);np.testing.assert_allclose(row['model'][0],scale*np.sqrt(q/cc.count),rtol=1e-12);repair.append(info);replay+=1
    assert len(repair)==84
    pure=read(ROOT/'noise_trials.json');count+=links(pure)
    for r in pure['rows']:
        n,rho,b=r['n'],r['true_rho'],r['draw'];pid={0.:0,.4:1,-.4:2}[rho];rng=np.random.default_rng(np.random.SeedSequence([960105,n,pid,b]));y=R.WF.ar_noise(rng,(n,),.2,[rho]);assert R.digest(y)==r['field_sha256'];checkfit(F6,y,r['model'],r['info']);replay+=1
    result=dict(status='PASS',screen_records=46,repair_variable_fits=84,repair_statuses=dict(Counter(i['status'] for i in repair)),pure_noise_draws=576,likelihood_score_replays=replay,hash_links=count,source_sha256=hashes('audit_followups.py'))
    dump(ROOT/'audit_followups.json',result);print(result,flush=True)
if __name__=='__main__':main()
