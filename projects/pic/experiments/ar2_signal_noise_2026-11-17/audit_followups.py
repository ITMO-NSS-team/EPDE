from pathlib import Path
import json,importlib.util
ROOT=Path(__file__).resolve().parent
import fit as Initial
import score_fit as Final
from score_core import profile_score
np=Final.np
spec=importlib.util.spec_from_file_location('ar2_followup_previous',Final.PREVIOUS/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R=C.S,C.R
from functools import lru_cache

@lru_cache(None)
def sha(p):return S.E.sha(p)

def links(d):
    count=0
    for key in ('source_sha256','input_sha256'):
        for p,h in d.get(key,{}).items():assert sha(p)==h,(key,p);count+=1
    return count

def check_fit(y,model,info,method):
    good=[f for f in info.get('fits',[]) if f['success'] and np.isfinite(f['value']) and f['value']<1e40]
    if model is None:assert not good and info['status']=='optimizer_failure';return 0,0
    best=min(good,key=lambda f:f['value']);t=np.asarray(best['theta']);dim=y.ndim;cc=method.Contrasts(y);scale=np.sqrt(sum(float(np.sum(d*d)) for d in cc.fields)/cc.count);cc.fields=[d/scale for d in cc.fields]
    rho=t[:dim];k=np.exp(t[dim:2*dim]);p1=t[2*dim:3*dim];p2=t[3*dim:];q,ld=cc.evaluate(rho,k,p1,p2);objective=np.log(q/cc.count)+ld/cc.count
    np.testing.assert_allclose(model[0],scale*np.sqrt(q/cc.count),rtol=1e-12);np.testing.assert_array_equal(model[1],rho);np.testing.assert_allclose(info['objective'],objective,atol=1e-12,rtol=1e-12);assert best['value']==info['objective']
    status='boundary' if np.any(np.abs(rho)>1-1e-5) or np.any(t[dim:2*dim]>19.99) else 'ok';assert info['status']==status
    gradients=0
    if method is Final:
        obj,gradient=profile_score(cc,rho,k,p1,p2,Final.covariance_band);np.testing.assert_allclose(obj,objective,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(gradient,best['gradient'],atol=2e-10,rtol=2e-8);gradients=1
    return 1,gradients

def main():
    counts=dict(hash_links=0,screen_records=0,likelihood_replays=0,gradient_replays=0,field_replays=0,paired_noise_draws=0)
    for name in ('checks.json','checks_score.json','screen_summary.json','score_screen_summary.json','noise_trials.json','noise_trials_score.json'):
        d=json.loads((ROOT/name).read_text());assert d['status'] in ('PASS','COMPLETE');counts['hash_links']+=links(d)
    for folder,frozen,method in [('screen','screen_frozen',Initial),('score_screen','score_frozen',Final)]:
        files=sorted((ROOT/folder).glob('*.json'));assert len(files)==23
        for path in files:
            d=json.loads(path.read_text());fp=ROOT/frozen/path.name;f=json.loads(fp.read_text());counts['hash_links']+=links(d)+links(f)+1;assert sha(str(fp))==d['frozen_sha256']
            first=d['rows'][0];e=C.Study(first['system'],first['seed'],first['profile'],'known_rho');assert {v:R.digest(y) for v,y in e.raw.items()}==f['raw_sha256']
            parent=Final.PREVIOUS/'results'/f"{first['system']}_{first['profile']}_curvature_seed{first['seed']}.json";prior=json.loads(parent.read_text())
            for row in d['rows']:
                v=row['variable'];y=e.raw[v] if row['mode']=='raw' else e.raw[v]-e.p.data[v]
                if row['mode']=='raw':assert row['model']==f['raw_fits'][v]['model'] and row['info']==f['raw_fits'][v]['info'];assert row['info']['warm']==prior['fit_info'][v]
                if row['model']:assert row['sigma_ratio']==row['model'][0]/(.05*float(np.std(e.p.data[v])))
                fits,grads=check_fit(y,row['model'],row['info'],method);counts['likelihood_replays']+=fits;counts['gradient_replays']+=grads;counts['field_replays']+=1
            counts['screen_records']+=1
    initial=json.loads((ROOT/'noise_trials.json').read_text());final=json.loads((ROOT/'noise_trials_score.json').read_text());prior=json.loads((Final.PREVIOUS/'noise_trials_correlated.json').read_text());assert len(initial['rows'])==len(final['rows'])==len(prior['rows'])==576
    for a,b,old in zip(initial['rows'],final['rows'],prior['rows']):
        for key in ('n','true_rho','draw','field_sha256'):assert a[key]==b[key]==old[key]
        pid={0.:0,.4:1,-.4:2}[a['true_rho']];rng=np.random.default_rng(np.random.SeedSequence([960105,a['n'],pid,a['draw']]));y=R.WF.ar_noise(rng,(a['n'],),.2,[a['true_rho']]);assert R.digest(y)==a['field_sha256'];counts['field_replays']+=1
        for row,method in ((a,Initial),(b,Final)):
            assert row['info']['warm']==old['info'];fits,grads=check_fit(y,row['model'],row['info'],method);counts['likelihood_replays']+=fits;counts['gradient_replays']+=grads
        counts['paired_noise_draws']+=1
    R.dump(ROOT/'audit_followups.json',dict(status='PASS',**counts,source_sha256={str(Path(__file__)):S.E.sha(__file__)}));print(counts,flush=True)

if __name__=='__main__':main()
