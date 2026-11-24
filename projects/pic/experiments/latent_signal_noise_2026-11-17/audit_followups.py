from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parent
import latent_fit as L
import correlated_fit as F
np=F.np
sys.path.insert(0,str(ROOT.parent/'spectral_noise_2026-11-17'))
spec=importlib.util.spec_from_file_location('previous_coverage',ROOT.parent/'spectral_noise_2026-11-17'/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R=C.S,C.R
from functools import lru_cache

@lru_cache(None)
def sha(p):return S.E.sha(p)

def links(d):
    n=0
    for key in ('source_sha256','input_sha256'):
        for p,h in d.get(key,{}).items():assert sha(p)==h,(key,p);n+=1
    return n

def evaluate_fit(y,model,info,method):
    good=[f for f in info.get('fits',[]) if f['success'] and np.isfinite(f['value']) and f['value']<1e40]
    if model is None:assert info['status']=='optimizer_failure' and not good;return 0
    best=min(good,key=lambda f:f['value']);theta=np.asarray(best['theta']);dim=y.ndim
    cc=method.Contrasts(y);scale=np.sqrt(sum(float(np.sum(d*d)) for d in cc.fields)/cc.count);cc.fields=[d/scale for d in cc.fields]
    args=[theta[:dim],np.exp(theta[dim:2*dim])]
    if method is F:args.append(theta[2*dim:])
    q,ld=cc.evaluate(*args);sigma=scale*np.sqrt(q/cc.count);objective=np.log(q/cc.count)+ld/cc.count
    np.testing.assert_allclose(sigma,model[0],rtol=1e-12);np.testing.assert_array_equal(theta[:dim],model[1]);np.testing.assert_allclose(objective,info['objective'],atol=1e-12,rtol=1e-12)
    assert info['objective']==best['value'];assert info['contrast_count']==cc.count and info['shape']==list(y.shape)
    status='boundary' if np.any(np.abs(theta[:dim])>1-1e-5) or np.any(theta[dim:2*dim]>19.99) else 'ok'
    assert info['status']==status
    return 1

def main():
    counts=dict(hash_links=0,exploration_records=0,likelihood_replays=0,field_replays=0,noise_trials=0)
    for name in ('checks.json','checks_correlated.json','explore_summary.json','correlated_summary.json','noise_trials.json','noise_trials_correlated.json'):
        d=json.loads((ROOT/name).read_text());assert d['status'] in ('PASS','COMPLETE');counts['hash_links']+=links(d)
    for folder,frozen,method in [('explore','explore_frozen',L),('correlated_explore','correlated_frozen',F)]:
        files=sorted((ROOT/folder).glob('*.json'));assert len(files)==69
        for path in files:
            d=json.loads(path.read_text());fp=ROOT/frozen/path.name;f=json.loads(fp.read_text());counts['hash_links']+=links(d)+links(f)+1;assert sha(fp)==d['frozen_sha256']
            first=d['rows'][0];e=C.Study(first['system'],first['seed'],first['profile'],'iid');assert {v:R.digest(y) for v,y in e.raw.items()}==f['raw_sha256']
            for row in d['rows']:
                v=row['variable'];y=e.raw[v] if row['mode']=='raw' else e.raw[v]-e.p.data[v]
                if row['mode']=='raw':assert row['model']==f['raw_fits'][v]['model'] and row['info']==f['raw_fits'][v]['info']
                if row['model']:assert row['sigma_ratio']==row['model'][0]/(.05*float(np.std(e.p.data[v])))
                counts['likelihood_replays']+=evaluate_fit(y,row['model'],row['info'],method);counts['field_replays']+=1
            counts['exploration_records']+=1
    old=json.loads((ROOT/'noise_trials.json').read_text());new=json.loads((ROOT/'noise_trials_correlated.json').read_text());assert len(old['rows'])==len(new['rows'])==576
    for a,b in zip(old['rows'],new['rows']):
        for key in ('n','true_rho','draw','field_sha256'):assert a[key]==b[key]
        pid={0.:0,.4:1,-.4:2}[a['true_rho']];rng=np.random.default_rng(np.random.SeedSequence([960105,a['n'],pid,a['draw']]));y=R.WF.ar_noise(rng,(a['n'],),.2,[a['true_rho']]);assert R.digest(y)==a['field_sha256']
        counts['field_replays']+=1;counts['likelihood_replays']+=evaluate_fit(y,a['latent_model'],a['latent_info'],L)+evaluate_fit(y,b['model'],b['info'],F);counts['noise_trials']+=1
    R.dump(ROOT/'audit_followups.json',dict(status='PASS',**counts,source_sha256={str(Path(__file__)):sha(str(Path(__file__)))}));print(counts,flush=True)

if __name__=='__main__':main()
