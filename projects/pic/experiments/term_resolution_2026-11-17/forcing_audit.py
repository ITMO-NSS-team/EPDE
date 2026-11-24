"""Paired independent evaluation of all three frozen readouts."""
import json
import numpy as np
from projected import ROOT, SOURCE, BANKS, sha


def main():
    out=ROOT/'forcing'
    lock=json.loads((out/'lock.json').read_text())
    assert sha(ROOT/'forcing_protocol.txt')==lock['protocol']
    assert sha(out/'directions.npz')==lock['directions']
    assert sha(ROOT/'projected'/'directions.npz')==lock['pilot_directions']
    src=SOURCE/'results'/'kdv_cossin_iid_seed57.json'
    d=json.loads(src.read_text())
    c=next(x for x in d['candidates'] if x['label']=='reference')
    idx=c['xi'][2]
    assert idx in d['deterministic']
    with np.load(src.with_suffix('.npz')) as z, np.load(out/'directions.npz') as f, np.load(ROOT/'projected'/'directions.npz') as p:
        forcing={b:f[b] for b in BANKS};pilot={b:p[b] for b in BANKS}
        np.testing.assert_array_equal(f['vectors'],p['vectors'])
        for b in BANKS:
            x=z[b+'_observed'][:,idx]
            np.testing.assert_allclose(x/np.linalg.norm(x),forcing[b],rtol=1e-13,atol=1e-15)
    rows=[]
    paths=sorted(out.glob('*_47*.json'));assert len(paths)==12
    for path in paths:
        d=json.loads(path.read_text()); vals={}
        assert d['directions']==lock['directions'] and d['seed'] in range(4705,4709)
        with np.load(path.with_suffix('.npz')) as z:
            for arm,dirs in [('whole',None),('pilot',pilot),('forcing',forcing)]:
                boot=[]; obs=[]; err=[]
                for b in BANKS:
                    def squared(x):
                        if dirs is None:
                            return np.sum(x*x,axis=-2)
                        return np.square(np.moveaxis(x,-2,-1)@dirs[b])
                    boot.append(squared(z[b+'_errors']))
                    obs.append(squared(z[b+'_observed']))
                    err.append(squared(z[b+'_observed']-z[b+'_clean']))
                boot=np.stack(boot,axis=1);scale=boot[:32].mean(axis=0)
                cutoff=float(sorted((boot[32:]/scale).max(axis=(1,2)))[122])
                ratio=np.max(np.array(obs)/scale,axis=0)/cutoff
                noise_ratio=np.max(np.array(err)/scale,axis=0)/cutoff
                vals[arm]=dict(accepted=(ratio<=1).tolist(),covered=(noise_ratio<=1).tolist(),
                               ratios=ratio.tolist(),noise_ratios=noise_ratio.tolist(),cutoff=cutoff)
                if arm!='pilot':
                    i=int(arm=='forcing')
                    np.testing.assert_allclose(cutoff,d['cutoff'][i],rtol=2e-12)
                    np.testing.assert_allclose(ratio,d['ratios'][i],rtol=2e-12)
        rows.append(dict(seed=d['seed'],profile=d['profile'],arms=vals,warnings=d['warnings'],
                         json_sha256=sha(path),array_sha256=sha(path.with_suffix('.npz'))))
    def tally(rr):
        return {arm:{k:np.sum([r['arms'][arm][k] for r in rr],axis=0).tolist() for k in ('accepted','covered')}
                for arm in ['whole','pilot','forcing']}
    summary={p:tally([r for r in rows if r['profile']==p]) for p in ('iid','positive','negative')}
    result=dict(status='PASS',candidate_order=['physical','dispersion_only'],total=tally(rows),
                by_profile=summary,records=rows,source=sha(ROOT/'forcing_audit.py'))
    (ROOT/'forcing_audit.json').write_text(json.dumps(result,indent=2))
    print(result['total'],result['by_profile'])


if __name__=='__main__':
    main()
