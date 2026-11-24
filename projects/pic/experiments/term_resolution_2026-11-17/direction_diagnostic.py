"""Post-hoc attribution on the completed projection cohort, with unchanged error draws."""
import json
import numpy as np
from projected import ROOT, OUT, SOURCE, BANKS, statistic, sha


def main():
    paths=sorted(OUT.glob('*_47*.npz'))
    assert len(paths)==6
    training=SOURCE/'results'/'kdv_cossin_iid_seed57.json'
    meta=json.loads(training.read_text())
    ref=next(c for c in meta['candidates'] if c['label']=='reference')
    forcing=ref['xi'][2]
    assert 'cos(t)sin(x)' in str(meta['labels'][forcing])
    with np.load(training.with_suffix('.npz')) as z:
        pattern={b:z[b+'_pilot'][:,forcing] for b in BANKS}
    with np.load(paths[0]) as z:
        oracle={b:z[b+'_clean'][:,1]-z[b+'_clean'][:,0] for b in BANKS}
    for dirs in [pattern,oracle]:
        for b in BANKS:
            dirs[b]=dirs[b]/np.linalg.norm(dirs[b])
    with np.load(OUT/'directions.npz') as z:
        pilot={b:z[b] for b in BANKS}
    cosine={k:{b:float(abs(d[b]@oracle[b])) for b in BANKS}
            for k,d in [('pilot',pilot),('forcing',pattern)]}
    out=[]
    for path in paths:
        d=json.loads(path.with_suffix('.json').read_text());row={}
        with np.load(path) as z:
            clean={b:z[b+'_clean'] for b in BANKS}
            observed={b:z[b+'_observed'] for b in BANKS}
            errors={b:z[b+'_errors'] for b in BANKS}
            for b in BANKS:
                delta=clean[b][:,1]-clean[b][:,0]
                np.testing.assert_allclose(delta/np.linalg.norm(delta),oracle[b],atol=1e-12)
            for name,dirs in [('pilot',pilot),('clean_oracle',oracle),('forcing',pattern)]:
                boot=statistic(errors,dirs)[:,1]
                scale=boot[:32].mean(axis=0)
                cutoff=float(np.partition((boot[32:]/scale).max(axis=(1,2)),122)[122])
                def ratio(fields):
                    return (statistic(fields,dirs)[1]/scale/cutoff).tolist()
                values=ratio(observed)
                row[name]=dict(cutoff=cutoff,observed_by_bank=values,clean_by_bank=ratio(clean),
                               passed=(np.max(values,axis=0)<=1).tolist())
                if name=='pilot':
                    np.testing.assert_allclose(np.max(values,axis=0),d['ratios'][1],rtol=1e-12)
        out.append(dict(profile=d['profile'],seed=d['seed'],arms=row))
    counts={k:np.sum([r['arms'][k]['passed'] for r in out],axis=0).tolist() for k in out[0]['arms']}
    record=dict(exploratory=True,source=sha(training),directions_to_clean_absolute_cosine=cosine,
                candidate_order=['physical','dispersion_only'],accepted=counts,records=out)
    (ROOT/'direction_diagnostic.json').write_text(json.dumps(record,indent=2))
    print(cosine,counts)


if __name__=='__main__':
    main()
