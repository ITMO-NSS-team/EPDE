"""Recompute the two decisions and noise containment from all saved row residuals."""
import json
import numpy as np
from projected import BANKS, OUT, SOURCE, sha


def main():
    lock=json.loads((OUT/'lock.json').read_text())
    path=SOURCE/'results'/'kdv_cossin_iid_seed57.npz'
    assert sha(path)==lock['training_array'] and sha(OUT/'directions.npz')==lock['direction_sha256']
    paths=sorted(p for p in OUT.glob('*_*.json') if p.stem.split('_')[-1].isdigit())
    assert len(paths)==6
    rows=[]
    with np.load(OUT/'directions.npz') as t, np.load(path) as training:
        directions={b:t[b] for b in BANKS}
        for b in BANKS:
            contrast=training[b+'_pilot']@t['vectors'][:,1]-training[b+'_pilot']@t['vectors'][:,0]
            np.testing.assert_allclose(contrast/np.linalg.norm(contrast),directions[b],atol=1e-12)
    for path in paths:
        d=json.loads(path.read_text());scores=np.empty((160,2,3,2));actual=np.empty((2,3,2))
        observed=np.empty_like(actual);clean_relative={}
        assert d['directions']==lock['direction_sha256'] and len(d['draw_hashes'])==160
        with np.load(path.with_suffix('.npz')) as z:
            for i,b in enumerate(BANKS):
                for j in range(2):
                    def energy(x):
                        return [x@x, (x@directions[b])**2]
                    errors=z[b+'_errors'][:,:,j]
                    scores[:,:,i,j]=np.array([energy(x) for x in errors])
                    obs=z[b+'_observed'][:,j]; clean=z[b+'_clean'][:,j]
                    actual[:,i,j]=energy(obs-clean); observed[:,i,j]=energy(obs)
                    clean_relative[b,j]=float(np.linalg.norm(clean)/z[b+'_target_norm'])
        scale=scores[:32].mean(axis=0)
        cutoff=np.partition((scores[32:]/scale).max(axis=(2,3)),122,axis=0)[122]
        ratios=(observed/scale).max(axis=1)/cutoff[:,None]
        noise_ratios=(actual/scale).max(axis=1)/cutoff[:,None]
        np.testing.assert_allclose(scale,d['scale'],rtol=2e-12)
        np.testing.assert_allclose(cutoff,d['cutoff'],rtol=2e-12)
        np.testing.assert_allclose(ratios,d['ratios'],rtol=2e-12)
        rows.append(dict(profile=d['profile'],seed=d['seed'],ratios=ratios.tolist(),
                         accepted=(ratios<=1).tolist(), covered_noise=(noise_ratios<=1).tolist(),
                         noise_ratios=noise_ratios.tolist(),warnings=d['warnings'],
                         relative_clean_residual={b:[clean_relative[b,j] for j in range(2)] for b in BANKS},
                         source_sha256=sha(path),array_sha256=sha(path.with_suffix('.npz'))))
    dest=OUT.parent/'projected_audit.json'
    dest.write_text(json.dumps(dict(status='PASS',records=rows,arm_order=['whole','projected'],
                                    candidate_order=['physical','dispersion_only']),indent=2))
    for r in rows:
        print(r['profile'],r['seed'],r['accepted'],r['covered_noise'],np.round(r['ratios'],3))


if __name__=='__main__':
    main()
