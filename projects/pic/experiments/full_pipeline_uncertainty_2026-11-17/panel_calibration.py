from bridge import *
from scipy.linalg import eigvalsh
import time


def qscore(g,c,idx):
    cc=c[np.ix_(idx,idx)];gg=g[np.ix_(idx,idx)];scale=np.sqrt(np.diag(cc))
    cc=cc/scale[:,None]/scale[None,:];gg=gg/scale[:,None]/scale[None,:]
    return float(eigvalsh((gg+gg.T)/2,(cc+cc.T)/2)[-1])


def one(path):
    d=read(path);key=path.stem;active=set(d['active'])
    supports=sorted(set(tuple(sorted(set([c['yi'],*c['xi']])&active)) for c in d['candidates']))
    assert all(supports)
    with np.load(path.with_suffix('.npz')) as z:
        grams={b:S.gram(z[b+'_observed']) for b in ('local','broad')}
        cov={b:z[b+'_anchor_covariance'] for b in grams}
        modes={};verification=0
        for mode in ('fixed_pilot','refit_pilot'):
            banks={}
            for b in grams:
                values=np.array([[qscore(g,c,idx) for idx in supports] for g,c in zip(z[b+'_bootstrap_grams'],z[mode+'_'+b+'_covariances'])])
                assert np.all(values.max(axis=1)<=z[mode+'_'+b+'_scores']*(1+3e-7)+3e-9)
                for j in (0,len(values)-1):
                    for k,idx in enumerate(supports):
                        w,info=S.metric(z[mode+'_'+b+'_covariances'][j],list(idx));assert w is not None
                        check=float(S.operator_scores(z[b+'_bootstrap_grams'][j:j+1],w,list(idx))[0])
                        np.testing.assert_allclose(check,values[j,k],rtol=3e-7,atol=3e-9);verification+=1
                banks[b]=values.max(axis=1)
            scores=np.maximum(banks['local'],banks['broad']);cut=float(np.sort(scores)[122])
            certs=[B.C.J.certificate([grams[b] for b in grams],[cov[b] for b in grams],cut,c,d['seed']) for c in d['candidates']]
            modes[mode]=dict(cutoff=cut,cutoff_ratio=cut/d['joint'][mode]['cutoff'],scores=scores.tolist(),certificates=certs)
        frozen=dict(system=d['system'],profile=d['profile'],seed=d['seed'],supports=supports,modes=modes,
                    independent_score_checks=verification,
                    source_sha256=sources('panel_calibration.py','panel_protocol.txt'),
                    input_sha256={str(path):sha(path),str(path.with_suffix('.npz')):sha(path.with_suffix('.npz'))})
        fp=ROOT/'panel_frozen'/path.name;dump(fp,dict(status='FROZEN_EXPLORATORY',**frozen));fh=sha(fp)
        actual={b:max(qscore(z[b+'_error_gram'],cov[b],idx) for idx in supports) for b in grams}
        coverage={m:max(actual.values())<=r['cutoff'] for m,r in modes.items()}
        refs=[]
        for m,r in modes.items():
            for c in d['candidates']:
                if c['label']!='reference':continue
                v=S.E.vector(c,c['exact'],len(d['labels']))
                refs.append(dict(mode=m,candidate=c['id'],noise_covered=all(float(v@z[b+'_error_gram']@v)<=r['cutoff']*float(v@cov[b]@v) for b in grams),
                                 zero_accepted=all(float(v@grams[b]@v)<=r['cutoff']*float(v@cov[b]@v) for b in grams)))
        assert sha(fp)==fh
        out=dict(status='COMPLETE_EXPLORATORY',**frozen,actual_scores=actual,coverage=coverage,references=refs,frozen_sha256=fh)
        dump(ROOT/'panel_results'/path.name,out);print(key,coverage,flush=True)


def main():
    done=set()
    while True:
        paths=sorted((ROOT/'results').glob('*.json'))
        for path in paths:
            if path.stem not in done:
                one(path);done.add(path.stem)
        if len(done)==69:break
        if (ROOT/'processes_fast.json').exists():break
        time.sleep(2)
    dump(ROOT/'panel_process.json',dict(status='COMPLETE' if len(done)==69 else 'INCOMPLETE',records=len(done),
                                      source_sha256=sources('panel_calibration.py','panel_protocol.txt')))


if __name__=='__main__':main()
