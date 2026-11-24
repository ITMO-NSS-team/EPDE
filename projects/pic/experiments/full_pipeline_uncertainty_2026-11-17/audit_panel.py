"""Independent whitening replay of exploratory fixed-panel calibration."""
from bridge import *
import time, traceback


def one(path):
    d=read(path);primary=ROOT/'results'/path.name;p=read(primary);fp=ROOT/'panel_frozen'/path.name;f=read(fp)
    assert sha(fp)==d['frozen_sha256']
    for obj in (d,f):
        for key in ('source_sha256','input_sha256'):
            for name,h in obj[key].items():assert sha(name)==h
    assert d['modes']==f['modes'];n=0;certificates=0
    with np.load(primary.with_suffix('.npz')) as z:
        grams={b:S.gram(z[b+'_observed']) for b in ('local','broad')};cov={b:z[b+'_anchor_covariance'] for b in grams}
        def score(g,c,idx):
            w,info=S.metric(c,idx);assert w is not None
            return float(S.operator_scores(g[None],w,idx)[0])
        actual={b:max(score(z[b+'_error_gram'],cov[b],idx) for idx in d['supports']) for b in grams}
        np.testing.assert_allclose(list(actual.values()),list(d['actual_scores'].values()),rtol=3e-7,atol=3e-9)
        for mode,m in d['modes'].items():
            values=[]
            for j in range(p['bootstrap']):
                values.append(max(score(z[b+'_bootstrap_grams'][j],z[mode+'_'+b+'_covariances'][j],idx) for b in grams for idx in d['supports']))
                n+=2*len(d['supports'])
            np.testing.assert_allclose(values,m['scores'],rtol=3e-7,atol=3e-9)
            np.testing.assert_allclose(np.sort(values)[122],m['cutoff'],rtol=3e-7,atol=3e-9)
            assert m['cutoff']<=p['joint'][mode]['cutoff']*(1+3e-7)
            assert d['coverage'][mode]==(max(d['actual_scores'].values())<=m['cutoff'])
            for c,cert in zip(p['candidates'],m['certificates']):
                replay=B.C.J.certificate(list(grams.values()),list(cov.values()),m['cutoff'],c,d['seed'])
                assert replay['geometry']==cert['geometry'];certificates+=1
            for r in d['references']:
                if r['mode']!=mode:continue
                c=next(c for c in p['candidates'] if c['id']==r['candidate']);v=S.E.vector(c,c['exact'],len(p['labels']))
                assert r['noise_covered']==all(float(v@z[b+'_error_gram']@v)<=m['cutoff']*float(v@cov[b]@v) for b in grams)
                assert r['zero_accepted']==all(float(v@grams[b]@v)<=m['cutoff']*float(v@cov[b]@v) for b in grams)
    return dict(status='PASS',record=path.stem,scores=n,certificates=certificates,result_sha256=sha(path))


def main():
    done=set();rows=[]
    while True:
        for path in sorted((ROOT/'panel_results').glob('*.json')):
            if path.stem in done:continue
            try:r=one(path)
            except Exception:r=dict(status='FAIL',record=path.stem,error=traceback.format_exc())
            dump(ROOT/'panel_audit_records'/path.name,r);rows.append(r);done.add(path.stem);print(path.stem,r['status'],flush=True)
        if len(done)==69:break
        if (ROOT/'panel_process.json').exists():break
        time.sleep(2)
    out=dict(status='PASS' if len(rows)==69 and all(r['status']=='PASS' for r in rows) else 'FAIL',records=len(rows),rows=rows,
             source_sha256=sources('audit_panel.py','panel_calibration.py','panel_protocol.txt'))
    dump(ROOT/'panel_audit.json',out);assert out['status']=='PASS'


if __name__=='__main__':main()
