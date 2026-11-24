import experiment as E
R,np,ROOT=E.R,E.np,E.ROOT
from pathlib import Path
from functools import lru_cache
import json
from run_suite import SYSTEMS

@lru_cache(None)
def sha(p):return E.sha(p)

def hashes(d):
    for key in ('source_sha256','input_sha256'):
        for p,h in d.get(key,{}).items():assert sha(p)==h,(key,p)

def direct_score(z,var):
    with np.errstate(divide='ignore',invalid='ignore'):s=z*z/var
    s=np.where((z==0)&(var==0),0.,s)
    return np.sum(s,axis=-1)/s.shape[-1]

def main():
    checks=json.loads((ROOT/'checks.json').read_text());assert checks['status']=='PASS';hashes(checks)
    runs=json.loads((ROOT/'process_results.json').read_text());expected={(n,s) for n in SYSTEMS for s in (24,25,26)}
    assert len(runs)==45 and {(r['system'],r['seed']) for r in runs}==expected and all(r['exit_code']==0 for r in runs)
    counts=dict(query_rows=0,scalar_scores=0,field_matrix_replays=0,fit_replays=0);maximum=0.
    for name,seed in sorted(expected):
        path=ROOT/'results'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());hashes(d)
        fp=ROOT/'frozen'/path.name;f=json.loads(fp.read_text());hashes(f)
        assert E.sha(fp)==d['frozen_sha256'] and E.sha(path.with_suffix('.npz'))==d['array_sha256']
        with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
        assert all(np.isfinite(v).all() for v in a.values())
        for k,h in f['calibration_hashes'].items():assert R.digest(a[k])==h
        e=E.Study(name,seed)
        assert {v:R.digest(y) for v,y in e.raw.items()}==f['raw_sha256']
        assert {v:R.digest(y) for v,y in e.pilot.items()}==f['pilot_sha256']
        assert d['actual_noise_seeds']==list(range(1200000+1000*seed,1200256+1000*seed))
        for arm,mat in e.matrices(e.raw).items():
            np.testing.assert_array_equal(mat,a[arm+'_observed']);counts['field_matrix_replays']+=1
        for bank,queries in d['queries'].items():
            for query in queries:
                c=next(c for c in e.cs if c['id']==query['candidate'])
                if query['kind']=='exact':np.testing.assert_array_equal(query['vector'],E.vector(c,c['exact'],len(e.labels)))
                else:
                    kind=query['kind'].split('_')[0];coef,info=E.N.E.ordinary(a[bank+'_'+kind+'_observed'],c)
                    assert info['status']==query['fit']['status']
                    if coef is not None:np.testing.assert_allclose(coef,query['coefficients'],rtol=2e-12,atol=2e-12)
                    counts['fit_replays']+=1
        for domain,label,key in [(830140,'calibration','calibration_innovation_hashes'),(830141,'check','check_innovation_hashes')]:
            rng=np.random.default_rng(np.random.SeedSequence([domain,seed]));hh=[]
            for b in range(256):
                noise={v:R.WF.ar_noise(rng,y.shape,*f['noise'][v][:2]) for v,y in e.raw.items()}
                if b in (0,255):
                    hh.append({v:R.digest(y) for v,y in noise.items()})
                    for arm,mat in e.matrices({v:e.pilot[v]+y for v,y in noise.items()}).items():
                        np.testing.assert_array_equal(mat,a[arm+'_'+label][b]);counts['field_matrix_replays']+=1
            assert hh==(f[key] if label=='calibration' else d[key])
        for b in (0,255):
            for arm,mat in e.matrices(e.p.noisy(5,1200000+1000*seed+b)).items():
                np.testing.assert_array_equal(mat,a[arm+'_actual'][b]);counts['field_matrix_replays']+=1
        for row in d['rows']:
            arm=row['arm'];bank=arm.split('_')[0];q=next(q for q in d['queries'][bank] if (q['candidate'],q['kind'])==(row['candidate'],row['kind']))
            if q['vector'] is None:assert row['status']=='infeasible_fit';continue
            v=np.asarray(q['vector']);var=np.var(a[arm+'_calibration']@v,axis=0,ddof=1)
            boot=np.einsum('brj,j->br',a[arm+'_check']-a[bank+'_pilot'],v)
            actual=np.einsum('brj,j->br',a[arm+'_actual']-a[bank+'_clean'],v)
            qb=direct_score(boot,var);qa=direct_score(actual,var);radius=np.quantile(qb,.95)
            np.testing.assert_allclose(radius,row['radius'],rtol=2e-11,atol=2e-12)
            assert row['noise_coverage_count']==np.sum(qa<=row['radius'])
            null=direct_score(a[arm+'_actual']@v,var);assert row['zero_acceptance_count']==np.sum(null<=row['radius'])
            obs=direct_score(a[arm+'_observed']@v,var);np.testing.assert_allclose(obs,row['anchor_score'],rtol=2e-12,atol=2e-12)
            assert row['anchor_acceptance']==bool(obs<=row['radius'])
            clean=direct_score(a[bank+'_clean']@v,var);np.testing.assert_allclose(clean,row['clean_score'],rtol=2e-12,atol=2e-12)
            counts['query_rows']+=1;counts['scalar_scores']+=len(qb)+len(qa)+len(null)+2
        print(name,seed,'AUDIT PASS',flush=True)
    out=dict(status='PASS',records=45,**counts,source_input_array_hashes=True,fresh_data_and_innovation_replay=True,
        source_sha256={str(Path(__file__)):E.sha(__file__),str(ROOT/'analyze.py'):E.sha(ROOT/'analyze.py')})
    R.dump(ROOT/'audit.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
