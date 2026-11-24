from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parent
import noise_fit as Initial
import stable_fit as F
np=F.np
spec=importlib.util.spec_from_file_location('followup_coverage',ROOT/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R=C.S,C.R
spec=importlib.util.spec_from_file_location('identification_audit',ROOT/'identification.py');I=importlib.util.module_from_spec(spec);spec.loader.exec_module(I)

def hashes(d):
    for key in ('source_sha256','input_sha256'):
        for p,h in d.get(key,{}).items():assert S.E.sha(p)==h,(p,key)

def model_close(a,b):
    if a is None or b is None:assert a is None and b is None;return
    np.testing.assert_allclose(a[0],b[0],rtol=2e-9,atol=2e-12);np.testing.assert_allclose(a[1],b[1],rtol=2e-8,atol=2e-9)

def main():
    exploratory=0;fields=0;fit_replays=0;oracle_scores=0
    files=sorted((ROOT/'explore').glob('*.json'));assert len(files)==69
    for path in files:
        d=json.loads(path.read_text());hashes(d);fp=ROOT/'explore_frozen'/path.name;f=json.loads(fp.read_text());hashes(f);assert S.E.sha(fp)==d['frozen_sha256']
        first=d['rows'][0];e=S.Study(first['system'],first['seed'],first['profile'],'ar');assert {v:R.digest(y) for v,y in e.raw.items()}==f['raw_hashes']
        for row in d['rows']:
            y=e.raw[row['variable']] if row['mode']=='raw' else e.raw[row['variable']]-e.p.data[row['variable']]
            model,info=Initial.estimate(y,row['design']);model_close(model,row['model']);assert info['status']==row['status'];fit_replays+=1
            if model is not None:np.testing.assert_allclose(model[0]/(.05*np.std(e.p.data[row['variable']])),row['sigma_ratio'],rtol=2e-9)
        exploratory+=1
    print('Exploratory replay PASS',exploratory,fit_replays,flush=True)
    ident=json.loads((ROOT/'identification.json').read_text());hashes(ident);assert len(ident['rows'])==576
    for row in ident['rows']:
        pid={0.:0,.4:1,-.4:2}[row['true_rho']];rng=np.random.default_rng(np.random.SeedSequence([950106,row['n'],pid,row['draw']]))
        y=R.WF.ar_noise(rng,(row['n'],),.2,[row['true_rho']]);assert R.digest(y)==row['field_sha256'];fields+=1
        sig,rho=I.oracle(y);np.testing.assert_array_equal([sig,rho],[row['oracle_sigma'],row['oracle_rho']])
        model,info=F.estimate(y,'strips');assert info['status']==row['spectral_status'];model_close(model,[row['spectral_sigma'],[row['spectral_rho']],[0]]);fit_replays+=1
    print('576 pure-noise fits PASS',flush=True)
    for seed in (33,34,35):
        ap=ROOT/f'compound_attribution_seed{seed}.npz';d=json.loads(ap.with_suffix('.json').read_text());hashes(d);assert S.E.sha(ap)==d['array_sha256']
        path=ROOT/'results'/f'pde_compound_iid_spectral_seed{seed}.json';p=json.loads(path.read_text());e=C.Study('pde_compound',seed,'iid','spectral')
        with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
        with np.load(ap) as z:extra={k:z[k] for k in z.files}
        true={v:[.05*float(np.std(y)),[0.]*y.ndim,[0]*y.ndim] for v,y in e.p.data.items()};cov=C.O.Covariance(e.axes,np.eye(len(e.axes[0])),true)
        def matrices(field,which):
            out={}
            for b,bank in e.banks.items():
                m=C.O.matrix(bank,field,which);out[b]=e.basis.convert(m) if b=='broad' else m.real
            return out
        clean0=matrices(e.p.data,e.cov);clean1=matrices(e.p.data,cov);sums={b:np.zeros_like(a[b+'_clean']) for b in e.banks};g={b:[] for b in e.banks}
        for i in range(256):
            fields0=e.true_field(i);fields+=1
            for b,m in matrices(fields0,cov).items():error=m-a[b+'_clean'];g[b].append(S.gram(error));sums[b]+=error
        for row in d['rows']:
            b=row['bank'];cut=p['metrics'][b]['cutoff'];w=a[b+'_white'];q=S.operator_scores(np.stack(g[b]),w,e.active)
            np.testing.assert_allclose(q,extra[b+'_oracle_scores'],rtol=2e-12,atol=2e-12);np.testing.assert_array_equal(sums[b]/256,extra[b+'_mean_oracle_error'])
            np.testing.assert_array_equal(clean0[b]-clean1[b],extra[b+'_analytic_bias']);bias=S.operator_scores(S.gram(extra[b+'_analytic_bias']),w,e.active)
            np.testing.assert_allclose(bias,row['analytic_bias_score'],rtol=2e-12);assert row['oracle_correction_coverage_count']==np.sum(q<=cut)
            assert row['original_coverage_count']==np.sum(a[b+'_actual_operator_scores']<=cut);oracle_scores+=256
        print('Compound attribution',seed,'PASS',flush=True)
    out=dict(status='PASS',exploratory_records=exploratory,noise_fit_replays=fit_replays,fresh_field_replays=fields,oracle_operator_scores=oracle_scores,
        source_sha256={str(Path(__file__)):S.E.sha(__file__)})
    R.dump(ROOT/'audit_followups.json',out);print(out,flush=True)

if __name__=='__main__':main()
