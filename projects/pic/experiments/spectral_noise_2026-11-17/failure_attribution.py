from pathlib import Path
import sys,json,importlib.util
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('spectral_failure_coverage',ROOT/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R,np=C.S,C.R,C.np

def main():
    rows=[];sources={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'failure_attribution_protocol.txt',ROOT/'coverage.py',ROOT/'noise_fit.py',ROOT/'stable_fit.py')}
    for seed in (33,34,35):
        path=ROOT/'results'/f'pde_compound_iid_spectral_seed{seed}.json';d=json.loads(path.read_text());f=json.loads((ROOT/'frozen'/path.name).read_text());e=C.Study('pde_compound',seed,'iid','spectral')
        with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
        truth={v:[.05*float(np.std(y)),[0.]*y.ndim,[0]*y.ndim] for v,y in e.p.data.items()};truecov=C.O.Covariance(e.axes,np.eye(len(e.axes[0])),truth)
        def matrices(fields,cov):
            out={}
            for tag,bank in e.banks.items():
                m=C.O.matrix(bank,fields,cov);out[tag]=e.basis.convert(m) if tag=='broad' else m.real
            return out
        miss=matrices(e.p.data,e.cov);oracle=matrices(e.p.data,truecov);bias={b:miss[b]-oracle[b] for b in e.banks}
        pure={v:C.F.estimate(e.raw[v]-e.p.data[v],'strips') for v in e.raw}
        sums={b:np.zeros_like(a[b+'_clean']) for b in e.banks};osums={b:np.zeros_like(a[b+'_clean']) for b in e.banks};gram={b:[] for b in e.banks};saved={};hh=[]
        for i in range(256):
            fields=e.true_field(i)
            if i in (0,255):hh.append({v:R.digest(y) for v,y in fields.items()})
            model=e.matrices(fields);corrected=matrices(fields,truecov)
            for b in e.banks:
                err=model[b]-a[b+'_clean'];oe=corrected[b]-a[b+'_clean'];sums[b]+=err;osums[b]+=oe;gram[b].append(S.gram(oe))
                np.testing.assert_array_equal(S.gram(err),a[b+'_actual_error_grams'][i])
        assert hh==d['actual_field_hashes']
        for b in e.banks:
            w=a[b+'_white'];q=S.operator_scores(np.stack(gram[b]),w,e.active);cut=d['metrics'][b]['cutoff'];bb=S.operator_scores(S.gram(bias[b]),w,e.active)
            mean=sums[b]/256;omean=osums[b]/256;saved[b+'_oracle_scores']=q;saved[b+'_mean_model_error']=mean;saved[b+'_mean_oracle_error']=omean;saved[b+'_analytic_bias']=bias[b]
            rows.append(dict(system=e.name,seed=seed,bank=b,raw_model=e.model,noise_only_model={v:x[0] for v,x in pure.items()},cutoff=cut,original_coverage_count=int(np.sum(a[b+'_actual_operator_scores']<=cut)),
                oracle_correction_coverage_count=int(np.sum(q<=cut)),analytic_bias_score=float(bb),analytic_bias_cutoff_ratio=float(bb/cut),
                empirical_mean_score=float(S.operator_scores(S.gram(mean),w,e.active)),oracle_mean_score=float(S.operator_scores(S.gram(omean),w,e.active)),
                mean_minus_bias_norm=float(np.linalg.norm(mean-bias[b])),oracle_mean_norm=float(np.linalg.norm(omean))))
        ap=ROOT/f'compound_attribution_seed{seed}.npz';np.savez_compressed(ap,**saved)
        R.dump(ap.with_suffix('.json'),dict(status='COMPLETE',seed=seed,rows=rows[-2:],source_sha256=sources,
            input_sha256={str(p):S.E.sha(p) for p in (path,path.with_suffix('.npz'),ROOT/'frozen'/path.name)},array_sha256=S.E.sha(ap),actual_hashes=hh))
        print(seed,rows[-2:],flush=True)
    R.dump(ROOT/'failure_attribution_summary.json',dict(status='COMPLETE',rows=rows,source_sha256=sources))

if __name__=='__main__':main()
