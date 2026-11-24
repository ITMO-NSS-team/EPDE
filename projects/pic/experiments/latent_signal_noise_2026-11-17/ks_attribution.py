from pathlib import Path
import sys,json,importlib.util,argparse
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('latent_failure_coverage',ROOT/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R,np=C.S,C.R,C.np
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(job):
    name,seed,profile=job;path=ROOT/'results'/f'{name}_{profile}_curvature_seed{seed}.json';d=json.loads(path.read_text())
    if d['status']!='COMPLETE':
        row=dict(system=name,seed=seed,profile=profile,status='unavailable_incomplete_noise_model',fit_info=d['fit_info'],input_sha256={str(path):S.E.sha(path)})
        R.dump(ROOT/f'{name}_{profile}_attribution_seed{seed}.json',dict(status='SKIPPED',reason=d['status'],rows=[row],source_sha256={str(Path(__file__)):S.E.sha(__file__)}))
        return [row]
    e=C.Study(name,seed,profile,'known_rho');e.model=d['model'];e.cov=C.O.Covariance(e.axes,np.eye(len(e.axes[0])),e.model);assert {v:R.digest(y) for v,y in e.raw.items()}==d['raw_sha256']
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    truth={v:[.05*float(np.std(y)),[e.rho]*y.ndim,[0]*y.ndim] for v,y in e.p.data.items()};truecov=C.O.Covariance(e.axes,np.eye(len(e.axes[0])),truth)
    def matrices(fields,cov):
        out={}
        for tag,bank in e.banks.items():
            m=C.O.matrix(bank,fields,cov);out[tag]=e.basis.convert(m) if tag=='broad' else m.real
        return out
    miss=matrices(e.p.data,e.cov);oracle=matrices(e.p.data,truecov);bias={b:miss[b]-oracle[b] for b in e.banks}
    sums={b:np.zeros_like(a[b+'_clean']) for b in e.banks};osums={b:np.zeros_like(a[b+'_clean']) for b in e.banks};gram={b:[] for b in e.banks};saved={};hh=[];rows=[]
    for i in range(256):
        fields=e.true_field(i)
        if i in (0,255):hh.append({v:R.digest(y) for v,y in fields.items()})
        model=e.matrices(fields);corrected=matrices(fields,truecov)
        for b in e.banks:
            err=model[b]-a[b+'_clean'];oe=corrected[b]-a[b+'_clean'];sums[b]+=err;osums[b]+=oe;gram[b].append(S.gram(oe))
            np.testing.assert_array_equal(S.gram(err),a[b+'_actual_error_grams'][i])
    assert hh==d['actual_field_hashes']
    for b in e.banks:
        w=a[b+'_white'];gg=np.stack(gram[b]);q=S.operator_scores(gg,w,e.active);cut=d['metrics'][b]['cutoff'];bb=S.operator_scores(S.gram(bias[b]),w,e.active)
        mean=sums[b]/256;omean=osums[b]/256;saved[b+'_oracle_scores']=q;saved[b+'_oracle_error_grams']=gg;saved[b+'_mean_model_error']=mean;saved[b+'_mean_oracle_error']=omean;saved[b+'_analytic_bias']=bias[b]
        rows.append(dict(system=name,seed=seed,profile=profile,bank=b,cutoff=cut,original_coverage_count=int(np.sum(a[b+'_actual_operator_scores']<=cut)),
            oracle_correction_coverage_count=int(np.sum(q<=cut)),analytic_bias_score=float(bb),analytic_bias_cutoff_ratio=float(bb/cut),
            empirical_mean_score=float(S.operator_scores(S.gram(mean),w,e.active)),oracle_mean_score=float(S.operator_scores(S.gram(omean),w,e.active)),
            mean_minus_bias_norm=float(np.linalg.norm(mean-bias[b])),oracle_mean_norm=float(np.linalg.norm(omean))))
    ap=ROOT/f'{name}_{profile}_attribution_seed{seed}.npz';np.savez_compressed(ap,**saved)
    sources={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'ks_attribution_protocol.txt',ROOT/'coverage.py',ROOT/'correlated_fit.py',ROOT/'latent_fit.py')}
    R.dump(ap.with_suffix('.json'),dict(status='COMPLETE',system=name,seed=seed,profile=profile,rows=rows,source_sha256=sources,
        input_sha256={str(p):S.E.sha(p) for p in (path,path.with_suffix('.npz'),ROOT/'frozen'/path.name)},array_sha256=S.E.sha(ap),actual_hashes=hh))
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);args=parser.parse_args();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one,('ks',36,p)) for p in ('iid','positive','negative')]):
            rr=f.result();rows.extend(rr);print(rr,flush=True)
    R.dump(ROOT/'ks_attribution_summary.json',dict(status='COMPLETE',workers=args.workers,rows=rows,source_sha256={str(Path(__file__)):S.E.sha(__file__)}))

if __name__=='__main__':main()
