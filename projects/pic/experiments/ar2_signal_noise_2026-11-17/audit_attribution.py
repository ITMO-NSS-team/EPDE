from pathlib import Path
import json,importlib.util
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('attribution_audit_coverage',ROOT/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R,np=C.S,C.R,C.np
from scipy.linalg import eigvalsh

def main():
    count=0;replays=0;skips=0;links=0
    for name in ('pde_compound',):
        for seed in (39,40,41):
            path=ROOT/f'{name}_attribution_seed{seed}.json';d=json.loads(path.read_text());parent=ROOT/'results'/f'{name}_iid_ar2_seed{seed}.json';base=json.loads(parent.read_text())
            if d['status']=='SKIPPED':assert base['status']=='incomplete_noise_model';assert d['rows'][0]['input_sha256'][str(parent)]==S.E.sha(parent);skips+=1;continue
            assert base['status']=='COMPLETE' and d['status']=='COMPLETE'
            for key in ('source_sha256','input_sha256'):
                for p,h in d[key].items():assert S.E.sha(p)==h;links+=1
            assert S.E.sha(path.with_suffix('.npz'))==d['array_sha256'];links+=1
            with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
            with np.load(parent.with_suffix('.npz')) as z:b={k:z[k] for k in z.files}
            assert all(np.isfinite(x).all() for x in a.values())
            e=C.Study(name,seed,'iid','iid');assert {v:R.digest(y) for v,y in e.raw.items()}==base['raw_sha256'];e.model=base['model'];e.cov=C.O.Covariance(e.axes,np.eye(len(e.axes[0])),e.model)
            truth={v:[.05*float(np.std(y)),[0.]*y.ndim,[0]*y.ndim] for v,y in e.p.data.items()};cov=C.O.Covariance(e.axes,np.eye(len(e.axes[0])),truth)
            for row in d['rows']:
                bank=row['bank'];sigma=b[bank+'_sigma'];w=b[bank+'_white'];active=e.active;sub=sigma[np.ix_(active,active)];scale=np.sqrt(np.diag(sub));sub=sub/scale[:,None]/scale[None,:]
                gg=a[bank+'_oracle_error_grams'];hh=gg[:,active,:][:,:,active]/scale[None,:,None]/scale[None,None,:]
                direct=np.array([eigvalsh((x+x.T)/2,sub)[-1] for x in hh]);np.testing.assert_allclose(direct,a[bank+'_oracle_scores'],rtol=3e-8,atol=3e-10);count+=len(direct)
                cut=base['metrics'][bank]['cutoff'];assert row['cutoff']==cut
                assert row['original_coverage_count']==np.sum(b[bank+'_actual_operator_scores']<=cut);assert row['oracle_correction_coverage_count']==np.sum(direct<=cut)
                for field,label in [('mean_model_error','empirical_mean_score'),('mean_oracle_error','oracle_mean_score'),('analytic_bias','analytic_bias_score')]:
                    value=S.operator_scores(S.gram(a[bank+'_'+field]),w,active);np.testing.assert_allclose(value,row[label],rtol=1e-12)
                assert row['analytic_bias_cutoff_ratio']==row['analytic_bias_score']/cut
            for j,k in enumerate((0,255)):
                fields=e.true_field(k);assert {v:R.digest(y) for v,y in fields.items()}==d['actual_hashes'][j]==base['actual_field_hashes'][j]
                for tag,bank in e.banks.items():
                    m=C.O.matrix(bank,fields,cov);m=e.basis.convert(m) if tag=='broad' else m.real
                    np.testing.assert_array_equal(S.gram(m-b[tag+'_clean']),a[tag+'_oracle_error_grams'][k]);replays+=1
    R.dump(ROOT/'audit_attribution.json',dict(status='PASS',records=3,skipped=skips,operator_scores=count,matrix_replays=replays,hash_links=links,
        source_sha256={str(Path(__file__)):S.E.sha(__file__)},input_sha256={str(ROOT/'compound_attribution_summary.json'):S.E.sha(ROOT/'compound_attribution_summary.json')}))
    print('PASS','scores',count,'matrix replays',replays,'skipped',skips)

if __name__=='__main__':main()
