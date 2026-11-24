"""Check reported aggregation and preserve a final artifact manifest."""
import run_outer as S
np,R,ROOT=S.np,S.R,S.ROOT
import json,csv


def main():
    folds=list(csv.DictReader((ROOT/'per_fold.csv').open()))
    cases=list(csv.DictReader((ROOT/'per_case.csv').open()))
    summary=json.loads((ROOT/'summary.json').read_text());metrics=['frozen_clean_energy']+[f'{stage}_{rule}' for stage in ('inner','outer','clean') for rule in ('native_broad','old_broad')]
    for r in cases:
        rr=[f for f in folds if all(f[k]==r[k] for k in ('system','seed','candidate','strategy'))]
        assert len(rr)==3
        good=[x for x in rr if x['status']=='ok'];assert int(r['successful_folds'])==len(good)
        if len(good)<3:
            assert all(not r[m] for m in metrics)
        else:
            for m in metrics:np.testing.assert_allclose(float(r[m]),np.mean([float(x[m]) for x in good]),rtol=1e-14,atol=1e-15)
    for r in summary['accuracy']:
        rr=[c for c in cases if c['system']==r['system'] and c['strategy']==r['strategy'] and c['candidate']=='e0:reference']
        good=[c for c in rr if int(c['successful_folds'])==3]
        assert r['complete_cases']==len(good)
        if len(good)==3:
            for m in metrics:np.testing.assert_allclose(r['median_'+m],np.median([float(c[m]) for c in good]),rtol=1e-14,atol=1e-15)
        else:assert 'median_frozen_clean_energy' not in r
    diag=json.loads((ROOT/'eiv_normal_score_diagnostic.json').read_text())
    assert diag['source_sha256']==S.sha(ROOT/'diagnose.py')
    assert all(S.sha(p)==h for p,h in diag['input_sha256'].items())
    assert len(diag['rows'])==26
    audit=json.loads((ROOT/'completion_audit.json').read_text())
    assert audit['system_runs']==12 and audit['independent_bootstrap_fits']==3456
    assert (ROOT/'findings.txt').exists()
    artifacts={str(p.relative_to(ROOT)):S.sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='report_audit.json'}
    R.dump(ROOT/'report_audit.json',dict(fold_rows=len(folds),case_rows=len(cases),
        complete_case_aggregation_verified=True,incomplete_case_averages_excluded=True,
        three_record_medians_verified=True,posthoc_diagnostic_hashes_verified=True,artifact_sha256=artifacts))
    print('PASS: record-level aggregation, incomplete cases, medians and final artifact hashes verified.')


if __name__=='__main__':main()
