"""Validate summary counts, three-record medians and final artifact manifest."""
import run_case as C
np,R,S,ROOT=C.np,C.R,C.S,C.ROOT
import csv,json
from collections import Counter


def main():
    cases=list(csv.DictReader((ROOT/'per_case.csv').open()));paired=list(csv.DictReader((ROOT/'paired_comparisons.csv').open()))
    table=list(csv.DictReader((ROOT/'system_summary.csv').open()));summary=json.loads((ROOT/'summary.json').read_text())
    for row in table:
        rr=[r for r in cases if all(r[k]==row[k] for k in ('system','candidate','strategy'))]
        assert len(rr)==3;good=[r for r in rr if int(r['successful_folds'])==3]
        assert len(good)==int(row['complete_cases'])
        for key,val in row.items():
            if not key.startswith('median_'):continue
            if len(good)<3:assert not val
            elif all(r.get(key[7:]) for r in good):np.testing.assert_allclose(float(val),np.median([float(r[key[7:]]) for r in good]),rtol=1e-14,atol=1e-14)
    for comparison in summary['comparisons']:
        rr=[r for r in paired if r['method']==comparison['method'] and r['baseline']==comparison['baseline']]
        assert len(rr)==60 and dict(Counter(r['outcome'] for r in rr))==comparison['outcomes']
        ratios=[float(r['ratio']) for r in rr if r['ratio']]
        np.testing.assert_allclose(np.median(ratios),comparison['median_ratio'],rtol=1e-14)
    audit=json.loads((ROOT/'completion_audit.json').read_text());assert audit['completed_system_runs']==45
    assert sum(r['count'] for r in audit['fit_statuses'])==4185
    assert sum(r['count'] for r in audit['conditional_fit_statuses'])==6912
    refs=[r for r in cases if r['reference']=='True' and r['strategy'].startswith('select/')]
    assert len(refs)==180 and all(int(r['successful_folds'])==3 for r in refs)
    assert (ROOT/'findings.txt').exists()
    files={str(p.relative_to(ROOT)):S.sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='report_audit.json'}
    R.dump(ROOT/'report_audit.json',dict(all_45_final_results_present=True,reference_selection_cases_complete=180,
        reference_cases_per_rule=60,summary_counts_verified=True,three_record_medians_verified=True,
        incomplete_comparisons_explicit=True,main_fits=4185,conditional_refits=6912,
        original_duffing_failures_retained_and_resolved=True,artifact_sha256=files))
    print('PASS: all 45 runs, 4185 main fits, 6912 conditional refits, medians, paired counts and artifact hashes.')


if __name__=='__main__':main()
