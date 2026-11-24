"""Check all runs, saved transforms, immutable selection, and aggregation."""
import run_case as C
S,A,N,R,np,ROOT=C.S,C.A,C.N,C.R,C.np,C.ROOT
import duffing_coordinates as D
import json,csv
from collections import Counter


def main():
    files=sorted((ROOT/'results').glob('*.json'))
    assert {p.stem for p in files}=={f'{n}_seed{s}' for n in C.SYSTEMS for s in (12,13,14)}
    statuses=Counter();decisions=0;forced=0;registered=0;references=0;hashes={};support_checks=0
    for path in files:
        d=json.loads(path.read_text());before=json.loads((ROOT/'frozen'/path.name).read_text())
        assert d['selection_freeze_sha256']==S.sha(ROOT/'frozen'/path.name)
        assert all(S.sha(p)==h for p,h in d['source_sha256'].items())
        assert all(S.sha(p)==h for p,h in d['input_sha256'].items())
        expected=C.sources()
        if d['system']=='duffing':expected[str(C.Path(D.__file__))]=S.sha(D.__file__)
        assert d['source_sha256']==expected
        assert d['seed'] in (12,13,14) and d['outer_seed']==d['seed']+3
        p=R.datasets.load(d['system']);axes=R.axes_of(p);panel,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
        registered+=len(d['candidates']);references+=sum(c['label']=='reference' for c in d['candidates'])
        assert [c['id'] for c in d['candidates']]==[c['id'] for c in panel if c['registered']]
        for seed,key in [(d['seed'],'raw_data_sha256'),(d['outer_seed'],'outer_data_sha256')]:
            assert d[key]=={v:R.digest(y) for v,y in p.noisy(5,seed).items()}
        _,old=C.F.baseline(d['system'],0);indices=[np.asarray(g['indices']) for g in old['groups']]
        axlist=[[axes[0][idx],*axes[1:]] for idx in indices]
        assert all(not np.intersect1d(a,b).size for i,a in enumerate(indices) for b in indices[:i])
        for desc in d['validation_specs'].values():
            spec=desc['spec'];mono=[t.monomials(psi_cap=spec['q']) for t in terms]
            cut=np.max([R.WF.leftover_orders(m,len(axes)) for m in mono],axis=0).astype(int)*R.WF.FD_REACH
            for ax in axlist:
                for a,c,centers,radius in zip(ax,cut,spec['centers'],spec['radii']):
                    assert min(centers)-radius>=a[c]-1e-10*max(1.,abs(a[c]))
                    assert max(centers)+radius<=a[-1-c]+1e-10*max(1.,abs(a[-1-c]));support_checks+=len(centers)
        for g,fold in enumerate(d['folds']):
            frozen=before['folds'][g];assert fold['metrics']==frozen['metrics']
            assert fold['fit_weighting']==frozen['fit_weighting'] and fold['validation_weighting']==frozen['validation_weighting']
            for row,original in zip(fold['rows'],frozen['rows']):
                assert not any(k.startswith(('outer_','clean_','frozen_clean')) for k in original)
                for k,v in original.items():assert row[k]==v
                statuses[row['method'],row['fit']['status']]+=1
                if row['fit']['status']=='ok':
                    assert row['coefficients'] is not None and np.isfinite(row['coefficients']).all()
                    for k,v in row.items():
                        if k.startswith(('inner_','outer_','clean_','frozen_clean')):assert np.isfinite(v)
                else:assert row['coefficients'] is None and 'frozen_clean_energy' not in row
            for decision,original in zip(fold['decisions'],frozen['decisions']):
                assert 'evaluation' not in original
                for k,v in A.choose(original['options']).items():assert original[k]==v
                for k,v in original.items():assert decision[k]==v
                decisions+=1;forced+=decision['forced']
                if decision['status']=='ok':
                    row=next(r for r in fold['rows'] if r['candidate']==decision['candidate'] and r['option']==decision['option'])
                    for k,v in decision['evaluation'].items():assert row[k]==v
            if d['system']=='duffing':
                ax=axlist[g]
                for tag in ('local','broad'):
                    spec=D.spec_from_plan(ax,fold['geometry'][tag],fold['geometry']['k'] if tag=='broad' else 0)
                    assert min(spec['centers'][0])-spec['radii'][0]>=ax[0][0]-1e-12
                    assert max(spec['centers'][0])+spec['radii'][0]<=ax[0][-1]+1e-12
        hashes[path.name]=S.sha(path)
    foldrows=list(csv.DictReader((ROOT/'per_fold.csv').open()));cases=list(csv.DictReader((ROOT/'per_case.csv').open()))
    metric_names=['frozen_clean_energy']+[f'{stage}_{rule}' for stage in ('inner','outer','clean') for rule in C.RULES]
    for case in cases:
        rr=[r for r in foldrows if all(r[k]==case[k] for k in ('system','seed','candidate','strategy'))]
        good=[r for r in rr if r['status']=='ok'];assert len(rr)==3 and len(good)==int(case['successful_folds'])
        for metric in metric_names:
            if len(good)<3:assert not case[metric]
            elif all(r[metric] for r in good):np.testing.assert_allclose(float(case[metric]),np.mean([float(r[metric]) for r in good]),rtol=1e-14,atol=1e-14)
    conditional=json.loads((ROOT/'conditional_results.json').read_text());conditional_status=Counter()
    assert all(S.sha(p)==h for p,h in conditional['input_sha256'].items())
    for row in conditional['rows']:
        assert len(row['samples'])==64
        for sample in row['samples']:conditional_status[row['method'],sample['status']]+=1
    checks=json.loads((ROOT/'extra_checks.json').read_text());assert checks['ode_simple_constants_preserved']
    failures=[r for r in json.loads((ROOT/'process_results.json').read_text()) if r['exit_code']!=0]
    assert {(r['system'],r['seed']) for r in failures}=={('duffing',s) for s in (12,13,14)}
    out=dict(completed_system_runs=45,reference_seed_cases=references,registered_equation_seed_cases=registered,
        source_and_input_hashes_verified=True,fresh_noise_replay_verified=True,frozen_selection_replayed=True,
        actual_score_transforms_saved_and_unchanged=True,finite_scored_fits_verified=True,registered_supports_preserved=True,
        selection_records=decisions,forced_selections=forced,validation_support_checks=support_checks,
        no_incomplete_case_averages=True,duffing_coordinate_repair_verified=True,initial_failed_attempts=failures,
        final_missing_results=0,fit_statuses=[dict(method=m,status=s,count=n) for (m,s),n in sorted(statuses.items())],
        conditional_fit_statuses=[dict(method=m,status=s,count=n) for (m,s),n in sorted(conditional_status.items())],
        ode_simple_identity_controls=3,result_sha256=hashes)
    R.dump(ROOT/'completion_audit.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='result_sha256'},indent=2))


if __name__=='__main__':main()
