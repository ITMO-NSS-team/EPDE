"""Completion audit: frozen fits, paired data, supports and score reproduction."""
import run_native as V
N,E,R,np,ROOT=V.N,V.E,V.R,V.np,V.ROOT
from pathlib import Path
import json,hashlib
from collections import Counter


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files=sorted((ROOT/'results').glob('*.json'))
    assert {p.stem for p in files}=={f'{n}_seed{s}' for n in ('ac','kdv_cossin','wave','ks') for s in (3,4,5)}
    expected_sources=V.sources();status=Counter();coefficients=0;iterations=0;weight_failures=[];support_count=0
    hashes={};reproduction=0.;registered=0;geometry_inputs={}
    for path in files:
        d=json.loads(path.read_text());assert d['source_sha256']==expected_sources
        assert all(sha(Path(p))==h for p,h in d['input_sha256'].items())
        p=R.datasets.load(d['system']);axes=R.axes_of(p);raw=p.noisy(5,d['seed'])
        assert d['raw_data_sha256']=={v:R.digest(y) for v,y in raw.items()}
        old_path=E.ROOT/'results'/path.name;prior=json.loads(old_path.read_text())
        assert d['raw_data_sha256']==prior['raw_data_sha256']
        assert d['candidates']==prior['candidates'];registered+=len(d['candidates'])
        _,geometry=V.F.baseline(d['system'],0)
        idx=[np.asarray(g['indices']) for g in geometry['groups']]
        group_axes=[[axes[0][i],*axes[1:]] for i in idx]
        _,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
        for key,desc in d['validation_specs'].items():
            spec=desc['spec'];monos=[t.monomials(psi_cap=spec['q']) for t in terms]
            cut=np.max([R.WF.leftover_orders(m,len(axes)) for m in monos],axis=0).astype(int)*R.WF.FD_REACH
            for ax in group_axes:
                for a,c,centers,radius in zip(ax,cut,spec['centers'],spec['radii']):
                    assert min(centers)-radius>=a[c]-1e-10*max(1.,abs(a[c]))
                    assert max(centers)+radius<=a[-1-c]+1e-10*max(1.,abs(a[-1-c]))
                    support_count+=len(centers)
        for g,fold in enumerate(d['folds']):
            assert fold['raw_group_sha256']==prior['groups'][g]['observed_sha256']
            ax=group_axes[g];geom=fold['geometry'];plan=geom['broad']
            assert plan['ms']==[(len(a)-2*c-1)//2 for a,c in zip(ax,plan['cut'])]
            for tag in ('local','broad'):
                native=N.spec_from_plan(ax,geom[tag],geom['k'] if tag=='broad' else 0)
                fine_axes=[axes[0][idx[g][0]:idx[g][-1]+1],*axes[1:]]
                fine=N.spec_from_plan(fine_axes,V.fine_plan(geom[tag]),geom['k'] if tag=='broad' else 0)
                for key in ('radii','frequencies','centers'):
                    for a,b in zip(native[key],fine[key]):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-11)
            for tag,w in fold['validation_weighting'].items():
                assert np.isfinite(w['eigenvalues']).all()
                if not w['feasible']:weight_failures.append(dict(system=d['system'],seed=d['seed'],training=g,bank=tag))
                else:assert w['rank']==w['rows'] and w['whitened_covariance_max_error']<2e-5
            for row in fold['rows']:
                if row['bank'].startswith(('native_','interpolated_')) and row['method'] in ('ols','irls'):
                    status[row['bank'],row['method'],row['fit']['status']]+=1
                    if row['method']=='irls':
                        for step in row['fit']['trace']:
                            iterations+=1
                            if 'prediction_change' in step:assert np.isfinite(step['prediction_change'])
                        if row['fit']['status']=='ok':
                            assert row['fit']['trace'][-1]['prediction_change']<=1e-8
                            np.testing.assert_array_equal(row['coefficients'],row['fit']['trace'][-1]['coefficients'])
                if row['coefficients'] is None:
                    assert row['fit']['status']!='ok' and 'frozen_clean_energy' not in row
                    continue
                assert np.isfinite(row['coefficients']).all();coefficients+=1
                for key,value in row.items():
                    if key in ('frozen_clean_energy','old_cross','old_clean') or key.endswith('_cross'):assert np.isfinite(value)
                if row['bank'].startswith('prior_'):
                    oldrow=next(r for r in prior['folds'][g]['rows'] if r['bank']=='_'.join(row['bank'].split('_')[1:]) and r['method']==row['method'] and r['candidate']==row['candidate'])
                    np.testing.assert_array_equal(row['coefficients'],oldrow['coefficients'])
                    np.testing.assert_allclose(row['frozen_clean_energy'],oldrow['frozen_clean_energy'],rtol=1e-12,atol=1e-14)
                    np.testing.assert_allclose(row['old_cross'],oldrow['cross'],rtol=2e-10,atol=2e-12)
        for a in d['aggregates']:
            rr=[r for f in d['folds'] for r in f['rows'] if all(r[k]==a[k] for k in ('bank','method','candidate'))]
            assert len(rr)==3
            good=[r for r in rr if r['fit']['status']=='ok' and r['coefficients'] is not None]
            assert a['successful_folds']==len(good)
            if len(good)<3:assert 'frozen_clean_energy' not in a
            else:
                for key,value in a.items():
                    if key in ('frozen_clean_energy','old_cross','old_clean') or key.endswith('_cross'):
                        np.testing.assert_allclose(value,np.mean([r[key] for r in good]),rtol=1e-14,atol=1e-15)
        reproduction=max(reproduction,d['baseline_score_reproduction_max_error']);hashes[path.name]=sha(path);geometry_inputs.update(d['input_sha256'])
    checks=json.loads((ROOT/'checks.json').read_text());iv=json.loads((ROOT/'invariant_checks.json').read_text())
    assert checks['constant_preserved'] and checks['unsafe_support_rejected'] and checks['shifted_grid_polynomial_agreement']
    assert iv['all_constants_preserved'] and len(iv['rows'])==3
    assert (ROOT/'findings.txt').exists()
    out=dict(system_runs=12,reference_seed_cases=12,registered_equation_seed_cases=registered,paired_reused_seeds=[3,4,5],
             source_and_input_hashes_verified=True,raw_data_replay_verified=True,registered_supports_unchanged=True,
             matched_physical_fitting_geometry_verified=True,common_validation_support_checks=support_count,
             finite_scored_fits=coefficients,checked_new_irls_iterations=iterations,
             new_fit_statuses=[dict(bank=b,method=m,status=s,count=n) for (b,m,s),n in sorted(status.items())],
             infeasible_validation_weights=weight_failures,prior_coefficients_and_scores_reproduced=True,
             maximum_baseline_score_error=reproduction,incomplete_averages_excluded=True,ode_simple_identity_controls=3,
             result_sha256=hashes,input_sha256=geometry_inputs,
             artifact_sha256={p.name:sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='completion_audit.json'})
    R.dump(ROOT/'completion_audit.json',out)
    print(json.dumps({k:v for k,v in out.items() if not k.endswith('_sha256')},indent=2))


if __name__=='__main__':main()
