"""Independent saved-array audit of oracle attribution and loss analysis."""
import metric as C
E,R,np,ROOT=C.E,C.R,C.np,C.ROOT
import json,csv
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('contrast_saved_array_audit',ROOT/'audit.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def qscore(delta,bias,var,cs,groups,axes):
    dd=delta-bias
    with np.errstate(divide='ignore',invalid='ignore'):ss=dd**2/(var*(1+1/256))
    ss=np.where((dd==0)&(var==0),0.,ss)
    return A.reduce(ss,cs,groups,axes)


def main():
    procs=json.loads((ROOT/'oracle_extension_processes.json').read_text());assert len(procs)==4 and all(p['exit_code']==0 for p in procs)
    totals=dict(oracle_score_values=0,oracle_coverage_rows=0,extension_score_values=0,field_replays=0,dominator_rows=0)
    summary=[];sources={str(Path(__file__)):E.S.sha(__file__)}
    for name in ('ac','kdv_cossin','ks','wave'):
        path=ROOT/'results'/f'{name}_seed21.json';meta=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'))
        f=json.loads((ROOT/'frozen'/path.name).read_text());cs=meta['candidates'];ref=next(j for j,c in enumerate(cs) if c['label']=='reference')
        op=ROOT/'oracle_diagnostic'/f'{name}.json';od=json.loads(op.read_text());o=np.load(op.with_suffix('.npz'));A.hashes(od)
        assert A.sha(str(op.with_suffix('.npz')))==od['array_sha256']
        for tag in ('clean_estimated_noise','clean_actual_noise'):
            bias=o[tag+'_calibration'].mean(axis=0)-a['clean_contrasts'];var=o[tag+'_calibration'].var(axis=0,ddof=1)
            np.testing.assert_array_equal(bias,o[tag+'_bias']);np.testing.assert_array_equal(var,o[tag+'_variance'])
            fields={'noise_q_check':o[tag+'_check']-a['clean_contrasts'],
                    'noise_q_actual':a['benchmark_contrasts']-a['clean_contrasts'],'point_q':a['observed_contrasts']}
            for key,delta in fields.items():
                q=qscore(delta,bias,var,cs,f['groups'],f['axis_groups']);np.testing.assert_allclose(q,o[tag+'_'+key],rtol=2e-12,atol=2e-12)
                totals['oracle_score_values']+=q.size
            for j,c in enumerate(cs):
                row=next(r for r in od['rows'] if r['candidate']==c['id'] and r['control']==tag)
                radius=np.quantile(o[tag+'_noise_q_check'][:,j],.95)
                assert radius==row['noise_q95'] and np.mean(o[tag+'_noise_q_actual'][:,j]<=radius)==row['noise_coverage']
                totals['oracle_coverage_rows']+=1
        ep=ROOT/'oracle_extension'/f'{name}.json';ed=json.loads(ep.read_text());x=np.load(ep.with_suffix('.npz'));A.hashes(ed)
        assert A.sha(str(ep.with_suffix('.npz')))==ed['array_sha256'];assert ed['actual_noise_seeds']==list(range(950000,951024))
        assert all(np.isfinite(x[key]).all() for key in x.files)
        gids=[i for i,g in enumerate(f['groups']) if g['candidate']==ref];ix=np.concatenate([np.arange(f['groups'][i]['start'],f['groups'][i]['stop']) for i in gids])
        groups=[];offset=0
        for i in gids:
            g=f['groups'][i];width=g['stop']-g['start'];groups.append(dict(g,start=offset,stop=offset+width,candidate=0));offset+=width
        axes=[[[gids.index(i) for i in axis] for axis in f['axis_groups'][ref]]]
        clean=a['clean_contrasts'][ix]
        for tag in ('pilot_estimated','clean_estimated_noise','clean_actual_noise'):
            bias=a['bias'][ix] if tag=='pilot_estimated' else o[tag+'_bias'][ix]
            var=a['variance'][ix] if tag=='pilot_estimated' else o[tag+'_variance'][ix]
            q=qscore(x['actual_contrasts']-clean,bias,var,[cs[ref]],groups,axes)[:,0]
            np.testing.assert_allclose(q,x[tag+'_actual_q'],rtol=2e-12,atol=2e-12);totals['extension_score_values']+=q.size
            row=next(r for r in ed['rows'] if r['control']==tag)
            radius=np.quantile(a['bootstrap_noise_q'][:,ref] if tag=='pilot_estimated' else o[tag+'_noise_q_check'][:,ref],.95)
            assert row['radius']==radius and row['coverage_count']==np.sum(q<=radius) and row['coverage']==np.mean(q<=radius)
            summary.append(dict(system=name,**row))
        q=qscore(x['check_contrasts']-clean,o['clean_actual_noise_bias'][ix],o['clean_actual_noise_variance'][ix],[cs[ref]],groups,axes)[:,0]
        np.testing.assert_allclose(q,x['clean_actual_noise_check_q'],rtol=2e-12,atol=2e-12);totals['extension_score_values']+=q.size
        row=next(r for r in ed['rows'] if r['radius_draws']==1024);radius=np.quantile(q,.95)
        np.testing.assert_allclose(radius,row['radius'],rtol=2e-12,atol=2e-12)
        assert row['coverage_count']==np.sum(x['clean_actual_noise_actual_q']<=row['radius'])
        summary.append(dict(system=name,**row))
        p=R.datasets.load(name);_,labels=R.panel(p);terms,opaque=R.symbols(p,labels);_,old=E.F.baseline(name,0)
        plan={k:tuple(v) for k,v in old['base_plan'].items()};C.K.common_width(max(len(c['xi']) for c in cs))
        bank=R.LocalBank(R.axes_of(p),terms,opaque,plan);ext=C.Extractor([cs[ref]],plan['shape'])
        rng=np.random.default_rng(np.random.SeedSequence([720132,21]))
        for b in range(1024):
            check={v:y+.05*float(np.std(y))*rng.standard_normal(y.shape) for v,y in p.data.items()}
            if b in (0,1023):
                for field,key in [(check,'check_contrasts'),(p.noisy(5,950000+b),'actual_contrasts')]:
                    np.testing.assert_array_equal(ext.extract(bank.matrix(field))[0],x[key][b]);totals['field_replays']+=1
    fail=json.loads((ROOT/'failure_summary.json').read_text());A.hashes(fail)
    dominators=list(csv.DictReader((ROOT/'loss_dominators.csv').open()));assert len(fail['cases'])==28
    pairratios=[]
    for case in fail['cases']:
        path=ROOT/'results'/f"{case['system']}_seed{case['seed']}.json";cs=json.loads(path.read_text())['candidates'];a=np.load(path.with_suffix('.npz'))
        ref=next(j for j,c in enumerate(cs) if c['id']==case['reference']);rows=[r for r in dominators if r['system']==case['system'] and int(r['seed'])==case['seed'] and int(r['slot'])==case['slot']]
        ix=[j for j,c in enumerate(cs) if c['slot']==case['slot']]
        d,q=a['observed_discrepancy'][0],a['observed_q_bias']
        ids=[cs[j]['id'] for j in ix if d[j]<=d[ref] and q[j]<=q[ref] and (d[j]<d[ref] or q[j]<q[ref])]
        assert ids==[r['dominator'] for r in rows] and len(ids)==case['dominators'];totals['dominator_rows']+=len(rows)
        for row in rows:
            if row['reference_superset']=='True':pairratios.append(float(row['shared_feature_median_variance_ratio']))
    assert len(pairratios)==fail['superset_dominator_pairs'] and np.median(pairratios)==fail['median_shared_feature_variance_ratio']
    with (ROOT/'oracle_extension_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)
    fig,ax=plt.subplots(figsize=(9,4.5));names=['ac','kdv_cossin','ks','wave'];xx=np.arange(4)
    labels={'pilot_estimated':'Reconstructed field, estimated noise (128-check radius)',
            'clean_estimated_noise':'Clean field, estimated noise (128-check radius)',
            'clean_actual_noise_extended_check':'Clean field, true noise (1024-check radius)'}
    for i,(tag,label) in enumerate(labels.items()):
        yy=[next(r['coverage'] for r in summary if r['system']==name and r['control']==tag) for name in names]
        ax.bar(xx+(i-1)*.24,yy,width=.24,label=label)
    ax.axhline(.95,color='grey',ls='--');ax.set_xticks(xx,['Allen-Cahn','Forced KdV','KS','Wave']);ax.set_ylim(0,1.22)
    ax.set_ylabel('Coverage on 1024 independent measurements');ax.set_title('Oracle attribution, anchor21; clean field is unavailable in practice')
    ax.legend(fontsize=8,loc='upper center',frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(ROOT/'oracle_attribution.png',dpi=180);fig.savefig(ROOT/'oracle_attribution.svg');plt.close(fig)
    out=dict(status='PASS',**totals,source_input_array_hashes=True,source_sha256=sources)
    R.dump(ROOT/'oracle_audit.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
