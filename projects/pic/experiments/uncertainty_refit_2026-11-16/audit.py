"""Independent arithmetic, provenance and completion audit."""
import experiment as E
R,np,ROOT=E.R,E.np,E.ROOT
import json,csv,sys,platform
from pathlib import Path
from functools import lru_cache
from run_suite import SYSTEMS


@lru_cache(None)
def sha(path):return E.S.sha(path)


def hashcheck(d):
    for group in ('source_sha256','input_sha256'):
        if group in d:
            for p,h in d[group].items():assert sha(p)==h,(group,p)


def front(dd,ii,cs,slot):
    ix=np.array([j for j,c in enumerate(cs) if c['slot']==slot]);v=np.column_stack([dd,ii])[ix]
    domination=np.all(v[:,None]<=v[None,:],axis=-1)&np.any(v[:,None]<v[None,:],axis=-1)
    return ix[~np.any(domination,axis=0)]


def main():
    checks=json.loads((ROOT/'checks.json').read_text());assert checks['status']=='PASS';hashcheck(checks)
    for n in ('ac','kdv_cossin','ks','wave'):
        d=json.loads((ROOT/'replay'/f'{n}.json').read_text());assert d['status']=='PASS';hashcheck(d)
    runs=json.loads((ROOT/'process_results.json').read_text());expected={(n,s) for n in SYSTEMS for s in (18,19,20)}
    assert len(runs)==45 and {(r['system'],r['seed']) for r in runs}==expected and all(r['exit_code']==0 for r in runs)
    pp=list(csv.DictReader((ROOT/'pair_calibration.csv').open()));ff=list(csv.DictReader((ROOT/'fronts.csv').open()))
    pair_checks=0;front_checks=0;support_checks=0;energy_checks=0;nonfinite=0
    for name,seed in sorted(expected):
        path=ROOT/'results'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());hashcheck(d)
        freeze=ROOT/'frozen'/path.name;frozen=json.loads(freeze.read_text());hashcheck(frozen)
        assert sha(str(freeze))==d['frozen_sha256'];assert sha(str(path.with_suffix('.npz')))==d['array_sha256']
        a=np.load(path.with_suffix('.npz'));p=R.datasets.load(name);cs,labels=R.panel(p);raw=p.noisy(5,seed)
        assert [c['id'] for c in cs]==[c['id'] for c in d['candidates']]
        assert {v:R.digest(y) for v,y in raw.items()}==frozen['raw_data_sha256']
        assert d['actual_noise_seeds']==list(range(620000+1000*seed,620128+1000*seed))
        rng=np.random.default_rng(np.random.SeedSequence([460713,seed]));innov=[]
        for b in range(128):
            z={v:R.WF.ar_noise(rng,y.shape,*frozen['model'][v][:2]) for v,y in raw.items()}
            if b in (0,127):innov.append({v:R.digest(y) for v,y in z.items()})
        assert innov==d['innovation_hashes']
        for key in a.files:nonfinite+=int(np.sum(~np.isfinite(a[key])))
        np.testing.assert_allclose(a['observed_vectors'],a['anchor_vectors'],rtol=5e-8,atol=2e-8)
        np.testing.assert_allclose(a['observed_discrepancy'][0],a['observed_discrepancy'][1],rtol=1e-8,atol=2e-8)
        scores={}
        for tag in ('observed','standard_center','twice_center','clean_center','standard','twice','benchmark'):
            dd=a[tag+'_discrepancy'];scores[tag]=np.stack([dd[...,0,:],dd[...,1,:],a[tag+'_dyad']],axis=-2)
            vectors=a[tag+'_vectors']
            for j,c in enumerate(cs):
                support={c['yi'],*c['xi']};off=[k for k in range(len(labels)) if k not in support]
                assert np.all(vectors[...,off,j]==0);assert np.all(vectors[...,c['yi'],j]==1)
                support_checks+=int(np.prod(vectors.shape[:-2]))
        _,old=E.F.baseline(name,0);terms,opaque=R.symbols(p,labels);grams=E.F.evaluation_grams(p,old,terms,opaque,labels)
        for tag in ('standard','twice','benchmark'):
            assert a[tag+'_discrepancy'].shape==(128,2,len(cs));assert a[tag+'_dyad'].shape==(128,len(cs))
            values=[]
            for g in range(3):
                v=a[tag+'_vectors'][:,g];gg=grams[f'{g}_loo_oracle'];scale=grams[f'{g}_loo_clean_scale'][[c['yi'] for c in cs]]
                values.append(np.einsum('bij,ik,bkj->bj',v,gg,v,optimize=True)/scale)
            np.testing.assert_allclose(np.mean(values,axis=0),a[tag+'_energy'],rtol=2e-7,atol=2e-8);energy_checks+=128*len(cs)
        for row in [r for r in pp if r['system']==name and int(r['seed'])==seed]:
            j=next(j for j,c in enumerate(cs) if c['id']==row['candidate']);ref=next(j for j,c in enumerate(cs) if c['slot']==int(row['slot']) and c['label']=='reference')
            axis=('fixed_D','refit_D','dyad').index(row['axis']);tag=row['pilot']
            be=scores[tag][:,axis,j]-scores[tag][:,axis,ref]-(scores[tag+'_center'][axis,j]-scores[tag+'_center'][axis,ref])
            ae=scores['benchmark'][:,axis,j]-scores['benchmark'][:,axis,ref]-(scores['clean_center'][axis,j]-scores['clean_center'][axis,ref])
            q=np.quantile(be,[.025,.975]);assert int(row['coverage_count'])==np.sum((ae>=q[0])&(ae<=q[1]))
            np.testing.assert_allclose([float(row['error_lo']),float(row['error_hi'])],q);pair_checks+=1
        for row in [r for r in ff if r['system']==name and int(r['seed'])==seed]:
            axis=0 if row['mode']=='fixed' else 1;slot=int(row['slot'])
            for tag,vals in scores.items():
                vals=vals[None] if vals.ndim==2 else vals;count=0
                for b in vals:
                    ids=front(b[axis],b[2] if row['criterion']=='dyad' else [len(c['xi']) for c in cs],cs,slot)
                    count+=any(cs[j]['registered'] for j in ids);front_checks+=1
                assert count==int(row[tag+'_registered_count'])
    assert nonfinite==0
    summary=json.loads((ROOT/'summary.json').read_text());assert summary['measurements']==17280 and summary['equation_seed_cases']==60
    out=dict(status='PASS',records=45,measurement_records=17280,pair_axis_pilot_coverage_checks=pair_checks,
        pareto_front_checks=front_checks,coefficient_support_checks=support_checks,clean_energy_checks=energy_checks,nonfinite=nonfinite,
        source_input_array_hashes=True,fresh_anchor_data_replay=True,shared_innovation_replay=True,
        conditional_scope='Refitted main coefficients and GCV; frozen bank choices, correction maps, weights and geometry.',
        interpreter=sys.executable,python=sys.version,platform=platform.platform(),
        auxiliary_source_sha256={str(p):sha(str(p)) for p in [Path(__file__),ROOT/'analyze.py',ROOT/'checks.py',ROOT/'run_suite.py',
            Path(R.WF._central4.__code__.co_filename),Path(R.datasets.__file__)]})
    R.dump(ROOT/'audit.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
