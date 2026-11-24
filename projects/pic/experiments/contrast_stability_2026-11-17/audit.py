import metric as C
E,R,np,ROOT=C.E,C.R,C.np,C.ROOT
import json,csv
from pathlib import Path
from functools import lru_cache
from run_suite import SYSTEMS


@lru_cache(None)
def sha(path):return E.S.sha(path)


def hashes(d):
    for key in ('source_sha256','input_sha256'):
        for p,h in d.get(key,{}).items():assert sha(p)==h,(key,p)


def reduce(values,cs,groups,axes):
    out=[]
    for c,aa in zip(cs,axes):
        if not c['xi']:out.append(np.zeros(values.shape[:-1]));continue
        levels=[]
        for ax in aa:levels.append(np.mean([np.mean(values[...,groups[i]['start']:groups[i]['stop']],axis=-1) for i in ax],axis=0))
        out.append(np.mean(levels,axis=0))
    return np.stack(out,axis=-1)


def main():
    checks=json.loads((ROOT/'checks.json').read_text());assert checks['status']=='PASS';hashes(checks)
    runs=json.loads((ROOT/'process_results.json').read_text());expected={(n,s) for n in SYSTEMS for s in (21,22,23)}
    assert len(runs)==45 and {(r['system'],r['seed']) for r in runs}==expected and all(r['exit_code']==0 for r in runs)
    rows=list(csv.DictReader((ROOT/'candidates.csv').open()));frontrows=list(csv.DictReader((ROOT/'fronts.csv').open()))
    scorechecks=0;coveragechecks=0;frontchecks=0;replaymax=0.
    for name,seed in sorted(expected):
        path=ROOT/'results'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());hashes(d);assert not d['nonfinite']
        fpath=ROOT/'frozen'/path.name;f=json.loads(fpath.read_text());hashes(f)
        assert sha(str(fpath))==d['frozen_sha256'] and sha(str(path.with_suffix('.npz')))==d['array_sha256']
        a=np.load(path.with_suffix('.npz'));assert all(np.isfinite(a[k]).all() for k in a.files)
        p=R.datasets.load(name);raw=p.noisy(5,seed);cs,labels=R.panel(p)
        assert {v:R.digest(y) for v,y in raw.items()}==f['raw_data_sha256']
        assert [c['id'] for c in cs]==[c['id'] for c in d['candidates']]
        assert d['actual_noise_seeds']==list(range(910000+1000*seed,910128+1000*seed))
        for draws,domain,key in [(256,720130,'calibration_innovation_hashes'),(128,720131,'check_innovation_hashes')]:
            rng=np.random.default_rng(np.random.SeedSequence([domain,seed]));hh=[]
            for b in range(draws):
                z={v:R.WF.ar_noise(rng,y.shape,*f['noise_model'][v][:2]) for v,y in raw.items()}
                if b in (0,draws-1):hh.append({v:R.digest(y) for v,y in z.items()})
            assert hh==(f[key] if domain==720130 else d[key])
        var=a['calibration_contrasts'].var(axis=0,ddof=1);bias=a['calibration_contrasts'].mean(axis=0)-a['pilot_contrasts']
        np.testing.assert_array_equal(var,a['variance']);np.testing.assert_array_equal(bias,a['bias'])
        np.testing.assert_allclose(a['observed_discrepancy'][0],a['observed_discrepancy'][1],rtol=1e-8,atol=2e-8)
        for tag in ('observed','bootstrap','benchmark'):
            for corrected in (False,True):
                delta=a[tag+'_contrasts']-bias if corrected else a[tag+'_contrasts'];variance=var*(1+1/256) if corrected else var
                with np.errstate(divide='ignore',invalid='ignore'):squared=delta**2/variance
                squared=np.where((delta==0)&(variance==0),0.,squared)
                q=reduce(squared,cs,f['groups'],f['axis_groups'])
                np.testing.assert_allclose(q,a[tag+('_q_bias' if corrected else '_q_raw')],rtol=2e-12,atol=2e-12);scorechecks+=q.size
            if tag!='observed':
                center=a['pilot_contrasts'] if tag=='bootstrap' else a['clean_contrasts'];delta=a[tag+'_contrasts']-center-bias
                with np.errstate(divide='ignore',invalid='ignore'):squared=delta**2/(var*(1+1/256))
                squared=np.where((delta==0)&(var==0),0.,squared)
                np.testing.assert_allclose(reduce(squared,cs,f['groups'],f['axis_groups']),a[tag+'_noise_q'],rtol=2e-12,atol=2e-12)
        # Fresh direct production replay, independent from the saved extraction outputs.
        _,old=E.F.baseline(name,0);terms,opaque=R.symbols(p,labels);plan={k:tuple(v) for k,v in old['base_plan'].items()}
        K=C.K;K.common_width(max(len(c['xi']) for c in cs));matrix=R.LocalBank(R.axes_of(p),terms,opaque,plan).matrix(raw)
        production=[float(np.sum(R.SV.dyad_scores(matrix[:,c['xi']],matrix[:,c['yi']],np.ones(len(matrix)),plan['shape'],fit_intercept=False))) if c['xi'] else 0. for c in cs]
        np.testing.assert_allclose(production,a['observed_dyad'],rtol=2e-9,atol=2e-10);replaymax=max(replaymax,float(np.max(np.abs(production-a['observed_dyad']))))
        rr=[r for r in rows if r['system']==name and int(r['seed'])==seed];assert len(rr)==len(cs)
        for j,c in enumerate(cs):
            row=next(r for r in rr if r['candidate']==c['id']);q95=np.quantile(a['bootstrap_noise_q'][:,j],.95)
            assert int(row['contrast_noise_coverage_count'])==np.sum(a['benchmark_noise_q'][:,j]<=q95);coveragechecks+=1
        # Batch all competitors and draws: independently verify reported Pareto counts.
        complexity=np.array([len(c['xi']) for c in cs])
        for row in [r for r in frontrows if r['system']==name and int(r['seed'])==seed]:
            slot=int(row['slot']);ix=np.array([j for j,c in enumerate(cs) if c['slot']==slot]);tag=row['ensemble']
            if tag=='anchor':
                dd=a['observed_discrepancy'][0][None];ii=a['observed_dyad'][None];qb=a['observed_q_bias'][None];qr=a['observed_q_raw'][None]
            else:
                axis=0 if row['mode']=='fixed' else 1;dd=a[tag+'_discrepancy'][:,axis];ii=a[tag+'_dyad'];qb=a[tag+'_q_bias'];qr=a[tag+'_q_raw']
            pp=np.broadcast_to(complexity,dd.shape)
            lookup={'dyad':[dd,ii],'complexity':[dd,pp],'Q_bias':[dd,qb],'Q_raw':[dd,qr],
                    'Q_bias_sum':[dd,qb*pp],'Q_raw_sum':[dd,qr*pp],'D_only':[dd],
                    'Q_bias_complexity':[dd,qb,pp],'dyad_complexity':[dd,ii,pp]}
            vv=np.stack(lookup[row['arm']],axis=-1)[:,ix]
            dom=np.all(vv[:,:,None,:]<=vv[:,None,:,:],axis=-1)&np.any(vv[:,:,None,:]<vv[:,None,:,:],axis=-1)
            front=~dom.any(axis=1);registered=np.array([cs[j]['registered'] for j in ix])
            assert int(row['registered_count'])==np.any(front[:,registered],axis=1).sum()
            np.testing.assert_allclose(float(row['mean_size']),front.sum(axis=1).mean(),rtol=0,atol=1e-14)
            if tag=='anchor':assert json.loads(row['candidate_ids'])==[cs[j]['id'] for j in ix[front[0]]]
            frontchecks+=len(dd)
    out=dict(status='PASS',records=45,calibration_fields=11520,independent_evaluation_fields=11520,
        source_input_array_hashes=True,fresh_data_and_innovation_replay=True,production_dyad_max_absolute_error=replaymax,
        individual_score_checks=scorechecks,candidate_coverage_checks=coveragechecks,pareto_front_checks=frontchecks,nonfinite=0,
        source_sha256={str(p):sha(str(p)) for p in [Path(__file__),ROOT/'analyze.py',ROOT/'run_suite.py']})
    R.dump(ROOT/'audit.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
