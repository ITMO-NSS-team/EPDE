import experiment as E
R,np,ROOT=E.R,E.np,E.ROOT
from pathlib import Path
import json,importlib.util
import iid_studentization as I
spec=importlib.util.spec_from_file_location('direct_moment_main_audit',ROOT/'audit.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
from run_suite import SYSTEMS

def contrast_reduce(values,cs,groups,axes):
    return np.stack([np.mean([np.mean([np.mean(values[...,groups[g]['start']:groups[g]['stop']],axis=-1) for g in axis],axis=0) for axis in aa],axis=0) for aa in axes],axis=-1)

def main():
    ck=json.loads((ROOT/'iid_checks.json').read_text());assert ck['status']=='PASS';A.hashes(ck)
    for filename,seeds in [('iid_processes.json',(24,25,26)),('iid_replication_processes.json',(27,28,29))]:
        runs=json.loads((ROOT/filename).read_text());assert len(runs)==45 and all(r['exit_code']==0 for r in runs)
        assert {(r['system'],r['seed']) for r in runs}=={(n,s) for n in SYSTEMS for s in seeds}
    counts=dict(iid_records=0,iid_query_rows=0,iid_scalar_scores=0,iid_matrix_replays=0,paired_records=0,paired_scalar_scores=0)
    for seed in (24,25,26,27,28,29):
        for name in SYSTEMS:
            path=ROOT/'iid'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());A.hashes(d)
            fp=path.with_name(path.stem+'_frozen.json');f=json.loads(fp.read_text());A.hashes(f)
            assert E.sha(fp)==d['frozen_sha256'] and E.sha(path.with_suffix('.npz'))==d['array_sha256'];assert d['correction_replay_max_error']==0
            with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
            assert all(np.isfinite(x).all() for x in a.values())
            for k,h in f['calibration_hashes'].items():assert R.digest(a[k])==h
            e=E.Study(name,seed);delta=I.amplitude(e.raw,f['scales'],f['orders']);assert delta==f['amplitude']==d['amplitude']
            assert {v:R.digest(y) for v,y in e.raw.items()}==f['raw_sha256']
            assert {v:R.digest(y) for v,y in e.pilot.items()}==f['pilot_sha256']
            def replay(fields,amp,tag,b=None):
                for kind,sig in [('fixed',delta),('studentized',amp)]:
                    cov=E.O.Covariance(e.axes,np.eye(len(e.axes[0])),{v:[sig*f['scales'][v],[0.]*y.ndim,[]] for v,y in fields.items()})
                    for bank,operator in e.banks.items():
                        mat=E.O.matrix(operator,fields,cov);mat=e.basis.convert(mat) if bank=='broad' else mat.real
                        expected=a[bank+'_'+kind+'_'+tag];expected=expected[b] if b is not None else expected
                        np.testing.assert_array_equal(mat,expected);counts['iid_matrix_replays']+=1
            replay(e.raw,delta,'observed')
            for domain,tag,key in [(830140,'calibration','calibration_innovation_hashes'),(830141,'check','check_innovation_hashes')]:
                rng=np.random.default_rng(np.random.SeedSequence([domain,seed]));hh=[]
                for b in range(256):
                    noise={v:delta*f['scales'][v]*rng.standard_normal(y.shape) for v,y in e.raw.items()}
                    if b in (0,255):
                        hh.append({v:R.digest(y) for v,y in noise.items()});field={v:e.pilot[v]+y for v,y in noise.items()};amp=I.amplitude(field,f['scales'],f['orders'])
                        assert amp==a[tag+'_amplitude'][b];replay(field,amp,tag,b)
                assert hh==(f[key] if tag=='calibration' else d[key])
            for b in (0,255):
                field=e.p.noisy(5,1200000+1000*seed+b);amp=I.amplitude(field,f['scales'],f['orders']);assert amp==a['actual_amplitude'][b];replay(field,amp,'actual',b)
            for row in d['rows']:
                assert row['status']=='ok';arm=row['arm'];bank,kind=arm.split('_',1);c=next(c for c in e.cs if c['id']==row['candidate']);v=E.vector(c,c['exact'],len(e.labels))
                def residual(tag,center):
                    z=np.einsum('brj,j->br',a[arm+'_'+tag]-a[bank+'_'+center],v)
                    return z/a[tag+'_amplitude'][:,None] if kind=='studentized' else z
                cal=residual('calibration','pilot');var=np.sum((cal-cal.mean(axis=0))**2,axis=0)/(len(cal)-1)
                qb=A.direct_score(residual('check','pilot'),var);qa=A.direct_score(residual('actual','clean'),var)
                radius=np.quantile(qb,.95);np.testing.assert_allclose(radius,row['radius'],rtol=3e-11,atol=3e-12)
                assert np.sum(qa<=row['radius'])==row['noise_coverage_count']
                null=a[arm+'_actual']@v;obs=a[arm+'_observed']@v
                if kind=='studentized':null=null/a['actual_amplitude'][:,None];obs=obs/delta
                assert np.sum(A.direct_score(null,var)<=row['radius'])==row['zero_acceptance_count']
                np.testing.assert_allclose(A.direct_score(obs,var),row['point_score'],rtol=3e-11,atol=3e-12)
                counts['iid_query_rows']+=1;counts['iid_scalar_scores']+=769
            counts['iid_records']+=1
        print('iid seed',seed,'AUDIT PASS',flush=True)
    runs=json.loads((ROOT/'paired_processes.json').read_text());assert len(runs)==12 and all(r['exit_code']==0 for r in runs)
    for path in sorted((ROOT/'paired_contrast').glob('*.json')):
        d=json.loads(path.read_text());A.hashes(d);assert E.sha(path.with_suffix('.npz'))==d['array_sha256']
        with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
        parent=json.loads((ROOT/'results'/path.name).read_text());cs=json.loads((ROOT/'frozen'/path.name).read_text())['candidates']
        for tag in ('raw','corrected'):
            bias=a[tag+'_calibration'].mean(axis=0)-a['pilot_contrasts'];var=a[tag+'_calibration'].var(axis=0,ddof=1)
            np.testing.assert_array_equal(bias,a[tag+'_bias']);np.testing.assert_array_equal(var,a[tag+'_variance'])
            for label,center in [('check','pilot'),('actual','clean')]:
                delta=a[tag+'_'+label]-a[center+'_contrasts']-bias
                with np.errstate(divide='ignore',invalid='ignore'):val=delta**2/(var*(1+1/256))
                val=np.where((delta==0)&(var==0),0.,val);q=contrast_reduce(val,cs,d['groups'],d['axis_groups'])
                np.testing.assert_allclose(q,a[tag+'_'+label+'_q'],rtol=2e-12,atol=2e-12);counts['paired_scalar_scores']+=q.size
            for j,c in enumerate(cs):
                row=next(r for r in d['rows'] if r['arm']==tag and r['candidate']==c['id']);radius=np.quantile(a[tag+'_check_q'][:,j],.95)
                assert row['radius']==radius and row['noise_coverage_count']==np.sum(a[tag+'_actual_q'][:,j]<=radius)
                direct=next(r for r in parent['rows'] if r['arm']=='local_'+tag and r['candidate']==c['id'] and r['kind']=='exact')
                assert row['direct_noise_coverage_count']==direct['noise_coverage_count']
        counts['paired_records']+=1
    out=dict(status='PASS',**counts,source_input_array_hashes=True,fresh_data_and_innovation_replay=True,
        source_sha256={str(Path(__file__)):E.sha(__file__),str(ROOT/'analyze_followups.py'):E.sha(ROOT/'analyze_followups.py')})
    R.dump(ROOT/'followup_audit.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
