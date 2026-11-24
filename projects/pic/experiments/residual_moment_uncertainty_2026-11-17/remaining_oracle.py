import experiment as E
import iid_studentization as I
R,O,np,ROOT=E.R,E.O,E.np,E.ROOT
from pathlib import Path
import json

def main():
    rows=[];hashes={};arrays={}
    for name,seed in [('ac',28),('pde_compound',28),('pde_compound',29)]:
        path=ROOT/'iid'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());fp=path.with_name(path.stem+'_frozen.json');f=json.loads(fp.read_text())
        with np.load(path.with_suffix('.npz')) as z:
            clean=z['local_clean'];actual_m=z['local_studentized_actual'];actual_amp=z['actual_amplitude']
        e=E.Study(name,seed);c=next(c for c in e.cs if c['label']=='reference');v=E.vector(c,c['exact'],len(e.labels));bank=e.banks['local']
        np.testing.assert_array_equal(bank.matrix(e.p.data).real,clean)
        cached=O.Covariance(e.axes,np.eye(len(e.axes[0])),f['noise']);stacks={}
        for label,domain in [('calibration',830140),('check',830141)]:
            rng=np.random.default_rng(np.random.SeedSequence([domain,seed]));rr=[]
            for b in range(256):
                fields={var:y+f['amplitude']*f['scales'][var]*rng.standard_normal(y.shape) for var,y in e.p.data.items()}
                amp=I.amplitude(fields,f['scales'],f['orders'])
                cov=O.Covariance(e.axes,cached.time_map,{var:[amp*f['scales'][var],[0.]*y.ndim,[]] for var,y in fields.items()})
                cov.maps=cached.maps;cov.axis_cov=cached.axis_cov
                rr.append(((O.matrix(bank,fields,cov).real-clean)@v)/amp)
            stacks[label]=np.stack(rr)
        actual=((actual_m-clean)@v)/actual_amp[:,None];var=stacks['calibration'].var(axis=0,ddof=1)
        check=E.score(stacks['check'],var);test=E.score(actual,var);radius=float(np.quantile(check,.95))
        baseline=next(r for r in d['rows'] if r['arm']=='local_studentized' and r['label']=='reference')
        rows.append(dict(system=name,seed=seed,baseline_coverage=baseline['noise_coverage_count']/256,
            oracle_coverage=float(np.mean(test<=radius)),oracle_radius=radius,
            oracle_median_sd_ratio=float(np.median(stacks['check'].std(axis=0,ddof=1)/actual.std(axis=0,ddof=1)))))
        prefix=f'{name}_{seed}_'
        for label,value in dict(**stacks,actual=actual,variance=var,check_q=check,actual_q=test).items():arrays[prefix+label]=value
        hashes.update({str(p):E.sha(p) for p in [path,path.with_suffix('.npz'),fp]})
    ap=ROOT/'remaining_oracle.npz';np.savez_compressed(ap,**arrays)
    # Independent arithmetic audit on the persisted arrays.
    with np.load(ap) as z:
        for row in rows:
            prefix=f"{row['system']}_{row['seed']}_";cal=z[prefix+'calibration'];variance=np.sum((cal-cal.mean(axis=0))**2,axis=0)/255
            np.testing.assert_array_equal(variance,z[prefix+'variance'])
            boot=np.mean(z[prefix+'check']**2/variance,axis=1);actual=np.mean(z[prefix+'actual']**2/variance,axis=1)
            np.testing.assert_array_equal(boot,z[prefix+'check_q']);np.testing.assert_array_equal(actual,z[prefix+'actual_q'])
            assert np.mean(actual<=np.quantile(boot,.95))==row['oracle_coverage']
    result=dict(status='PASS',scope='Exploratory clean-center attribution only; no method change.',rows=rows,
        source_sha256={str(Path(__file__)):E.sha(__file__),I.__file__:E.sha(I.__file__),str(ROOT/'remaining_oracle_protocol.txt'):E.sha(ROOT/'remaining_oracle_protocol.txt')},
        input_sha256=hashes,array_sha256=E.sha(ap))
    R.dump(ROOT/'remaining_oracle.json',result);print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
