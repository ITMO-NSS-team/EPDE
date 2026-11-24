import experiment as E
R,O,np,ROOT=E.R,E.O,E.np,E.ROOT
from pathlib import Path
from math import comb
import json,time,sys,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed


def amplitude(fields,scales,orders):
    total=0.;count=0
    for v,y in fields.items():
        for axis,k in enumerate(orders[v]):
            if k:
                d=np.diff(y,n=k,axis=axis);total+=float(np.sum(d*d))/(comb(2*k,k)*scales[v]**2);count+=d.size
    if not count or not total>0:raise ValueError('No positive iid amplitude estimate')
    return float(np.sqrt(total/count))


def run(name,seed):
    start=time.monotonic();e=E.Study(name,seed);parent=ROOT/'results'/f'{name}_seed{seed}.json';base=np.load(parent.with_suffix('.npz'))
    parentmeta=json.loads(parent.read_text());frozen=json.loads((ROOT/'frozen'/parent.name).read_text())
    assert {v:R.digest(y) for v,y in e.raw.items()}==frozen['raw_sha256']
    scales={v:float(np.std(y)) for v,y in e.raw.items()};orders={v:list(e.model[v][2]) for v in e.raw};delta=amplitude(e.raw,scales,orders)
    model={v:[delta*scales[v],[0.]*y.ndim,[]] for v,y in e.raw.items()};cov=O.Covariance(e.axes,np.eye(len(e.axes[0])),model)
    checks=[]
    def matrices(fields,amp,raw=None,replay=False):
        out={}
        for banktag,bank in e.banks.items():
            convert=e.basis.convert if banktag=='broad' else lambda x:x.real
            plain=convert(bank.matrix(fields)) if raw is None else raw[banktag]
            fixed=convert(O.matrix(bank,fields,cov));student=plain+(amp/delta)**2*(fixed-plain)
            if replay:
                cc=O.Covariance(e.axes,np.eye(len(e.axes[0])),{v:[amp*scales[v],[0.]*y.ndim,[]] for v,y in fields.items()})
                direct=convert(O.matrix(bank,fields,cc));np.testing.assert_allclose(student,direct,rtol=2e-12,atol=2e-12)
                checks.append(float(np.max(np.abs(student-direct))))
            out[banktag+'_fixed']=fixed;out[banktag+'_studentized']=student
        return out
    arrays={};obs=matrices(e.raw,delta)
    for arm,mat in obs.items():arrays[arm+'_observed']=mat
    for bank in e.banks:arrays[bank+'_pilot']=e.pilot_moments[bank+'_raw']
    rng=np.random.default_rng(np.random.SeedSequence([830140,seed]));stacks={arm:[] for arm in obs};aa=[];ih=[]
    for b in range(256):
        z={v:delta*scales[v]*rng.standard_normal(y.shape) for v,y in e.raw.items()}
        if b in (0,255):ih.append({v:R.digest(y) for v,y in z.items()})
        field={v:e.pilot[v]+y for v,y in z.items()};amp=amplitude(field,scales,orders);aa.append(amp)
        for arm,mat in matrices(field,amp,replay=b==0).items():stacks[arm].append(mat)
    arrays['calibration_amplitude']=np.array(aa)
    for arm,ss in stacks.items():arrays[arm+'_calibration']=np.stack(ss)
    dest=ROOT/'iid';dest.mkdir(exist_ok=True);fp=dest/f'{name}_seed{seed}_frozen.json'
    src={**E.sources(),__file__:E.sha(__file__),str(ROOT/'iid_protocol.txt'):E.sha(ROOT/'iid_protocol.txt')}
    R.dump(fp,dict(system=name,seed=seed,amplitude=delta,scales=scales,orders=orders,noise=model,source_sha256=src,
        input_sha256={str(p):E.sha(p) for p in [parent,parent.with_suffix('.npz'),ROOT/'frozen'/parent.name]},
        calibration_hashes={k:R.digest(v) for k,v in arrays.items() if k.endswith('_calibration')},calibration_innovation_hashes=ih))
    fh=E.sha(fp)
    for bank in e.banks:arrays[bank+'_clean']=base[bank+'_clean']
    rawactual={bank:base[bank+'_raw_actual'] for bank in e.banks}
    stacks={tag:{arm:[] for arm in obs} for tag in ('check','actual')};amps={tag:[] for tag in stacks}
    rng=np.random.default_rng(np.random.SeedSequence([830141,seed]));ih=[]
    for b in range(256):
        z={v:delta*scales[v]*rng.standard_normal(y.shape) for v,y in e.raw.items()}
        if b in (0,255):ih.append({v:R.digest(y) for v,y in z.items()})
        fields={'check':{v:e.pilot[v]+y for v,y in z.items()},'actual':e.p.noisy(5,1200000+1000*seed+b)}
        for tag,field in fields.items():
            amp=amplitude(field,scales,orders);amps[tag].append(amp)
            raw={bank:rawactual[bank][b] for bank in e.banks} if tag=='actual' else None
            for arm,mat in matrices(field,amp,raw,replay=b==0).items():stacks[tag][arm].append(mat)
    for tag,st in stacks.items():
        arrays[tag+'_amplitude']=np.array(amps[tag])
        for arm,ss in st.items():arrays[arm+'_'+tag]=np.stack(ss)
    rows=[]
    for arm in obs:
        bank,kind=arm.split('_',1);pilot=arrays[bank+'_pilot'];clean=arrays[bank+'_clean']
        student=kind=='studentized'
        norm={tag:arrays[tag+'_amplitude'][:,None] if student else 1. for tag in ('calibration','check','actual')}
        for c in e.cs:
            v=E.vector(c,c['exact'],len(e.labels));cal=((arrays[arm+'_calibration']-pilot)@v)/norm['calibration'];var=cal.var(axis=0,ddof=1)
            boot=((arrays[arm+'_check']-pilot)@v)/norm['check'];actual=((arrays[arm+'_actual']-clean)@v)/norm['actual']
            qb=E.score(boot,var);qa=E.score(actual,var);radius=float(np.quantile(qb,.95))
            null=E.score((arrays[arm+'_actual']@v)/norm['actual'],var);point=E.score((obs[arm]@v)/(delta if student else 1.),var)
            intervals=np.quantile(boot,[.025,.975],axis=0);actualsd=actual.std(axis=0,ddof=1);valid=actualsd>0
            finite=all(np.isfinite(x).all() for x in (qb,qa,null,point))
            rows.append(dict(arm=arm,candidate=c['id'],label=c['label'],slot=c['slot'],status='ok' if finite else 'nonfinite',
                noise_coverage_count=int(np.sum(qa<=radius)),zero_acceptance_count=int(np.sum(null<=radius)),anchor_acceptance=bool(point<=radius),
                radius=radius,point_score=float(point),mean_shift_score=float(E.score(actual.mean(axis=0),var)),
                coordinate_coverage=float(np.mean((actual>=intervals[0])&(actual<=intervals[1]))),
                median_sd_ratio=float(np.median(boot.std(axis=0,ddof=1)[valid]/actualsd[valid])) if valid.any() else None,
                zero_variance_rows=int(np.sum(var==0))))
    assert all(np.isfinite(x).all() for x in arrays.values())
    ap=dest/f'{name}_seed{seed}.npz';np.savez_compressed(ap,**arrays);assert E.sha(fp)==fh
    R.dump(ap.with_suffix('.json'),dict(system=name,seed=seed,rows=rows,amplitude=delta,
        actual_amplitude_mean=float(arrays['actual_amplitude'].mean()),actual_amplitude_sd=float(arrays['actual_amplitude'].std(ddof=1)),
        source_sha256=src,input_sha256={str(p):E.sha(p) for p in [parent,parent.with_suffix('.npz'),ROOT/'frozen'/parent.name]},
        frozen_sha256=fh,array_sha256=E.sha(ap),check_innovation_hashes=ih,correction_replay_max_error=max(checks),seconds=time.monotonic()-start))


def worker(job):
    name,seed=job;path=ROOT/'logs'/f'iid_{name}_seed{seed}.txt'
    with path.open('w',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__)),name,str(seed)],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row=dict(system=name,seed=seed,exit_code=r.returncode);print(json.dumps(row),flush=True);return row


if __name__=='__main__':
    if len(sys.argv)>1:run(sys.argv[1],int(sys.argv[2]))
    else:
        from run_suite import SYSTEMS
        with ThreadPoolExecutor(max_workers=7) as pool:rr=[f.result() for f in as_completed([pool.submit(worker,(n,s)) for s in (24,25,26) for n in SYSTEMS])]
        R.dump(ROOT/'iid_processes.json',rr);sys.exit(any(r['exit_code'] for r in rr))
