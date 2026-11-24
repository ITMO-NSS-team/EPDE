import experiment as E
R,np,ROOT=E.R,E.np,E.ROOT
import argparse,json,time


def run(name,seed):
    start=time.monotonic();e=E.Study(name,seed);arrays={};observed=e.matrices(e.raw);fits={}
    for bank in ('local','broad'):
        arrays[bank+'_pilot']=e.pilot_moments[bank+'_raw'];fits[bank]=[]
        for arm in ('raw','corrected'):
            for c in e.cs:
                coef,info=E.N.E.ordinary(observed[bank+'_'+arm],c)
                fits[bank].append(dict(candidate=c['id'],label=c['label'],slot=c['slot'],kind=arm+'_ols',coefficients=coef,fit=info,
                    vector=None if coef is None else E.vector(c,coef,len(e.labels))))
    for arm,value in observed.items():arrays[arm+'_observed']=value
    innovations={};stacks={arm:[] for arm in observed}
    rng=np.random.default_rng(np.random.SeedSequence([830140,seed]));hh=[]
    for b in range(256):
        z={v:R.WF.ar_noise(rng,y.shape,*e.model[v][:2]) for v,y in e.raw.items()}
        if b in (0,255):hh.append({v:R.digest(y) for v,y in z.items()})
        mm=e.matrices({v:e.pilot[v]+y for v,y in z.items()})
        for arm,value in mm.items():stacks[arm].append(value)
    for arm,ss in stacks.items():arrays[arm+'_calibration']=np.stack(ss)
    innovations['calibration']=hh
    frozen=dict(system=name,seed=seed,candidates=e.cs,labels=e.labels,specs=e.specs,noise=e.model,pilot_info=e.info,
        raw_sha256={v:R.digest(y) for v,y in e.raw.items()},pilot_sha256={v:R.digest(y) for v,y in e.pilot.items()},
        bank_metadata={k:b.metadata for k,b in e.banks.items()},fits=fits,calibration_hashes={k:R.digest(v) for k,v in arrays.items() if k.endswith('_calibration')},
        source_sha256=E.sources(),input_sha256=e.inputs,calibration_innovation_hashes=hh)
    fp=ROOT/'frozen'/f'{name}_seed{seed}.json';R.dump(fp,frozen);fh=E.sha(fp)
    # Geometry, estimates, simulated variance inputs and anchor fits frozen.
    clean=e.matrices(e.p.data,corrected=False)
    queries={bank:[dict(candidate=c['id'],label=c['label'],slot=c['slot'],kind='exact',vector=E.vector(c,c['exact'],len(e.labels))) for c in e.cs]+fits[bank] for bank in ('local','broad')}
    for bank in ('local','broad'):arrays[bank+'_clean']=clean[bank+'_raw']
    print(name,seed,'calibrated',round(time.monotonic()-start,1),'s',flush=True)
    stacks={tag:{arm:[] for arm in observed} for tag in ('check','actual')};rng=np.random.default_rng(np.random.SeedSequence([830141,seed]));hh=[]
    for b in range(256):
        z={v:R.WF.ar_noise(rng,y.shape,*e.model[v][:2]) for v,y in e.raw.items()}
        if b in (0,255):hh.append({v:R.digest(y) for v,y in z.items()})
        fields={'check':{v:e.pilot[v]+y for v,y in z.items()},'actual':e.p.noisy(5,1200000+1000*seed+b)}
        for tag,field in fields.items():
            for arm,value in e.matrices(field).items():stacks[tag][arm].append(value)
        if (b+1)%64==0:print(name,seed,'pairs',b+1,round(time.monotonic()-start,1),'s',flush=True)
    for tag,st in stacks.items():
        for arm,ss in st.items():arrays[arm+'_'+tag]=np.stack(ss)
    assert all(np.isfinite(v).all() for v in arrays.values())
    rows=E.statistics(arrays,queries);path=ROOT/'results'/f'{name}_seed{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
    assert E.sha(fp)==fh
    R.dump(path.with_suffix('.json'),dict(system=name,seed=seed,draws=256,calibration_draws=256,rows=rows,queries=queries,
        source_sha256=E.sources(),input_sha256=e.inputs,frozen_sha256=fh,array_sha256=E.sha(path),
        check_innovation_hashes=hh,actual_noise_seeds=list(range(1200000+1000*seed,1200256+1000*seed)),seconds=time.monotonic()-start))
    print(name,seed,'DONE',round(time.monotonic()-start,1),'s',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--system',required=True);ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    p=ROOT/'results'/f'{args.system}_seed{args.seed}.json'
    if not p.exists() or json.loads(p.read_text())['source_sha256']!=E.sources():run(args.system,args.seed)
