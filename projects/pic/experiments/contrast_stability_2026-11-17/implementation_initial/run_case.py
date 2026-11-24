import metric as C
E,R,np,ROOT=C.E,C.R,C.np,C.ROOT
from pathlib import Path
import json,time,argparse
CAL=256;B=128


def sources():
    paths={Path(__file__),Path(C.__file__),ROOT/'protocol.txt',Path(E.__file__)}
    paths.update(Path(p) for p in E.sources())
    return {str(p):E.S.sha(p) for p in sorted(paths)}


def run(name,seed):
    e=E.Experiment(name,seed);extract=C.Extractor(e.cs,e.dshape)
    observed=e.evaluate(e.raw);obs,dyad,defect=extract.extract(e.dbank.matrix(e.raw))
    np.testing.assert_allclose(dyad,observed['dyad'],rtol=2e-9,atol=2e-10)
    pilot,_,pilot_defect=extract.extract(e.dbank.matrix(e.pilot))
    rng=np.random.default_rng(np.random.SeedSequence([720130,seed]));cal=[];cal_defect=[];innovations=[]
    for b in range(CAL):
        z={v:R.WF.ar_noise(rng,y.shape,*e.model[v][:2]) for v,y in e.raw.items()}
        if b in (0,CAL-1):innovations.append({v:R.digest(y) for v,y in z.items()})
        d,_,df=extract.extract(e.dbank.matrix({v:e.pilot[v]+y for v,y in z.items()}));cal.append(d);cal_defect.append(df)
    cal=np.stack(cal);noise=C.NoiseModel(extract,pilot,cal)
    frozen=dict(system=name,seed=seed,candidates=e.cs,choices=e.choices,anchor_vectors=e.anchor_vectors,
        groups=extract.groups,axis_groups=extract.axis_groups,calibration_draws=CAL,noise_model=e.model,
        contrast_bias=noise.bias,contrast_variance=noise.variance,
        raw_data_sha256={v:R.digest(y) for v,y in e.raw.items()},pilot_sha256={v:R.digest(y) for v,y in e.pilot.items()},
        calibration_innovation_hashes=innovations,input_sha256=e.inputs,source_sha256=sources())
    fpath=ROOT/'frozen'/f'{name}_seed{seed}.json';R.dump(fpath,frozen);fhash=E.S.sha(fpath)
    # Every method input freezes before the clean diagnostic controls below.
    clean,clean_dyad,clean_defect=extract.extract(e.dbank.matrix(e.p.data))
    e.clean_grams=E.F.evaluation_grams(e.p,e.old,e.terms,e.opaque,e.labels)
    arrays=dict(observed_contrasts=obs,observed_dyad=dyad,observed_deficient=defect,
        observed_discrepancy=observed['discrepancy'],anchor_vectors=e.anchor_vectors,anchor_energy=e.energy(e.anchor_vectors),
        pilot_contrasts=pilot,pilot_deficient=pilot_defect,clean_contrasts=clean,clean_dyad=clean_dyad,clean_deficient=clean_defect,
        calibration_contrasts=cal,calibration_deficient=np.stack(cal_defect),bias=noise.bias,variance=noise.variance,
        observed_q_bias=noise.score(obs),observed_q_raw=noise.score(obs,bias=False))
    samples={k:[] for k in ('bootstrap','benchmark')};rng=np.random.default_rng(np.random.SeedSequence([720131,seed]));shared=[]
    print(name,seed,'calibrated',extract.width,'contrasts',round(time.monotonic()-e.start,1),'s',flush=True)
    for b in range(B):
        z={v:R.WF.ar_noise(rng,y.shape,*e.model[v][:2]) for v,y in e.raw.items()}
        if b in (0,B-1):shared.append({v:R.digest(y) for v,y in z.items()})
        fields={'bootstrap':{v:e.pilot[v]+y for v,y in z.items()},'benchmark':e.p.noisy(5,910000+1000*seed+b)}
        for tag,field in fields.items():
            st=e.evaluate(field);d,ii,df=extract.extract(e.dbank.matrix(field))
            np.testing.assert_allclose(ii,st['dyad'],rtol=2e-9,atol=2e-10)
            st.update(contrasts=d,q_bias=noise.score(d),q_raw=noise.score(d,bias=False),slab_deficient=df,
                noise_q=noise.score(d-(pilot if tag=='bootstrap' else clean)),energy=e.energy(st['vectors']))
            samples[tag].append(st)
        if (b+1)%32==0:print(name,seed,'measurement pairs',b+1,round(time.monotonic()-e.start,1),'s',flush=True)
    for tag,ss in samples.items():
        for key in ss[0]:arrays[tag+'_'+key]=np.stack([s[key] for s in ss])
    path=ROOT/'results'/f'{name}_seed{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
    assert E.S.sha(fpath)==fhash
    R.dump(path.with_suffix('.json'),dict(system=name,seed=seed,draws=B,calibration_draws=CAL,candidates=e.cs,
        source_sha256=sources(),input_sha256=e.inputs,frozen_sha256=fhash,array_sha256=E.S.sha(path),
        check_innovation_hashes=shared,actual_noise_seeds=[910000+1000*seed+b for b in range(B)],
        nonfinite={k:int(np.sum(~np.isfinite(v))) for k,v in arrays.items() if np.any(~np.isfinite(v))},
        seconds=time.monotonic()-e.start))
    print(name,seed,'DONE',round(time.monotonic()-e.start,1),'s',flush=True)


if __name__=='__main__':
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--system',required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    if not (ROOT/'results'/f'{a.system}_seed{a.seed}.json').exists():run(a.system,a.seed)
