"""Paired pilot/noise factorial for the first observation."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import json
from paired import ROOT,ANCHORS,CHUNK,E,np,geometry,anchor,records

CELLS=("pilot_iid","clean_fitted","clean_iid")
N=64


def run(job):
    cell,start=job
    center,noise=cell.split("_")
    geo=geometry()
    a=ANCHORS[0]
    pilot,model=anchor(a)
    signal=pilot if center=="pilot" else geo.p.data
    inputs=[]
    hashes=[]
    for i in range(start,start+CHUNK):
        rng=np.random.default_rng(np.random.SeedSequence([48283,a,i]))
        innovation={v:(E.R.WF.ar_noise(rng,y.shape,*model[v][:2]) if noise=="fitted"
                       else .05*np.std(geo.p.data[v])*rng.standard_normal(y.shape)) for v,y in signal.items()}
        field={v:y+innovation[v] for v,y in signal.items()}
        fit=geo.estimate(field,800000+i)
        inputs.append(fit)
        hashes.append(E.R.digest(field["u"]))
    path=ROOT/"generator"/f"{cell}_{start:03}.npz"
    np.savez_compressed(path,**{k:np.stack([f[k] for f in inputs]) for k in inputs[0]},field_hashes=hashes)
    return path.stem


def report():
    from importlib.util import module_from_spec, spec_from_file_location
    spec=spec_from_file_location("ac_repeat_summary",ROOT/"report.py")
    summary=module_from_spec(spec)
    spec.loader.exec_module(summary)
    meta=json.loads((ROOT/"replay"/f"{ANCHORS[0]}.json").read_text())
    vals={}
    audit={}
    for cell in ("pilot_fitted",*CELLS):
        margins=[]
        for i in range(0,N,CHUNK):
            path=(ROOT/"draws"/f"{ANCHORS[0]}_{i:03}.npz" if cell=="pilot_fitted"
                  else ROOT/"generator"/f"{cell}_{i:03}.npz")
            with np.load(path) as z:
                summary.verify(z,meta)
                scores=z["scores"].mean(axis=1)
                margins.extend(scores[:,1]-scores[:,0])
            audit[path.name]=records.sha(path)
        vals[cell]=np.asarray(margins)
    out={k:dict(positive=int(sum(x>0)),n=N,median=float(np.median(x)),
                iqr=np.quantile(x,[.25,.75]).tolist()) for k,x in vals.items()}
    shifts={}
    for left,right in (("pilot_fitted","pilot_iid"),("clean_fitted","clean_iid"),
                       ("pilot_fitted","clean_fitted"),("pilot_iid","clean_iid")):
        x=vals[right]-vals[left]
        shifts[left+"_to_"+right]=dict(median=float(np.median(x)),increased=int(sum(x>0)))
    (ROOT/"generator_summary.json").write_text(json.dumps(dict(cells=out,paired_shifts=shifts,archives=audit,
        source_sha256={k:records.sha(ROOT/k) for k in ("generator.py","generator_protocol.txt")}),indent=2))
    print(out,shifts)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["run","report"])
    stage=ap.parse_args().stage
    if stage=="report":
        report()
    else:
        (ROOT/"generator").mkdir(exist_ok=True)
        jobs=[(cell,i) for i in range(0,N,CHUNK) for cell in CELLS
              if not (ROOT/"generator"/f"{cell}_{i:03}.npz").exists()]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for f in as_completed([pool.submit(run,j) for j in jobs]):
                print(f.result(),flush=True)
