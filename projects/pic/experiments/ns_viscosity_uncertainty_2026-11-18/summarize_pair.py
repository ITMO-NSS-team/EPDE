"""Score replay distributions and verify the fixed decision rule."""
import csv
from importlib.util import module_from_spec,spec_from_file_location
import json
from ns_pair import ROOT,AC,NEW,ANCHORS,B,CHUNK,np,records
from scipy.linalg import eigh

spec=spec_from_file_location("ac_predictive_statistics",AC/"report.py")
stats=module_from_spec(spec)
spec.loader.exec_module(stats)


def verify(data,meta):
    vec=data["vec"]
    assert vec.shape[1:]==(3,4,6)
    truth=np.asarray(meta["truth"])
    miss,target=meta["missing"],meta["target"]
    assert truth[miss]==-.01 and truth[target]==1.
    np.testing.assert_array_equal(vec[:,:,2,:],np.broadcast_to(truth,vec[:,:,2,:].shape))
    reduced=truth.copy()
    reduced[miss]=0.
    np.testing.assert_array_equal(vec[:,:,3,:],np.broadcast_to(reduced,vec[:,:,3,:].shape))
    assert np.all(vec[:,:,1,miss]==0.)
    scale=data["scale"]
    denom=np.sum(np.square(vec*scale[:,:,None,:]),axis=-1)
    values=np.einsum("...ki,...ij,...kj->...k",vec,data["cross"],vec)/denom
    np.testing.assert_allclose(values,data["scores"],atol=2.e-12,rtol=2.e-8)
    max_score=float(np.max(abs(values-data["scores"])))
    worst_angle=0.
    for j in range(len(vec)):
        for g in range(3):
            for k,ids in enumerate((np.arange(len(truth)),np.delete(np.arange(len(truth)),miss))):
                w=data["norm"][j,g,ids]
                gram=data["gram"][j,g][np.ix_(ids,ids)]/w[:,None]/w[None,:]
                _,basis=eigh((gram+gram.T)/2,driver="evx")
                est=vec[j,g,k,ids]*w
                est/=np.linalg.norm(est)
                dist=min(np.linalg.norm(est-basis[:,0]),np.linalg.norm(est+basis[:,0]))
                worst_angle=max(worst_angle,float(2*np.arcsin(dist/2)))
    assert worst_angle<1.e-7
    return max_score,worst_angle


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    samples=[]
    forecasts=[]
    results={}
    manifest={}
    seen=set()
    worst_score=worst_angle=0.
    for tag in (*map(str,ANCHORS),"fresh"):
        chunks=[]
        for start in range(0,B,CHUNK):
            path=ROOT/"draws"/f"{tag}_{start:03}.npz"
            manifest[path.name]=records.sha(path)
            with np.load(path) as z:
                chunks.append({k:z[k] for k in z.files})
        data={k:np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
        first=NEW if tag=="fresh" else 810000+(int(tag)-ANCHORS[0])*1000
        np.testing.assert_array_equal(data["seeds"],np.arange(first,first+B))
        hashes=set(data["field_hashes"])
        assert len(hashes)==B and not hashes&seen
        seen.update(hashes)
        a=ANCHORS[0] if tag=="fresh" else int(tag)
        meta=json.loads((ROOT/"replay"/f"{a}.json").read_text())
        ds,ang=verify(data,meta)
        worst_score=max(ds,worst_score)
        worst_angle=max(ang,worst_angle)
        v=data["vec"][:,:,0,:]
        viscosity=np.median(-v[:,:,meta["missing"]]/v[:,:,meta["target"]],axis=1)
        scores=data["scores"].mean(axis=1)
        diffs=dict(refitted=scores[:,1]-scores[:,0],fixed=scores[:,3]-scores[:,2],
                   reduced_fit_minus_fixed_full=scores[:,1]-scores[:,2],
                   full_fit_minus_fixed_full=scores[:,0]-scores[:,2])
        results[tag]={key:stats.describe(d) for key,d in diffs.items()}
        results[tag]["median_viscosity"]=float(np.median(viscosity))
        results[tag]["paired_correlation"]=float(np.corrcoef(scores[:,0],scores[:,1])[0,1])
        for i in range(B):
            samples.append(dict(cohort=tag,draw=i,seed=int(data["seeds"][i]),viscosity=viscosity[i],
                                **{k:vals[i] for k,vals in diffs.items()}))
    fresh=np.array([r["refitted"] for r in samples if r["cohort"]=="fresh"])
    for a in ANCHORS:
        path=ROOT/"replay"/f"{a}.npz"
        meta=json.loads(path.with_suffix(".json").read_text())
        assert meta["archive"]==records.sha(path)
        with np.load(path) as z:
            scores=z["scores"].mean(axis=0)
        s=results[str(a)]["refitted"]
        lo,hi=s["central_range"]
        forecasts.append(dict(anchor=a,observed=scores[1]-scores[0],lower=lo,upper=hi,
            bootstrap_median=s["median"],positive=s["positive"]["count"],
            predicts="retain" if lo>0 else "delete" if hi<0 else "unresolved",
            covered=int(sum((fresh>=lo)&(fresh<=hi))),below=int(sum(fresh<lo)),above=int(sum(fresh>hi)),fresh_total=B))
    write_csv(ROOT/"replicates.csv",samples)
    write_csv(ROOT/"forecast_check.csv",forecasts)
    summary=dict(cohorts=results,forecasts=forecasts,distinct_fields=len(seen),verified_scores=12*len(samples),
                 maximum_score_error=worst_score,maximum_angle=worst_angle,archive_sha256=manifest,
                 sources={str(p):records.sha(p) for p in (ROOT/"ns_pair.py",ROOT/"protocol.txt",ROOT/"summarize_pair.py",
                             AC/"paired.py",AC/"report.py")})
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print("fresh",results["fresh"])
    for f in forecasts:
        print(f)
    print("verification",len(seen),worst_score,worst_angle)


if __name__=="__main__":
    main()
