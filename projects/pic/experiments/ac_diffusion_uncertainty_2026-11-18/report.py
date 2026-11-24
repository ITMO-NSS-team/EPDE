"""Check per-fold results, estimate repeat ranges, and assess fresh coverage."""
import csv
import json
from pathlib import Path
from paired import ANCHORS, B, CHUNK, NEW, ROOT, np, records
from scipy.stats import binom, binomtest


def ordered_range(x):
    s=np.sort(x)
    assert len(s)==128
    return s[[2,125]].tolist()


def median_interval(values):
    n=len(values)
    k=max(k for k in range(1,n//2+1) if 2*binom.cdf(k-1,n,.5)<=.05)
    return np.sort(values)[[k-1,n-k]].tolist()


def rate(count,n):
    it=binomtest(int(count),int(n)).proportion_ci(confidence_level=.95,method="exact")
    return dict(count=int(count),total=n,lower=it.low,upper=it.high)


def describe(x):
    return dict(median=float(np.median(x)),median_ci=median_interval(x),mean=float(np.mean(x)),
        central_range=ordered_range(x),positive=rate(sum(x>0),len(x)),ties=int(sum(x==0)))


def verify(arr,meta):
    v=arr["vec"]
    cross=arr["cross"]
    denom=np.sum((v*arr["scale"][...,None,:])**2,axis=-1)
    independent=np.einsum("...ki,...ij,...kj->...k",v,cross,v)/denom
    ds=np.max(abs(independent-arr["scores"]))
    np.testing.assert_allclose(independent,arr["scores"],atol=2e-12,rtol=2e-8)
    miss=meta["diffusion"]
    assert np.all(v[...,1,miss]==0.) and np.all(v[...,3,miss]==0.)
    keep=np.delete(np.arange(v.shape[-1]),miss)
    np.testing.assert_array_equal(v[...,2,keep],v[...,3,keep])
    assert np.all(v[...,2,miss]==-0.0001) and np.all(v[...,2,meta["target"]]==1.)
    max_angle=0.
    for k,ix in enumerate((np.arange(v.shape[-1]),keep)):
        w=arr["norm"][...,ix]
        g=arr["gram"][...,ix,:][...,ix]/w[...,None,:]/w[..., :,None]
        _,u=np.linalg.eigh((g+np.swapaxes(g,-2,-1))/2)
        observed=v[...,k,:][...,ix]*w
        observed=observed/np.linalg.norm(observed,axis=-1,keepdims=True)
        diff=np.minimum(np.linalg.norm(observed-u[...,0],axis=-1),np.linalg.norm(observed+u[...,0],axis=-1))
        max_angle=max(max_angle,float(np.max(2*np.arcsin(diff/2))))
    assert max_angle<1.e-7
    return float(ds),max_angle


def main():
    out={}
    samples=[]
    archive_hashes={}
    raw_hashes=set()
    max_score=max_angle=0.
    for tag in (*map(str,ANCHORS),"fresh"):
        chunks=[]
        for start in range(0,B,CHUNK):
            path=ROOT/"draws"/f"{tag}_{start:03}.npz"
            archive_hashes[path.name]=records.sha(path)
            with np.load(path) as z:
                chunks.append({k:z[k] for k in z.files})
        data={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        firstseed=NEW if tag=="fresh" else 800000+(int(tag)-73001)*1000
        assert np.array_equal(data["seeds"],np.arange(firstseed,firstseed+B))
        hashes=set(data["field_hashes"].tolist())
        assert len(hashes)==B and not hashes & raw_hashes
        raw_hashes.update(hashes)
        a=ANCHORS[0] if tag=="fresh" else int(tag)
        meta=json.loads((ROOT/"replay"/f"{a}.json").read_text())
        err,ang=verify(data,meta)
        max_score,max_angle=max(max_score,err),max(max_angle,ang)
        means=data["scores"].mean(axis=1)
        coeff=-data["vec"][:,:,0,meta["diffusion"]]/data["vec"][:,:,0,meta["target"]]
        contrasts=dict(refitted=means[:,1]-means[:,0],fixed=means[:,3]-means[:,2],
                       reduced_fit_minus_fixed_full=means[:,1]-means[:,2],
                       full_fit_minus_fixed_full=means[:,0]-means[:,2])
        out[tag]={k:describe(x) for k,x in contrasts.items()}
        out[tag]["diffusion_median"]=float(np.median(np.median(coeff,axis=1)))
        out[tag]["paired_score_correlation"]=float(np.corrcoef(means[:,0],means[:,1])[0,1])
        for i,seed in enumerate(data["seeds"]):
            samples.append(dict(cohort=tag,draw=i,seed=int(seed),**{k:float(x[i]) for k,x in contrasts.items()},
                                diffusion=float(np.median(coeff[i]))))
    fresh=np.array([s["refitted"] for s in samples if s["cohort"]=="fresh"])
    forecasts=[]
    for a in ANCHORS:
        tag=str(a)
        path=ROOT/"replay"/f"{a}.npz"
        meta=json.loads(path.with_suffix(".json").read_text())
        assert records.sha(path)==meta["archive"]
        with np.load(path) as z:
            scores=z["scores"].mean(axis=0)
        observed=float(scores[1]-scores[0])
        lo,hi=out[tag]["refitted"]["central_range"]
        center=out[tag]["refitted"]["median"]
        within=int(sum((lo<=fresh)&(fresh<=hi)))
        forecasts.append(dict(anchor=a,observed=observed,lower=lo,upper=hi,bootstrap_median=center,
            positive=out[tag]["refitted"]["positive"]["count"],predicts="delete" if hi<0 else "retain" if lo>0 else "unresolved",
            covered=within,below=int(sum(fresh<lo)),above=int(sum(fresh>hi)),fresh_total=len(fresh)))
    for name,rows in (("replicates",samples),("forecast_check",forecasts)):
        with (ROOT/(name+".csv")).open("w",newline="") as f:
            w=csv.DictWriter(f,rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    summary=dict(conditional=out,forecasts=forecasts,raw_fields=len(raw_hashes),
        verified_fold_scores=len(samples)*3*4,worst_score_error=max_score,worst_projective_angle=max_angle,
        source_sha256={k:records.sha(ROOT/k) for k in ("paired.py","protocol.txt","report.py")},
        archive_sha256=archive_hashes)
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print("fresh",out["fresh"])
    for r in forecasts:
        print(r)
    print("checks",len(raw_hashes),max_score,max_angle)


if __name__=="__main__":
    main()
