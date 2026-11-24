"""Matched physical-time Fourier held-outs and independent null calibration."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from common import ROOT, np, E, score, records
from measure import geometry, temporal_weights
from compare import fit
from confirmation_study import NEW,DEST,fixture
from outcomes import dump_csv

FRESH=range(796001,796129)
BANDS={"low":np.arange(0,4),"high":np.arange(4,13),"all":np.arange(0,13)}
METRICS=(*BANDS,"abs_diffusion")
WAVE=np.r_[0,1:13,1:13]


@lru_cache(maxsize=1)
def design():
    _,axes,folds,_,_=geometry("ac")
    t,x=axes
    indices=[f[0] for f in folds]
    stride=np.diff(indices[0])[0]
    assert all(np.all(np.diff(idx)==stride) for idx in indices)
    common=np.arange(max(idx[0] for idx in indices),min(idx[-1] for idx in indices)+1)
    interior=t[common]
    temporal=[]
    for radius in (3,5,7):
        m=int(radius*stride)
        centers=np.arange(m,len(interior)-m,stride)
        (w,der),_=E.Q.axis_weights(interior,0,centers,m,3,np.zeros(1),1,3)
        temporal.append((w,-der))
    operators=[]
    for idx in indices:
        interp=E.O.interpolation(t,idx)[common]
        assert interior[0]>=t[idx[0]] and interior[-1]<=t[idx[-1]]
        bank=[(w@interp,dt@interp) for w,dt in temporal]
        for w,dt in bank:
            np.testing.assert_allclose(dt@t[idx],w@np.ones(len(idx)),atol=2e-12)
        operators.append(bank)
    return t,x,indices,operators


def channel(u,spatial_origin):
    var=(np.median(abs(np.roll(u,1,axis=1)+np.roll(u,-1,axis=1)-2*u))/.6744897501960817)**2/6
    def fft(a):
        m=np.fft.rfft(a,axis=1)[:,:13]/a.shape[1]
        m*=np.exp(-1j*np.pi*spatial_origin*np.arange(13))
        return np.concatenate((m[:,:1].real,np.sqrt(2)*m[:,1:].real,-np.sqrt(2)*m[:,1:].imag),axis=1)
    return fft(u),fft(u**3-3*var*u)


def blocks(mom,operators,band):
    # Column order is u_t, u³, u, u_xx. Every candidate uses the same test rows.
    out=[]
    take=np.isin(WAVE,band)
    field,cube=mom
    for w,dt in operators:
        u=w@field[:,take]
        mat=np.stack((dt@field[:,take],w@cube[:,take],u,-(np.pi*WAVE[take])**2*u),axis=2).reshape(-1,4)
        out.append(mat/np.sqrt(3*len(mat)))
    return np.vstack(out)


def one(problem,seed):
    t,x,indices,operators=design()
    raw=problem.noisy(5,seed)["u"]
    vectors=[]
    val=[]
    for g,idx in enumerate(indices):
        sides=[channel(raw[idx,offset::2],x[offset]) for offset in range(2)]
        l,r=[blocks(side,temporal_weights(tuple(t[idx])),BANDS["all"]) for side in sides]
        gram=(l.T@r+r.T@l)/2
        norms=np.sqrt(np.sum(l*l+r*r,axis=0)/2)
        # Keep the full four columns for the held-out term norms and residuals.
        support_vectors=[]
        for count in (4,3):
            vec,status=fit((gram[:count,:count],norms[:count]))
            assert status=="ok"
            support_vectors.append(np.pad(vec,(0,4-count)))
        vectors.append(support_vectors)
        avg=tuple((a+b)/2 for a,b in zip(*sides))
        val.append({b:blocks(avg,operators[g],band) for b,band in BANDS.items()})
    coeff=[-s[0][3]/s[0][0] for s in vectors]
    row=dict(seed=seed,raw_hash=E.R.digest(raw),diffusion=float(np.median(coeff)),abs_diffusion=float(abs(np.median(coeff))))
    for band in BANDS:
        deltas=[]
        for g in range(3):
            l,r=[v[band] for j,v in enumerate(val) if j!=g]
            scale=np.linalg.norm(val[g][band],axis=0)
            full,small=[score(l,r,scale,v) for v in vectors[g]]
            deltas.append(small-full)
        row[band]=float(np.mean(deltas))
    return row,coeff


def batch(job):
    group,kind,seeds=job
    problem=fixture(kind)
    out=[]
    for seed in seeds:
        row,coeff=one(problem,seed)
        if group=="calibration":
            original=json.loads((DEST/kind/"results"/f"ac_{seed}.json").read_text())
            ref=next(r for r in original["rows"] if r["label"]=="reference")
            np.testing.assert_allclose(coeff,[f["high_only"]["coeff"]["laplacian"] for f in ref["folds"]],
                                       atol=2e-12,rtol=2e-8)
            assert row["raw_hash"]==original["raw_hash"]
        out.append(dict(group=group,trajectory=kind,**row))
    return out


if __name__=="__main__":
    jobs=[]
    for group,kind,seeds in (("calibration","zero",NEW),("test","zero",FRESH),("test","positive",FRESH)):
        for i in range(seeds.start,seeds.stop,8):
            jobs.append((group,kind,range(i,min(i+8,seeds.stop))))
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows=[r for batch_rows in pool.map(batch,jobs) for r in batch_rows]
    assert len({r["raw_hash"] for r in rows})==len(rows)==320
    dump_csv(ROOT/"readout_fields.csv",rows)
    summary={}
    for k in METRICS:
        def values(group,kind):
            return np.array([r[k] for r in rows if r["group"]==group and r["trajectory"]==kind])
        null=values("calibration","zero")
        pos=values("test","positive")
        neg=values("test","zero")
        q=float(np.sort(null)[61])
        summary[k]=dict(cutoff=q,detected=int(sum(pos>q)),false_calls=int(sum(neg>q)),
            positive_sign=int(sum(pos>0)),null_sign=int(sum(neg>0)),
            paired_larger=int(sum(pos>neg)),auc=float(np.mean((pos[:,None]>neg)+.5*(pos[:,None]==neg))),
            positive_median=float(np.median(pos)),zero_median=float(np.median(neg)))
    (ROOT/"readout_summary.json").write_text(json.dumps(dict(metrics=summary,protocol=records.sha(ROOT/"readout_protocol.txt")),indent=2))
    print(summary)
