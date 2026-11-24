"""Independent replay with separate linear algebra routines and Pareto masks."""
import csv
import json
from panel import ROOT, REPLAY, FRESH, SYSTEMS, record_path, records
from evaluate import ARMS, load_npz, matrices, projective_error, np
from train import TrainingMoments, E
from scipy import linalg


def correction_controls():
    rng = np.random.default_rng(730)
    for complex_ in (False,True):
        data = rng.normal(size=(129,20,4))
        if complex_:
            data = data+1j*rng.normal(size=data.shape)
        data[:,:,-1] = 1
        st = TrainingMoments(data[0])
        for m in data[1:]:
            st.update(m)
        expected = E.J.prepare(data,128)["point_clsm"]
        np.testing.assert_allclose(st.corrected(),expected,rtol=3e-12,atol=3e-13)
        assert np.all(st.scatter[-1,:]==0)
    t=np.linspace(0,4*np.pi,201)
    u=np.sin(t)+1.3*np.cos(t)
    du=np.cos(t)-1.3*np.sin(t)
    for a,b,k in ((du*np.cos(t),u*np.sin(t),1.),(u*np.cos(t),-du*np.sin(t),1.3),(du*du,u*u,2.69)):
        m=np.column_stack((a,b,np.ones(len(t))))
        for g in range(3):
            train=m[g::3][:66].reshape(11,6,3).mean(axis=1)
            st=TrainingMoments(train)
            for _ in range(128):
                st.update(train)
            norm=np.linalg.norm(train,axis=0)
            _,v=linalg.eigh(st.corrected()/norm[:,None]/norm[None,:],driver="evx")
            vec=v[:,0]/norm
            np.testing.assert_allclose(vec/vec[0],[1.,1.,-k],rtol=1e-11,atol=1e-12)


def main():
    correction_controls()
    summary=json.loads((ROOT/"summary.json").read_text())
    slots={(s["system"],s["seed"],s["slot"]):s for s in json.loads((ROOT/"slots.json").read_text())}
    max_score=max_angle=max_ns=0.
    fits=0
    fronts=0
    unresolved=0
    with (ROOT.parent/"ns_viscosity_contrast_2026-11-18"/"paired.csv").open() as f:
        ns={(int(r["seed"]),r["bank"],r["fit"]):r for r in csv.DictReader(f)}
    from collections import Counter
    verified=Counter()
    for seed in (*REPLAY,*FRESH):
        cohort="replay" if seed in REPLAY else "fresh"
        for system in SYSTEMS:
            path=ROOT/"results"/f"{system}_{seed}.json"
            assert records.sha(path)==summary["results_sha256"][path.name]
            doc=json.loads(path.read_text())
            obs_path=record_path(system,seed)
            train_path=ROOT/"training"/path.with_suffix(".npz").name
            assert (records.sha(obs_path),records.sha(train_path))==(doc["record"],doc["training"])
            data=load_npz(obs_path)
            tr=load_npz(train_path)
            cs={c["id"]:(j,c) for j,c in enumerate(json.loads(obs_path.with_suffix(".json").read_text())["candidates"])}
            layouts={(g,bank):matrices(tr,g,tags) for g in range(3)
                     for bank,tags in (("local_time",("local","time")),("broad",("broad",)))}
            for slot_id in {r["slot"] for r in doc["rows"]}:
                rows=[r for r in doc["rows"] if r["slot"]==slot_id]
                outcomes=slots[system,seed,slot_id]["arms"]
                for arm in ARMS:
                    means=[]
                    for row in rows:
                        idx=row["ids"]
                        ds=[]
                        for g,f in enumerate(row["folds"]):
                            result=f[arm]
                            if result["D"] is None:
                                unresolved+=1
                                continue
                            v=np.asarray(result["vector"])
                            sc=data[f"scale{g}"][idx]
                            l,r=[data[f"{key}{g}"][:,idx] for key in ("left","right")]
                            denom=np.sum((sc*v)**2)
                            if arm=="legacy":
                                cindex,c=cs[row["id"]]
                                assert np.array_equal(v,data["old_coefficients"][g,idx,cindex])
                                denom=sc[idx.index(c["yi"])]**2
                            else:
                                bank,corr=arm.rsplit("_",1)
                                full,gram=layouts[g,bank]
                                m=full[:,idx]
                                norm=np.linalg.norm(m,axis=0)
                                if corr=="on":
                                    mat=gram[np.ix_(idx,idx)]/norm[:,None]/norm[None,:]
                                    _,eigvec=linalg.eigh((mat+mat.T)/2,driver="evx")
                                    other=eigvec[:,0]/norm
                                else:
                                    real=np.vstack((m.real,m.imag)) if np.iscomplexobj(m) else m
                                    full_basis=len(real)<len(idx)
                                    _,_,vt=linalg.svd(real/norm,full_matrices=full_basis,lapack_driver="gesvd")
                                    other=vt[-1]/norm
                                max_angle=max(max_angle,projective_error(v,other,norm))
                            check=float(v@(l.T@r)@v)/denom
                            max_score=max(max_score,abs(check-result["D"])/max(1.,abs(check),abs(result["D"])))
                            # Compare front memberships using the recorded dot products to preserve their tie convention.
                            ds.append(result["D"])
                            fits+=1
                        means.append(np.mean(ds) if len(ds)==3 else np.nan)
                    ids=[r["id"] for r in rows]
                    d=np.array(means)
                    y=np.array([r["legacy_dyad" if arm=="legacy" else "dyad"] for r in rows])
                    valid=np.isfinite(d)
                    dominated=((d[:,None]<=d[None,:])&(y[:,None]<=y[None,:])&
                               ((d[:,None]<d[None,:])|(y[:,None]<y[None,:]))).any(axis=0)
                    possible=[k for k in range(len(rows)) if not dominated[k]]
                    expected=outcomes[arm]
                    assert sorted(ids[k] for k in possible)==expected["possible"]
                    if valid.all():
                        assert expected["possible"]==expected["definite"]
                        reg=any(rows[k]["registered"] for k in possible)
                        adds=sum(rows[k]["label"].startswith("add_") and not rows[k]["registered"] for k in possible)
                        ref=any(rows[k]["label"]=="reference" for k in possible)
                        for key,value in (("registered",reg),("add",bool(adds)),("add_count",adds),("reference",ref),
                                          ("front",len(possible))):
                            verified[cohort,arm,key]+=value
                    fronts+=1
                    if seed in REPLAY and system=="ns" and slot_id==0 and arm!="legacy":
                        bank,corr=arm.rsplit("_",1)
                        old=ns[seed,("time" if bank=="local_time" else "broad0"),("ols" if corr=="off" else "clsm")]
                        for name,key in (("e0:reference","full"),("e0:drop_3","drop")):
                            max_ns=max(max_ns,abs(d[ids.index(name)]-float(old[key])))
    assert fronts==900 and max_ns<1e-10 and max_angle<1e-5 and max_score<1e-7
    for cohort,arms in summary["cohorts"].items():
        for arm,res in arms.items():
            if res["unresolved"]==0:
                for key in ("registered","reference","add","add_count","front"):
                    assert res[key]==[verified[cohort,arm,key]]*2
    report=dict(verified_fold_scores=fits,unresolved_folds=unresolved,verified_fronts=fronts,
        max_scaled_score_difference=max_score,max_projective_angle=max_angle,max_ns_replay_error=max_ns,
        noiseless_constant_identities=9,sha256={p:records.sha(ROOT/p) for p in
            ("protocol.txt","panel.py","train.py","evaluate.py","summarize.py","verify.py","summary.json","slots.json")})
    (ROOT/"verification.json").write_text(json.dumps(report,indent=2))
    print(report)


if __name__=="__main__":
    main()
