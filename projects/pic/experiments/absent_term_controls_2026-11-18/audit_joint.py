"""Rebuild both KdV objectives by explicit projection into a common term space."""
import json
from controls import ROOT,np,E
from evaluate import matrices
from scipy.linalg import eigh


def main():
    dest=ROOT/"kdv_joint"
    saved=json.loads((dest/"folds.json").read_text())
    uc,labels=E.R.panel(E.R.datasets.load("kdv"))
    uref=next(c for c in uc if c["label"]=="reference")
    uids=[uref["yi"],*uref["xi"]]
    uterms=[json.dumps(labels[i]) for i in uids]
    worst=0.
    n=0
    shares=[]
    for row in saved:
        forced=f'kdv_cossin_{row["forced_seed"]}'
        unforced=f'kdv_{row["unforced_seed"]}'
        fm=json.loads((dest/"records"/(forced+".json")).read_text())
        with (np.load(dest/"records"/(forced+".npz")) as obs,
              np.load(dest/"training"/(forced+".npz")) as ft,
              np.load(dest/"training"/(unforced+".npz")) as ut):
            for g in range(3):
                F,fg=matrices(ft,g,("local","time"))
                U,ug=matrices(ut,g,("local","time"))
                for case in fm["candidates"]:
                    if case["id"] not in row["scores"]["joint"]:
                        continue
                    ids=sorted([case["yi"],*case["xi"]])
                    terms=[json.dumps(fm["labels"][i]) for i in ids]
                    projection=np.zeros((len(uids),len(ids)))
                    for i,x in enumerate(uterms):
                        projection[i,terms.index(x)]=1.
                    fgram=fg[np.ix_(ids,ids)]
                    ugram=projection.T@ug[np.ix_(uids,uids)]@projection
                    if case["label"]=="reference":
                        unforced_energy=np.sum(abs(U[:,uids]@projection)**2,axis=0)
                        forced_energy=np.sum(abs(F[:,ids])**2,axis=0)
                        share=unforced_energy/(forced_energy+unforced_energy)
                        shares.append([share[ids.index(k)] for k in (case["yi"],*case["xi"][:2])])
                    cross=obs[f"left{g}"][:,ids].T@obs[f"right{g}"][:,ids]
                    sc=obs[f"scale{g}"][ids]
                    for arm in ("single","joint"):
                        G=fgram if arm=="single" else fgram+ugram
                        P=F[:,ids] if arm=="single" else np.vstack([F[:,ids],U[:,uids]@projection])
                        mass=np.diag(np.sum(abs(P)**2,axis=0))
                        _,vec=eigh((G+G.T)/2,mass,driver="gvx",subset_by_index=(0,0))
                        v=vec[:,0]
                        d=float(v@cross@v/np.sum((sc*v)**2))
                        expected=row["scores"][arm][case["id"]][g]
                        diff=abs(d-expected)
                        np.testing.assert_allclose(d,expected,rtol=5e-8,atol=3e-10)
                        worst=max(worst,diff)
                        n+=1
    out=dict(scores=n,maximum_absolute_difference=worst,
             median_unforced_pilot_energy_fraction=dict(zip(("time_derivative","nonlinearity","dispersion"),
                 np.median(shares,axis=0).tolist())))
    (dest/"audit.json").write_text(json.dumps(out,indent=2))
    print(out)


if __name__=="__main__":
    main()
