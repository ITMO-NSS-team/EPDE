"""Independent decision audit with reordered, rescaled columns and a generalized eigensolve."""
import json
from setup_study import ROOT, SYSTEMS, SEEDS, np
from scipy.linalg import eigh


def audit(system,seed):
    name=f"{system}_{seed}"
    decisions=json.loads((ROOT/"decisions"/(name+".json")).read_text())["folds"]
    meta=json.loads((ROOT/"training"/(name+".json")).read_text())
    largest=0.
    fits=shared_changes=0
    changes=[]
    with np.load(ROOT/"training"/(name+".npz")) as data:
        for g,d in enumerate(decisions):
            gram,norm,cross,scale=[data[f"{g}_{k}"] for k in ("inner_gram","inner_norm","inner_cross","inner_scale")]
            risks=[]
            for support,recorded in zip(d["distinct_supports"],d["risks"]):
                ids=list(reversed(support))
                ix=np.ix_(ids,ids)
                units=(-1.)**np.arange(len(ids))*np.geomspace(.2,4,len(ids))
                actual=[]
                for w,old in enumerate(recorded):
                    if old is None:
                        actual.append(np.inf)
                        continue
                    parts=[]
                    for r in range(3):
                        G=gram[r,w][ix]*np.outer(units,units)
                        mass=norm[r,w,ids]*abs(units)
                        v=eigh((G+G.T)/2,np.diag(mass**2),driver="gvx",subset_by_index=(0,0))[1][:,0]*units
                        parts.append(np.einsum("i,ij,j",v,cross[r][ix],v)/np.sum((v*scale[r,ids])**2))
                    value=np.mean(parts)
                    largest=max(largest,abs(value-old)/max(1,abs(old)))
                    actual.append(value)
                    fits+=3
                if len(ids)==1 and np.isfinite(actual).all():
                    actual=[actual[0]]*3
                risks.append(actual)
            risks=np.array(risks)
            # Count smaller values and half the tied alternatives, including all-infeasible rows.
            ranks=np.array([1+np.sum(risks<risks[:,w,None],axis=1)
                            +0.5*(np.sum(risks==risks[:,w,None],axis=1)-1) for w in range(3)]).T
            shared_changes+=int(np.argmin(ranks.mean(axis=0))!=d["shared"])
            lookup={tuple(support):j for j,support in enumerate(d["distinct_supports"])}
            own=[int(np.argmin(risks[lookup[tuple(s)]])) for s in meta["supports"]]
            for j in np.flatnonzero(np.array(own)!=d["personal"]):
                i=lookup[tuple(meta["supports"][j])]
                left,right=d["personal"][j],own[j]
                changes.append(dict(system=system,seed=seed,fold=g,candidate=int(j),support=meta["supports"][j],
                    choices=[left,right],difference=abs(d["risks"][i][left]-d["risks"][i][right])))
    return fits,largest,shared_changes,changes


if __name__=="__main__":
    rows=[audit(s,n) for n in SEEDS for s in SYSTEMS]
    out=dict(inner_fits=sum(r[0] for r in rows),maximum_scaled_risk_difference=max(r[1] for r in rows),
             changed_shared_decisions=sum(r[2] for r in rows),changed_individual_decisions=sum(len(r[3]) for r in rows),
             changes=[change for r in rows for change in r[3]])
    (ROOT/"selection_audit.json").write_text(json.dumps(out,indent=2))
    print({k:v for k,v in out.items() if k!="changes"})
    print(out["changes"][:3])
