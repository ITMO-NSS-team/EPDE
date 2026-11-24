"""Independent Fourier, subgrid, field-provenance and Pareto replay checks."""
from collections import Counter
import json
from scipy.io import loadmat
from common import ROOT, SEEDS, E, np, records
from measure import temporal_weights, geometry, paired_measurement, label_names
from confirmation_study import DEST, NEW, METHODS, fixture
from compare import ARMS, ORACLES, bundle, fit


def fft_moments(z,t,x,period,names,modes):
    def transform(a):
        b=np.fft.rfft(a,axis=1)[:,:modes+1]/len(x)
        b*=np.exp(-2j*np.pi*x[0]*np.arange(modes+1)/period)
        return np.concatenate([b[:,:1].real,np.sqrt(2)*b[:,1:].real,-np.sqrt(2)*b[:,1:].imag],axis=1)
    var=(np.median(np.abs(np.roll(z,1,axis=1)+np.roll(z,-1,axis=1)-2*z))/.6744897501960817)**2/6
    moment=dict(u=transform(z),square=transform(z**2-var),cube=transform(z**3-3*var*z))
    freq=np.r_[0,1:modes+1,1:modes+1]*2*np.pi/period
    moment["laplacian"]=-freq**2*moment["u"]
    for key,values in dict(constant=np.ones_like(t),time=t,time2=t*t,cos=np.cos(2*t),sin=np.sin(2*t)).items():
        moment[key]=np.zeros_like(moment["u"])
        moment[key][:,0]=values
    out=[]
    for w,dt in temporal_weights(tuple(t)):
        columns=[dt@moment["u"] if name=="dt" else w@moment[name] for name in names]
        matrix=np.stack(columns,axis=2).reshape(-1,len(names))
        out.append(matrix/np.sqrt(len(matrix)*3))
    return np.vstack(out)


def main():
    ac=fixture("positive")
    zero=fixture("zero")
    t,x=E.R.axes_of(ac)
    source=ROOT.parents[1]/"data"/"ac"/"AC.mat"
    dense=loadmat(source)
    np.testing.assert_array_equal(dense["uu"][::4,::4].T,ac.data["u"])
    np.testing.assert_allclose(dense["tt"].ravel()[::4],t,atol=1e-14,rtol=0)
    np.testing.assert_allclose(dense["x"].ravel()[::4],x,atol=1e-14,rtol=0)
    _,labels=E.R.panel(ac)
    names=label_names(labels)
    cols=[names.index(k) for k in ("dt","cube","u","laplacian")]
    _,_,folds,_,_=geometry("ac")
    errors=Counter()
    front_checks=0
    arrays=0
    invariant=0
    archive_hashes=set()
    angles={}
    for path,seeds,problem,arms in ((ROOT,SEEDS,ac,ARMS),(DEST/"positive",NEW,ac,METHODS),(DEST/"zero",NEW,zero,METHODS)):
        for seed in seeds:
            doc=json.loads((path/"results"/f"ac_{seed}.json").read_text())
            raw=problem.noisy(5,seed)["u"]
            assert E.R.digest(raw)==doc["raw_hash"] and doc["raw_hash"] not in archive_hashes
            archive_hashes.add(doc["raw_hash"])
            for arm in (*arms,*ORACLES):
                scores={r["id"]:float(np.mean([f[arm]["D"] for f in r["folds"]])) for r in doc["rows"]}
                dyad={r["id"]:r["dyad"] for r in doc["rows"]}
                got={j for j in scores if not any(scores[k]<=scores[j] and dyad[k]<=dyad[j] and
                                                  (scores[k]<scores[j] or dyad[k]<dyad[j]) for k in scores)}
                assert got==set(doc["fronts"][arm]["definite"])
                front_checks+=1
            if seed not in (seeds.start,seeds.stop-1):
                continue
            z,meta=bundle("measurements",seed,base=path)
            for g,(idx,*_) in enumerate(folds):
                np.testing.assert_array_equal(idx,z[f"{g}_indices"])
                for tag,m,channels in (("full_mean",0,((0,2),(1,2))),("full_low",3,((0,2),(1,2))),
                                      ("even_low",3,((0,4),(2,4))),("full_high",12,((0,2),(1,2)))):
                    if f"{g}_{tag}_left" not in z:
                        continue
                    lr=[]
                    for key,(off,step) in zip(("left","right"),channels):
                        got=fft_moments(raw[idx,off::step],t[idx],x[off::step],2.,names,m)
                        stored=z[f"{g}_{tag}_{key}"]
                        err=float(np.max(abs(got-stored))/max(1,np.max(abs(stored))))
                        errors["FFT_relative_to_matrix_scale"]=max(errors["FFT_relative_to_matrix_scale"],err)
                        np.testing.assert_allclose(got,stored,rtol=2e-11,atol=3e-13)
                        lr.append(got)
                    cross=lr[0].T@lr[1]
                    np.testing.assert_allclose((cross+cross.T)/2,z[f"{g}_{tag}_gram"],rtol=2e-10,atol=1e-11)
                    arrays+=2
                # Arbitrary changes on odd columns leave every even-only reaction measurement identical.
                disturbed=raw[idx].copy()
                disturbed[:,1::2]=-13*disturbed[:,1::2]+17
                actual=paired_measurement(disturbed,t[idx],x,names,((0,4),(2,4)))
                np.testing.assert_array_equal(actual["mean_gram"],z[f"{g}_even_mean_gram"])
                invariant+=1
        co={}
        for tag,channels in (("full",((0,2),(1,2))),("even",((0,4),(2,4)))):
            for modes in (0,3,12):
                if tag=="even" and modes==12:
                    continue
                cv=[]
                for idx,*_ in folds:
                    mm=paired_measurement(problem.data["u"][idx],t[idx],x,names,channels,False,max(modes,1))
                    key="mean" if modes==0 else "low"
                    sel=cols if modes else cols[:3]
                    v,status=fit((mm[key+"_gram"][np.ix_(sel,sel)],mm[key+"_norm"][sel]))
                    assert status=="ok"
                    cv.append(-v[1:]/v[0])
                co[f"{tag}_{modes}"]=np.median(cv,axis=0).tolist()
        angles[path.name]=co
    positive_sd=.05*np.std(ac.data["u"])
    null_sd=.05*np.std(zero.data["u"])
    for n in NEW:
        a=(ac.noisy(5,n)["u"]-ac.data["u"])/positive_sd
        b=(zero.noisy(5,n)["u"]-zero.data["u"])/null_sd
        np.testing.assert_allclose(a,b,atol=3e-14,rtol=0)
    result=dict(fronts=front_checks,field_hashes=len(archive_hashes),fft_matrices=arrays,disjoint_checks=invariant,
        errors=errors,dense_source=records.sha(source),matched_noise_pairs=len(NEW),clean_coefficient_medians=angles)
    (ROOT/"audit.json").write_text(json.dumps(result,indent=2))
    print(result)


if __name__=="__main__":
    main()
