"""Bootstrap training Grams on a common set of reconstructed field draws."""
from concurrent.futures import ProcessPoolExecutor
import json
from holdouts import E, FRESH, NAMES, REPLAY, ROOT, digest, np


def read_case():
    problem = E.R.datasets.load("ns")
    axes = E.R.axes_of(problem)
    cs, labels = E.R.panel(problem)
    terms, opaque = E.R.symbols(problem,labels)
    full,drop = [next(c for c in cs if c["id"]==name) for name in NAMES]
    ids = sorted([full["yi"],*full["xi"]])
    missing, = set(full["xi"])-set(drop["xi"])
    assert full["exact"][full["xi"].index(missing)]==0.01
    return problem,axes,[terms[i] for i in ids],opaque,ids,full,drop,missing


def training(seed):
    problem,axes,terms,opaque,ids,full,drop,missing = read_case()
    raw = problem.noisy(5,seed)
    baseline,old = E.F.baseline("ns",0)
    paths = [baseline,E.M.ROOT/"results/ns_seed0.json",E.J.ROOT/"results/ns_seed0.json",
             E.F.ROOT/"fit_results/ns_seed0.json"]
    validation,joint,local = [json.loads(p.read_text()) for p in paths[1:]]
    common = slice(*validation["segment"])
    common_axes = [axes[0][common],*axes[1:]]
    common_opaque = {k:v[common] for k,v in opaque.items()}
    saved = dict(ids=ids, target=ids.index(full["yi"]), missing=ids.index(missing))
    draws = old["draws"]
    assert draws==128
    for g,group in enumerate(old["groups"]):
        choices = {x["candidate"]:x["selected"]["all"] for x in joint["folds"][g]["choices"]
                   if x["draws"]==draws}
        assert all(choices[name]=="broad0_clsm" for name in NAMES)
        idx = np.asarray(group["indices"])
        obs = {v:y[idx] for v,y in raw.items()}
        pilot,_ = E.R.reconstruct(obs,axes,idx)
        noise = {v:E.R.WF.noise_model(y) for v,y in obs.items()}
        segment,tax,opq,plan = E.Q.mapped_plan(old,g,local["plans"][g]["plan"],axes,opaque)
        local_bank = E.Q.Bank(tax,terms,opq,plan=plan)
        broad_meta = joint["folds"][g]["banks"]["broad0"]
        broad_bank = E.S.N.V.make_broad(tax,terms,opq,broad_meta["plan"],broad_meta["k"])
        key = validation["folds"][g]["selection"]["q0"]["key"]
        plan = validation["banks"][key]["plan"]
        modes = [np.r_[np.arange(k+1),-np.arange(1,k+1)] for k in (4,2,2)]
        freqs = [f*np.pi/(m*(a[1]-a[0])) for f,m,a in zip(modes,plan["ms"],common_axes)]
        probe = E.Q.Bank(common_axes,terms,common_opaque,plan=plan,frequencies=freqs)
        assert probe.shape==(9,5,5)
        mats = {tag:[] for tag in ("local","broad0","time","x1","x2","xy")}
        rng = np.random.default_rng(np.random.SeedSequence([1171,seed,g]))
        for b in range(draws+1):
            if b:
                sample = {v:pilot[v][idx]+E.R.WF.ar_noise(rng,y.shape,*noise[v][:2])
                          for v,y in obs.items()}
                field,_ = E.R.reconstruct(sample,axes,idx)
            else:
                field = pilot
            segment_field = {v:y[segment] for v,y in field.items()}
            for tag,bank in (("local",local_bank),("broad0",broad_bank)):
                mats[tag].append(bank.matrix(segment_field))
            tensor = probe.matrix({v:y[common] for v,y in field.items()}).reshape(9,5,5,len(ids))
            for tag,piece in (("time",tensor[:,0,0]),("x1",tensor[:,:,0]),
                              ("x2",tensor[:,0,:]),("xy",tensor)):
                mats[tag].append(piece.reshape(-1,len(ids)))
        for tag,values in mats.items():
            prepared = E.J.prepare(np.stack(values),draws)
            # For combined banks these per-row Grams give equal block weights.
            # The two correction operations commute with stacking disjoint rows.
            for kind in ("ols","clsm"):
                saved[f"{g}_{tag}_{kind}"] = prepared["point_"+kind]/len(values[0])
            saved[f"{g}_{tag}_point"] = values[0]
    dest = ROOT/"training"/f"{seed}.npz"
    np.savez_compressed(dest,**saved)
    dest.with_suffix(".json").write_text(json.dumps(dict(raw={v:E.R.digest(y) for v,y in raw.items()},
        inputs={str(p):digest(p) for p in paths},sha256=digest(dest))))
    return seed


if __name__=="__main__":
    (ROOT/"training").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for seed in pool.map(training,(*REPLAY,*FRESH)):
            print(seed,flush=True)
