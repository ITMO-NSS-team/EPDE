"""Build three full-width fits and nested, training-only selection measurements."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
from setup_study import ROOT, LEGACY, SYSTEMS, SEEDS, WIDTHS, TrainingMoments, E, np, records


def subfield(field, seg):
    return {k: v[seg] for k, v in field.items()}


def plans(axes, local):
    capacity = np.array([(len(a)-1-2*c)//2 for a, c in zip(axes, local["cut"])])
    small = np.minimum(local["ms"], capacity)
    assert np.all(small > 0)
    large = np.maximum(small, (3*capacity)//4)
    if len(axes) > 1:
        large[0] = small[0]
    middle = np.rint(np.sqrt(small*large)).astype(int)
    result = []
    for radius in (small, middle, large):
        bounds = [1+(len(a)-1-2*(c+m))//s for a, c, m, s in zip(axes,local["cut"],radius,local["ss"])]
        result.append(dict(local, ms=tuple(map(int,radius)), shape=tuple(map(int,bounds))))
    return result


def temporal(axes, terms, opaque, plan):
    freq = [(np.r_[0:5, -1:-5:-1] if i == 0 else np.zeros(1)) * np.pi/(m*(a[1]-a[0]))
            for i,(m,a) in enumerate(zip(plan["ms"],axes))]
    return E.Q.Bank(axes,terms,opaque,plan=plan,frequencies=freq)


@lru_cache(maxsize=1)
def geometry(system):
    p = E.R.datasets.load(system)
    axes = E.R.axes_of(p)
    cases, labels = E.R.panel(p)
    terms, opaque = E.R.symbols(p, labels)
    base_path, old = E.F.baseline(system,0)
    paths = [E.F.ROOT/"fit_results"/f"{system}_seed0.json", E.M.ROOT/"results"/f"{system}_seed0.json"]
    local, val = [json.loads(path.read_text()) for path in paths]
    folds = []
    for g, group in enumerate(old["groups"]):
        idx = np.array(group["indices"])
        seg, ax, op, plan = E.Q.mapped_plan(old,g,local["plans"][g]["plan"],axes,opaque)
        scales = plans(ax,plan)
        # The original bank retains its exact grid, including any deliberately unused tail.
        scales[0] = plan
        banks = [E.Q.Bank(ax,terms,op,plan=pl) for pl in scales]
        common = slice(*val["segment"])
        key = val["folds"][g]["selection"]["q0"]["key"]
        tplan = val["banks"][key]["plan"]
        time = temporal([axes[0][common],*axes[1:]],terms,subfield(opaque,common),tplan)
        inner = [idx[j::3] for j in range(3)]
        assert sum(map(len,inner)) == len(set(np.concatenate(inner))) == len(idx)
        interior = slice(max(z[0] for z in inner),min(z[-1] for z in inner)+1)
        iax = [axes[0][interior],*axes[1:]]
        iop = subfield(opaque,interior)
        iplans = plans(iax,plan)
        ibanks = [E.Q.Bank(iax,terms,iop,plan=pl) for pl in iplans]
        cap = (len(iax[0])-1-2*tplan["cut"][0])//2
        tms = [min(tplan["ms"][0],cap),*tplan["ms"][1:]]
        assert tms[0]>0 and tplan["shape"][0]==1
        itime = temporal(iax,terms,iop,dict(tplan,ms=tms))
        folds.append((idx,seg,banks,common,time,inner,interior,iax,ibanks,itime,scales))
    assert sum(len(f[0]) for f in folds)==len(set(np.concatenate([f[0] for f in folds])))==len(axes[0])
    supports = [sorted({c["yi"], *c["xi"]}) for c in cases]
    return p, axes, folds, supports, {str(a):records.sha(a) for a in [base_path,*paths]}


def stack(a,b):
    return np.vstack([v / np.sqrt(len(v)) for v in (a,b)])


def nested(axes,raw,parts,interior,iax,banks,time):
    grams, norms, anchor_scale, measurements = [], [], [], []
    for idx in parts:
        obs = subfield(raw,idx)
        model = {k:E.R.WF.noise_model(v) for k,v in obs.items()}
        pilot,_ = E.R.reconstruct(obs,axes,idx)
        signal = subfield(pilot,interior)
        mats = [bank.matrix(signal) for bank in banks]
        temporal_matrix = time.matrix(signal)
        mixed = [stack(m,temporal_matrix) for m in mats]
        grams.append([E.J.grams(m) for m in mixed])
        norms.append([np.linalg.norm(m,axis=0) for m in mixed])
        anchor_scale.append(np.linalg.norm(stack(mats[0],mats[-1]),axis=0))
        mapping = E.O.interpolation(axes[0],idx)[interior]
        cov = E.O.Covariance(iax,mapping,model)
        measured = E.O.map_values(mapping,obs)
        measurements.append(stack(*[E.O.matrix(bank,measured,cov) for bank in (banks[0],banks[-1])]))
    cross = []
    for i in range(3):
        l,r = [m for j,m in enumerate(measurements) if i!=j]
        cross.append(l.T@r)
    return dict(inner_gram=grams,inner_norm=norms,inner_scale=anchor_scale,inner_cross=cross)


def estimate(system,seed):
    p, axes, folds, supports, sources = geometry(system)
    path = ROOT/"training"/f"{system}_{seed}.npz"
    if path.with_suffix(".json").exists():
        return
    raw = p.noisy(5,seed)
    result = {}
    for g,(idx,seg,banks,common,time,parts,interior,iax,ibanks,itime,_) in enumerate(folds):
        values = nested(axes,raw,parts,interior,iax,ibanks,itime)
        obs = subfield(raw,idx)
        model = {k:E.R.WF.noise_model(v) for k,v in obs.items()}
        pilot,_ = E.R.reconstruct(obs,axes,idx)
        def matrices(field):
            local = subfield(field,seg)
            return [bank.matrix(local) for bank in banks]+[time.matrix(subfield(field,common))]
        stats = [TrainingMoments(v) for v in matrices(pilot)]
        rng = np.random.default_rng(np.random.SeedSequence([1171,seed,g]))
        for _ in range(128):
            draw = {k:pilot[k][idx]+E.R.WF.ar_noise(rng,v.shape,*model[k][:2]) for k,v in obs.items()}
            rec,_ = E.R.reconstruct(draw,axes,idx)
            for st,m in zip(stats,matrices(rec)):
                st.update(m)
        gr = [st.corrected()/len(st.point) for st in stats]
        values["gram"] = [m+gr[-1] for m in gr[:-1]]
        values["norm"] = [np.linalg.norm(stack(st.point,stats[-1].point),axis=0) for st in stats[:-1]]
        for k,v in values.items():
            result[f"{g}_{k}"] = v
    np.savez_compressed(path,**result)
    metadata = dict(raw={k:E.R.digest(v) for k,v in raw.items()}, archive=records.sha(path),supports=supports,
                    plans=[f[-1] for f in folds],inputs=sources, protocol=records.sha(ROOT/"protocol.txt"))
    path.with_suffix(".json").write_text(json.dumps(metadata, default=lambda x:x.tolist()))


def batch(system):
    for seed in SEEDS:
        estimate(system,seed)
        print(system,seed,flush=True)


if __name__ == "__main__":
    (ROOT/"training").mkdir(exist_ok=True)
    heavy = ["ks","ns","kdv","lorenz"]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(batch,s) for s in heavy+[k for k in SYSTEMS if k not in heavy]]):
            f.result()
