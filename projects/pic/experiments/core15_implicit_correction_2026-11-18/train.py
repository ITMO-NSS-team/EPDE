"""Build paired pilot and noise-corrected training moments on frozen geometry."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from panel import ROOT, SYSTEMS, REPLAY, FRESH, records
sys.path.insert(0,str(records.LEGACY))
import experiment as E
import numpy as np


class TrainingMoments:
    def __init__(self,point):
        self.point = point
        self.mean = np.zeros_like(point)
        self.scatter = np.zeros((point.shape[1],point.shape[1]))
        self.count = 0

    def update(self,matrix):
        self.count += 1
        change = matrix-self.mean
        self.mean += change/self.count
        self.scatter += ((self.count-1)/self.count)*E.J.grams(change)

    def corrected(self):
        n = self.count
        unbiased = 2*self.point-self.mean
        return E.J.grams(unbiased)-(1+1/n)*self.scatter/(n-1)


def run(job):
    system,seed = job
    p = E.R.datasets.load(system)
    axes = E.R.axes_of(p)
    _,labels = E.R.panel(p)
    terms,opaque = E.R.symbols(p,labels)
    baseline,old = E.F.baseline(system,0)
    inputs = [baseline,E.F.ROOT/"fit_results"/f"{system}_seed0.json",
              E.M.ROOT/"results"/f"{system}_seed0.json",E.J.ROOT/"results"/f"{system}_seed0.json"]
    local,validation,joint = [json.loads(f.read_text()) for f in inputs[1:]]
    common = slice(*validation["segment"])
    ax = [axes[0][common],*axes[1:]]
    op = {k:y[common] for k,y in opaque.items()}
    raw = p.noisy(5,seed)
    saved = {}
    draws = 128
    for g,group in enumerate(old["groups"]):
        idx = np.asarray(group["indices"])
        obs = {v:y[idx] for v,y in raw.items()}
        noise = {v:E.R.WF.noise_model(y) for v,y in obs.items()}
        pilot,_ = E.R.reconstruct(obs,axes,idx)
        seg,tax,topq,plan = E.Q.mapped_plan(old,g,local["plans"][g]["plan"],axes,opaque)
        bank = E.Q.Bank(tax,terms,topq,plan=plan)
        meta = joint["folds"][g]["banks"]["broad0"]
        assert meta["feasible"]
        broad = E.S.N.V.make_broad(tax,terms,topq,meta["plan"],meta["k"])
        key = validation["folds"][g]["selection"]["q0"]["key"]
        plan = validation["banks"][key]["plan"]
        modes = np.r_[np.arange(5),-np.arange(1,5)]
        freqs = [(modes if j==0 else np.array([0]))*np.pi/(m*(a[1]-a[0]))
                 for j,(a,m) in enumerate(zip(ax,plan["ms"]))]
        time_bank = E.Q.Bank(ax,terms,op,plan=plan,frequencies=freqs)
        def matrices(field):
            part = {v:y[seg] for v,y in field.items()}
            return dict(local=bank.matrix(part),broad=broad.matrix(part),
                        time=time_bank.matrix({v:y[common] for v,y in field.items()}))
        stats = {tag:TrainingMoments(point) for tag,point in matrices(pilot).items()}
        rng = np.random.default_rng(np.random.SeedSequence([1171,seed,g]))
        for _ in range(draws):
            draw = {v:pilot[v][idx]+E.R.WF.ar_noise(rng,y.shape,*noise[v][:2]) for v,y in obs.items()}
            reconstructed,_ = E.R.reconstruct(draw,axes,idx)
            for tag,m in matrices(reconstructed).items():
                stats[tag].update(m)
        for tag,st in stats.items():
            saved[f"{g}_{tag}_point"] = st.point
            saved[f"{g}_{tag}_corrected"] = st.corrected()/len(st.point)
    path = ROOT/"training"/f"{system}_{seed}.npz"
    np.savez_compressed(path,**saved)
    path.with_suffix(".json").write_text(json.dumps(dict(raw={v:E.R.digest(y) for v,y in raw.items()},
        inputs={str(f):records.sha(f) for f in inputs},sha256=records.sha(path))))
    return path.stem


if __name__=="__main__":
    (ROOT/"training").mkdir(exist_ok=True)
    jobs = [(s,n) for n in (*REPLAY,*FRESH) for s in SYSTEMS
            if not (ROOT/"training"/f"{s}_{n}.json").exists()]
    with ProcessPoolExecutor(max_workers=5) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            print(f.result(),flush=True)
