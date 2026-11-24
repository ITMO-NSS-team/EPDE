"""Replay the unmodified baseline builder for an ODE, a 1D PDE and a 2D system."""
from concurrent.futures import ProcessPoolExecutor
import json
from setup_study import ROOT, np
import train
from evaluate import matrices


def replay(system):
    seed=792001
    train.ROOT=ROOT/"baseline_replay"
    (train.ROOT/"training").mkdir(parents=True,exist_ok=True)
    train.run((system,seed))
    return system


if __name__=="__main__":
    systems=("duffing","kdv_cossin","ns")
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(replay,systems))
    out={}
    for s in systems:
        filename=f"{s}_792001.npz"
        maximum=0.
        with np.load(ROOT/"baseline_replay"/"training"/filename) as old, np.load(ROOT/"training"/filename) as z:
            for g in range(3):
                m,G=matrices(old,g,("local","time"))
                for a,b in ((G,z[f"{g}_gram"][0]),(np.linalg.norm(m,axis=0),z[f"{g}_norm"][0])):
                    np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-12)
                    maximum=max(maximum,float(np.max(abs(a-b)/np.maximum(1,abs(a)))))
        out[s]=maximum
    (ROOT/"baseline_replay"/"checks.json").write_text(json.dumps(out,indent=2))
    print(out)
