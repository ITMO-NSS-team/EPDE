"""Freeze width decisions using only the outer training partition."""
import json
from scipy.stats import rankdata
from setup_study import ROOT, SYSTEMS, SEEDS, WIDTHS, corrected_fit, np, records


def decisions(data, fold, supports):
    ix = sorted(set(tuple(p) for p in supports))
    grams, norms, scales, cross = [data[f"{fold}_{key}"] for key in
                                   ("inner_gram","inner_norm","inner_scale","inner_cross")]
    risks = np.full((len(ix),len(WIDTHS)),np.inf)
    for j,support in enumerate(ix):
        ids = list(support)
        idx = np.ix_(ids,ids)
        for w in range(len(WIDTHS)):
            rotation_risks = []
            for r in range(3):
                v,_,status = corrected_fit(grams[r,w][idx],norms[r,w,ids])
                if status != "ok":
                    break
                energy = np.sum((v * scales[r,ids])**2)
                if energy == 0:
                    break
                rotation_risks.append(v @ cross[r][idx] @ v / energy)
            if len(rotation_risks)==3:
                risks[j,w] = np.mean(rotation_risks)
        if len(ids)==1 and np.isfinite(risks[j]).all():
            # A one-column homogeneous direction has no adjustable ratio.
            # Its scale cancels exactly from the common risk for every width.
            k=ids[0]
            risks[j,:]=np.mean(cross[:,k,k]/scales[:,k]**2)
    rank = rankdata(risks,axis=1,method="average").mean(axis=0)
    own = np.argmin(risks,axis=1)
    return dict(shared=int(np.argmin(rank)),mean_ranks=rank.tolist(),
                personal=[int(own[ix.index(tuple(s))]) for s in supports],distinct=len(ix),
                unavailable=int(np.isinf(risks).sum()),all_unavailable=int(np.isinf(risks).all(axis=1).sum()),
                risks=[[None if not np.isfinite(v) else float(v) for v in row] for row in risks],
                distinct_supports=ix)


def freeze(system,seed):
    path=ROOT/"training"/f"{system}_{seed}.npz"
    meta=json.loads(path.with_suffix(".json").read_text())
    assert records.sha(path)==meta["archive"] and meta["protocol"]==records.sha(ROOT/"protocol.txt")
    with np.load(path) as z:
        folds=[decisions(z,g,meta["supports"]) for g in range(3)]
    dest=ROOT/"decisions"/f"{system}_{seed}.json"
    dest.write_text(json.dumps(dict(training=meta["archive"],folds=folds),allow_nan=False))


if __name__=="__main__":
    (ROOT/"decisions").mkdir(exist_ok=True)
    for seed in SEEDS:
        for system in SYSTEMS:
            freeze(system,seed)
