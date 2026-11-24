"""Replay the live fronts and insert the canonical law into their union on independent noise."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

from search import ROOT, ARMS, SEEDS, SYSTEMS, U, np
from blocked import block_specs, fit, wls
from study import R
from recheck import labels_of
from failure_audit import REFERENCES
from epde.operators.common.objectives import LOSS_NAN_VAL
from epde.operators.common.survival import dyad_scores

N_FRESH = 16
LAWS = dict(REFERENCES, pde_compound='1 * du/dx1{power: 2.0} + 1 * u{power: 1.0} * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}')


def tokens(equation):
    target, = labels_of('0 = '+equation.split('=')[1])
    return labels_of(equation), target


def native_dyad(m,pivot,shape):
    x = np.delete(m,pivot,axis=1)
    return float(np.sum(dyad_scores(x,m[:,pivot],np.ones(len(m)),shape,fit_intercept=False)))


def run_pair(job,axis_override=None,validation_fields=None):
    system,seed = job
    rr = {arm:json.loads((ROOT/'results'/f'{system}_{seed}_{arm}.json').read_text()) for arm in ARMS}
    base = rr['wape']
    assert all(r['settings']==base['settings'] and r['raw_sha256']==base['raw_sha256'] and r['geometry']==base['geometry']
               and r['epochs']==10 and r['vwsr_calls']==0 and r['calls']>0 for r in rr.values())
    axes, clean = U.load(system)
    index = list(U.TRUTH).index(system)
    def noisy(entropy):
        rng = np.random.default_rng(entropy)
        return {v: y + .05*y.std()*rng.standard_normal(y.shape) for v,y in clean.items()}
    original = noisy([seed,index])
    assert hashlib.sha256(original['u'].tobytes()).hexdigest()==base['raw_sha256']['u']
    cfg = R.active_config()
    R.set_active_config(replace(cfg,search_space=replace(cfg.search_space,system_rps_mode='plain',equation_terms_max_number=7)))
    ref,physical_target = tokens(LAWS[system])
    variants = {}
    for arm,r in rr.items():
        for row in r['front']:
            support, target = tokens(row['equation'])
            variant = variants.setdefault((support,target),dict(text=row['equation'],recorded=[]))
            variant['recorded'].append((arm,row['objectives']))
    for term in ref:
        if len(term)==1 and '/dx' in term[0][0] and dict(term[0][1])['power']==1:
            variants.setdefault((ref,term),dict(text=None,recorded=[]))
    labels = sorted(set().union(*(support for support,_ in variants)))
    idx = {label:i for i,label in enumerate(labels)}
    grids = np.meshgrid(*axes,indexing='ij')
    problem = SimpleNamespace(shape=original['u'].shape,grids=grids,data=original,named_arrays={})
    terms,opaque = R.symbols(problem,labels)
    plan = {k:tuple(v) for k,v in base['geometry']['native'].items()}
    split_axis = base['geometry']['axis'] if axis_override is None else axis_override
    blocks = block_specs(problem.shape,plan,split_axis)
    assert blocks
    slices = [(tuple(slice(None) for _ in axes),plan),*blocks]
    banks = [(s,R.LocalBank([axis[z] for axis,z in zip(axes,s)],terms,{k:v[s] for k,v in opaque.items()},p)) for s,p in slices]
    def matrices(raw):
        ms = []
        for s,b in banks:
            m = b.matrix({k:v[s] for k,v in raw.items()})
            if () in idx:
                np.testing.assert_allclose(m[:,idx[()]],1,atol=1e-12)
                m[:,idx[()]]=1.
            ms.append(m)
        return ms
    train = matrices(original)
    validation = (noisy([984102,seed,index,j]) for j in range(N_FRESH)) if validation_fields is None else validation_fields
    checks = [train, *(matrices(fields) for fields in validation)]
    result = []
    worst = 0.
    for (support,target), v in variants.items():
        ids = [idx[t] for t in support]
        pivot = support.index(target)
        full = train[0][:,ids]
        native, status = wls(full,pivot)
        assert status=='ok'
        native = -native/native[pivot]
        dyad = native_dyad(full,pivot,plan['shape'])
        scores = {'wape':[float(np.sum(np.abs(m[0][:,ids]@native))/np.sum(np.abs(m[0][:,ids[pivot]]))) for m in checks]}
        for arm in ('wls','tls'):
            pieces = []
            for g in (1,2):
                m = train[g][:,ids]
                coef, status = wls(m,pivot) if arm=='wls' else fit(m)
                if status!='ok':
                    break
                scale = np.linalg.norm(np.linalg.norm(m,axis=0)*coef)/np.sqrt(len(m))
                pieces.append([np.mean((mats[3-g][:,ids]@(coef/scale))**2) for mats in checks])
            scores[arm] = (np.mean(pieces,axis=0).tolist() if len(pieces)==2 else [float(LOSS_NAN_VAL)]*len(checks))
        for arm,obj in v['recorded']:
            if max(obj)<LOSS_NAN_VAL:
                if arm=='wape' or split_axis==base['geometry']['axis']:
                    diff = abs(obj[0]-scores[arm][0])/max(1.,abs(obj[0]))
                    worst = max(worst,diff)
                    np.testing.assert_allclose(scores[arm][0],obj[0],rtol=2e-6,atol=2e-9)
                np.testing.assert_allclose(dyad,obj[1],rtol=2e-4,atol=3e-7)
        amp = (np.abs(native)*np.linalg.norm(full,axis=0)).sum()/np.linalg.norm(full[:,pivot])-1
        result.append(dict(reference=support==ref,target=target,equation=v['text'],source=[a for a,o in v['recorded']],
                           eligible_amplification=bool(amp<=cfg.search_space.rps_amplification_cap),dyad=dyad,scores=scores))
    expected = len(variants)
    assert len(result)==expected
    survival = {}
    for arm in ARMS:
        won = []
        for i in range(len(checks)):
            pool = [r for r in result if r['eligible_amplification'] and r['scores'][arm][i]<=1.]
            won.append(any(not any(c['scores'][arm][i]<=r['scores'][arm][i] and c['dyad']<=r['dyad'] and
                                   (c['scores'][arm][i]<r['scores'][arm][i] or c['dyad']<r['dyad']) for c in pool)
                           for r in pool if r['reference']))
        survival[arm] = won
    physical_ids = [idx[t] for t in ref]
    i = ref.index(physical_target)
    m = train[0][:,physical_ids]
    native,_ = wls(m,i)
    tls,_ = fit(m)
    coefs = dict(labels=ref,pivot=physical_target,wls=(-native/native[i]).tolist(),tls=(-tls/tls[i]).tolist())
    return dict(system=system,seed=seed,variants=result,survival=survival,reference_coefficients=coefs,max_score_replay_error=worst)


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run_pair,[(s,n) for s in SYSTEMS for n in SEEDS]))
    (ROOT/'transfer.json').write_text(json.dumps(rows,allow_nan=False))
    for s in SYSTEMS:
        print(s,{a:(sum(r['survival'][a][0] for r in rows if r['system']==s),
                    sum(sum(r['survival'][a][1:]) for r in rows if r['system']==s)) for a in ARMS})
