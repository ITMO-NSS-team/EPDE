"""Score recorded native coefficients on independent waveforms and build complete fronts."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import hashlib
import json
from types import SimpleNamespace
import sys

from capture import ROOT, SEEDS
from transfer import R, U, block_specs, labels_of, np, wls, LOSS_NAN_VAL
from scipy.sparse import csr_matrix

RESULTS = ROOT/'results'
REPEATS = 8


def validation_profiles(axes, original):
    t, x = np.meshgrid(*axes, indexing='ij')
    profiles = {'fresh_original': original}
    for order in (3, 4):
        mode = np.sin(order*np.pi*x)*np.cos(.2*order*np.pi*t)
        profiles[f'added_{order}'] = original + .5*original.std()*mode / np.sqrt(np.mean(mode**2))
    k = np.pi*np.array([1, 3, 4])[:,None,None]
    a = np.array([.7, -.55, .42])[:,None,None]
    b = np.array([-.2, .24, .32])[:,None,None]
    standing = np.sin(k*x)
    def mixed(speed):
        return np.sum(standing * (a*np.cos(speed*k*t) + (.2/speed)*b*np.sin(speed*k*t)), axis=0)
    standard = mixed(.2)
    amplitude = original.std()/standard.std()
    offset = original.mean() - amplitude*standard.mean()
    for name, speed in [('independent', .2), ('speed_022', .22), ('speed_040', .4)]:
        profiles[name] = offset + amplitude*mixed(speed)
        np.testing.assert_allclose(profiles[name][0], profiles['independent'][0])
    return profiles


def make_banks(axes, original, names, geometry):
    labels = [labels_of('0 = ' + name)[0] for name in names]
    problem = SimpleNamespace(shape=original.shape, grids=np.meshgrid(*axes,indexing='ij'),
                              data={'u': original}, named_arrays={})
    terms, opaque = R.symbols(problem, labels)
    plan = {k:tuple(v) for k,v in geometry['native'].items()}
    specs = [(tuple(slice(None) for _ in axes), plan), *block_specs(original.shape, plan, geometry['axis'])]
    banks = [(s,R.LocalBank([a[z] for a,z in zip(axes,s)],terms,{k:v[s] for k,v in opaque.items()},p)) for s,p in specs]
    def matrices(field):
        output = []
        for s, bank in banks:
            m = bank.matrix({'u': field[s]})
            np.testing.assert_allclose(m[:,0],1.,atol=1e-12)
            m[:,0] = 1.
            output.append(m)
        return output
    return matrices


def coefficient_matrices(rows, train):
    shape = (len(rows),train[0].shape[1])
    native, block1, block2 = [], [], []
    @lru_cache(maxsize=None)
    def half_fit(ids, pivot, half):
        matrix = train[half][:,ids]
        coef, status = wls(matrix, ids.index(pivot))
        if status!='ok':
            return np.full(len(ids), np.nan)
        return coef / np.linalg.norm(np.linalg.norm(matrix,axis=0)*coef/np.sqrt(len(matrix)))
    row_indices, col_indices = [], []
    for i, r in enumerate(rows):
        ids = tuple(r['ids'])
        assert len(ids)==len(set(ids)) and r['weights'][ids.index(r['pivot'])]==-1.
        row_indices.extend([i]*len(ids))
        col_indices.extend(ids)
        native.extend(r['weights'])
        block1.extend(half_fit(ids,r['pivot'],1))
        block2.extend(half_fit(ids,r['pivot'],2))
    return [csr_matrix((values, (row_indices, col_indices)),shape=shape) for values in (native, block1, block2)]


def scores(matrices, coefs, targets):
    full, first, second = matrices
    native, a, b = coefs
    with np.errstate(divide='ignore',invalid='ignore'):
        wape = np.mean(np.abs(native @ full.T),axis=1)/np.mean(np.abs(full),axis=0)[targets]
        block = .5*(np.mean((a @ second.T)**2,axis=1) + np.mean((b @ first.T)**2,axis=1))
    return wape, block


def nondominated(ids, score, dyad):
    """Minimize both objectives. Preserve all exact ties at a nondominated point."""
    if len(ids)==0:
        return ids
    order = ids[np.lexsort((dyad[ids],score[ids]))]
    starts = np.r_[0,1+np.flatnonzero(np.diff(score[order])!=0)]
    group = np.cumsum(np.isin(np.arange(len(order)), starts))-1
    minimum = dyad[order[starts]]
    previous = np.r_[np.inf,np.minimum.accumulate(minimum)[:-1]]
    keep = (dyad[order]==minimum[group]) & (dyad[order]<previous[group])
    return order[keep]


def validate(seed):
    with gzip.open(RESULTS/f'archive_{seed}.json.gz','rt') as f:
        arc = json.load(f)
    run = json.loads((RESULTS/f'wave_{seed}_wape.json').read_text())
    axes, clean = U.load('wave')
    field = clean['u']
    rng = np.random.default_rng([seed,list(U.TRUTH).index('wave')])
    train_field = field + .05*field.std()*rng.standard_normal(field.shape)
    assert hashlib.sha256(train_field.tobytes()).hexdigest()==run['raw_sha256']['u']
    matrices = make_banks(axes,field,arc['labels'],run['geometry'])
    train = matrices(train_field)
    raw_ids, rows = zip(*[(i,r) for i,r in enumerate(arc['rows']) if max(r['dyad'],r['wape'])<LOSS_NAN_VAL])
    coefs = coefficient_matrices(rows,train)
    targets = np.array([r['pivot'] for r in rows])
    dyad = np.array([r['dyad'] for r in rows])
    support_keys = [(tuple(r['ids']),r['pivot']) for r in rows]
    terms = {'d^2u/dx0^2{power: 1.0}', 'd^2u/dx1^2{power: 1.0}'}
    reference = np.array([{arc['labels'][i] for i in r['ids']}==terms and len(r['ids'])==2 for r in rows])
    source, source_block = scores(train, coefs, targets)
    original_scores = [r['wape'] for r in rows]
    np.testing.assert_allclose(source, original_scores, rtol=1e-6, atol=2e-9)
    pools = {'returned':np.array([r['returned'] for r in rows]),
             'terminal':np.array([r['terminal'] for r in rows]), 'archive':np.ones(len(rows),bool)}
    primary = nondominated(np.flatnonzero(pools['terminal']), np.array(original_scores), dyad)
    assert {support_keys[i] for i in primary}=={support_keys[i] for i in np.flatnonzero(pools['returned'])}
    output = []
    def assess(profile, repeat, score, block):
        for pool, members in pools.items():
            for gated in (False,True):
                valid = members & np.isfinite(score) & (score<=1.)
                if gated:
                    valid &= np.isfinite(block) & (block<=1.)
                front = nondominated(np.flatnonzero(valid), score, dyad)
                best = np.flatnonzero(reference & members)
                passing = np.any(valid[best])
                points = [[raw_ids[i],float(score[i]),float(block[i]) if np.isfinite(block[i]) else None] for i in front]
                output.append(dict(profile=profile, repeat=repeat, pool=pool, gated=gated, candidates=int(members.sum()),
                                   eligible=int(valid.sum()), reference_present=bool(len(best)), reference_eligible=bool(passing),
                                   exact_hit=bool(np.any(reference[front])), size=len({support_keys[i] for i in front}),
                                   supports=len({support_keys[i][0] for i in front}), front=points,
                                   reference_points=[[raw_ids[i],float(score[i]),float(block[i])] for i in best]))
    assess('training',-1,np.array(original_scores),source_block)
    profiles = validation_profiles(axes,field)
    for repeat in range(REPEATS):
        noise = .05*field.std()*np.random.default_rng([511963,seed,repeat]).standard_normal(field.shape)
        for profile, clean_profile in profiles.items():
            assess(profile,repeat,*scores(matrices(clean_profile+noise),coefs,targets))
    result = dict(seed=seed, valid=len(rows), scored=len(arc['rows']), unique_valid=len(set(support_keys)),
                  max_wape_replay_error=float(np.max(np.abs(source-original_scores))), results=output)
    dest = RESULTS/f'validation_{seed}.json.gz'
    with gzip.open(dest.with_suffix('.pending'),'wt',encoding='utf8',compresslevel=4) as f:
        json.dump(result,f,allow_nan=False,separators=(',',':'))
    dest.with_suffix('.pending').replace(dest)
    print(seed, len(rows), 'eligible points evaluated', flush=True)


if __name__=='__main__':
    todo = [int(s) for s in sys.argv[1:]] or list(SEEDS)
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(validate, todo))
