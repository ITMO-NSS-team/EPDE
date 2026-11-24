"""Replay held-out WAPE and count support generation and front membership."""
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations

from live import ARMS, BUDGET, DATASETS, ROOT, SEEDS, U, R, LOSS_NAN_VAL, folder, gzip, json, load_data, np, split, sys
from scipy.sparse import csr_matrix
from validate import nondominated


def label_key(label, frequencies):
    keys = []
    for name, raw_params in label:
        params = dict(raw_params)
        text = name
        if params['power'] != 1:
            text += '^%g' % params['power']
        if 'freq' in params:
            freq = params['freq']
            matched = [f for f in frequencies if abs(freq-f) <= .01*abs(f)]
            text += '~%.12g' % (min(matched, key=lambda f: abs(freq-f)) if matched else freq)
        if 'dim' in params:
            text += '@%d' % int(params['dim'])
        keys.append(text)
    return '*'.join(sorted(keys)) if keys else '1'


def registry(name, problem):
    """Each allowed system is a tuple of distinct target-independent law supports."""
    forms = []
    frequencies = set()
    for row in problem.truth_systems:
        parsed = []
        for equation in row:
            target, rhs, _ = R.parse(equation)
            support = [target,*rhs]
            parsed.append(support)
            frequencies.update(dict(p)['freq'] for t in support for _,p in t if 'freq' in dict(p))
        forms.append(parsed)
    allowed = [tuple(frozenset(label_key(t,frequencies) for t in support) for support in row) for row in forms]
    if name == 'ode_simple':
        frequencies.add(1.)
        for _,target,terms in U.truth_forms('ode')['u']:
            allowed.append((frozenset((target,*terms)),))
    if name == 'ns':
        # Two equivalent rewrites of the convective term using continuity.
        forms = {k:frozenset([t,*xs]) for values in U.truth_forms('ns').values() for k,t,xs in values}
        assert allowed[0] == (forms['x_mom'],forms['y_mom'],forms['continuity'])
        allowed.extend((forms[x],forms[y],forms['continuity']) for x in ('x_mom','x_mom_cont')
                       for y in ('y_mom','y_mom_cont'))
    return allowed, frequencies


def replay(problem, raw, archive):
    axis,a,b = split(R.axes_of(problem))
    assert archive['axis'] == axis and archive['calls'] == BUDGET
    assert archive['train_hashes'] == {k:R.digest(v[a[0]]) for k,v in raw.items()}
    plan = {k:tuple(v) for k,v in archive['plan'].items()}
    terms, opaque = R.symbols(problem,archive['labels'])
    points = archive['points']
    ii,jj,ww = [],[],[]
    for i,(_,_,pairs,_,_,_) in enumerate(points):
        for j,w in pairs:
            ii.append(i)
            jj.append(j)
            ww.append(w)
    coef = csr_matrix((ww,(ii,jj)),shape=(len(points),len(terms)))
    pivot = np.array([p[1] for p in points])
    def score(side):
        cuts,axes = side
        bank = R.LocalBank(axes,terms,{k:v[cuts] for k,v in opaque.items()},plan)
        fields = {k:v[cuts] for k,v in raw.items()}
        mat = bank.matrix(fields)
        np.testing.assert_allclose(mat[:,0],1.,atol=1.e-12)
        mat[:,0] = 1.
        scale = np.abs(mat).mean(axis=0)[pivot]
        chunks = []
        for start in range(0,len(points),256):
            selection = slice(start,start+256)
            with np.errstate(divide='ignore',invalid='ignore'):
                chunks.append(np.abs(coef[selection] @ mat.T).mean(axis=1)/scale[selection])
        return np.concatenate(chunks)
    source = np.array([p[3] for p in points])
    dyad = np.array([p[4] for p in points])
    valid = (source<LOSS_NAN_VAL) & (dyad<LOSS_NAN_VAL)
    reconstructed = score(a)
    np.testing.assert_allclose(reconstructed[valid],source[valid],rtol=1e-6,atol=2e-9)
    return source, score(b), dyad, valid, float(np.max(np.abs(reconstructed[valid]-source[valid])))


def assess(job):
    name,seed,arm = job
    dest = folder(*job)
    with gzip.open(dest/'archive.json.gz','rt') as stream:
        rec = json.load(stream)
    problem = load_data(name)
    source, held, dyad, valid, error = replay(problem,problem.noisy(5,seed),rec)
    allowed, freq = registry(name,problem)
    keys = [label_key(t,freq) for t in rec['labels']]
    points = rec['points']
    supports = [frozenset(keys[i] for i,w in p[2]) for p in points]
    slots = np.array([p[0] for p in points])
    first = np.array([p[5] for p in points])
    variables = rec['variables']
    cohorts = {'source':source,'held':held}
    fronts = {}
    for key, score in cohorts.items():
        mask = np.zeros(len(points),dtype=bool)
        for v in variables:
            ids = np.flatnonzero(valid & (slots==v) & np.isfinite(score) & (score<=1.))
            mask[nondominated(ids,score,dyad)] = True
        fronts[key] = mask
    laws = []
    for i in range(len(allowed[0])):
        alternatives = {row[i] for row in allowed}
        match = np.array([key in alternatives for key in supports])
        exact = np.array([key == allowed[0][i] for key in supports])
        def fate(key):
            scores = cohorts[key]
            if not match.any():
                return 'never scored'
            if not np.any(match & valid):
                return 'failed native eligibility'
            if not np.any(match & valid & np.isfinite(scores) & (scores <= 1.)):
                return 'outside WAPE cutoff'
            return 'on front' if np.any(match & fronts[key]) else 'dominated'
        laws.append(dict(law=i, generated=int(match.any()), exact_generated=int(exact.any()),
                         first=int(first[match].min()) if match.any() else None,
                         **{k:int((match & mask).any()) for k,mask in fronts.items()},
                         **{k+'_fate':fate(k) for k in fronts},
                         **{'exact_'+k:int((exact & mask).any()) for k,mask in fronts.items()}))
    combos = np.array(rec['systems'],dtype=int).reshape(-1,len(variables))
    legal = {perm for system in allowed for perm in permutations(system)}
    true_combo = np.array([tuple(supports[i] for i in combo) in legal for combo in combos],dtype=bool)
    system = {'assembled':int(true_combo.any()), 'candidates':len(combos)}
    for key,score in cohorts.items():
        vals = np.column_stack((score[combos],dyad[combos]))
        eligible = np.isfinite(vals).all(axis=1) & (score[combos]<=1.).all(axis=1)
        members = vals[eligible]
        system[key] = int(any(not np.any(np.all(members<=r,axis=1) & np.any(members<r,axis=1))
                              for r in vals[true_combo & eligible]))
    hits = dict(system=name,seed=seed,arm=arm,laws=laws,joint=system,replay_error=error,
                source_size=int(fronts['source'].sum()),held_size=int(fronts['held'].sum()),
                calls=rec['calls'],seconds=rec['seconds'],gain_columns=rec['operator']['gain_columns'],
                guided=rec['operator']['counters'].get('guided_additions',0),
                opportunities=rec['operator']['counters']['ordinary_add_opportunities'],
                trig=rec['trig'],plan=rec['plan'],train_hashes=rec['train_hashes'])
    if name == 'ode_simple':
        from ode_identities import identity
        def factors(i):
            return tuple(tuple((n,tuple(tuple(v) for v in args)) for n,args in rec['labels'][j]) for j,w in points[i][2])
        hits['oscillator_identity'] = {k:dict(literal=any(identity(factors(i),False) for i in np.flatnonzero(mask)),
                                            snapped=any(identity(factors(i),True) for i in np.flatnonzero(mask)))
                                      for k,mask in fronts.items()}
    R.dump(dest/'assessment.json',hits)
    return hits


def collect():
    rows, runs = [], []
    for name in DATASETS:
        for seed in SEEDS:
            pair = [json.loads((folder(name,seed,a)/'assessment.json').read_text()) for a in ARMS]
            for k in ('train_hashes','plan','calls','trig'):
                assert pair[0][k] == pair[1][k]
            for r in pair:
                ids = dict(system=name,seed=seed,arm=r['arm'])
                rows.extend(dict(**ids,**law) for law in r['laws'])
                runs.append(dict(**ids,**r['joint'],**{k:r[k] for k in ('calls','seconds','replay_error',
                                 'source_size','held_size','gain_columns','guided','opportunities','trig')}))
    for filename, data in [('laws.csv',rows),('systems.csv',runs)]:
        with (ROOT/filename).open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=data[0])
            writer.writeheader()
            writer.writerows(data)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh',action='store_true')
    args = parser.parse_args()
    jobs = [(s,n,a) for n in SEEDS for s in DATASETS for a in ARMS]
    todo = [j for j in jobs if (folder(*j)/'archive.json.gz').exists()
            and (args.refresh or not (folder(*j)/'assessment.json').exists())]
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(assess,todo))
    if all((folder(*j)/'assessment.json').exists() for j in jobs):
        collect()
