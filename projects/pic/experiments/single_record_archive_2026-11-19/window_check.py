"""Exploratory wave check with a wider validation window and frozen search coefficients."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import sys

from run_split import SEEDS, folder, measurements
from validate import coefficient_matrices, make_banks, nondominated, scores
from transfer import LAWS, LOSS_NAN_VAL, U, labels_of, np, tokens


def temporal_geometry(shape, geometry):
    native = geometry['native']
    m = native['ms'][0]
    cut, stride = native['cut'][0],native['ss'][0]
    def centers(radius):
        return 1+(shape[0]-1-2*cut-2*radius)//stride
    while centers(2*m)>=2:
        m *= 2
    assert centers(m)>=2
    plan = dict(native,ms=[m,*native['ms'][1:]],shape=[centers(m),*native['shape'][1:]])
    return dict(geometry,native=plan)


def run(job):
    seed,part = job
    dest = folder('wave',part)
    with gzip.open(dest/f'archive_{seed}.json.gz','rt') as f:
        archive = json.load(f)
    source = json.loads((dest/f'wave_{seed}_wape.json').read_text())
    primary = json.loads((dest/f'validation_{seed}.json').read_text())
    parts = measurements('wave',seed)
    ax,f = parts[part]
    other = 'late' if part=='early' else 'early'
    bx,y = parts[other]
    f,y = f['u'],y['u']
    _,clean = U.load('wave')
    clean_eval = clean['u'][-len(y):] if other=='late' else clean['u'][:len(y)]
    geo = source['geometry']
    wider = temporal_geometry(y.shape,geo)
    names = archive['labels']
    labels = [labels_of('0 = '+n)[0] for n in names]
    ref,_ = tokens(LAWS['wave'])
    raw_ids = [i for i,r in enumerate(archive['rows']) if max(r['wape'],r['dyad'])<LOSS_NAN_VAL]
    rows = [archive['rows'][i] for i in raw_ids]
    have = np.array([tuple(sorted(labels[j] for j in r['ids']))==ref for r in rows])
    src = make_banks(ax,f,names,geo)(f)
    coefs = coefficient_matrices(rows,src)
    targets = np.array([r['pivot'] for r in rows])
    dyad = np.array([r['dyad'] for r in rows])
    native = make_banks(bx,y,names,geo)(y)
    _,block = scores(native,coefs,targets)
    output = []
    for method, plan in [('native',geo),('wider',wider)]:
        full = native[0] if method=='native' else make_banks(bx,y,names,plan)(y)[0]
        score = np.mean(np.abs(coefs[0]@full.T),axis=1)/np.mean(np.abs(full),axis=0)[targets]
        idx = [names.index(n) for n in primary['coefficients']['terms']]
        q = full[:,idx]
        physical = np.array(primary['coefficients']['known'])
        # Only this diagnostic sees the clean benchmark.
        clean_names = ['1',*primary['coefficients']['terms']]
        q_clean = make_banks(bx,clean_eval,clean_names,plan)(clean_eval)[0][:,1:]
        def error(matrix, vector):
            return float(np.mean(np.abs(matrix@vector))/np.mean(np.abs(matrix[:,0])))
        cases = []
        for which, mask in [('returned',[r['returned'] for r in rows]),('terminal',[r['terminal'] for r in rows]),
                            ('archive',np.ones(len(rows),bool))]:
            for gate in (False,True):
                eligible = np.array(mask) & np.isfinite(score) & (score<=1.)
                if gate:
                    eligible &= np.isfinite(block) & (block<=1.)
                front = nondominated(np.flatnonzero(eligible),score,dyad)
                hit = bool(np.any(have[front]))
                size = len({(tuple(rows[i]['ids']),rows[i]['pivot']) for i in front})
                entry = dict(pool=which,gated=gate,hit=hit,eligible_reference=bool(np.any(have & eligible)),size=size,
                             front=[[raw_ids[i],float(score[i]),float(dyad[i])] for i in front])
                if method=='native':
                    original, = [r for r in primary['outcomes'] if r['pool']==which and r['gated']==gate and r['evaluation']=='heldout']
                    assert hit==original['hit'] and size==original['size']
                cases.append(entry)
        output.append(dict(method=method,ms=plan['native']['ms'],shape=plan['native']['shape'],
                           physical=error(q,physical), fitted=error(q,primary['coefficients']['weights']),
                           clean_physical=error(q_clean,physical),cases=cases))
    (dest/f'window_{seed}.json').write_text(json.dumps(output,allow_nan=False),encoding='utf8')
    print(seed,part,'window check complete',flush=True)


if __name__=='__main__':
    jobs = [(n,p) for n in SEEDS for p in ('early','late')]
    if '--ready' in sys.argv:
        jobs = [(n,p) for n,p in jobs if (folder('wave',p)/f'validation_{n}.json').exists()]
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(run,[(n,p) for n,p in jobs if not (folder('wave',p)/f'window_{n}.json').exists()]))
