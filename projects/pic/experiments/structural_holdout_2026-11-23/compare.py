"""Evaluate complete systems and replay selected equations on common fresh measurements."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from itertools import permutations
import json
from time import sleep
from unittest.mock import patch
import argparse

from experiment import ROOT, alternate, directory, full_directory, live, np, trial, whole_record
from assess_swap import previous, read_archive, system_front, write_csv
from scipy.stats import rankdata


def system_objectives(score, stability, valid, combos):
    eligible = valid & np.isfinite(score) & (score<=1)
    pool = np.flatnonzero(eligible[combos].all(axis=1))
    values = np.column_stack((score[combos[pool]],stability[combos[pool]]))
    return pool, values


def select(score, stability, valid, combos):
    pool, values = system_objectives(score, stability, valid, combos)
    ranks = rankdata(values,axis=0).sum(axis=1) if len(pool) else []
    rest = np.arange(len(pool))
    front = []
    chosen = []
    while len(rest) and len(chosen)<6:
        mask = system_front(values[rest])
        layer = rest[mask]
        if not chosen:
            front = pool[layer].tolist()
        # combos are archived in tuple order. Subsetting retains this tie breaker.
        ordered = sorted(layer,key=lambda i:(ranks[i],pool[i]))
        chosen.extend(pool[i] for i in ordered[:6-len(chosen)])
        rest = rest[~mask]
    return front, chosen


def fresh_discrepancy(problem, seed, archive, combos, leaders, common_plan):
    ids = sorted({int(p) for leader in leaders if leader is not None for p in combos[leader]})
    if not ids:
        return {}
    points = archive['points']
    cols = sorted({j for i in ids for j,_ in points[i][2]})
    indices = {j: n for n,j in enumerate(cols)}
    _,_,(cuts,axes) = live.split(live.R.axes_of(problem))
    terms, opaque = live.R.symbols(problem,[archive['labels'][j] for j in cols])
    bank = live.R.LocalBank(axes,terms,{k:v[cuts] for k,v in opaque.items()},common_plan)
    fresh = problem.noisy(5,seed+44000)
    matrix = bank.matrix({k:v[cuts] for k,v in fresh.items()})
    if 0 in indices:
        assert not archive['labels'][0]
        np.testing.assert_allclose(matrix[:,indices[0]],1.,atol=1e-12)
        matrix[:,indices[0]] = 1.
    errors = {}
    for i in ids:
        _,pivot,pairs,*_ = points[i]
        residual = sum(w*matrix[:,indices[j]] for j,w in pairs)
        denom = np.mean(abs(matrix[:,indices[pivot]]))
        errors[i] = float(np.mean(abs(residual))/denom) if denom else np.inf
    return {k:float(np.mean([errors[p] for p in combos[k]])) for k in leaders if k is not None}


def assess(job):
    name,seed,design = job
    base = trial.folder(name,seed,'candidate')
    source = {'contiguous':base,'alternate':directory(name,seed,'candidate'),'full':full_directory(name,seed,'candidate'),
              'slabs':full_directory(name,seed,'candidate'),'dyad_rows':full_directory(name,seed,'candidate')}[design]
    archive = read_archive(source)
    original = json.loads((base/'assessment.json').read_text())
    common_plan = {k:tuple(v) for k,v in original['windows'].items()}
    problem = live.load_data(name)
    partition = {'contiguous':live.split,'alternate':alternate,'full':whole_record,'slabs':whole_record,
                 'dyad_rows':whole_record}[design]
    assert archive['settings']==original['settings']
    with patch.object(previous,'split',partition):
        train,held,dyad,valid,replay_error = previous.replay(problem,problem.noisy(5,seed),archive)
    assert archive['calls']==live.BUDGET and archive['operator']['full_cost'][-1]>=live.BUDGET
    budget = np.array(archive['operator']['system_cost'])<=live.BUDGET
    combos = np.asarray(archive['systems'],dtype=int)[budget]
    assert len(combos)>0
    rules = (dict(train=train) if design=='full' else
             dict(held=held,train=train,mean=(held+train)/2,worst=np.maximum(held,train)))
    details=[]
    if design=='slabs':
        from slabs import slab_scores
        rules,details=slab_scores(problem,problem.noisy(5,seed),archive,combos)
    if design=='dyad_rows':
        from dyad_discrepancy import native_row_scores
        (halves, multiscale), details = native_row_scores(problem,problem.noisy(5,seed),archive,combos)
        rules = {name+'_'+objective:score for name,score in
                 [('ordinary',train),('halves',halves),('multiscale',multiscale)]
                 for objective in ('only','stability')}
    returns = {rule:select(score,np.zeros_like(dyad) if rule.endswith('_only') else dyad,valid,combos)
               for rule,score in rules.items()}
    leaders = [fixed[0] if len(fixed) else None for _,fixed in returns.values()]
    errors = fresh_discrepancy(problem,seed,archive,combos,leaders,common_plan)

    matches, literal, hit = structural_matches(name, problem, archive, combos)
    rows = []
    for rule,(front,fixed) in returns.items():
        lead = fixed[0] if len(fixed) else None
        row = dict(dataset=name,seed=seed,design=design,rule=rule,assembled=int(matches.any()),available=len(combos),
                   search_seconds=archive['seconds'],
                   front_size=len(front),front_hit=hit(front),six_size=len(fixed),six_hit=hit(fixed),
                   primary_six=int(any(literal[i] for i in fixed)),one_hit=hit(fixed[:1]),
                   fresh_wape=errors.get(lead),replay_error=replay_error,
                   selected=json.dumps([combos[i].tolist() for i in fixed]))
        if details:
            refits = sum(d['refits'] for d in details if not rule.startswith('halves_') or d['slabs']==2)
            row.update(slab_geometry=json.dumps(details),validation_refits=0 if rule=='slab_frozen'
                       or rule.startswith('ordinary_') else refits)
        if design=='contiguous' and rule in ('held','train'):
            tag = 'source' if rule=='train' else 'held'
            known = original['total']['system']
            assert (row['front_size'],row['front_hit'],row['assembled'],len(combos)) == (
                known[tag+'_system_front_size'],known[tag],known['assembled'],known['candidates'])
        rows.append(row)
    return rows


def structural_matches(name, problem, archive, combos):
    allowed, frequencies = previous.registry(name,problem)
    labels = [previous.label_key(t,frequencies) for t in archive['labels']]
    supports = [frozenset(labels[j] for j,_ in p[2]) for p in archive['points']]
    accepted = {perm for system in allowed for perm in permutations(system)}
    primary = set(permutations(allowed[0]))
    patterns = [tuple(supports[i] for i in combo) for combo in combos]
    matches = np.array([p in accepted for p in patterns])
    literal = np.array([p in primary for p in patterns])
    @lru_cache(maxsize=None)
    def control_match(index):
        from ode_identities import identity
        point = archive['points'][combos[index][0]]
        def freeze(term):
            return tuple((n,tuple(map(tuple,pars))) for n,pars in term)
        return identity(tuple(freeze(archive['labels'][j]) for j,_ in point[2]),True)
    def hit(ids):
        return int(any(matches[i] for i in ids) or (name=='ode_simple' and any(control_match(i) for i in ids)))
    return matches, literal, hit


def as_ready(jobs, workers):
    """Assess committed archives as discovery finishes and reuse completed evaluations."""
    pending=list(jobs)
    active={}
    done={}
    def paths(job):
        name,seed,design=job
        locate={'contiguous':trial.folder,'alternate':directory,'full':full_directory,
                'slabs':full_directory,'dyad_rows':full_directory}[design]
        dest=locate(name,seed,'candidate')
        return dest/'archive.json.gz',dest/(design+'_structural_assessment.json')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            for job in pending.copy():
                archive,assessment=paths(job)
                if assessment.exists():
                    done[job]=json.loads(assessment.read_text())
                    pending.remove(job)
                elif archive.exists() and len(active)<workers:
                    active[pool.submit(assess,job)]=job
                    pending.remove(job)
            if active:
                finished,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for future in finished:
                    job=active.pop(future)
                    rows=future.result()
                    _,dest=paths(job)
                    temp=dest.with_suffix('.pending')
                    temp.write_text(json.dumps(rows,allow_nan=False))
                    temp.replace(dest)
                    done[job]=rows
            elif pending:
                sleep(5)
    return [row for job in jobs for row in done[job]]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('design',choices=['contiguous','alternate','full','slabs','dyad_rows'])
    parser.add_argument('cohort',choices=['screen','confirmation'])
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--as-ready',action='store_true')
    args = parser.parse_args()
    seeds = live.SEEDS if args.cohort=='screen' else trial.CONFIRMATION
    jobs = [(name,seed,args.design) for seed in seeds for name in live.DATASETS]
    if args.as_ready:
        rows=as_ready(jobs,args.workers)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [row for result in pool.map(assess,jobs) for row in result]
    write_csv(ROOT/(args.design+'_'+args.cohort+'.csv'), rows)
