"""Score each excluded slab and aggregate structures discovered independently across folds."""
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
import json
from scipy.stats import rankdata
from blocked_search import fold_path, slabs
from blocked_control import dest as control_path
from compare import select
from experiment import ROOT, live, np, trial
from assess_swap import previous, read_archive, write_csv
from scipy.sparse import csr_matrix


def structural_key(label):
    factors=[]
    for name,params in label:
        ordinary=tuple((k,v) for k,v in params if k!='freq')
        frequencies=tuple(v for k,v in params if k=='freq')
        factors.append(((name,ordinary),frequencies))
    factors.sort()
    return tuple(k for k,f in factors),tuple(v for k,f in factors for v in f)


class ObservedFamilies:
    def __init__(self):
        self.anchors=defaultdict(list)
        self.members=[]

    def find(self, pattern, frequencies):
        for anchor, key in self.anchors[pattern]:
            if all(abs(x-y)<=.01*abs(x) for x,y in zip(anchor,frequencies,strict=True)):
                return key
        key=len(self.members)
        self.members.append([])
        self.anchors[pattern].append((frequencies,key))
        return key


def correct_systems(archive, combos, problem):
    allowed,freq=previous.registry(problem.name,problem)
    legal={p for system in allowed for p in permutations(system)}
    labels=[previous.label_key(term,freq) for term in archive['labels']]
    supports=[frozenset(labels[j] for j,w in p[2]) for p in archive['points']]
    def hit(index):
        combo=combos[index]
        if tuple(supports[i] for i in combo) in legal:
            return True
        if problem.name=='ode_simple':
            from ode_identities import identity
            factors=tuple(tuple((n,tuple(map(tuple,pars))) for n,pars in archive['labels'][j])
                          for j,w in archive['points'][combo[0]][2])
            return identity(factors,True)
        return False
    return hit


def matrices(archive,problem,raw,parts):
    plan={k:tuple(v) for k,v in archive['plan'].items()}
    terms,extra=live.R.symbols(problem,archive['labels'])
    points=archive['points']
    r,c,v=[],[],[]
    for i,p in enumerate(points):
        for j,w in p[2]:
            r.append(i)
            c.append(j)
            v.append(w)
    coef=csr_matrix((v,(r,c)),shape=(len(points),len(terms)))
    pivots=[p[1] for p in points]
    scores=[]
    for cuts,axes in parts:
        bank=live.R.LocalBank(axes,terms,{k:v[cuts] for k,v in extra.items()},plan)
        m=bank.matrix({k:v[cuts] for k,v in raw.items()})
        np.testing.assert_allclose(m[:,0],1.,atol=1e-12)
        m[:,0]=1.
        den=np.abs(m).mean(axis=0)[pivots]
        err=np.abs(coef@m.T).mean(axis=1)
        scores.append(np.divide(err,den,out=np.full_like(den,np.inf),where=den>0))
    return scores


def evaluate(job):
    name,seed=job
    p=live.load_data(name)
    raw=p.noisy(5,seed)
    _,parts=slabs(live.R.axes_of(p))
    hashes=[{k:live.R.digest(v[cut]) for k,v in raw.items()} for cut,axes in parts]
    families=ObservedFamilies()
    fold_rows=[]
    truth=[]
    for f in range(4):
        dest=fold_path(name,seed,f)
        meta=json.loads((dest/'fold.json').read_text())
        training=meta['training']
        assert meta['sample_hashes']==hashes and set(training)==set(range(4))-{f}
        status=meta.get('status','completed')
        if status!='completed':
            assert 'no feasible weak-form window' in status
            truth.append(None)
            fold_rows.append(dict(dataset=name,seed=seed,fold=f,status=status,assembled=0,train_six=0,train_one=0,
                                  out_of_fold_wape=None,held_six=0,held_one=0,handovers=0))
            continue
        a=read_archive(dest)
        assert a['train_hashes']==hashes[training[0]] and a['calls']==750
        scores=matrices(a,p,raw,parts)
        held=scores[f]
        train=np.array([q[3] for q in a['points']])
        dyad=np.array([q[4] for q in a['points']])
        valid=(train<live.LOSS_NAN_VAL)&(dyad<live.LOSS_NAN_VAL)
        np.testing.assert_allclose(np.mean([scores[i] for i in training],axis=0)[valid],train[valid],rtol=1e-6,atol=2e-9)
        combos=np.array(a['systems'],dtype=int)
        hit=correct_systems(a,combos,p)
        truth.append(hit)
        _,train_six=select(train,dyad,valid,combos)
        _,held_six=select(held,dyad,valid,combos)
        fold_rows.append(dict(dataset=name,seed=seed,fold=f,status=status,
                              assembled=int(any(map(hit,range(len(combos))))) if name!='ode_simple' else None,
                              train_six=int(any(map(hit,train_six))),
                              train_one=int(bool(train_six) and hit(train_six[0])),
                              out_of_fold_wape=float(held[combos[train_six[0]]].mean()) if train_six else None,
                              held_six=int(any(map(hit,held_six))),held_one=int(bool(held_six) and hit(held_six[0])),
                              handovers=len(combos)))
        labs=[structural_key(x) for x in a['labels']]
        keys=[]
        for point in a['points']:
            row=sorted(labs[j] for j,w in point[2])
            keys.append((tuple(k for k,fs in row),tuple(w for k,fs in row for w in fs)))
        pool=np.flatnonzero((valid & (train<=1.))[combos].all(axis=1))
        values=np.column_stack((train[combos[pool]],dyad[combos[pool]]))
        tr=rankdata(values,axis=0).mean(axis=1)/max(1,len(pool))
        representative={}
        for i,rank in zip(pool,tr):
            joint=[keys[k] for k in combos[i]]
            idx=families.find(tuple(key for key,fs in joint),tuple(w for key,fs in joint for w in fs))
            representative[idx]=min(representative.get(idx,(np.inf,np.inf)),(rank,int(i)))
        candidates=[(g,rank,i) for g,(rank,i) in representative.items() if np.all(held[combos[i]]<=1.)]
        if candidates:
            ids=[i for g,r,i in candidates]
            vals=np.column_stack((held[combos[ids]],dyad[combos[ids]]))
            hr=rankdata(vals,axis=0).mean(axis=1)/len(ids)
            for (g,trank,i),hrank in zip(candidates,hr):
                families.members[g].append((f,i,trank,hrank))
    observed=[(i,m) for i,m in enumerate(families.members) if m]
    ordered=sorted(observed,key=lambda item:(-len(item[1]),np.mean([m[3] for m in item[1]]),item[0]))[:6]
    returns=[]
    for rank,(g,members) in enumerate(ordered,1):
        f,i,_,_=min(members,key=lambda r:(r[2],r[0],r[1]))
        returns.append(dict(dataset=name,seed=seed,rank=rank,fold=f,system_index=i,coverage=len(members),
                            correct=int(truth[f](i))))
    result=dict(dataset=name,seed=seed,consensus_six=int(any(r['correct'] for r in returns)),
                consensus_one=returns[0]['correct'] if returns else 0,returned=len(returns),
                feasible_folds=sum(r['status']=='completed' for r in fold_rows),
                folds_assembled=sum(r['assembled'] or 0 for r in fold_rows) if name!='ode_simple' else None,
                fold_train_hits=sum(r['train_six'] for r in fold_rows),fold_held_hits=sum(r['held_six'] for r in fold_rows))
    for tag,path in [('baseline',trial.folder(name,seed,'native')),('matched',control_path(name,seed))]:
        base=read_archive(path)
        bt,bh,bd,bv,_=previous.replay(p,raw,base)
        for limit in (750,3000):
            combos=np.asarray(base['systems'])[np.asarray(base['operator']['system_cost'])<=limit]
            trace=correct_systems(base,combos,p)
            _,fixed=select(bh,bd,bv,combos)
            _,train_fixed=select(bt,bd,bv,combos)
            result[tag+'_six_'+str(limit)]=int(any(map(trace,fixed)))
            result[tag+'_one_'+str(limit)]=int(bool(fixed) and trace(fixed[0]))
            result[tag+'_train_six_'+str(limit)]=int(any(map(trace,train_fixed)))
            result[tag+'_held_wape_of_train_leader_'+str(limit)]=(float(bh[combos[train_fixed[0]]].mean())
                                                                  if train_fixed else None)
    return result,fold_rows,returns


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        outputs=list(pool.map(evaluate,[(n,s) for s in live.SEEDS for n in live.DATASETS]))
    write_csv(ROOT/'blocked_systems.csv',[r for r,_,_ in outputs])
    write_csv(ROOT/'blocked_folds.csv',[r for _,folds,_ in outputs for r in folds])
    write_csv(ROOT/'blocked_returns.csv',[r for _,_,rows in outputs for r in rows])
