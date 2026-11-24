"""Sparse expanded EPDE search, training-only proposal and cross-recording transfer scoring."""
import argparse
from dataclasses import replace
import hashlib
import json

from expanded_data import CACHE, DEST, fingerprint
from expanded_native import FACTORS, RHS, read_screen
from expanded_terms import BENCHMARK, IDS, NATIVE_VECTORS, PRIORITY, TERMS
from pareto import pareto_flags, write_rows
from epde.interface.search_config import active_config, load_search_config, set_active_config
from epde.operators.common.subset_selection import subset_table
from epde.operators.common.survival import dyad_subset_scores
import numpy as np
import pandas as pd


P = len(IDS)
BASE = tuple(sorted(IDS.index(v) for v in BENCHMARK))
ALL = np.arange(P)
CONTACT = np.array([i for i,t in enumerate(TERMS) if t[2]!='Robot'])
OLD = np.array([n-1 for n in [1,2,81,82,20,35,48,100,101,15,16,17,18,19]])
SCOPES = [('original14',OLD),('contact',CONTACT),('joint',ALL)]
SCOPES += [('contact_catalog',np.array([i for i in CONTACT if IDS[i].startswith('C') and '*' not in IDS[i]]))]
SCOPES += [(f'native_grammar_{cap}',np.array([i for i,label in enumerate(IDS)
            if label.split('*')[0] in NATIVE_VECTORS and len(label.split('*'))<=cap])) for cap in (1,2,3)]


def key(s):
    return '+'.join(IDS[j] for j in sorted(s))


def build_beam(g, allowed, benchmark=BASE):
    """Exact marginal least-squares improvements of every admissible one-column extension."""
    scale = np.sqrt(np.maximum(np.diag(g)[:-1],1e-30))
    corr = g[:-1,:-1]/scale[:,None]/scale[None,:]
    cov = g[:-1,-1]/scale
    parents = [()]
    archive = set()
    for size in range(1,RHS+1):
        pool = {}
        if benchmark and size==len(benchmark)+1:
            parents = list(set(parents)|{benchmark})
        for s in parents:
            idx = list(s)
            if idx:
                sub = corr[np.ix_(idx,idx)]
                if np.linalg.eigvalsh(sub)[0]<1e-9:
                    continue
                cross = corr[idx,:]
                fit = np.linalg.solve(sub,cov[idx])
                residual = cov-fit@cross
                den = np.diag(corr)-(cross*np.linalg.solve(sub,cross)).sum(0)
                loss = g[-1,-1]-fit@cov[idx]
            else:
                residual,den,loss = cov,np.diag(corr),g[-1,-1]
            possible = allowed[den[allowed]>1e-8]
            improvement = residual[possible]**2/den[possible]
            for k in np.argsort(-improvement,kind='stable')[:10]:
                candidate = tuple(sorted((*s,int(possible[k]))))
                pool[candidate] = loss-improvement[k]
        archive.update(pool)
        parents = sorted(pool,key=lambda s:(pool[s],s))[:8]
    return archive


def stored(held,role):
    values = []
    for path in sorted(CACHE.glob(f'{held}__*__{role}.npz')):
        with np.load(path) as z:
            assert str(z['digest'])==fingerprint()
            values.append({k:z[k] for k in z.files if k not in ('digest','sample','ids')})
    assert len(values)==(1 if role=='test' else 4)
    return values


def fitted(g,s):
    ix = list(s)
    scale = np.sqrt(np.maximum(np.diag(g)[ix],1e-30))
    corr = g[np.ix_(ix,ix)]/scale[:,None]/scale[None,:]
    if np.linalg.eigvalsh(corr)[0]<1e-9:
        return None
    return np.linalg.solve(corr,g[ix,-1]/scale)/scale


def error(g,s,b):
    v = np.r_[b,-1]
    ix = [*s,P]
    return float(np.sqrt(max(0.,v@g[np.ix_(ix,ix)]@v)))


def metrics(test,s,b):
    rmse = error(test['g'],s,b)
    return dict(rmse=rmse,r2=float(1-rmse**2/test['variance']),
                normal=error(test['normal'],s,b),tangent=error(test['tangent'],s,b))


def evaluate(held,include_native):
    train = np.mean([z['g'] for z in stored(held,'train')],axis=0)
    valid = np.mean([z['g'] for z in stored(held,'valid')],axis=0)
    test, = stored(held,'test')
    proposals = {(i,) for i in range(P)}
    proposals.update(tuple(sorted(set(BASE)|{i})) for i in range(P))
    for _,allowed in SCOPES:
        proposals.update(build_beam(train,allowed))
    # Each fold receives the same predeclared nested set of the top catalog priorities.
    chain = BASE
    for i in PRIORITY[:RHS-len(BASE)]:
        chain = tuple(sorted((*chain,i-1)))
        proposals.add(chain)
    native_rows = []
    sample = read_screen(held)
    screen_sha256 = hashlib.sha256(sample.tobytes()).hexdigest()
    for library in (['focused','broad'] if include_native else []):
        prefix = '' if library=='focused' else 'broad_'
        native = json.loads((DEST/f'{held}_{prefix}nsga2.json').read_text())
        assert native['digest']==fingerprint() or native.get('screen_sha256')==screen_sha256
        native_front = {frozenset(q['terms']) for q in native['front']}
        for q in native['evaluated']:
            s = tuple(sorted(IDS.index(name) for name in q['terms']))
            proposals.add(s)
            b = [q['terms'][IDS[i]] for i in s]
            native_rows.append(dict(held=held,library=library,terms=key(s),size=len(s),native_front=frozenset(q['terms']) in native_front,
                               wape=q['wape'],dyad=q['dyad'],equation=json.dumps(q['terms']),
                               validation=error(valid,s,b),**metrics(test,s,b)))
    structs,coefficients,wape,dyad,eligible = score_structures(train,sample,proposals)
    front = pareto_flags(wape,dyad,eligible)
    rows = []
    for j,(s,b) in enumerate(zip(structs,coefficients)):
        rows.append(dict(held=held,terms=key(s),size=len(s),old=set(s)<=set(OLD),
                         contact=set(s)<=set(CONTACT),baseline=s==BASE,
                         refit=json.dumps(dict(zip([IDS[i] for i in s],b))),
                         wape=wape[j],dyad=dyad[j],eligible=bool(eligible[j]),epde_front=bool(front[j]),
                         validation=error(valid,s,b),**metrics(test,s,b)))
    data = pd.DataFrame(rows)
    winners = []
    for name,allowed in SCOPES:
        domain = set(allowed)
        candidate = data[[set(s)<=domain for s in structs]]
        candidate = candidate[candidate.eligible]
        for budget in range(1,RHS+1):
            group = candidate[candidate['size']<=budget]
            best = group.sort_values(['validation','size','terms']).iloc[0].to_dict()
            winners.append(dict(scope=name,budget=budget,**best))
    return data,pd.DataFrame(winners),pd.DataFrame(native_rows)


def score_structures(train,sample,proposals):
    structs = []
    coefficients = []
    for s in sorted(proposals,key=lambda s:(len(s),s)):
        b = fitted(train,s)
        if b is not None:
            structs.append(s)
            coefficients.append(b)
    width = train.shape[0]-1
    x,y = sample[:,:width].astype(float),sample[:,width].astype(float)
    # Avoid exactly dependent or inactive subsets on the proposal screen as well.
    sg = np.column_stack((x,y))
    sg = sg.T@sg/len(y)
    screen_rank = np.array([fitted(sg,s) is not None for s in structs])
    weights = np.ones(len(y))
    masks = [sum(1<<i for i in s) for s in structs]
    wape,amp,_ = subset_table(x,y,weights,subsets=masks,discrepancy='wape')
    eligible = screen_rank & (amp<=active_config().search_space.rps_amplification_cap)
    dyad = np.full(len(structs),np.nan)
    if eligible.any():
        with np.errstate(over='ignore'):
            dyad[eligible] = dyad_subset_scores(x,y,weights,(len(y),),[s for j,s in enumerate(structs) if eligible[j]])
    eligible &= np.isfinite(dyad) & (dyad<np.finfo(float).max)
    return structs,coefficients,wape,dyad,eligible


def configure():
    cfg = load_search_config()
    space = replace(cfg.search_space,system_rps_mode='plain',equation_terms_max_number=RHS+1,
                    equation_factors_max_number=FACTORS)
    set_active_config(replace(cfg,search_space=space))


def run(include_native):
    configure()
    all_data,win,native = [],[],[]
    tags = sorted(p.name.split('__')[0] for p in CACHE.glob('*__test.npz'))
    assert len(tags)==5
    for held in tags:
        d,w,n = evaluate(held,include_native)
        print(held,len(d),'supports;',sum(d.epde_front),'nondominated',flush=True)
        all_data.append(d)
        win.append(w)
        native.append(n)
    data = pd.concat(all_data)
    win = pd.concat(win)
    data.to_csv(DEST/'evaluated.csv.gz',index=False)
    data[data.epde_front].to_csv(DEST/'fold_pareto.csv',index=False)
    win.to_csv(DEST/'validation_selected.csv',index=False)
    selected = win[win.budget==RHS]
    data[data.baseline].to_csv(DEST/'baseline.csv',index=False)
    if include_native:
        pd.concat(native).to_csv(DEST/'nsga2_evaluated.csv.gz',index=False)
    common = data[data.eligible].groupby('terms').agg(folds=('held','nunique'),size=('size','first'),
            contact=('contact','first'),wape=('wape','mean'),dyad=('dyad','mean'),
            native_front_folds=('epde_front','sum'),validation=('validation','mean'),rmse=('rmse','mean'),
            r2=('r2','mean'),worst_r2=('r2','min'),normal=('normal','mean'),tangent=('tangent','mean')).reset_index()
    common = common[common.folds==len(tags)].copy()
    common['merged_front'] = pareto_flags(common.wape.to_numpy(),common.dyad.to_numpy(),np.ones(len(common),bool))
    common['size_validation_front'] = pareto_flags(common['size'].to_numpy(),common.validation.to_numpy(),np.ones(len(common),bool))
    common.sort_values(['size','validation','terms']).to_csv(DEST/'common_structures.csv',index=False)
    common[common.merged_front].sort_values('wape').to_csv(DEST/'merged_pareto.csv',index=False)
    add_one = data[data.terms.map(lambda s:set(BENCHMARK)<set(s.split('+')) and len(s.split('+'))==5)].copy()
    baseline = data[data.baseline].set_index('held')
    add_one['addition'] = add_one.terms.map(lambda s:next(iter(set(s.split('+'))-set(BENCHMARK))))
    for k in ('rmse','r2','normal','tangent','validation'):
        add_one['delta_'+k] = add_one[k]-add_one.held.map(baseline[k])
    add_one.to_csv(DEST/'one_addition.csv',index=False)
    write_rows(DEST/'term_dictionary.csv',[dict(id=id_,family=fam,scope=scope,weight=weight,direction=direct)
               for id_,fam,scope,weight,direct in TERMS])
    print(selected.groupby('scope')[['rmse','r2','normal','tangent']].mean().to_string(),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--without-native',action='store_true')
    run(not p.parse_args().without_native)
