"""Normal-velocity EPDE search, leave-one-recording-out selection, and matched predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys

from expanded_data import moments
from discovery import write_json
from expanded_native import search_screen
from expanded_search import build_beam, configure, fitted, pareto_flags, score_structures
from pair_normal_data import STUDY, GATES, MODIFIABLE
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


COMPACT = ('bias','limit6','bend6')
NATIVE = (*MODIFIABLE,'turnLimit3','limitBend10','rawIndent','originShift','spinMagnitude','oppositeClosing',
          'facingIn','H_cap','c_entry','active_age')


def tags():
    result = sorted(p.name.split('__')[0] for p in STUDY.glob('*__train.npz'))
    assert len(result)==5
    return result


def read(tag,role):
    with np.load(STUDY/f'{tag}__{role}.npz') as z:
        return {k:z[k] for k in z.files}


def sample(held):
    screens = []
    for index,tag in enumerate(tags()):
        if tag==held:
            continue
        d = read(tag,'train')
        edges = np.linspace(0,len(d['y']),513,dtype=int)
        rows = np.random.default_rng(22031+index).integers(edges[:-1],edges[1:])
        screens.append(np.column_stack((d['x'][rows],d['y'][rows])).astype(np.float32))
    return list(d['names']),np.concatenate(screens)


def native(held):
    names,a = sample(held)
    def col(name):
        return a[:,names.index(name)]
    receipt = search_screen(a[:,-1],{n:col(n) for n in NATIVE},{n:col(n) for n in GATES},MODIFIABLE,col)
    write_json(STUDY/f'{held}_nsga2.json',dict(held=held,screen_sha256=hashlib.sha256(a.tobytes()).hexdigest(),**receipt))


def launch(held):
    with (STUDY/f'{held}_nsga2.log').open('w',encoding='utf-8') as log:
        cmd = [sys.executable,'-u',__file__,'--native',held]
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,PYTHONHASHSEED='0'))
    print(held,'completed native normal-velocity search',flush=True)


def metric(y,p):
    mse = np.mean((y-p)**2)
    return dict(rmse=float(np.sqrt(mse)),mae=float(np.abs(y-p).mean()),r2=float(1-mse/y.var()))


def gram(d):
    return moments(np.column_stack((d['x'],d['y'])))/len(d['y'])


def loss(g,s,b):
    ix = [*s,len(g)-1]
    w = np.r_[b,-1]
    return float(np.sqrt(max(0,w@g[np.ix_(ix,ix)]@w)))


def tree_predict(train,valid,test,names):
    # The validation tails of the four training recordings choose the tree capacity.
    cols = [i for i,k in enumerate(names) if '*' not in k and k!='bias']
    x = np.concatenate([d['x'][:,cols] for d in train])
    y = np.concatenate([d['y'] for d in train])
    sample_weights = np.concatenate([np.full(len(d['y']),len(y)/(len(train)*len(d['y']))) for d in train])
    candidates = []
    with threadpool_limits(limits=2):
        for leaves in (7,23):
            model = HistGradientBoostingRegressor(max_leaf_nodes=leaves,max_iter=160,learning_rate=.08,
                        min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=923701)
            model.fit(x,y,sample_weight=sample_weights)
            mse = np.mean([np.mean((d['y']-model.predict(d['x'][:,cols]))**2) for d in valid])
            candidates.append((mse,leaves,model))
        mse,leaves,model = min(candidates,key=lambda t:t[:2])
        return model.predict(test['x'][:,cols]),leaves,np.sqrt(mse)


def assess():
    configure()
    records = tags()
    blocks = {role:{tag:read(tag,role) for tag in records} for role in ('train','valid','test')}
    gs = {role:{tag:gram(d) for tag,d in blocks[role].items()} for role in blocks}
    names = list(blocks['train'][records[0]]['names'])
    assert all(list(d['names'])==names for role in blocks.values() for d in role.values())
    compact = tuple(sorted(names.index(n) for n in COMPACT))
    allcols = np.arange(len(names))
    local,stats = {},{}
    for held in records:
        others = [tag for tag in records if tag!=held]
        train,valid = [np.mean([gs[role][tag] for tag in others],axis=0) for role in ('train','valid')]
        _,screen = sample(held)
        native_data = json.loads((STUDY/f'{held}_nsga2.json').read_text())
        assert native_data['screen_sha256']==hashlib.sha256(screen.tobytes()).hexdigest()
        proposals = {tuple(sorted(names.index(k) for k in item['terms'])) for item in native_data['evaluated']}
        proposals.update((int(i),) for i in allcols)
        proposals.add(compact)
        proposals.update(build_beam(train,allcols,benchmark=compact))
        local[held] = proposals
        stats[held] = train,valid,screen
    union = set.union(*local.values())
    rows, selections = [],[]
    for held in records:
        train,valid,screen = stats[held]
        test = blocks['test'][held]
        y = test['y']
        structs,coefs,wape,dyad,ok = score_structures(train,screen,union)
        for j,(s,b) in enumerate(zip(structs,coefs)):
            err = loss(gs['test'][held],s,b)
            rows.append(dict(held=held,terms='+'.join(names[i] for i in s),size=len(s),eligible=bool(ok[j]),
                   proposed_locally=s in local[held],wape=wape[j],dyad=dyad[j],validation=loss(valid,s,b),
                   rmse=err,r2=1-err**2/y.var(),refit=json.dumps(dict(zip([names[i] for i in s],b)))))
        current = pd.DataFrame(rows[-len(structs):])
        winner = current[current.proposed_locally & current.eligible].sort_values(['validation','size','terms']).iloc[0]
        beta = json.loads(winner.refit)
        sparse = test['x'][:,[names.index(k) for k in beta]]@list(beta.values())
        compact_b = fitted(train,compact)
        fixed = test['x'][:,compact]@compact_b
        other_tags = [t for t in records if t!=held]
        tree,leaves,tree_validation = tree_predict([blocks['train'][t] for t in other_tags],
            [blocks['valid'][t] for t in other_tags],test,names)
        predictions = dict(previous=test['previous'],persistence=test['x'][:,names.index('rawU10')],
                           compact=fixed,sparse=sparse,boosted=tree)
        for model,pred in predictions.items():
            desc = {'sparse':winner.terms,'compact':'+'.join(names[i] for i in compact),'boosted':f'{leaves} leaves'}
            validation = {'sparse':winner.validation,'compact':loss(valid,compact,compact_b),'boosted':tree_validation}
            selections.append(dict(held=held,model=model,structure=desc.get(model,''),validation=validation.get(model),
                pairs=len(y),**metric(y,pred)))
        np.savez_compressed(STUDY/f'{held}_predictions.npz',origins=test['origins'],y=y,**predictions)
        print(held,'pair-normal validation winner',winner.terms,flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(STUDY/'candidate_errors.csv.gz',index=False)
    common = folds[folds.eligible].groupby('terms',as_index=False).agg(folds=('held','nunique'),size=('size','first'),
        wape=('wape','mean'),dyad=('dyad','mean'),validation=('validation','mean'),rmse=('rmse','mean'),
        r2=('r2','mean'),worst_r2=('r2','min'))
    common = common[common.folds==len(records)].copy()
    common['epde_front'] = pareto_flags(common.wape.to_numpy(),common.dyad.to_numpy(),np.ones(len(common),bool))
    common['size_error_front'] = pareto_flags(common['size'].to_numpy(),common.validation.to_numpy(),np.ones(len(common),bool))
    shared = common.sort_values(['validation','size','terms']).iloc[0].terms
    common.sort_values('validation').to_csv(STUDY/'common_equations.csv',index=False)
    pooled = np.mean(list(gs['train'].values()),axis=0)
    equations = common[common.epde_front|common.size_error_front|(common.terms==shared)].copy()
    functions = {}
    for terms in {*equations.terms,'+'.join(names[i] for i in compact)}:
        s = [names.index(k) for k in terms.split('+')]
        functions[terms] = dict(zip(terms.split('+'),fitted(pooled,s)))
    equations['pooled_coefficients'] = equations.terms.map(lambda terms:json.dumps(functions[terms]))
    equations.sort_values('validation').to_csv(STUDY/'pareto_equations.csv',index=False)
    write_json(STUDY/'common_law.json',dict(target='raw next 0.4 s relative velocity on the origin pair normal [px/s]',
        pooled_prefixes=records,shared=functions[shared],compact=functions['+'.join(names[i] for i in compact)],
        candidate_structures=len(union),common_eligible=len(common),epde_front=int(common.epde_front.sum()),
        size_validation_front=int(common.size_error_front.sum()),candidate_terms=names))
    for held in records:
        row = folds[(folds.held==held)&(folds.terms==shared)].iloc[0]
        beta = json.loads(row.refit)
        d = blocks['test'][held]
        p = d['x'][:,[names.index(k) for k in beta]]@list(beta.values())
        selections.append(dict(held=held,model='shared',structure=shared,validation=row.validation,
                               pairs=len(p),**metric(d['y'],p)))
    pd.DataFrame(selections).to_csv(STUDY/'heldout_scores.csv',index=False)


def run(workers=3):
    from pair_normal_data import prepare
    prepare()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(launch,tags()))
    assess()
    from pair_normal_dynamics import prepare as prepare_dynamics
    from pair_normal_correction import run as correct
    prepare_dynamics()
    correct(workers)
    from pair_normal_report import report
    report()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native')
    args = parser.parse_args()
    if args.native:
        native(args.native)
    else:
        run()
