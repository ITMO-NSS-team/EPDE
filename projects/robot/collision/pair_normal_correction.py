"""Search a sparse correction to the existing vector law using the pair-normal residual."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys

from discovery import write_json
from expanded_native import search_screen
from expanded_search import build_beam, configure, fitted, pareto_flags, score_structures
from pair_normal_data import GATES, STUDY
from pair_normal_dynamics import DIR, NAMES
from pair_normal_search import gram, loss, metric, read, tags, tree_predict
import numpy as np
import pandas as pd


NATIVE = ('prior','capCorrection','capCurvedCorrection','priorBend','freeNormal','driveNormal','driveCapCorrection',
          'individualIn','turnedDrive','turnLimit6','limit6','u3','originShift','gapSpeed','otherPush','oppositeClosing')
MODIFIED = ('prior','capCorrection','driveNormal','turnedDrive')
FIXED = ('bias','prior','capCorrection')
OUTPUT = STUDY/'correction'


def design(held,tag,role):
    block = read(tag,role)
    with np.load(DIR/f'{held}__{tag}__{role}.npz') as d:
        assert tuple(d['names'])==NAMES
        dyn = d['x']
    names = [*block['names'],*NAMES]
    columns = [block['x'],dyn]
    for term in MODIFIED:
        for mods in [(GATES[0],),(GATES[1],),GATES]:
            v = dyn[:,NAMES.index(term)]
            for m in mods:
                v = v*block['x'][:,names.index(m)]
            columns.append(v[:,None])
            names.append('*'.join((term,*mods)))
    return dict(names=names,x=np.column_stack(columns),y=block['y']-dyn[:,0],prior=dyn[:,0],observed=block['y'])


def screen(held,train):
    screens = []
    for index,tag in enumerate(tags()):
        if tag!=held:
            d = train[tag]
            edges = np.linspace(0,len(d['y']),513,dtype=int)
            rows = np.random.default_rng(22031+index).integers(edges[:-1],edges[1:])
            screens.append(np.column_stack((d['x'][rows],d['y'][rows])).astype(np.float32))
    return np.concatenate(screens)


def native(held):
    train = {t:design(held,t,'train') for t in tags() if t!=held}
    a = screen(held,train)
    names = next(iter(train.values()))['names']
    def col(name):
        return a[:,names.index(name)]
    result = search_screen(a[:,-1],{k:col(k) for k in NATIVE},{k:col(k) for k in GATES},MODIFIED,col)
    write_json(OUTPUT/f'{held}_nsga2.json',dict(held=held,screen_sha256=hashlib.sha256(a.tobytes()).hexdigest(),**result))


def launch(held):
    with (OUTPUT/f'{held}_nsga2.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-u',__file__,'--native',held],stdout=log,stderr=subprocess.STDOUT,
                       check=True,env=dict(os.environ,PYTHONHASHSEED='0'))


def run(workers=3):
    OUTPUT.mkdir(exist_ok=True)
    records = tags()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(launch,records))
    configure()
    stats,candidates,trees = {},{},{}
    for held in records:
        train = {t:design(held,t,'train') for t in records if t!=held}
        valid = [design(held,t,'valid') for t in train]
        test = design(held,held,'test')
        names = test['names']
        fixed = tuple(sorted(names.index(k) for k in FIXED))
        g = np.mean([gram(d) for d in train.values()],axis=0)
        a = screen(held,train)
        v = np.mean([gram(d) for d in valid],axis=0)
        stats[held] = (g,v,gram(test),a,test['observed'].var())
        receipt = json.loads((OUTPUT/f'{held}_nsga2.json').read_text())
        assert receipt['screen_sha256']==hashlib.sha256(a.tobytes()).hexdigest()
        models = {tuple(sorted(names.index(k) for k in q['terms'])) for q in receipt['evaluated']}
        models.update(build_beam(g,np.arange(len(names)),benchmark=fixed))
        models.update((i,) for i in range(len(names)))
        models.add(fixed)
        candidates[held] = models
        prediction,leaves,err = tree_predict(list(train.values()),valid,test,names)
        trees[held] = (prediction+test['prior'],leaves,err)
        print(held,'normal-residual proposals and nonlinear reference fitted',flush=True)
    union = set.union(*candidates.values())
    forecasts,rows = [],[]
    for held in records:
        g,v,tt,a,var = stats[held]
        structs,weights,wape,dyad,ok = score_structures(g,a,union)
        part = []
        for j,(s,b) in enumerate(zip(structs,weights)):
            rmse = loss(tt,s,b)
            part.append(dict(held=held,terms='+'.join(names[i] for i in s),size=len(s),eligible=bool(ok[j]),
                 local=s in candidates[held],wape=wape[j],dyad=dyad[j],validation=loss(v,s,b),
                 rmse=rmse,r2=1-rmse**2/var,refit=json.dumps(dict(zip([names[i] for i in s],b)))))
        part = pd.DataFrame(part)
        rows.append(part)
        best = part[part.local & part.eligible].sort_values(['validation','size','terms']).iloc[0]
        d = design(held,held,'test')
        beta = json.loads(best.refit) if best.validation < np.sqrt(v[-1,-1]) else {}
        pred = d['prior'] + (d['x'][:,[names.index(k) for k in beta]]@list(beta.values()) if beta else 0.)
        predictions = dict(residual_sparse=pred,residual_compact=d['prior']+d['x'][:,fixed]@fitted(g,fixed),
                           residual_boosted=trees[held][0])
        for k,p in predictions.items():
            terms,validation = dict(residual_sparse=('+'.join(beta),min(best.validation,np.sqrt(v[-1,-1]))),
                residual_compact=('+'.join(names[i] for i in fixed),loss(v,fixed,fitted(g,fixed))),
                residual_boosted=(f'{trees[held][1]} leaves',trees[held][2]))[k]
            forecasts.append(dict(held=held,model=k,structure=terms,validation=validation,pairs=len(p),**metric(d['observed'],p)))
        np.savez_compressed(OUTPUT/f'{held}_predictions.npz',**predictions)
    table = pd.concat(rows)
    table.to_csv(OUTPUT/'candidate_errors.csv.gz',index=False)
    common = table[table.eligible].groupby('terms',as_index=False).agg(folds=('held','nunique'),size=('size','first'),
            wape=('wape','mean'),dyad=('dyad','mean'),validation=('validation','mean'),r2=('r2','mean'),
            rmse=('rmse','mean'),worst_r2=('r2','min'))
    common = common[common.folds==len(records)].copy().sort_values(['validation','size','terms'])
    common['epde_front'] = pareto_flags(common.wape.to_numpy(),common.dyad.to_numpy(),np.ones(len(common),bool))
    common['size_error_front'] = pareto_flags(common['size'].to_numpy(),common.validation.to_numpy(),np.ones(len(common),bool))
    common.to_csv(OUTPUT/'common_equations.csv',index=False)
    shared = common.iloc[0].terms
    pool = np.mean([gram(design('pooled',t,'train')) for t in records],axis=0)
    equations = common[common.epde_front|common.size_error_front|(common.terms==shared)].copy()
    def pooled(terms):
        parts = terms.split('+')
        return dict(zip(parts,fitted(pool,[names.index(k) for k in parts])))
    equations['pooled_coefficients'] = equations.terms.map(lambda terms:json.dumps(pooled(terms)))
    equations.to_csv(OUTPUT/'pareto_equations.csv',index=False)
    write_json(OUTPUT/'common_law.json',dict(offset='previous shared vector law projected on the observed pair normal',
            compact=pooled('+'.join(names[i] for i in fixed)),shared=pooled(shared),candidate_terms=names,
            structures=len(union),common_eligible=len(common),epde_front=int(common.epde_front.sum())))
    for held in records:
        d = design(held,held,'test')
        row = table[(table.held==held)&(table.terms==shared)].iloc[0]
        b = json.loads(row.refit)
        yhat = d['prior']+d['x'][:,[names.index(k) for k in b]]@list(b.values())
        forecasts.append(dict(held=held,model='residual_shared',structure=shared,validation=row.validation,
                              pairs=len(yhat),**metric(d['observed'],yhat)))
        path = OUTPUT/f'{held}_predictions.npz'
        with np.load(path) as z:
            prev = dict(z)
        np.savez_compressed(path,**prev,residual_shared=yhat)
    pd.DataFrame(forecasts).to_csv(OUTPUT/'heldout_scores.csv',index=False)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--native')
    args = p.parse_args()
    if args.native:
        native(args.native)
    else:
        run()
