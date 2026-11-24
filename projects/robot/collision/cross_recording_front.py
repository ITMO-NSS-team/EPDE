"""Refit the union of archived raw and corrected collision-law proposals in every recording fold."""
import gzip
import json

import numpy as np
import pandas as pd

from expanded_data import DEST, PROJECTED
from expanded_search import IDS, configure, error, key, metrics, pareto_flags, score_structures
from physics import OUT


def proposals(directories, references):
    names = set(IDS)
    names.update(references)
    for directory in directories:
        for archive in ['evaluated.csv.gz','nsga2_evaluated.csv.gz']:
            names.update(pd.read_csv(directory/archive,usecols=['terms']).terms.unique())
    return sorted({tuple(sorted(IDS.index(t) for t in name.split('+'))) for name in names},
                  key=lambda s:(len(s),s))


def statistics(directory,held,role):
    files = sorted((directory/'statistics').glob(f'{held}__*__{role}.npz'))
    assert len(files)==(1 if role=='test' else 4)
    rows = []
    for path in files:
        with np.load(path) as z:
            assert list(z['ids'])==IDS
            rows.append({k:z[k] for k in z.files if k not in ('digest','ids')})
    return rows


def score_fold(directory,held,supports):
    train_files = statistics(directory,held,'train')
    train = np.mean([z['g'] for z in train_files],axis=0)
    valid = np.mean([z['g'] for z in statistics(directory,held,'valid')],axis=0)
    test, = statistics(directory,held,'test')
    sample = np.concatenate([z['sample'] for z in train_files])
    rows = []
    for start in range(0,len(supports),4096):
        subset = supports[start:start+4096]
        structs,weights,wape,dyad,eligible = score_structures(train,sample,subset)
        rankable = set(structs)
        for s in subset:
            if s not in rankable:
                rows.append(dict(held=held,terms=key(s),size=len(s),estimable=False,eligible=False))
        for k,(s,b) in enumerate(zip(structs,weights)):
            rows.append(dict(held=held,terms=key(s),size=len(s),estimable=True,eligible=bool(eligible[k]),
                wape=wape[k],dyad=dyad[k],validation=error(valid,s,b),**metrics(test,s,b),
                refit=json.dumps(dict(zip([IDS[i] for i in s],b)))))
    result = pd.DataFrame(rows)
    assert len(result)==len(supports) and result.terms.is_unique
    print(directory.name,held,len(supports),'structures,',result.eligible.sum(),'eligible',flush=True)
    return result


def compare_archive(table,previous):
    metrics_cols = ['wape','dyad','validation','rmse','r2','normal','tangent']
    old = pd.read_csv(previous/'evaluated.csv.gz').set_index(['held','terms']).sort_index()
    replay = table.set_index(['held','terms']).loc[old.index]
    assert (old.eligible == replay.eligible).all()
    np.testing.assert_allclose(replay.loc[old.eligible,metrics_cols],old.loc[old.eligible,metrics_cols],
                               rtol=1e-8,atol=1e-8)


def run(references):
    if not PROJECTED:
        raise ValueError('Select the projected geometry for the matched study.')
    folder = DEST/'cross_recording'
    folder.mkdir(exist_ok=True)
    configure()
    directories = {'raw':OUT/'expanded_factors','projected':DEST}
    supports = proposals(directories.values(),references)
    tags = sorted(p.name.split('__')[0] for p in (DEST/'statistics').glob('*__test.npz'))
    assert len(tags)==5
    for mode,directory in directories.items():
        parts = [score_fold(directory,held,supports) for held in tags]
        folds = pd.concat(parts,ignore_index=True)
        compare_archive(folds,directory)
        with gzip.open(folder/f'{mode}_by_recording.csv.gz','wt',encoding='utf-8',newline='',compresslevel=3) as stream:
            folds.to_csv(stream,index=False)
        common = folds[folds.eligible].groupby('terms').agg(folds=('held','nunique'),size=('size','first'),
                    wape=('wape','mean'),dyad=('dyad','mean'),validation=('validation','mean'),r2=('r2','mean'),
                    worst_r2=('r2','min'),rmse=('rmse','mean'),normal=('normal','mean'),tangent=('tangent','mean'))
        common = common[common.folds==5].reset_index()
        common['merged_front'] = pareto_flags(common.wape.to_numpy(),common.dyad.to_numpy(),np.ones(len(common),bool))
        common.sort_values('wape').to_csv(folder/f'{mode}_common.csv.gz',index=False)
        common[common.merged_front].sort_values('wape').to_csv(folder/f'{mode}_pareto.csv',index=False)
        print(mode,len(common),'eligible in all five;',int(common.merged_front.sum()),'on front',flush=True)
    return len(supports)
