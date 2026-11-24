"""Exhaust zero, one and two additions to the previously reported common Pareto law."""
from itertools import combinations
from dataclasses import replace
import json

from expanded_data import CACHE, DEST
from expanded_search import IDS, TERMS, metrics, stored, fitted
from expanded_native import FACTORS, RHS, read_screen
from epde.interface.search_config import active_config, load_search_config, set_active_config
from epde.operators.common.subset_selection import subset_table
from epde.operators.common.survival import dyad_subset_scores
import numpy as np
import pandas as pd


ANCHOR = ('C012', 'C108', 'C109', 'C111')
OUTPUT = DEST / 'common_extensions'


def errors(g, cols, beta):
    """Component or projected RMSE from a per-recording second-moment matrix."""
    v = np.column_stack((beta, -np.ones(len(beta))))
    indices = np.column_stack((cols, np.full(len(beta), len(IDS))))
    sub = g[indices[:, :, None], indices[:, None, :]]
    return np.sqrt(np.maximum(0., np.einsum('bi,bij,bj->b', v, sub, v)))


def fit_group(train, cols):
    sub = train[cols[:, :, None], cols[:, None, :]]
    scales = np.sqrt(np.maximum(np.diagonal(sub, axis1=1, axis2=2), 1e-30))
    corr = sub / scales[:, :, None] / scales[:, None, :]
    keep = np.linalg.eigvalsh(corr)[:, 0] >= 1e-9
    beta = np.full(cols.shape, np.nan)
    rhs = train[cols[keep], -1] / scales[keep]
    beta[keep] = np.linalg.solve(corr[keep], rhs[..., None])[..., 0] / scales[keep]
    return beta


def choose(frame, score, scopes):
    """Validation-only choice at each complexity budget in the two physical libraries."""
    winners = {}
    for scope in scopes:
        pool = frame[frame.contact] if scope == 'contact' else frame
        for budget in (0, 1, 2):
            eligible = pool[(pool.additions <= budget) & np.isfinite(pool[score])]
            best = eligible.sort_values([score, 'additions', 'terms']).iloc[0]
            winners[scope, budget] = int(best.name)
    return winners


def run_stage(names, directory, contact_only=False, reference=None):
    anchor = tuple(IDS.index(name) for name in names)
    scopes = ('contact',) if contact_only else ('contact','joint')
    extra = [j for j in range(len(IDS)) if j not in anchor and (not contact_only or TERMS[j][2]!='Robot')]
    groups = [np.array([anchor + add for add in combinations(extra, size)]) for size in range(3)]
    supports = [tuple(row) for group in groups for row in group]
    candidates = pd.DataFrame(dict(
        terms=['+'.join(IDS[j] for j in row) for row in supports],
        additions=[len(row)-len(anchor) for row in supports],
        contact=[all(TERMS[j][2] != 'Robot' for j in row[len(anchor):]) for row in supports]))
    tags = sorted(f.name.split('__')[0] for f in CACHE.glob('*__test.npz'))
    assert len(tags) == 5 and len(candidates) == 1 + len(extra) + len(extra)*(len(extra)-1)//2
    fits, fold_winners = {}, {}
    for held in tags:
        train = np.mean([z['g'] for z in stored(held,'train')],axis=0)
        valid = np.mean([z['g'] for z in stored(held,'valid')],axis=0)
        fits[held] = [fit_group(train, group) for group in groups]
        candidates[held] = np.concatenate([errors(valid, g, b) for g,b in zip(groups, fits[held])])
        fold_winners[held] = choose(candidates, held, scopes)

    candidates['validation_rmse'] = candidates[tags].mean(axis=1, skipna=False)
    shared = choose(candidates, 'validation_rmse', scopes)
    # Freeze the supports above before opening any omitted-recording targets.
    by_fold, summary = [], []
    targets = {tag: stored(tag, 'test')[0] for tag in tags}
    if reference is None:
        replay = pd.read_csv(DEST/'evaluated.csv.gz')
        reference = replay[replay.terms == '+'.join(names)]
    old = reference.set_index('held')
    for held,test in targets.items():
        scores = {}
        for name, mat in [('rmse','g'), ('normal','normal'), ('tangent','tangent')]:
            scores[name] = np.concatenate([errors(test[mat],g,b) for g,b in zip(groups,fits[held])])
        scores['r2'] = 1 - scores['rmse']**2 / test['variance']
        np.testing.assert_allclose([scores[k][0] for k in scores],old.loc[held,list(scores)].to_numpy(float),rtol=1e-8)
        summary.append(pd.DataFrame(scores))
        for mode, winners in [('outer_fold',fold_winners[held]), ('shared',shared)]:
            for (scope,budget),idx in winners.items():
                size = candidates.at[idx,'additions']
                offset = sum(len(g) for g in groups[:size])
                b = fits[held][size][idx-offset]
                cols = supports[idx]
                m = metrics(test, cols, b)
                np.testing.assert_allclose(list(m.values()),[scores[k][idx] for k in m],rtol=1e-9)
                by_fold.append(dict(held=held,selection=mode,scope=scope,budget=budget,
                               terms=candidates.at[idx,'terms'],coefs=json.dumps(dict(zip([IDS[j] for j in cols],b))),
                               validation_rmse=candidates.at[idx,held],**m,delta_r2=m['r2']-scores['r2'][0]))
    for metric in ('rmse','r2','normal','tangent'):
        values = np.stack([f[metric] for f in summary])
        candidates['mean_'+metric] = values.mean(axis=0)
        if metric=='r2':
            candidates['worst_r2'] = values.min(axis=0)
            candidates['improved_folds'] = (values > values[:,:1]).sum(axis=0)
    directory.mkdir(parents=True,exist_ok=True)
    candidates.to_csv(directory/'all_extensions.csv.gz',index=False)
    folds = pd.DataFrame(by_fold)
    folds.to_csv(directory/'selected_by_recording.csv',index=False)
    report = folds.groupby(['selection','scope','budget'],as_index=False).agg(
        structures=('terms','nunique'), mean_r2=('r2','mean'),worst_r2=('r2','min'),
        mean_rmse=('rmse','mean'),normal_rmse=('normal','mean'),tangent_rmse=('tangent','mean'),
        improved_folds=('delta_r2',lambda x:(x>0).sum()))
    report.to_csv(directory/'comparison.csv',index=False)
    print(report.round(5).to_string(index=False))
    print(folds.query("selection=='shared' and budget>0")[['scope','budget','terms']].drop_duplicates().to_string(index=False))
    return folds


def run():
    first = run_stage(ANCHOR, OUTPUT)
    chosen = first.query("selection=='outer_fold' and scope=='joint' and budget==2")
    stages = [first]
    # Only extend a common structure when each outer fold selected that same
    # structure using its own validation segments.
    if chosen.terms.nunique() == 1:
        second = run_stage(chosen.terms.iloc[0].split('+'), OUTPUT/'after_motion', True, chosen)
        stages.append(second)
    tables = pd.concat(stages)
    shared = tables[tables.selection=='shared'].drop_duplicates(['held','terms'])
    laws = shared.groupby('terms',as_index=False).agg(mean_r2=('r2','mean'),worst_r2=('r2','min'),
             velocity_rmse=('rmse','mean'),normal_rmse=('normal','mean'),tangent_rmse=('tangent','mean'))
    objectives = finalist_objectives(laws.terms, sorted(shared.held.unique()))
    laws = laws.merge(objectives, on='terms',validate='one_to_one')
    from expanded_report import pooled_law
    pooled_law(chosen.terms.iloc[0].split('+') if chosen.terms.nunique()==1 else ANCHOR, laws,
               directory=OUTPUT, table_filename='pooled_equations.csv')


def finalist_objectives(structures, tags):
    """Replay the original EPDE objectives on exactly the same training screens."""
    previous = active_config()
    cfg = load_search_config()
    space = replace(cfg.search_space,system_rps_mode='plain',equation_terms_max_number=RHS+1,
                    equation_factors_max_number=FACTORS)
    cols = [sorted(IDS.index(name) for name in s.split('+')) for s in structures]
    records = []
    try:
        set_active_config(replace(cfg,search_space=space))
        for held in tags:
            screen = read_screen(held).astype(float)
            x, y = screen[:,:len(IDS)], screen[:,len(IDS)]
            w = np.ones(len(y))
            mask = [sum(1<<j for j in row) for row in cols]
            wape, amp, _ = subset_table(x,y,w,subsets=mask,discrepancy='wape')
            dyad = dyad_subset_scores(x,y,w,(len(y),),cols)
            sample = screen[:,:len(IDS)+1]
            gram = sample.T@sample / len(y)
            for s,c,a,d,l in zip(structures,cols,amp,dyad,wape):
                eligible = (fitted(gram,c) is not None and a<=space.rps_amplification_cap and
                            np.isfinite(d) and d<np.finfo(float).max)
                records.append(dict(terms=s,held=held,wape=l,dyad=d,eligible=eligible))
    finally:
        set_active_config(previous)
    frame = pd.DataFrame(records)
    frame.to_csv(OUTPUT/'finalist_objectives.csv',index=False)
    result = frame.groupby('terms',as_index=False).agg(mean_wape=('wape','mean'),mean_dyad=('dyad','mean'),
                                                     eligible_folds=('eligible','sum'))
    name = '+'.join(ANCHOR)
    old = pd.read_csv(DEST/'merged_pareto.csv').set_index('terms').loc[name]
    current = result.set_index('terms').loc[name]
    np.testing.assert_allclose([current.mean_wape,current.mean_dyad],[old.wape,old.dyad],rtol=1e-8)
    return result


if __name__=='__main__':
    run()
