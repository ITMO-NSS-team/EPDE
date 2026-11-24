"""Independent subset analysis and native EPDE verification of the two swarm field libraries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import hashlib
import json
import os
import subprocess
import sys
from unittest.mock import patch

from swarm_term_fields import COMBINED, DEST, LIB
import numpy as np
import pandas as pd
from discovery import write_json

SEEDS = (923701, 923702)
CAP_TERMS = 6
HISTORY = False
JOINT = False
DENSITY_BASE = ('Drive','Diffusion','Contact','Wall','Memory')
POLAR_BASE = ('IndividualTurn','Damping')


def run_native(held, target, seed):
    import expanded_native as engine
    with np.load(DEST/f'{held}_moments.npz') as data:
        screen = np.concatenate(data['train_'+target+'_screen'])
    columns = dict(zip(LIB[target], screen[:, :-1].T))
    with patch.multiple(engine, SEED=seed, RHS=CAP_TERMS, FACTORS=1):
        receipt = engine.search_screen(screen[:, -1], columns, {}, (), columns.__getitem__)
    write_json(DEST/f'{held}_{target}_{seed}.json', dict(held=held, target=target,
               screen_rows=len(screen), sha256=hashlib.sha256(screen.tobytes()).hexdigest(), **receipt))


def launch(job):
    held, target, seed = job
    log = DEST/f'{held}_{target}_{seed}.log'
    with log.open('w', encoding='utf-8') as stream:
        cmd = [sys.executable, '-u', __file__, '--native', held, target, str(seed)]
        if HISTORY:
            cmd.append('--history')
        if JOINT:
            cmd.append('--combined')
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                              env=dict(os.environ, PYTHONHASHSEED='0'))
    if proc.returncode:
        raise RuntimeError(log.read_text(errors='replace')[-3000:])
    print('Finished native EPDE', held, target, seed, flush=True)


def fit(gram, idx):
    sub = gram[np.ix_(idx,idx)]
    scale = np.sqrt(sub.diagonal())
    corr = sub / scale[:,None] / scale
    if np.linalg.eigvalsh(corr)[0] < 1e-9:
        return None
    return np.linalg.solve(corr, gram[idx,-1]/scale)/scale


def rmse(gram, idx, coef):
    indices = [*idx, gram.shape[-1]-1]
    v = np.r_[coef, -1.]
    return np.sqrt(np.maximum(v @ gram[:,indices][:,:,indices] @ v, 0.))


def screen_objectives(frame, names, screen):
    from pareto import configure, pareto_flags
    from epde.operators.common.subset_selection import subset_table
    from epde.operators.common.survival import dyad_subset_scores
    space = configure(CAP_TERMS+1)
    x, y = screen[:, :-1], screen[:, -1]
    subsets = [tuple(names.index(k) for k in entry.split(' + ')) for entry in frame.terms]
    masks = [sum(1 << i for i in subset) for subset in subsets]
    weights = np.ones(len(y))
    wape, amp, _ = subset_table(x,y,weights,subsets=masks,discrepancy='wape')
    dyad = dyad_subset_scores(x,y,weights,(len(y),),subsets)
    allowed = (amp <= space.rps_amplification_cap) & np.isfinite(dyad) & (dyad < np.finfo(float).max)
    frame['screen_wape'],frame['screen_dyad'],frame['eligible'] = wape,dyad,allowed
    frame['exact_front'] = pareto_flags(wape,dyad,allowed)


def evaluate(target, held, data, include_native):
    names = LIB[target]
    train = data['train_'+target+'_gram'].mean(axis=0)
    valid = data['valid_'+target+'_gram']
    test = data['test_'+target+'_gram']
    variance = data['test_'+target+'_variance'].item()
    rows, winners, native = [], [], []
    def measure(idx, coef, model):
        loss = rmse(test, idx, coef)[0]
        return dict(target=target, held=held, model=model, terms=' + '.join(names[i] for i in idx),
                    size=len(idx), coefficients=json.dumps(dict(zip([names[i] for i in idx],coef))),
                    valid_rmse=rmse(valid, idx, coef).mean(), rmse=loss, r2=1-loss**2/variance)

    for size in range(1,CAP_TERMS+1):
        for subset in combinations(range(len(names)),size):
            idx = list(subset)
            coef = fit(train, idx)
            if coef is not None:
                rows.append(measure(idx,coef,'subset'))
    frame = pd.DataFrame(rows)
    def record_best(df, model):
        chosen = df.sort_values(['valid_rmse','size','terms']).iloc[0].to_dict()
        winners.append(dict(chosen,model=model))

    def fixed(keys, name):
        idx = sorted(names.index(k) for k in keys)
        winners.append(measure(idx,fit(train,idx),name))
    baseline = DENSITY_BASE if target == 'density' else POLAR_BASE
    fixed(baseline, 'free baseline')
    if target == 'polarization':
        # ΔP = IndividualTurn + (1-a) Damping, identical to a R P - P.
        i, d = [names.index(k) for k in POLAR_BASE]
        beta_d = (train[d,-1]-train[d,i])/train[d,d]
        winners.append(measure([i,d],[1.,beta_d],'previous benchmark'))
    else:
        fixed(baseline,'previous benchmark')
    for term in names:
        if term not in baseline:
            fixed([*baseline, term],'baseline plus '+term)
    if target == 'polarization' and 'DriveTransport' in names:
        fixed([*baseline,'SpinMemory','DriveTransport'],'spin history and transport')
        if 'SlowTurn' in names:
            fixed([*baseline,'SlowTurn','SpinMemory','DriveTransport'],'two spin histories and transport')
    for size in range(1,CAP_TERMS+1):
        record_best(frame[frame['size']<=size], f'independent at most {size}')
    if include_native:
        screen = np.concatenate(data['train_'+target+'_screen'])
        digest = hashlib.sha256(screen.tobytes()).hexdigest()
        screen_objectives(frame,names,screen)
        record_best(frame[frame.exact_front], 'exhaustive EPDE front refit')
        objectives = frame.set_index('terms')
        for seed in SEEDS:
            receipt = json.loads((DEST/f'{held}_{target}_{seed}.json').read_text())
            assert receipt['sha256'] == digest and receipt['epochs'] > 0
            front = {tuple(sorted(c['terms'])) for c in receipt['front']}
            for cand in receipt['evaluated']:
                idx = sorted(names.index(k) for k in cand['terms'])
                raw = measure(idx,[cand['terms'][names[i]] for i in idx],'native screen coefficients')
                refit = measure(idx,fit(train,idx),'native full-training refit')
                check = objectives.loc[refit['terms']]
                np.testing.assert_allclose([cand['wape'],cand['dyad']], [check.screen_wape,check.screen_dyad],
                                           rtol=1e-4,atol=2e-6)
                native.append(dict(refit,seed=seed,wape=cand['wape'],dyad=cand['dyad'],
                                   exact_front=check.exact_front,
                                   native_front=tuple(sorted(cand['terms'])) in front,
                                   native_r2=raw['r2'],native_rmse=raw['rmse'],
                                   native_validation=raw['valid_rmse'],native_coefficients=raw['coefficients']))
        archive = pd.DataFrame(native)
        record_best(archive, 'native evaluated')
        front_rows = archive[archive.native_front]
        record_best(front_rows, 'native front refit')
        raw = front_rows.sort_values(['native_validation','size','terms']).iloc[0].to_dict()
        winners.append(dict(raw,model='native front as returned',coefficients=raw['native_coefficients'],
                            r2=raw['native_r2'],rmse=raw['native_rmse'],valid_rmse=raw['native_validation']))
    return frame.to_dict('records'), winners, native


def analyze(tags, native):
    subsets, winners, candidates = [], [], []
    for tag in tags:
        with np.load(DEST/f'{tag}_moments.npz') as data:
            for target in LIB:
                rows, choices, proposals = evaluate(target,tag,data,native)
                subsets.extend(rows)
                winners.extend(choices)
                candidates.extend(proposals)
    all_subsets = pd.DataFrame(subsets)
    all_subsets.to_csv(DEST/'independent_subsets.csv.gz',index=False)
    selected = pd.DataFrame(winners)
    selected.to_csv(DEST/'selected.csv',index=False)
    if native:
        from pareto import pareto_flags
        df = pd.DataFrame(candidates)
        df.to_csv(DEST/'native_candidates.csv.gz',index=False)
        df[df.native_front].to_csv(DEST/'native_fronts.csv',index=False)
        all_subsets[all_subsets.exact_front].to_csv(DEST/'exact_fronts.csv',index=False)
        common = all_subsets.groupby(['target','terms'],as_index=False).agg(
            folds=('held','nunique'), eligible=('eligible','all'),size=('size','first'),
            wape=('screen_wape','mean'),dyad=('screen_dyad','mean'),front_folds=('exact_front','sum'),
            validation_rmse=('valid_rmse','mean'),mean_r2=('r2','mean'),mean_rmse=('rmse','mean'))
        fronts = []
        for _, f in common.groupby('target'):
            flag = pareto_flags(f.wape.to_numpy(),f.dyad.to_numpy(),f.eligible.to_numpy() & (f.folds==len(tags)))
            fronts.append(f[flag])
        pd.concat(fronts).sort_values(['target','wape']).to_csv(DEST/'merged_front.csv',index=False)
    summary = selected.groupby(['target','model'],sort=False).agg(
        mean_r2=('r2','mean'),min_r2=('r2','min'),max_r2=('r2','max'),mean_rmse=('rmse','mean'))
    summary.to_csv(DEST/'summary.csv')
    print(summary.round(4).to_string(), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--native', nargs=3, metavar=('RECORDING','TARGET','SEED'))
    library = cli.add_mutually_exclusive_group()
    library.add_argument('--history',action='store_true')
    library.add_argument('--combined',action='store_true')
    cli.add_argument('--stage',choices=['independent','search','score'])
    args = cli.parse_args()
    if args.history:
        HISTORY = True
        LIB = {'polarization': ('IndividualTurn','Damping','SpinMemory','DelayedTurn','SlowTurn')}
        DEST = DEST/'spin_history'
        CAP_TERMS = 5
    if args.combined:
        JOINT = True
        LIB = COMBINED
        DEST = DEST/'combined'
    if args.native:
        run_native(args.native[0],args.native[1],int(args.native[2]))
    else:
        tags = sorted(p.name.replace('_moments.npz','') for p in DEST.glob('*_moments.npz'))
        assert len(tags)==5
        if args.stage == 'search':
            jobs = [(t,k,s) for s in SEEDS for t in tags for k in LIB]
            with ThreadPoolExecutor(3) as pool:
                list(pool.map(launch,jobs))
        else:
            analyze(tags,args.stage=='score')
