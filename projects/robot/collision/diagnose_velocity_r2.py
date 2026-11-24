"""Attribute forecast error on the exact eight-term holdouts and compare matched time-block refits."""
import json

from expanded_data import CACHE, DEST, calibration
from expanded_terms import IDS, WARMUP, columns, history
from physics import CUTOFF, LAG, arena_for, fit_self, recordings
import numpy as np
import pandas as pd


RESULTS = DEST / 'common_extensions'


def design(track, params, terms, rows, memory):
    data = [[], [], [], []]
    ix = [IDS.index(name) for name in terms]
    for start in range(0, len(rows), 32):
        slots = np.arange(start, min(start+32, len(rows)))
        x, base, y, valid, _, _ = columns(track, rows[slots], slots, memory, params)
        for dst, arr in zip(data, (x[..., ix].astype(float), base, y, valid)):
            dst.append(arr)
    return tuple(np.concatenate(v) for v in data)


def inputs(track):
    rows = np.arange(WARMUP, len(track.pos)-LAG, LAG)
    return rows, history(track, rows)


def refit(x, base, y, ok):
    a = x[ok].reshape(-1, x.shape[-1])
    return np.linalg.lstsq(a, (y-base)[ok].ravel(), rcond=None)[0]


def neighbors(pos):
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    d2 = np.einsum('...i,...i->...', diff, diff)
    return (d2>0) & (d2<CUTOFF**2)


def score(rows, held, group, prediction, y, mask, total_sse=None):
    actual = y[mask].ravel()
    residual = (y-prediction)[mask].ravel()
    sse = float(residual@residual)
    mse = sse/len(actual)
    rows.append(dict(held=held, group=group, components=len(actual), target_sd=actual.std(),
                     target_mean=actual.mean(), prediction_mean=prediction[mask].mean(),
                     rmse=np.sqrt(mse), r2=1-mse/actual.var(), sse=sse,
                     sse_fraction=sse/total_sse if total_sse is not None else np.nan))
    return rows[-1]


def gram_fit(g, names):
    ix = [IDS.index(n) for n in names]
    scale = np.sqrt(np.maximum(np.diag(g)[ix],1e-30))
    corr = g[np.ix_(ix,ix)] / scale[:,None] / scale[None,:]
    cross = g[ix,-1] / scale
    beta, _, rank, _ = np.linalg.lstsq(corr,cross,rcond=1e-12)
    return beta/scale, rank


def run():
    selected = pd.read_csv(RESULTS/'after_motion'/'selected_by_recording.csv')
    selected = selected.query("selection=='shared' and budget==2").set_index('held')
    detail, tail, capacity = [], [], []
    for held, tr in recordings().items():
        saved = selected.loc[held]
        terms = saved.terms.split('+')
        coef = np.array([json.loads(saved.coefs)[n] for n in terms])
        foreign = calibration(held)
        origin, memory = inputs(tr)
        x, base, y, ok = design(tr, foreign, terms, origin, memory)
        pred = base+x@coef
        motion = np.array([n in ('C180','C183') for n in terms])
        motion_pred = base+x[...,motion]@coef[motion]
        train_grams = []
        for path in sorted(CACHE.glob(f'{held}__*__train.npz')):
            with np.load(path) as cache:
                train_grams.append(cache['g'])
        assert len(train_grams)==4
        motion_weights,_ = gram_fit(np.mean(train_grams,axis=0),np.array(terms)[motion])
        refit_motion = base+x[...,motion]@motion_weights
        all_rows = score(detail,held,'all',pred,y,ok)
        np.testing.assert_allclose([all_rows['r2'],all_rows['rmse']], [saved.r2,saved.rmse], rtol=1e-8)
        current = neighbors(tr.pos[origin])
        count = current.sum(-1)
        changed = np.zeros_like(ok)
        for step in range(1,LAG+1):
            changed |= (neighbors(tr.pos[origin+step]) != current).any(axis=-1)
        for name, mask in [('zero_neighbors', count==0), ('one_neighbor', count==1),
                           ('multiple_neighbors', count>=2), ('same_neighbors', ~changed),
                           ('changed_neighbors', changed),('no_neighbors_throughout',(count==0)&~changed)]:
            score(detail,held,name,pred,y,ok&mask,all_rows['sse'])
        score(detail,held,'motion_only_ablation',motion_pred,y,ok)
        score(detail,held,'motion_only_trained',refit_motion,y,ok)
        f,i,j = np.nonzero(np.triu(current,k=1))
        use = ok[f,i]&ok[f,j]
        f,i,j = f[use],i[use],j[use]
        d = tr.pos[origin[f],i]-tr.pos[origin[f],j]
        n = d/np.linalg.norm(d,axis=-1,keepdims=True)
        for name, axis in [('normal',n),('tangent',np.stack((-n[:,1],n[:,0]),axis=1))]:
            rel_y = np.einsum('ij,ij->i', y[f,i]-y[f,j],axis)
            for model, v in [('full',pred),('motion_only',motion_pred),('trained_motion_only',refit_motion)]:
                rel_p = np.einsum('ij,ij->i',v[f,i]-v[f,j],axis)
                result = score(detail,held,'pair_'+name+'_'+model,rel_p,rel_y,np.ones(len(f),bool))
                if model=='full':
                    np.testing.assert_allclose(result['rmse'],saved[name],rtol=1e-8)

        with np.load(CACHE/f'{held}__{held}__test.npz') as cache:
            gram, variance = cache['g'], float(cache['variance'])
        for names in (terms, IDS):
            oracle, rank = gram_fit(gram,names)
            ix = [IDS.index(n) for n in names]+[len(IDS)]
            v = np.r_[oracle,-1.]
            mse = v@gram[np.ix_(ix,ix)]@v
            capacity.append(dict(held=held, score_type='same_rows_fit_and_score',
                                 candidate_terms=len(names), numerical_rank=rank,r2=1-mse/variance))

        cut = 3*len(tr.pos)//4
        train, test = tr.part(0,cut), tr.part(cut,None)
        arenas = arena_for([train])
        lookup, common = fit_self([train],arenas)
        train_inputs, test_inputs = inputs(train), inputs(test)
        for label, params in [('foreign_calibration',foreign),('local_calibration',(lookup,common,arenas))]:
            b = refit(*design(train,params,terms,*train_inputs))
            xx,bb,yy,mask = design(test,params,terms,*test_inputs)
            if label=='foreign_calibration':
                score(tail,held,'transferred',bb+xx@coef,yy,mask)
            score(tail,held,'local_refit_'+label,bb+xx@b,yy,mask)
        print(held, 'verified',flush=True)
    for path, entries in [('r2_error_breakdown.csv',detail),('r2_transfer_check.csv',tail)]:
        df = pd.DataFrame(entries)
        df.to_csv(RESULTS/path,index=False)
        print(df.groupby('group')[['components','target_sd','rmse','r2','sse_fraction']].mean().round(4).to_string())
    pd.DataFrame(capacity).to_csv(RESULTS/'r2_capacity_check.csv',index=False)


if __name__=='__main__':
    run()
