"""Matched raw-target transfer test of frame-local cap projection and tracking-quality masks."""
import json

from expanded_data import calibration
from expanded_terms import IDS, WARMUP, columns, history
from physics import CUTOFF, DT, LAG, OUT, arena_for, arena_envelope, fit_self, geometry, input_at, rotate, wall_vectors
from preprocess_tracks import DEST, load_product
import numpy as np
import pandas as pd

TERMS = ('C012','C108','C109','C111','C180','C183','C025','C156')
MODES = ('raw', 'projected', 'raw_qc', 'projected_qc')


def clear(flags, rows, before, after):
    counts = np.vstack((np.zeros((1,flags.shape[1]),int),np.cumsum(flags!=0,axis=0)))
    return counts[rows+after+1]==counts[rows-before]


def masked_calibration(tracks, flags):
    keys = sorted({(tr.pwm,int(rid)) for tr in tracks for rid in tr.ids})
    index = {key:i for i,key in enumerate(keys)}
    width = 2*len(keys)+2
    g = np.zeros((width,width))
    cross = np.zeros(width)
    per_size, spins = {}, {}
    for tr in tracks:
        keep = flags[tr.tag][::5]==0
        coords = tr.pos[::5][keep]
        per_size.setdefault(int(tr.tag[-2:]),[]).append(arena_envelope(*coords.T))
    arenas = {s:np.median(values,axis=0) for s,values in per_size.items()}
    for tr in tracks:
        rows = np.arange(LAG,len(tr.pos)-LAG,LAG)
        p,e,v,omega = input_at(tr,rows)
        wall = wall_vectors(p,arenas[int(tr.tag[-2:])])
        y = (tr.pos[rows+LAG]-p)/DT
        usable = clear(flags[tr.tag],rows,LAG,LAG)
        free = ~(geometry(p)[0]<CUTOFF).any(axis=-1) & usable
        for j,rid in enumerate(tr.ids):
            key = tr.pwm,int(rid)
            spins.setdefault(key,[]).append(omega[usable[:,j],j])
            m = free[:,j]
            a = np.stack((e[m,j],rotate(e[m,j]),v[m,j],wall[m,j]),axis=-1).reshape(-1,4)
            ix = [2*index[key],2*index[key]+1,width-2,width-1]
            g[np.ix_(ix,ix)] += a.T@a
            cross[ix] += a.T@y[m,j].ravel()
    fit = np.linalg.lstsq(g,cross,rcond=1e-12)[0]
    lookup = {}
    for key,i in index.items():
        rates = np.concatenate(spins[key])
        lookup[key] = np.r_[fit[2*i:2*i+2],rates.mean(),rates.std()*DT]
    for pwm in {key[0] for key in keys}:
        lookup[pwm,-1] = np.mean([lookup[k] for k in keys if k[0]==pwm],axis=0)
    return lookup, fit[-2:], arenas


def matched_mask(flags, rows, raw, clean):
    # Mask corrupt input histories in direct and third-body neighbors at the
    # origin. Only the focal track's future observations enter its target.
    adjacency = (geometry(raw.pos[rows])[0]<CUTOFF) | (geometry(clean.pos[rows])[0]<CUTOFF)
    bad_input = ~clear(flags,rows,2*LAG,0)
    for _ in range(2):
        bad_input |= np.any(adjacency & bad_input[:,None,:],axis=2)
    return ~bad_input & clear(flags,rows,0,LAG)


def design(tr,rows,memory,params):
    ix = [IDS.index(n) for n in TERMS]
    xs,bs,ys = [],[],[]
    for start in range(0,len(rows),32):
        slots = np.arange(start,min(start+32,len(rows)))
        # The fixed eight terms use only instantaneous geometry and the past
        # two velocity windows. Raw memory fixes the common availability mask.
        x,b,y,*_ = columns(tr,rows[slots],slots,memory,params)
        xs.append(x[...,ix].astype(float))
        bs.append(b)
        ys.append(y)
    return tuple(np.concatenate(v) for v in (xs,bs,ys))


def scores(pred, raw_y, ok, raw_pos, origin_shift):
    e = pred-raw_y
    mse = (e[ok]**2).mean()
    d,n = geometry(raw_pos)
    f,i,j = np.nonzero(np.triu((d>0)&(d<CUTOFF),k=1))
    keep = ok[f,i]&ok[f,j]
    f,i,j = f[keep],i[keep],j[keep]
    err = e[f,i]-e[f,j]
    normal = n[f,i,j]
    return dict(robot_origins=int(ok.sum()),pair_origins=len(f),rmse=np.sqrt(mse),
                r2=1-mse/raw_y[ok].var(),normal_rmse=np.sqrt(((err*normal).sum(1)**2).mean()),
                tangent_rmse=np.sqrt(((err*rotate(normal)).sum(1)**2).mean()),
                raw_endpoint_component_rmse_px=np.sqrt(((DT*e+origin_shift)[ok]**2).mean()))


def run():
    raw, clean, flags = {}, {}, {}
    for path in sorted(DEST.glob('N*_D*.npz')):
        observed, corrected, mask = load_product(path)
        tag = corrected.tag
        raw[tag],clean[tag],flags[tag] = observed,corrected,mask
    assert len(raw)==5
    cuts = {tag:3*len(tr.pos)//4 for tag,tr in raw.items()}
    params, grams = {}, {}
    for held in raw:
        for mode in MODES:
            data = clean if mode.startswith('projected') else raw
            train = [tr.part(0,cuts[tag]) for tag,tr in data.items() if tag!=held]
            if mode.endswith('qc'):
                pars = masked_calibration(train,{t.tag:flags[t.tag][:len(t.pos)] for t in train})
            elif mode=='raw':
                pars = calibration(held)
            else:
                arenas = arena_for(train)
                pars = (*fit_self(train,arenas),arenas)
            params[mode,held] = pars
            grams[mode,held] = []
    fits, outcomes = {}, []
    for role in ('train','test'):
        for tag,tr in raw.items():
            stop = cuts[tag] if role=='train' else len(tr.pos)
            original,projected = tr.part(0,stop),clean[tag].part(0,stop)
            rows = np.arange(WARMUP,stop-LAG,LAG)
            memory = history(original,rows)
            all_ok = memory[-1]
            qc = all_ok & matched_mask(flags[tag][:stop],rows,original,projected)
            raw_y = (original.pos[rows+LAG]-original.pos[rows])/DT
            for held in (sorted(set(raw)-{tag}) if role=='train' else [tag]):
                for mode in MODES:
                    source = projected if mode.startswith('projected') else original
                    x,b,y = design(source,rows,memory,params[mode,held])
                    if role=='train':
                        use = qc if mode.endswith('qc') else all_ok
                        a = np.concatenate((x,(y-b)[...,None]),axis=-1)[use].reshape(-1,len(TERMS)+1)
                        grams[mode,held].append(a.T@a/len(a))
                    else:
                        fit = fits[mode,held]
                        pred = b+x@fit
                        shift = source.pos[rows]-original.pos[rows]
                        for mask_name,ok in [('all',all_ok),('qc',qc)]:
                            outcomes.append(dict(held=held,training=mode,test_mask=mask_name,
                                **scores(pred,raw_y,ok,original.pos[rows],shift),
                                coefficients=json.dumps(dict(zip(TERMS,fit)))))
            print(role,tag,flush=True)
        if role=='train':
            for key,moments in grams.items():
                assert len(moments)==4
                g = np.mean(moments,axis=0)
                scale = np.sqrt(np.diag(g)[:-1])
                fits[key] = np.linalg.solve(g[:-1,:-1]/scale[:,None]/scale[None,:],g[:-1,-1]/scale)/scale
    table = pd.DataFrame(outcomes)
    prior = pd.read_csv(OUT/'expanded_factors'/'common_extensions'/'after_motion'/'selected_by_recording.csv')
    ref = prior.query("selection=='shared' and budget==2").set_index('held').sort_index()
    replay = table.query("training=='raw' and test_mask=='all'").set_index('held').sort_index()
    np.testing.assert_allclose(replay[['r2','rmse','normal_rmse','tangent_rmse']],ref[['r2','rmse','normal','tangent']],rtol=1e-8)
    table.to_csv(DEST/'heldout_raw_target.csv',index=False)
    summary = table.groupby(['training','test_mask'],as_index=False).agg(
        mean_r2=('r2','mean'),worst_r2=('r2','min'),rmse=('rmse','mean'),normal_rmse=('normal_rmse','mean'),
        tangent_rmse=('tangent_rmse','mean'),endpoint_rmse_px=('raw_endpoint_component_rmse_px','mean'),
        robot_origins=('robot_origins','sum'))
    summary.to_csv(DEST/'comparison.csv',index=False)
    print(summary.round(5).to_string(index=False))


if __name__=='__main__':
    run()
