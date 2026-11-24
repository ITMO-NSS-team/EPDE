"""Isolated pair episodes, finite-window outcomes, and recording-level transfer."""
from collections import Counter
import csv
import json

import numpy as np

from physics import CAP, CUTOFF, FPS, OUT, SIGMA, heading, partition, recordings, rotate
from discovery import write_json

SETTINGS = [(label, threshold, width) for label, threshold in
            [('cap_minus_2', CAP-2), ('cap', CAP), ('cap_plus_2', CAP+2), ('neighborhood', CUTOFF),
             ('physical_clearance', CAP)]
            for width in (3, 5)]
VARIABLES = ['vn_in','vt_in','relative_heading_n','relative_heading_t','surface_spin']
MODELS = {'linear': [0,1], 'heading': [0,1,2,3], 'spin': [0,1,4], 'heading_spin': list(range(5))}


def episodes(distance, enter, release):
    """First inward crossing per approach after release. A track starting inside is left censored."""
    entries = np.flatnonzero((distance[1:] < enter) & (distance[:-1] >= enter)) + 1
    clear = np.flatnonzero(distance >= release)
    if not len(clear):
        return np.array([], int), np.array([], int)
    entries = entries[entries > clear[0]]
    group = np.searchsorted(clear, entries)
    first = np.r_[True, np.diff(group) != 0] if len(group) else np.array([], bool)
    return entries[first], np.r_[clear, len(distance)][group[first]]


def slope(series, first, width):
    times = np.arange(width) - (width-1)/2
    return np.einsum('w,ew...->e...', times*FPS/(times@times), series[first[:,None]+np.arange(width)])


def pairs(track, arena):
    p = track.pos
    a, b = np.triu_indices(len(track.ids), 1)
    # Preserve exactly the same cutoff comparison for the pair and the neighbor count.
    distance = np.empty((len(p),len(a)), np.float64)
    counts = {limit:np.empty(p.shape[:2], dtype=np.int16) for limit in (CUTOFF, 1.03*CAP)}
    for i in range(0,len(p),64):
        sl = slice(i,i+64)
        d = np.linalg.norm(p[sl,:,None,:]-p[sl,None,:,:], axis=-1)
        distance[sl] = d[:,a,b]
        for limit,value in counts.items():
            value[sl] = (d<limit).sum(-1)-1
    wall = np.linalg.norm(p-arena[:2],axis=-1) > arena[2]-.25*SIGMA
    return a,b,distance,counts,wall


def collect(track, arena):
    a,b,dist,neighbors,wall = pairs(track, arena)
    output = {}
    for label,threshold,width in SETTINGS:
        clearance=1.03*CAP if label=='physical_clearance' else CUTOFF
        near=neighbors[clearance]
        counts, rows = Counter(), []
        for k,(i,j) in enumerate(zip(a,b)):
            d = dist[:,k]
            start,end = episodes(d,threshold,threshold+.03*CAP)
            counts['approaches'] += len(start)
            incomplete = (start<width) | (end+width>len(d))
            counts['incomplete'] += int(incomplete.sum())
            start,end = start[~incomplete],end[~incomplete]
            if not len(start):
                continue
            lo,hi = start-width,end+width
            def in_window(mask):
                prefix = np.r_[0,np.cumsum(mask)]
                return prefix[hi] > prefix[lo]
            near_wall = in_window(wall[:,i]|wall[:,j])
            other = in_window(np.maximum(near[:,i],near[:,j]) > (d<clearance).astype(int))
            long = end-start > FPS  # one-second separation limit for the short-impact estimand
            counts['wall'] += int(near_wall.sum())
            counts['third_robot'] += int(other.sum())
            counts['prolonged'] += int(long.sum())
            keep = ~(near_wall|other|long)
            start,end = start[keep],end[keep]
            if not len(start):
                continue
            rel = track.pos[:,j]-track.pos[:,i]
            n = rel[start]/np.maximum(np.linalg.norm(rel[start],axis=-1,keepdims=True),1e-12)
            t = rotate(n)
            before,after = slope(rel,start-width,width),slope(rel,end,width)
            vn = np.sum(before*n, axis=1)
            # All covariates end before entry apart from the geometry defining the bond frame.
            h = heading(track.phi[start-1,j])-heading(track.phi[start-1,i])
            omega = slope(track.phi[:,i]+track.phi[:,j],start-width,width)
            table = np.column_stack((vn, np.sum(before*t,1),np.sum(h*n,1),np.sum(h*t,1),CAP/2*omega,
                                     np.sum(after*n,1),np.sum(after*t,1),start,end,
                                     np.full(len(start),track.ids[i]),np.full(len(start),track.ids[j])))
            counts['nonapproaching'] += int((vn>=0).sum())
            rows.append(table[vn<0])
        table = np.concatenate(rows) if rows else np.zeros((0,11))
        counts['resolved'] = len(table)
        output[f'{label}_{width}'] = (dict(counts),table)
    return output


def coefficients(rows, name):
    # Give each represented recording equal total weight.
    if name=='persistence':
        return np.eye(5,2)
    blocks = [r for r in rows if len(r)]
    if not blocks:
        return None
    data = np.concatenate([r/np.sqrt(len(r)) for r in blocks])
    if name=='restitution':
        coef = np.eye(5,2)
        coef[0,0] = -max(0., -data[:,0]@data[:,5] / max(data[:,0]@data[:,0],1e-12))
        return coef
    ix = MODELS[name]
    if len(data)<2*len(ix) or np.linalg.matrix_rank(data[:,ix])<len(ix):
        return None
    coef = np.zeros((5,2))
    coef[ix] = np.linalg.lstsq(data[:,ix], data[:,5:7], rcond=None)[0]
    return coef


def score(rows, coef):
    if coef is None or not len(rows):
        return None
    err = rows[:,:5]@coef-rows[:,5:7]
    return dict(mae=np.abs(err).mean(0).tolist(), rmse=np.sqrt(np.mean(err**2,0)).tolist(),
                mse=float(np.mean(err**2)))


def analyze(held):
    tr,va,test = partition(recordings(),held)
    meta = json.loads((OUT/held/'training.json').read_text())
    def extract(track):
        return collect(track,np.array(meta['arenas'][track.tag[-2:]]))
    training,validation = [extract(t) for t in tr],[extract(t) for t in va]
    # Model choice uses these two groups only. The held-out event labels are used only below.
    observed = extract(test)
    result = {}
    for key,(counts,actual) in observed.items():
        train = [items[key][1] for items in training]
        val = [items[key][1] for items in validation]
        fits, losses = {}, {}
        for name in ['persistence','restitution',*MODELS]:
            coef = coefficients(train,name)
            if coef is not None:
                fits[name] = coef
            metrics = [s['mse'] for v in val if (s:=score(v,coef)) is not None]
            if metrics:
                losses[name] = np.mean(metrics)
        picked = min(losses,key=lambda name:(losses[name],len(MODELS.get(name,[])))) if losses else None
        result[key] = dict(counts=counts, distance_precision='float64',
                           train_counts=[len(r) for r in train], valid_counts=[len(v) for v in val],
                           selected=picked, coefficients={k:v.tolist() for k,v in fits.items()},
                           validation_mse=losses, test={k:score(actual,v) for k,v in fits.items()},
                           positive_outgoing_fraction=float(np.mean(actual[:,5]>0)) if len(actual) else None)
        if key=='cap_3':
            path = OUT/held/'heldout_events.csv'
            with path.open('w',newline='') as fh:
                writer=csv.writer(fh)
                writer.writerow(VARIABLES+['vn_out','vt_out','entry_frame','release_frame','id_i','id_j'])
                writer.writerows(actual)
    write_json(OUT/held/'impacts.json',result)
