"""Causal, exchange-symmetric features for close-pair velocity along the observed origin normal."""
import json

import numpy as np
import pandas as pd

from expanded_terms import IDS, V, WARMUP, columns, dot, history, pos
from physics import CAP, CUTOFF, DT, FPS, LAG, OUT, geometry, heading, rotate
from projected_data import fold_calibrations, projected_tracks
from run import check_inputs

STUDY = OUT/'preprocessed'/'pair_normal'
PREVIOUS = OUT/'preprocessed'/'search'
GATES = ('Near','Crowded')
MODIFIABLE = ('u3','u6','u10','opening6','closing6','limit6','turn6','bend6','otherPush','gapSpeed')


def features(raw, clean, origins, i, j, f, n, measured, degree, other_push, memory, slots):
    # All coordinate and heading reads in this function stop at the forecast origin.
    at = origins[f]
    def relative(p,h=0):
        return p[at-h,i]-p[at-h,j]
    now = relative(clean.pos)
    tangent = rotate(n)
    r = np.linalg.norm(now,axis=1)
    gap = np.maximum(r-CAP,0)
    crowd = degree[f,i]+degree[f,j]-2
    near = np.clip((CUTOFF-r)/(CUTOFF-CAP),0,1)
    x = {}
    def add(name,value):
        x[name] = np.broadcast_to(value,at.shape)
    add('bias',1.)
    add('gapSpeed',gap/DT)
    add('rawIndent',pos(CAP-measured)/DT)
    add('originShift',dot(now-relative(raw.pos),n)/DT)
    add('near2',V*near**2)
    add('otherCount',crowd)
    add('otherPush',other_push)
    for h in (2,3,6,10,20):
        vi = (clean.pos[at,i]-clean.pos[at-h,i])*(FPS/h)
        vj = (clean.pos[at,j]-clean.pos[at-h,j])*(FPS/h)
        u,s = dot(vi-vj,n),dot(vi-vj,tangent)
        add(f'u{h}',u)
        add(f'opening{h}',pos(u))
        add(f'closing{h}',pos(-u))
        add(f'limit{h}',np.maximum(u,-gap/DT))
        # Leading curvature term from translating around a disk at constant clearance.
        bend = s*s*DT/(2*r)
        add(f'bend{h}',bend)
        add(f'limitBend{h}',np.maximum(u,-gap/DT-bend))
        add(f'rawU{h}',dot(relative(raw.pos)-relative(raw.pos,h),n)*(FPS/h))
        def turn(v,robot):
            rate = (clean.phi[at,robot]-clean.phi[at-h,robot])*(FPS/h)
            angle = rate*(DT+h/FPS)/2
            # Ratio of the future and past circular-arc mean speeds; bounded near a full turn.
            gain = np.sinc(rate*DT/(2*np.pi))/np.maximum(np.sinc(rate*h/FPS/(2*np.pi)),.35)
            return gain[:,None]*(v*np.cos(angle)[:,None]+rotate(v)*np.sin(angle)[:,None])
        turned = turn(vi,i)-turn(vj,j)
        projected = dot(turned,n)
        add(f'turn{h}',projected)
        add(f'turnLimit{h}',np.maximum(projected,-gap/DT-dot(turned,tangent)**2*DT/(2*r)))
    e_i,e_j = heading(clean.phi[at,i]),heading(clean.phi[at,j])
    wi = (clean.phi[at,i]-clean.phi[at-LAG,i])/DT
    wj = (clean.phi[at,j]-clean.phi[at-LAG,j])/DT
    add('driveAlong',dot(e_i-e_j,n))
    add('facingIn',pos(-dot(e_i,n))+pos(dot(e_j,n)))
    add('headingAlign',dot(e_i,e_j))
    add('spinMagnitude',(abs(wi)+abs(wj))/2)
    add('oppositeClosing',(wi*wj<0)*x['closing6'])
    hist,index,*_ = memory
    for name in ('active_age','fresh','c_entry','H_cap','H_c','H_spin','repeat_count'):
        add(name,hist[name][slots[f],index[i,j]])
    add('Near',near)
    add('Crowded',(crowd>0).astype(float))
    for term in MODIFIABLE:
        for mod in [(GATES[0],),(GATES[1],),GATES]:
            value = x[term]
            for gate in mod:
                value = value*x[gate]
            add('*'.join((term,*mod)),value)
    return x


def pair_inputs(raw,clean,origins,memory,slots):
    d,axes = geometry(raw.pos[origins])
    close = (d>0)&(d<CUTOFF)
    f,i,j = np.nonzero(np.triu(close,k=1))
    ready = memory[-1][slots]
    keep = ready[f,i]&ready[f,j]
    f,i,j = f[keep],i[keep],j[keep]
    n = axes[f,i,j]
    p = clean.pos[origins]
    dist,cn = geometry(p)
    v = (p-clean.pos[origins-6])*(FPS/6)
    closing = pos(-np.sum((v[:,:,None]-v[:,None,:])*cn,axis=-1))
    load = np.clip((CUTOFF-dist)/(CUTOFF-CAP),0,1)*closing
    pushes = np.sum(load[...,None]*cn,axis=2)
    other = dot(pushes[f,i]-pushes[f,j],n)-2*load[f,i,j]*dot(cn[f,i,j],n)
    degree = close.sum(2)
    x = features(raw,clean,origins,i,j,f,n,d[f,i,j],degree,other,memory,slots)
    return x,f,i,j,n


def prepare():
    check_inputs()
    observed, projected = projected_tracks()
    calibrations = fold_calibrations(projected)
    summary = json.loads((PREVIOUS/'summary.json').read_text())
    previous_terms = summary['minimum_wape_terms']
    indices = [IDS.index(t) for t in previous_terms]
    refits = pd.read_csv(PREVIOUS/'cross_recording'/'projected_by_recording.csv.gz',usecols=['terms','held','refit'])
    refits = refits[refits.terms=='+'.join(previous_terms)].set_index('held').refit
    reference = pd.read_csv(PREVIOUS/'comparison_by_recording.csv').query(
        "geometry=='projected' and selection=='min_wape_shared'").set_index('held')
    STUDY.mkdir(exist_ok=True,parents=True)
    for tag,track in projected.items():
        split = len(track.pos)*3//4
        weights = [json.loads(refits[tag])[name] for name in previous_terms]
        for role,lo,hi in [('train',0,split),('valid',split,None),('test',0,None)]:
            clean,raw = track.part(lo,hi),observed[tag].part(lo,hi)
            rows = np.arange(WARMUP,len(raw.pos)-LAG,LAG)
            memory = history(clean,rows,raw)
            chunks,targets,metadata,old = [],[],[],[]
            vector_checks = []
            for start in range(0,len(rows),48):
                slots = np.arange(start,min(start+48,len(rows)))
                origins = rows[slots]
                x,f,i,j,n = pair_inputs(raw,clean,origins,memory,slots)
                chunks.append(np.column_stack(list(x.values())).astype(np.float32))
                # The response is the displacement difference projected on today's raw axis.
                at = origins[f]
                delta = (raw.pos[at+LAG,i]-raw.pos[at,i])-(raw.pos[at+LAG,j]-raw.pos[at,j])
                targets.append(dot(delta,n)/DT)
                metadata.append(np.column_stack([at+lo,clean.ids[i],clean.ids[j]]))
                if role=='test':
                    xx,base,*_ = columns(clean,origins,slots,memory,calibrations[tag],raw)
                    pred = base + xx[...,indices]@weights
                    old.append(dot(pred[f,i]-pred[f,j],n))
                    measured_v = (raw.pos[origins+LAG]-raw.pos[origins])/DT
                    err = pred-measured_v
                    vector_checks.append(np.column_stack((n,err[f,i],err[f,j],dot(delta,rotate(n))/DT)))
            y = np.concatenate(targets)
            array = np.concatenate(chunks)
            assert np.isfinite(array).all() and np.isfinite(y).all()
            extra = {}
            if old:
                baseline = np.concatenate(old)
                assert len(y)==summary['matched_counts'][tag]['pair_origins']
                np.testing.assert_allclose(np.sqrt(np.mean((baseline-y)**2)),reference.loc[tag,'normal'],rtol=1e-8)
                extra['previous'] = baseline
                extra['vector_checks'] = np.concatenate(vector_checks)
            path = STUDY/f'{tag}__{role}.npz'
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,names=list(x),x=array,y=y,origins=np.concatenate(metadata),**extra)
            tmp.replace(path)
        print(tag,'matched normal forecasts prepared',flush=True)


if __name__=='__main__':
    prepare()
