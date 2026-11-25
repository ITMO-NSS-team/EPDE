"""Common-gate pair kinematics and isolated robot controls in body diameters."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from impacts import episodes
from new_pair_interactions import read_stage
from physics import ROBOT, OUT

DEST = ROBOT / 'results/shared_interaction'
BASE = OUT / 'preprocessed'
MS = (40, 80, 400)
DRIVE = ('recent', 'long', 'phase', 'motor', 'rate')
D_MM = 48.7


def stages():
    scales = pd.read_csv(ROBOT/'results/pair_interactions_sweeps/physical/scale.csv')
    for path in sorted(BASE.glob('N*_PWM*_D*.npz')):
        with np.load(path) as z:
            p = z['raw_pos']
            dia = float(z['cap_px'])
            tag = path.stem
            arena = json.loads((OUT/tag/'training.json').read_text())['arenas'][tag[-2:]]
            pos = (p[...,0]+1j*p[...,1]-complex(*arena[:2])) / dia
            yield tag, 25, int(z['pwm']), z['ids'], pos, z['phi'], z['quality_flags']!=0, arena[2]/dia
    for path in sorted(ROBOT.glob('Experiment_[123].zip')):
        with zipfile.ZipFile(path) as z:
            for key in sorted(s for s in z.namelist() if '/data_N=' in s):
                count = int(key.split('/data_N=')[1].split('_')[0])
                _,ids,pos,phi,bad = read_stage(z,key,count)
                # All stages below N=20 have the same independently measured dense-stage calibration.
                a = scales[(scales.experiment==path.stem)&(scales.N==max(count,2))].iloc[0]
                pos = (pos-complex(a.center_x,a.center_y)) / a.diameter_archive_units
                jumps = (abs(np.diff(pos,axis=0))>.25)|(abs(np.diff(phi,axis=0))>np.pi/3)
                bad[1:] |= jumps
                bad[:-1] |= jumps
                yield path.stem,50,20,ids,pos,phi,bad,a.radius/a.diameter_archive_units


def counts(mask):
    return np.vstack((np.zeros((1,mask.shape[1]),int),np.cumsum(mask,axis=0)))


def geometry(pos):
    nearest = np.full(pos.shape,np.inf)
    others = np.zeros(pos.shape,complex)
    for start in range(0,len(pos),100):
        sl = slice(start,start+100)
        delta = pos[sl,:,None]-pos[sl,None,:]
        dist = abs(delta)
        dist[:,np.arange(pos.shape[1]),np.arange(pos.shape[1])] = np.inf
        nearest[sl] = np.min(np.where(np.isfinite(dist),dist,np.inf),axis=2)
        unit = np.divide(delta,dist,out=np.zeros_like(delta),where=(dist>0)&np.isfinite(dist))
        others[sl] = (np.maximum(1.30-np.nan_to_num(dist,nan=np.inf),0)*unit).sum(2)
    return nearest,others


def features(pos,phi,hz,pwm):
    # History finishes 40 ms before the gate so predictor and target increments share no endpoint.
    def lag(x,seconds):
        return np.roll(x,round(hz*seconds),axis=0)
    short = (lag(pos,.04)-lag(pos,.12))/.08
    long = (lag(pos,.12)-lag(pos,.44))/.32
    spin = (lag(phi,.04)-lag(phi,.20))/.16
    # Heading-phase memory is insensitive to camera handedness and heading zero.
    phase = np.zeros_like(pos)
    for a in (.04,.12,.20,.28,.36):
        v = (lag(pos,a)-lag(pos,a+.08))/.08
        phase += np.cos(lag(phi,.04)-lag(phi,a+.04))*v/5
    return [short,long,phase,short*((pwm-20)/10),short*np.minimum(abs(spin),30)/10],spin


def extract(tag,hz,pwm,ids,pos,phi,bad,rad,mode):
    nearest,other = geometry(pos)
    wall = abs(pos)>rad-.25
    # A missing third pose prevents establishing isolation on that frame.
    unknown = ~np.isfinite(pos).all(axis=1)
    bad = bad | ~np.isfinite(pos) | unknown[:,None]
    invalid = counts(bad)
    free_invalid = counts(bad | wall | (nearest<1.5))
    drive,omega = features(pos,phi,hz,pwm)
    past,future = round(.44*hz),round(.4*hz)
    n = pos.shape[1]
    def common(t):
        return dict(recording=tag,stage=n,mode=mode,hz=hz,pwm=pwm,frame=t,
                    seconds=t/hz,segment_end=(len(pos)-1)/hz)
    # The full history and response must be wall- and neighbor-free for baseline fitting.
    grid = np.arange(past,len(pos)-future,round(.08*hz))
    loc,j = np.where(free_invalid[grid+future+1]==free_invalid[grid-past])
    t = grid[loc]
    f = dict(**common(t),robot=ids[j],heading=phi[t,j],spin=omega[t,j])
    for name,v in zip(DRIVE,drive):
        f[name+'_x'],f[name+'_y'] = v[t,j].real,v[t,j].imag
    for ms in MS:
        v = (pos[t+round(ms*hz/1000),j]-pos[t,j])/(ms/1000)
        f[f'y{ms}_x'],f[f'y{ms}_y'] = v.real,v.imag
    tables = []
    audit = []
    for kind,entry,reset in [('near',1.10,1.30),('distant',1.70,1.90)]:
        totals = np.zeros(5,int)
        for i,j in zip(*np.triu_indices(n,1)):
            r = pos[:,j]-pos[:,i]
            dist = abs(r)
            gates,_ = episodes(dist,entry,reset)
            # A gap in this pair's trace invalidates an open episode. Require a fresh reset.
            if not np.isfinite(dist).all():
                frames = np.arange(len(pos))
                reset_at = np.maximum.accumulate(np.where(dist>=reset,frames,-1))
                lost_at = np.maximum.accumulate(np.where(~np.isfinite(dist),frames,-1))
                gates = gates[reset_at[gates]>lost_at[gates]]
            totals[0] += len(gates)
            t = gates[(gates>=past)&(gates+future<len(pos))]
            totals[1] += len(t)
            t = t[(invalid[t+future+1,i]==invalid[t-past,i])&(invalid[t+future+1,j]==invalid[t-past,j])]
            totals[2] += len(t)
            t = t[~(wall[t,i]|wall[t,j])]
            totals[3] += len(t)
            u = r[t]/dist[t]
            before = (drive[0][t,j]-drive[0][t,i])*u.conj()
            totals[4] += int((before.real<0).sum())
            if not len(t):
                continue
            depth = np.maximum(1.30-dist[t],0)
            third = (other[t,j]-other[t,i])*u.conj()-2*depth
            recent = (dist[t[:,None]-np.arange(1,round(.4*hz)+1)]<1.30).mean(1)
            data = dict(**common(t),kind=kind,i=ids[i],j=ids[j],r=dist[t],side=before.imag,
                        closing=-before.real,spin_sum=(omega[t,i]+omega[t,j])/2,
                        same=(omega[t,i]*omega[t,j]>0),recent_near=recent,
                        third_n=third.real,third_t=third.imag)
            for name,v in zip(DRIVE,drive):
                diff = (v[t,j]-v[t,i])*u.conj()
                data[name+'_n'],data[name+'_t'] = diff.real,diff.imag
            for ms in MS:
                v = (r[t+round(ms*hz/1000)]-r[t])*u.conj()/(ms/1000)
                data[f'y{ms}_n'],data[f'y{ms}_t'] = v.real,v.imag
            tables.append(pd.DataFrame(data))
        audit.append(dict(recording=tag,stage=n,mode=mode,kind=kind,free_windows=len(f['frame']),
                          **dict(zip(('crossings','complete','quality_ok','interior','approaching'),totals))))
    return pd.DataFrame(f),tables,audit


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    cache = DEST/'data'
    cache.mkdir(exist_ok=True)
    free,events,audit = [],[],[]
    def save():
        pd.concat(free,ignore_index=True).to_parquet(cache/f'{previous}_free.parquet',index=False)
        pd.concat(events,ignore_index=True).to_parquet(cache/f'{previous}_pairs.parquet',index=False)
        pd.DataFrame(audit).to_csv(cache/f'{previous}_audit.csv',index=False)
        print(previous,sum(map(len,events)),'pair windows',sum(map(len,free)),'free',flush=True)
    previous = None
    for tag,hz,pwm,ids,pos,phi,bad,rad in stages():
        if previous is not None and tag!=previous:
            save()
            free,events,audit = [],[],[]
        previous = tag
        for decimation in ((1,2) if hz==50 else (1,)):
            # Preserve flags from intermediate native frames in the 25 Hz sensitivity run.
            mask = bad.copy()
            if decimation==2:
                mask[1:] |= bad[:-1]
                mask[:-1] |= bad[1:]
            f,e,a = extract(tag,hz//decimation,pwm,ids,pos[::decimation],phi[::decimation],
                            mask[::decimation],rad,'native' if decimation==1 else 'down25')
            free.append(f)
            events.extend(e)
            audit.extend(a)
    save()


if __name__=='__main__':
    main()
