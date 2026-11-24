"""Measured pair encounters in the three 50 Hz population sweeps, in empirical spacing units."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import re
import zipfile

import numpy as np
import pandas as pd

from physics import ROBOT
from impacts import episodes

DEST = ROBOT/'results'/'pair_interactions_sweeps'
HZ = 50
H = 20
GATES = (1.00,1.05,1.10,1.15)
OUTER = 1.30


class TrackReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('scalar','_reconstruct'):
            return getattr(np._core.multiarray,name)
        if module=='numpy' and name in ('dtype','ndarray'):
            return getattr(np,name)
        raise pickle.UnpicklingError(f'Unexpected class {module}.{name}')


def read_stage(z, name, size):
    with z.open(name) as stream:
        frames = TrackReader(stream).load()
    with z.open(name.replace('/data_', '/time_')) as stream:
        times = np.asarray(TrackReader(stream).load(),float)
    if times.shape!=(len(frames),) or not np.allclose(np.diff(times),1/HZ,rtol=0,atol=1e-6):
        raise ValueError(f'{name} has irregular timing')
    ids = np.array(sorted({int(row[0]) for frame in frames for row in frame}))
    if len(ids)!=size:
        raise ValueError(f'{name} contains {len(ids)} distinct robot IDs')
    lookup = {rid:i for i,rid in enumerate(ids)}
    xyz = np.full((len(frames),size,3),np.nan)
    for t,frame in enumerate(frames):
        for rid,angle,(x,y) in frame:
            j = lookup[int(rid)]
            if np.isfinite(xyz[t,j,0]):
                raise ValueError(f'Duplicate ID in {name} at {t}')
            xyz[t,j] = (x,y,angle)
    # Complex positions simplify projections. Wrapped angular differences tolerate a missing pose.
    pos = xyz[...,0]+1j*xyz[...,1]
    a = np.deg2rad(xyz[...,2])
    da = np.angle(np.exp(1j*np.diff(a,axis=0)))
    phi = np.nan_to_num(a[0]) + np.vstack([np.zeros(len(ids)),np.cumsum(np.nan_to_num(da),axis=0)])
    return times,ids,pos,phi,~np.isfinite(xyz).all(axis=-1)


def project(q,axis):
    return q*axis.conj()


def clip_rows(frames, pair, distance, rel, left, right, ids, pos, phi, bad, gate):
    possible = (frames>=22) & (frames+H<len(pos))
    frames,pair = frames[possible],pair[possible]
    i,j = left[pair],right[pair]
    ok = (bad[frames+H+1,i]==bad[frames-22,i]) & (bad[frames+H+1,j]==bad[frames-22,j])
    t,pair,i,j = frames[ok],pair[ok],i[ok],j[ok]
    separation = distance[t,pair]
    n = rel[t,pair]/separation
    def velocity(start,end):
        return project(rel[t+end,pair]-rel[t+start,pair],n)*HZ/(end-start)

    before,earlier = velocity(-11,-1),velocity(-22,-12)
    after = velocity(0,H)
    spin_i,spin_j = (phi[t-1,i]-phi[t-21,i])*HZ/H,(phi[t-1,j]-phi[t-21,j])*HZ/H
    # Resolve third-robot distances and directions at the forecast origin only.
    def others(k):
        d = pos[t,k,None]-pos[t]
        d[np.arange(len(t)),i] = np.inf
        d[np.arange(len(t)),j] = np.inf
        r = abs(d)
        unit = np.divide(d,r,out=np.zeros_like(d),where=np.isfinite(r)&(r>0))
        return np.min(np.where(np.isfinite(r),r,np.inf),axis=1), (np.maximum(OUTER-r,0)*unit).sum(axis=1)

    di,force_i = others(i)
    dj,force_j = others(j)
    third = project(force_j-force_i,n)/.30
    near = (np.minimum(di,dj)<1.15)
    g = np.maximum(separation-1.,0.)
    z = np.maximum(OUTER-separation,0.)/(OUTER-1)
    same = spin_i*spin_j>0
    short = velocity(-3,-1).real
    return pd.DataFrame(dict(frame=t,id_i=ids[i],id_j=ids[j],gate=gate,r=separation,
        incoming=before.real,side=before.imag,prior=earlier.real,prior_side=earlier.imag,
        outward=after.real,tangent=after.imag,immediate=velocity(0,2).real,incoming_short=short,
        after_100ms=velocity(0,5).real,third=near,spin_i=spin_i,spin_j=spin_j,same=same,
        bias=np.ones(len(t)),gap=g,proximity=z,closing=np.maximum(-before.real,0),
        hinge=np.maximum(-g/(H/HZ)-before.real,0.),bend=before.imag**2*(H/HZ)/(2*separation),
        crowd=near.astype(float),third_normal=third.real,third_tangent=third.imag,
        spin_slide=z*(spin_i+spin_j)/2,same_near=z*same,slide_near=z*before.imag))


def run():
    DEST.mkdir(parents=True,exist_ok=True)
    origin_tables,event_tables,audit,receipt = [],[],[],{}
    for archive in sorted(ROBOT.glob('Experiment_[123].zip')):
        with zipfile.ZipFile(archive) as z:
            names = {int(m[1]):s for s in z.namelist() if (m:=re.search(r'/data_N=(\d+)_',s))}
            assert sorted(names)==list(range(1,31))
            clips = {n:read_stage(z,name,n) for n,name in names.items()}
        # Calibration uses just the starting configuration of each dense stage.
        reference = []
        for n in range(20,31):
            p = clips[n][2][0]
            d = abs(p[:,None]-p[None,:])
            np.fill_diagonal(d,np.inf)
            reference.append(np.nanquantile(np.nanmin(d,axis=1),.05))
        scale = float(np.median(reference))
        receipt[archive.stem] = dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            reference_spacing=scale,starting_dense_stage_quantiles=reference)
        for size,(times,ids,raw,phi,missing) in sorted(clips.items()):
            pos = raw/scale
            jumps = (abs(np.diff(pos,axis=0))>.25) | (abs(np.diff(phi,axis=0))>np.pi/3)
            suspicious = missing.copy()
            suspicious[1:] |= jumps
            suspicious[:-1] |= jumps
            bad = np.vstack([np.zeros(size,int),suspicious.cumsum(axis=0)])
            pairrows = 0
            n_events = 0
            if size>1:
                left,right = np.triu_indices(size,1)
                rel = pos[:,right]-pos[:,left]
                distance = abs(rel)
                grid = np.arange(22,len(raw)-H,H)
                it,pair = np.where((distance[grid]>0)&(distance[grid]<OUTER))
                regular = clip_rows(grid[it],pair,distance,rel,left,right,ids,pos,phi,bad,0.)
                origin_tables.append(regular.assign(experiment=archive.stem,N=size))
                pairrows = len(regular)
                for gate in GATES:
                    entries = [episodes(s,gate,OUTER)[0] for s in distance.T]
                    t = np.concatenate(entries)
                    p = np.repeat(np.arange(len(entries)),[len(x) for x in entries])
                    event = clip_rows(t,p,distance,rel,left,right,ids,pos,phi,bad,gate)
                    event_tables.append(event.assign(experiment=archive.stem,N=size))
                    if gate==1.1:
                        n_events = int((event.incoming<0).sum())
            audit.append(dict(experiment=archive.stem,N=size,frames=len(raw),start=times[0],end=times[-1],
                reference_spacing=scale,missing=int(missing.sum()),bad_robot_frames=int(suspicious.sum()),
                spike_steps=int(jumps.sum()),predictions=pairrows,approaches_110=n_events))
        print(archive.stem,scale,flush=True)
    for tables,name in [(origin_tables,'origins'),(event_tables,'encounters')]:
        data = pd.concat(tables,ignore_index=True)
        assert np.isfinite(data.select_dtypes('number').to_numpy()).all()
        data.to_csv(DEST/(name+'.csv.gz'),index=False,float_format='%.9g')
    pd.DataFrame(audit).to_csv(DEST/'audit.csv',index=False)
    (DEST/'inputs.json').write_text(json.dumps(receipt,indent=1),encoding='utf-8')


def separation():
    """Distance-based endpoints alongside displacement along the initial normal."""
    events = pd.read_csv(DEST/'encounters.csv.gz')
    events = events[events.incoming<0]
    refs = json.loads((DEST/'inputs.json').read_text())
    tables = []
    for experiment,stages in events.groupby('experiment'):
        with zipfile.ZipFile(ROBOT/(experiment+'.zip')) as z:
            for size,sub in stages.groupby('N'):
                name = next(s for s in z.namelist() if f'/data_N={size}_' in s)
                _,ids,p,_,_ = read_stage(z,name,size)
                t,i,j = sub.frame.to_numpy(),np.searchsorted(ids,sub.id_i),np.searchsorted(ids,sub.id_j)
                at = t[:,None]+np.arange(H+1)
                dist = abs(p[at,j[:,None]]-p[at,i[:,None]])/refs[experiment]['reference_spacing']
                np.testing.assert_allclose(dist[:,0],sub.r,atol=1.e-7)
                tables.append(sub[['experiment','gate','N','third']].assign(
                    range_increased=dist[:,-1]>dist[:,0],cleared_any=(dist>=OUTER).any(axis=1),
                    clear_at_end=dist[:,-1]>=OUTER,fixed_axis_opening=sub.outward>0))
    records = []
    for gate,part in pd.concat(tables).groupby('gate'):
        for exp,d in [('all',part),*part.groupby('experiment')]:
            for group,s in [('all',d),('two robots',d[d.N==2]),('without third',d[~d.third]),('with third',d[d.third])]:
                if len(s):
                    props = s[['range_increased','cleared_any','clear_at_end','fixed_axis_opening']].mean().to_dict()
                    records.append(dict(experiment=exp,gate=gate,group=group,events=len(s),**props))
    pd.DataFrame(records).to_csv(DEST/'separation.csv',index=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--separation',action='store_true')
    if parser.parse_args().separation:
        separation()
    else:
        run()
