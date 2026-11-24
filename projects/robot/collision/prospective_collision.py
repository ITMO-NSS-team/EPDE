"""Freeze, replay and prospectively score the shared collision law without refitting on the input."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from expanded_terms import BOOK, IDS, WARMUP, columns, history
from pair_normal_data import PREVIOUS, STUDY, pair_inputs
from pair_normal_dynamics import DIR, NAMES, normal_dynamics
from pair_normal_report import reconcile
from physics import CAP, CUTOFF, DT, FPS, LAG, OUT, Track, robot_parameters
from preprocess_tracks import load_observations, project_centers
from projected_data import projected_tracks


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE/'prospective'/'frozen_v1.json'
RESULTS = OUT/'preprocessed'/'prospective'
FILES = [Path(__file__).resolve(),BOOK,*[HERE/(n+'.py') for n in (
    'physics','expanded_terms','preprocess_tracks','pair_normal_data','pair_normal_dynamics','pair_normal_report')],
    HERE.parent/'swarm_data.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track_digest(tr):
    order = np.argsort(tr.ids)
    h = hashlib.sha256()
    for a in (tr.ids[order].astype('<i8'),tr.pos[:,order].astype('<f8'),tr.phi[:,order].astype('<f8')):
        h.update(np.asarray(a.shape,dtype='<i8').tobytes())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def freeze():
    tracks,_ = projected_tracks()
    pooled = json.loads((PREVIOUS/'cross_recording'/'pooled_common_law.json').read_text())
    correction = json.loads((STUDY/'correction'/'common_law.json').read_text())
    grams = []
    for tag in sorted(tracks):
        with np.load(DIR/f'pooled__{tag}__train.npz') as b,np.load(STUDY/f'{tag}__train.npz') as d:
            prior = b['x'][:,list(b['names']).index('prior')]
            a = np.column_stack([np.ones(len(prior)),prior,d['y']])
            grams.append(a.T@a/len(a))
    g = np.mean(grams,axis=0)
    law = dict(version=1,fps=FPS,cap_px=CAP,cutoff_px=CUTOFF,horizon_frames=LAG,
        warmup_frames=WARMUP,ridge=.1,baseline=pooled,affine=np.linalg.solve(g[:2,:2],g[:2,2]).tolist(),
        shared=correction['shared'],compact=correction['compact'],
        primary='shared_coherent',comparator='affine_coherent',block_seconds=10,bootstrap_replicates=2000,
        training_files=json.loads((OUT/'inputs.json').read_text()),
        training_tracks={k:track_digest(v) for k,v in sorted(tracks.items())},
        implementation={p.name:digest(p) for p in FILES})
    payload = json.dumps(law,indent=2,allow_nan=False).encode()+b'\n'
    ARTIFACT.parent.mkdir(exist_ok=True)
    if ARTIFACT.exists():
        if ARTIFACT.read_bytes()!=payload:
            raise ValueError('Frozen v1 already exists with different contents.')
    else:
        with ARTIFACT.open('xb') as stream:
            stream.write(payload)
    print('Frozen model SHA-256',digest(ARTIFACT))


def frozen():
    law = json.loads(ARTIFACT.read_text())
    if law['implementation']!={p.name:digest(p) for p in FILES}:
        raise ValueError('Scoring source differs from the frozen version.')
    return law


def predict(raw,clean,law):
    base = law['baseline']
    lookup = {tuple(map(int,k.split('/'))):np.array(v) for k,v in base['lookup'].items()}
    params = lookup,np.array(base['base_persistence_wall']),{int(k):np.array(a) for k,a in base['arenas'].items()}
    if (raw.pwm,-1) not in lookup or int(raw.tag[-2:]) not in params[2]:
        raise ValueError('No frozen PWM or camera/arena calibration for this input.')
    indices = [IDS.index(term) for term in base['terms']]
    b = list(base['terms'].values())
    calibration = robot_parameters(clean,lookup)
    rows = np.arange(WARMUP,len(raw.pos)-LAG,LAG)
    if not len(rows):
        raise ValueError('Recording ends before the first eligible forecast.')
    mem = history(clean,rows,raw)
    blocks = []
    for start in range(0,len(rows),48):
        slots = np.arange(start,min(start+48,len(rows)))
        rr = rows[slots]
        pair,f,i,j,n = pair_inputs(raw,clean,rr,mem,slots)
        if not len(i):
            continue
        x,free,*_ = columns(clean,rr,slots,mem,params,raw)
        prior_v = free+x[...,indices]@b
        dyn = normal_dynamics(clean,rr,f,i,j,n,calibration,free,prior_v)
        features = {k:v.astype(np.float32) for k,v in pair.items()}
        features.update(dict(zip(NAMES,dyn.astype(np.float32).T)))
        u = features['prior']
        a0,a1 = law['affine']
        forecasts = dict(prior=u,affine=a0+a1*u,clearance=np.maximum(u,-features['gapSpeed']))
        for name in ('shared','compact'):
            beta = law[name]
            parts = []
            for term in beta:
                value = features.get(term)
                if value is None:
                    value = np.prod([features[k] for k in term.split('*')],axis=0)
                parts.append(value)
            forecasts[name] = u+np.column_stack(parts)@list(beta.values())
        forecast_frame = rr[f]
        ids = np.column_stack((raw.ids[i],raw.ids[j]))
        for name in ('affine','shared','compact','clearance'):
            result = dyn[:,0].copy()
            for ix in np.split(np.arange(len(f)),np.flatnonzero(np.diff(f))+1):
                dv,one,two = reconcile(ids[ix],n[ix],forecasts[name][ix]-dyn[ix,0],law['ridge'])
                result[ix] += np.sum((dv[one]-dv[two])*n[ix],axis=1)
            forecasts[name+'_coherent'] = result
        # Observations enter only after all forecasts for this block have been evaluated.
        observed = (raw.pos[forecast_frame+LAG,i]-raw.pos[forecast_frame,i]
                    -raw.pos[forecast_frame+LAG,j]+raw.pos[forecast_frame,j])/DT
        blocks.append(pd.DataFrame(dict(frame=forecast_frame,i=ids[:,0],j=ids[:,1],
                    multiple_neighbors=features['Crowded'].astype(bool),observed=(observed*n).sum(1),**forecasts)))
    if not blocks:
        raise ValueError('No close pairs with resolved histories in the scoring interval.')
    return pd.concat(blocks,ignore_index=True)


def metrics(table,law):
    rows = []
    for label,df in [('all',table),('isolated',table[~table.multiple_neighbors]),('multiple',table[table.multiple_neighbors])]:
        for name in table.columns[5:]:
            y = df.observed.to_numpy()
            err = df[name].to_numpy()-y
            mse = np.mean(err**2) if len(y) else np.nan
            rows.append(dict(group=label,model=name,pairs=len(y),rmse=np.sqrt(mse),
                         r2=1-mse/y.var() if len(y) and y.var()>0 else np.nan))
    y = table.observed.to_numpy()
    e0 = table[law['comparator']].to_numpy()-y
    e1 = table[law['primary']].to_numpy()-y
    blocks = np.array([np.sum(z,axis=0) for z in np.split(np.column_stack([np.ones(len(y)),y,y*y,e0*e0,e1*e1]),
                          1+np.flatnonzero(np.diff(table.frame.to_numpy()//int(FPS*law['block_seconds']))))])
    rng = np.random.default_rng(25481)
    interval = None
    if len(blocks)>1:
        n,sy,sy2,old,new = blocks[rng.integers(len(blocks),size=(law['bootstrap_replicates'],len(blocks)))].sum(1).T
        variance = sy2/n-(sy/n)**2
        good = variance>0
        if good.any():
            interval = np.quantile((old[good]-new[good])/n[good]/variance[good],[.025,.975]).tolist()
    return pd.DataFrame(rows),interval


def load_input(path):
    if path.suffix.lower()=='.pickle':
        # Existing trusted lab tracking export, with the standard swarm filename.
        return load_observations(path)
    with np.load(path,allow_pickle=False) as z:
        if float(z['fps'])!=FPS or not np.isclose(float(z['cap_px']),CAP,rtol=1e-7):
            raise ValueError('Use the frozen 25 Hz camera scale and coordinates.')
        ids = z['ids']
        p,phi = z['raw_pos'].astype(float),z['phi'].astype(float)
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids) or len(ids)<2
            or p.shape!=(len(phi),len(ids),2) or phi.shape!=p.shape[:2] or not np.isfinite(p).all()
            or not np.isfinite(phi).all() or not np.array_equal(z['frame'],np.arange(len(p)))):
            raise ValueError('Expected contiguous frames, unique integer IDs and complete finite poses.')
        tag = str(z['recording'])
        suffix = re.search(r'D(\d+)$',tag)
        if 'arena_cm' not in z and suffix is None:
            raise ValueError('Supply arena_cm or a recording label ending in D40 or D41.')
        arena = int(z['arena_cm']) if 'arena_cm' in z else int(suffix.group(1))
        return Track(f'{tag}_D{arena}',int(z['pwm']),ids,p,phi)


def score(path):
    law = frozen()
    raw = load_input(path)
    signature = track_digest(raw)
    known_source = set(law['training_files'].values())
    source_sha = digest(path)
    if path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as z:
            known_export = 'raw_sha256' in z and str(z['raw_sha256']) in known_source
    else:
        known_export = False
    if signature in law['training_tracks'].values() or source_sha in known_source or known_export:
        raise ValueError('This is a development recording. Use verify for software replay.')
    clean = Track(raw.tag,raw.pwm,raw.ids,project_centers(raw.pos)[0],raw.phi)
    table = predict(raw,clean,law)
    scores,ci = metrics(table,law)
    folder = RESULTS/signature
    folder.mkdir(parents=True,exist_ok=True)
    table.to_csv(folder/'predictions.csv.gz',index=False)
    scores.to_csv(folder/'scores.csv',index=False)
    receipt = dict(input=path.name,track_sha256=signature,source_sha256=source_sha,frozen_sha256=digest(ARTIFACT),
                primary=law['primary'],comparator=law['comparator'],paired_r2_gain_95=ci)
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(scores[scores.group=='all'].to_string(index=False))
    print(folder)


def verify():
    law = frozen()
    raw,clean = projected_tracks()
    results = []
    for tag in sorted(raw):
        cut = 3*len(raw[tag].pos)//4
        table = predict(raw[tag].part(0,cut),clean[tag].part(0,cut),law)
        with np.load(STUDY/f'{tag}__train.npz') as a,np.load(DIR/f'pooled__{tag}__train.npz') as d:
            np.testing.assert_array_equal(table[['frame','i','j']],a['origins'])
            np.testing.assert_allclose(table.observed,a['y'],atol=1e-9)
            features = {**dict(zip(a['names'],a['x'].T)),**dict(zip(d['names'],d['x'].T))}
            np.testing.assert_array_equal(table.prior,features['prior'])
            for name in ('compact','shared'):
                beta = law[name]
                matrix = np.column_stack([features[t] if t in features else np.prod(
                    [features[k] for k in t.split('*')],axis=0) for t in beta])
                np.testing.assert_allclose(table[name],features['prior']+matrix@list(beta.values()),rtol=1e-7,atol=2e-6)
        scores,_ = metrics(table,law)
        assert len(scores)==27 and np.isfinite(table.select_dtypes('number')).all().all()
        # Independent primal least squares check of the primary simultaneous-contact update.
        for frame,sub in list(table.groupby('frame'))[::100]:
            ids = raw[tag].ids
            robots = {rid:k for k,rid in enumerate(ids)}
            i = np.array([robots[rid] for rid in sub.i])
            j = np.array([robots[rid] for rid in sub.j])
            rel = raw[tag].pos[frame,i]-raw[tag].pos[frame,j]
            n = rel/np.linalg.norm(rel,axis=1)[:,None]
            a = np.zeros((len(sub),len(ids),2))
            a[np.arange(len(a)),i],a[np.arange(len(a)),j] = n,-n
            a = a.reshape(len(sub),-1)
            system = np.vstack((a,np.sqrt(law['ridge'])*np.eye(a.shape[1])))
            rhs = np.r_[sub.shared-sub.prior,np.zeros(a.shape[1])]
            dv = np.linalg.lstsq(system,rhs,rcond=None)[0]
            np.testing.assert_allclose(sub.shared_coherent,sub.prior+a@dv,rtol=2e-6,atol=2e-5)
        results.append(dict(recording=tag,verified_pair_origins=len(table),split='development prefix'))
    RESULTS.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS/'software_replay.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action',required=True)
    commands.add_parser('freeze')
    commands.add_parser('verify')
    commands.add_parser('score').add_argument('recording',type=Path)
    args = parser.parse_args()
    if args.action=='score':
        score(args.recording)
    else:
        {'freeze':freeze,'verify':verify}[args.action]()
