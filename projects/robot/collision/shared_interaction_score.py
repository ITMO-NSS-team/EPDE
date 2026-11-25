"""Score a calibrated recording with frozen pair equations and references."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np
import pandas as pd

from new_pair_interactions import read_stage
from shared_interaction_data import D_MM, MS, extract
from shared_interaction_fit import free_basis, history_basis, library

MODEL = Path(__file__).parent/'prospective/shared_pair_v1.json'
REFERENCES = ['persistence','clamp','history','learned_free','learned_clamp']
MODELS = ['full','compact',*REFERENCES]
KEY = ['recording','stage','frame','seconds','i','j']


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def predict(events,model):
    """Forecasts depend only on the positions/kinematics through the gate."""
    results=[]
    for ms in MS:
        h=ms/1000
        block=model['horizons'][str(ms)]
        for a in 'nt':
            fit=block[a]
            free=events['recent_'+a].to_numpy()
            x=library(events,h,a,free)
            learned=free_basis(events,a)@block['isolated_motion']
            gap=np.maximum(events.r.to_numpy()-1,0)
            floor=-gap/h
            out=events[KEY].assign(horizon_ms=ms,axis=a)
            for name in ('full','compact'):
                out[name]=(free+x@fit[name])*D_MM
            out['persistence']=free*D_MM
            out['clamp']=(np.maximum(free,floor) if a=='n' else free)*D_MM
            out['learned_free']=learned*D_MM
            out['learned_clamp']=(np.maximum(learned,floor) if a=='n' else learned)*D_MM
            out['history']=(history_basis(events,a)@fit['history'])*D_MM
            results.append(out)
    return pd.concat(results,ignore_index=True)


def evaluate(events,model):
    prediction=predict(events,model)
    prediction['observed']=np.concatenate([events[f'y{ms}_{a}'].to_numpy()*D_MM for ms in MS for a in 'nt'])
    return prediction


def summaries(predictions):
    rows=[]
    for (ms,a),g in predictions.groupby(['horizon_ms','axis']):
        y=g.observed.to_numpy()
        losses=(g[MODELS].to_numpy()-y[:,None])**2
        rmse=np.sqrt(losses.mean(0))
        # New recording, one weight per gate. Re-select the lowest-error frozen
        # reference in each resample when quantifying uncertainty on that comparison.
        best=2+rmse[2:].argmin()
        blocks=pd.DataFrame(losses,columns=MODELS).assign(
            stage=g.stage.to_numpy(),block=g.seconds.to_numpy()//10,n=1).groupby(['stage','block']).sum()
        ci=np.full((2,2),np.nan)
        if len(blocks)>1:
            v=blocks.to_numpy()
            rng=np.random.default_rng(731)
            draws=rng.integers(len(v),size=(2000,len(v)))
            sums=v[draws].sum(1)
            sampled=np.sqrt(sums[:,:-1]/sums[:,-1,None])
            ci=np.quantile(sampled[:,2:].min(1)[:,None]-sampled[:,:2],[.025,.975],axis=0)
        for k,name in enumerate(MODELS):
            rows.append(dict(horizon_ms=ms,axis=a,model=name,events=len(g),blocks=len(blocks),
                rmse_mm_s=rmse[k],mae_mm_s=np.mean(np.sqrt(losses[:,k])),
                r2=1-losses[:,k].mean()/np.var(y) if np.var(y)>0 else np.nan,
                strongest_reference=MODELS[best],gain_mm_s=rmse[best]-rmse[k],
                gain_ci_low=ci[0,k] if k<2 else np.nan,gain_ci_high=ci[1,k] if k<2 else np.nan))
    return pd.DataFrame(rows)


def calibrated_stage(rec,hz,pwm,ids,xy,phi,bad,cal):
    cx,cy,rad,d=cal
    if not np.isfinite(cal).all() or min(rad,d)<=0:
        raise ValueError('Calibration needs a finite center, positive center-envelope radius and diameter.')
    if hz not in (25,50) or not np.isfinite(pwm):
        raise ValueError('Expected a uniformly sampled 25 or 50 Hz recording and a finite PWM.')
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or xy.ndim!=2 or xy.shape[1]!=len(ids):
        raise ValueError('Expected one unique robot ID per position column.')
    if phi.shape!=xy.shape or bad.shape!=xy.shape:
        raise ValueError('Positions, headings, and quality flags must use identical frame and robot axes.')
    mask=bad|~np.isfinite(xy)|~np.isfinite(phi)
    pos=(xy-complex(cx,cy))/d
    return rec,hz,pwm,ids,pos,phi,mask,rad/d


def input_stages(path,arena=None,diameter=None,calibration=None,pwm=None):
    if path.suffix.lower()=='.npz':
        if calibration is not None:
            raise ValueError('For an NPZ use its embedded arena or --arena CX CY R.')
        with np.load(path,allow_pickle=False) as z:
            frames=z['frame']
            raw=z['raw_pos']
            if raw.ndim!=3 or raw.shape[2]!=2 or frames.shape!=(len(raw),) or not np.all(np.diff(frames)==1) or (len(frames) and frames[0]!=0):
                raise ValueError('NPZ needs contiguous zero-based frames and [frame, robot, x/y] raw_pos.')
            if arena is None and 'arena' not in z:
                raise ValueError('Supply --arena CX CY R or include the arena field in the NPZ.')
            geom=np.asarray(z['arena'] if arena is None else arena,dtype=float)
            if geom.shape!=(3,):
                raise ValueError('Use --arena CX CY R in the coordinate units of raw_pos.')
            d=diameter if diameter is not None else float(z['diameter'] if 'diameter' in z else z['cap_px'])
            # A decoded and calibrated NPZ carries the upstream tracker quality decisions.
            yield calibrated_stage(str(z['recording']),float(z['fps']),pwm if pwm is not None else float(z['pwm']),
                z['ids'],raw[...,0]+1j*raw[...,1],z['phi'],z['quality_flags']!=0,[*geom,d])
        return
    if path.suffix.lower()!='.zip':
        raise ValueError('Expected a calibrated NPZ or a population-sweep ZIP.')
    if calibration is not None and (arena is not None or diameter is not None):
        raise ValueError('Use a stage calibration CSV or a common arena and diameter.')
    scales=pd.read_csv(calibration) if calibration is not None else None
    if scales is not None and 'experiment' in scales and scales.experiment.nunique()>1:
        scales=scales[scales.experiment==path.stem]
    with zipfile.ZipFile(path) as z:
        members={}
        for name in z.namelist():
            match=re.search(r'/data_N=(\d+)_',name)
            if match:
                n=int(match[1])
                if n in members:
                    raise ValueError(f'Duplicate occupancy stage {n} in {path.name}.')
                members[n]=name
        if not members:
            raise ValueError('No population-sweep stages in the ZIP.')
        for n,name in sorted(members.items()):
            if n<2:
                continue
            if scales is None:
                if arena is None or diameter is None:
                    raise ValueError('Supply --calibration CSV or both --arena CX CY R and --diameter D for this sweep.')
                cal=[*arena,diameter]
            else:
                row=scales[scales.N==n]
                if len(row)!=1:
                    raise ValueError(f'Expected one calibration row for {path.stem}, occupancy {n}.')
                cal=row.iloc[0][['center_x','center_y','radius','diameter_archive_units']].to_numpy(float)
            setting=re.search(r'PWM_(\d+)',name)
            if pwm is None and setting is None:
                raise ValueError('Supply the recorded --pwm setting.')
            _,ids,xy,phi,bad=read_stage(z,name,n)
            jump=(abs(np.diff(xy,axis=0))>.25*cal[-1])|(abs(np.diff(phi,axis=0))>np.pi/3)
            bad[1:] |= jump
            bad[:-1] |= jump
            yield calibrated_stage(path.stem,50,float(setting[1]) if pwm is None else pwm,ids,xy,phi,bad,cal)


def extract_input(path,**kwargs):
    events,audit=[],[]
    for args in input_stages(path,**kwargs):
        _,parts,counts=extract(*args,mode='native')
        events.extend(p for p in parts if p.kind.iloc[0]=='near')
        audit.extend(c for c in counts if c['kind']=='near')
    return pd.concat(events,ignore_index=True) if events else pd.DataFrame(),pd.DataFrame(audit)


def score_file(path,output,model_path=MODEL,**kwargs):
    model=json.loads(model_path.read_text())
    if model.get('version')!='shared-pair-v1':
        raise ValueError('Expected the frozen shared-pair-v1 model.')
    events,audit=extract_input(path,**kwargs)
    file_hash=digest(path)
    known=model['development_inputs'].get(file_hash)
    # ZIP member digests also identify the same tracks after repacking.
    if known is None and path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if '/data_N=' in name:
                    with z.open(name) as stream:
                        fingerprint=hashlib.file_digest(stream,'sha256').hexdigest()
                    if fingerprint in model['development_members']:
                        known=model['development_members'][fingerprint]
                        break
    if known is None and path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as z:
            sig=pose_hash(z['raw_pos'],z['ids'])
        known=model['development_poses'].get(sig)
    receipt=dict(input_file=path.name,input_sha256=file_hash,model_sha256=digest(model_path),
                 identity='development overlap' if known else 'no known input match',
                 matched_development_recording=known,events=len(events),prediction_unit='mm/s',
                 calibration={k:str(v) for k,v in kwargs.items() if v is not None},
                 calibration_sha256=digest(kwargs['calibration']) if kwargs.get('calibration') else None)
    receipt['implementation_matches_freeze']=all(digest(Path(__file__).with_name(n))==sha
        for n,sha in model['implementation_sha256'].items())
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'Output directory already exists: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage=Path(tmp)/'score'
        stage.mkdir()
        audit.to_csv(stage/'coverage.csv',index=False)
        if len(events):
            predictions=evaluate(events,model)
            predictions.to_parquet(stage/'predictions.parquet',index=False)
            summaries(predictions).to_csv(stage/'scores.csv',index=False)
        (stage/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        stage.rename(output)
    return receipt


def pose_hash(xy,ids):
    order=np.argsort(ids)
    arr=np.ascontiguousarray(xy[:,order],dtype='<f8')
    h=hashlib.sha256(np.array(arr.shape,dtype='<i8').tobytes())
    h.update(arr)
    return h.hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arena',nargs=3,type=float,metavar=('CX','CY','CENTER_RADIUS'))
    parser.add_argument('--diameter',type=float)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--pwm',type=float)
    a=parser.parse_args()
    print(json.dumps(score_file(a.recording,a.output,arena=a.arena,diameter=a.diameter,
                                calibration=a.calibration,pwm=a.pwm),indent=2))
