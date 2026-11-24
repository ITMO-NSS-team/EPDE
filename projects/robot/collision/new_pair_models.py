"""Archive-held-out ablations, native EPDE fronts, and event-level normal responses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
import subprocess
import sys
from unittest.mock import patch

for setting in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[setting]='1'

import numpy as np
import pandas as pd

from new_pair_interactions import DEST, HZ, H

HISTORY = {'outward':['incoming','prior','bend','abs_side'],
           'tangent':['side','prior_side']}
GAP = {'outward':['proximity','hinge','closing'], 'tangent':['slide_near','side_closing']}
THIRD = {'outward':['third_normal','crowd'], 'tangent':['third_tangent','crowd_side']}
SPIN = {'outward':['same_near','spin_magnitude'], 'tangent':['spin_slide','same_slide']}
SEEDS = (923701,923702)
USE_DRIVE = False
PHYSICAL = False
OUTPUT = DEST


def features():
    source = DEST/'physical' if PHYSICAL else DEST
    df = pd.read_csv(source/'origins.csv.gz')
    df['abs_side'] = df.side.abs()
    df['spin_magnitude'] = df.spin_slide.abs()
    df['side_closing'] = df.side * df.closing
    df['crowd_side'] = df.side * df.crowd
    df['same_slide'] = df.slide_near * df.same
    return df


def split(data,held):
    # Endpoints and incoming histories of the validation tails do not touch training rows.
    other = data.experiment!=held
    cut = .75 * data.end * HZ
    return data[other & (data.frame+H<cut)],data[other & (data.frame-22>=cut)],data[~other]


def terms(target):
    return HISTORY[target]+GAP[target]+THIRD[target]+SPIN[target]


def inputs(held=None):
    manifest = pd.read_csv(DEST/'audit.csv')[['experiment','N','end']]
    data = features()
    if 'end' not in data:
        data = data.merge(manifest,on=['experiment','N'],validate='many_to_one')
    if USE_DRIVE:
        drive = pd.read_csv(DEST/f'{held}_drive.csv.gz')
        if PHYSICAL:
            refs = json.loads((DEST/'inputs.json').read_text())
            scales = pd.read_csv(DEST/'physical'/'scale.csv')
            scales['ratio'] = scales.experiment.map({e:r['reference_spacing'] for e,r in refs.items()})/scales.diameter_archive_units
            drive = drive.merge(scales[['experiment','N','ratio']],on=['experiment','N'],validate='many_to_one')
            for col in ('free_normal','free_side'):
                drive[col] *= drive.ratio
        expected = len(data)
        data = data.merge(drive,on=['experiment','N','frame','id_i','id_j'],validate='one_to_one')
        if len(data)!=expected:
            raise ValueError(f'Missing free-motion features for {expected-len(data)} origins')
        data['free_hinge'] = np.maximum(-data.gap/(H/HZ)-data.free_normal,0.)
        data['clipped_free'] = np.maximum(data.free_normal,-data.gap/(H/HZ))
    return data


def fit(df,target,names):
    x = np.column_stack([np.ones(len(df)),df[names].to_numpy()])
    w = 1/np.sqrt(df.groupby('experiment').frame.transform('count').to_numpy())
    return np.linalg.lstsq(w[:,None]*x,w*df[target],rcond=None)[0]


def metrics(y,p):
    loss = np.mean((y-p)**2)
    return dict(r2=1-loss/y.var() if y.var()>0 else np.nan,rmse=np.sqrt(loss),mae=np.mean(abs(y-p)))


def predict(df,names,coef):
    return coef[0]+df[names].to_numpy()@coef[1:]


def native(held,target,seed):
    import expanded_native as engine
    from discovery import write_json
    train,_,_ = split(inputs(held),held)
    screen = pd.concat([d.sample(min(640,len(d)),random_state=542+i).sort_values(['N','frame','id_i','id_j'])
                        for i,(_,d) in enumerate(train.groupby('experiment'))])
    names = terms(target)
    center = screen[names].mean()
    x = {n:(screen[n]-center[n]).to_numpy() for n in names}
    y = screen[target].to_numpy()
    with patch.multiple(engine,SEED=seed,RHS=6,FACTORS=1):
        receipt = engine.search_screen(y-y.mean(),x,{},(),x.__getitem__)
    write_json(OUTPUT/f'{held}_{target}_{seed}.json',
               dict(held=held,target=target,screen_center=center.to_dict(),target_mean=y.mean(),**receipt))


def launch(job):
    held,target,seed = job
    path = OUTPUT/f'{held}_{target}_{seed}.log'
    with path.open('w',encoding='utf-8') as log:
        cmd = [sys.executable,'-u',__file__,'--native',held,target,str(seed)]
        if USE_DRIVE:
            cmd.append('--drive')
        if PHYSICAL:
            cmd.append('--physical')
        proc = subprocess.run(cmd,stdout=log,
                              stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED='0'))
    if proc.returncode:
        raise RuntimeError(path.read_text(errors='replace')[-2500:])
    print('Searched',*job,flush=True)


def selection():
    rows,fronts = [],[]
    experiments = sorted(pd.read_csv(DEST/'audit.csv').experiment.unique())
    for held,target in product(experiments,HISTORY):
        data = inputs(held)
        tr,va,te = split(data,held)
        base = HISTORY[target]
        gap = base+GAP[target]
        sets = {'mean':[], 'motion history':base, 'history and clearance':gap,
                'clearance and third robot':gap+THIRD[target], 'clearance and spin':gap+SPIN[target],
                'full':terms(target)}
        if USE_DRIVE:
            sets['compact interaction'] = (['free_normal','free_hinge','proximity','third_normal']
                    if target=='outward' else ['free_side','side','slide_near','third_tangent','spin_slide'])
            if target=='outward':
                sets['clearance clamp'] = ['clipped_free','proximity','third_normal']
        proposals = []
        for seed in SEEDS:
            receipt = json.loads((OUTPUT/f'{held}_{target}_{seed}.json').read_text())
            for eq in receipt['front']:
                keys = sorted(eq['terms'])
                beta = fit(tr,target,keys)
                valid = np.mean([metrics(d[target].to_numpy(),predict(d,keys,beta))['rmse']
                                 for _,d in va.groupby('experiment')])
                raw_b = np.array([eq['terms'][k] for k in keys])
                raw = np.r_[receipt['target_mean']-raw_b@np.array([receipt['screen_center'][k] for k in keys]),raw_b]
                fronts.append(dict(held=held,target=target,seed=seed,terms=' + '.join(keys),wape=eq['wape'],
                    dyad=eq['dyad'],validation_rmse=valid,refit=json.dumps(beta.tolist()),
                    raw_r2=metrics(te[target].to_numpy(),predict(te,keys,raw))['r2'],
                    **metrics(te[target].to_numpy(),predict(te,keys,beta))))
                proposals.append((valid,len(keys),keys))
        sets['native EPDE front'] = min(proposals)[-1]
        if USE_DRIVE:
            free = te.free_normal.to_numpy() if target=='outward' else te.free_side.to_numpy()
            candidates = {'free motion only':free}
            if target=='outward':
                candidates['free motion clipped to clearance'] = np.maximum(free,-te.gap.to_numpy()/(H/HZ))
            for label,pred in candidates.items():
                rows.append(dict(held=held,target=target,model=label,group='all',count=len(te),terms=label,
                                 coef='{}',**metrics(te[target].to_numpy(),pred)))
        for label,keys in sets.items():
            beta = fit(tr,target,keys)
            for group,sub in [('all',te),('without third',te[~te.third]),('with third',te[te.third]),
                              ('two robots',te[te.N==2]),('approaching',te[te.incoming<0])]:
                if sub.empty:
                    continue
                rows.append(dict(held=held,target=target,model=label,group=group,count=len(sub),
                    terms=' + '.join(keys),coef=json.dumps(dict(zip(['constant',*keys],beta))),
                    **metrics(sub[target].to_numpy(),predict(sub,keys,beta))))
    pd.DataFrame(fronts).to_csv(OUTPUT/'native_fronts.csv',index=False)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT/'heldout.csv',index=False)
    print(results[results.group=='all'].groupby(['target','model'])[['r2','rmse']].mean().round(4))


def describe():
    events = pd.read_csv(DEST/'encounters.csv.gz')
    starts = events[events.incoming<0].copy()
    starts['reversed'] = starts.outward>0
    starts['reversed_100ms'] = starts.after_100ms>0
    starts['normal_change'] = starts.outward-starts.incoming
    starts['side_change'] = starts.tangent-starts.side
    starts['spin_aligned_change'] = np.where(starts.same,np.sign(starts.spin_i+starts.spin_j)*starts.side_change,np.nan)
    rows = []
    for gate,d in starts.groupby('gate'):
        for exp,part in [('all',d),*d.groupby('experiment')]:
            for group,sub in [('all',part),('two robots',part[part.N==2]),('without third',part[~part.third]),
                              ('with third',part[part.third]),('co-rotating',part[part.same]),
                              ('counter-rotating',part[part.spin_i*part.spin_j<0]),
                              ('N 2-10',part[part.N<=10]),('N 11-20',part[part.N.between(11,20)]),
                              ('N 21-30',part[part.N>20])]:
                if len(sub):
                    rows.append(dict(experiment=exp,gate=gate,group=group,events=len(sub),
                        inward_median=sub.incoming.median(),outward_median=sub.outward.median(),
                        change_median=sub.normal_change.median(),reversed=sub.reversed.mean(),
                        reversed_100ms=sub.reversed_100ms.mean(),tangent_change=sub.side_change.mean(),
                        spin_aligned=sub.spin_aligned_change.mean()))
    pd.DataFrame(rows).to_csv(DEST/'event_summary.csv',index=False)
    # Test a through-zero restitution model at 40 ms separately from the 0.4 s forecast.
    scores = []
    for gate,df in events[events.incoming_short<0].groupby('gate'):
        for held in sorted(df.experiment.unique()):
            tr,te = df[df.experiment!=held],df[df.experiment==held]
            w = 1/tr.groupby('experiment').frame.transform('count').to_numpy()
            closing,y = -tr.incoming_short.to_numpy(),tr.immediate.to_numpy()
            e = max(0.,np.sum(w*closing*y)/np.sum(w*closing**2))
            for name,pred,coef in [('restitution',-e*te.incoming_short,e),
                    ('training mean',np.average(y,weights=w),np.average(y,weights=w))]:
                scores.append(dict(held=held,gate=gate,model=name,events=len(te),coef=coef,
                                   reversal_fraction=np.mean(te.immediate>0),
                                   **metrics(te.immediate.to_numpy(),np.asarray(pred))))
    pd.DataFrame(scores).to_csv(DEST/'short_restitution.csv',index=False)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--native',nargs=3)
    p.add_argument('--stage',choices=['describe','search','score'])
    p.add_argument('--drive',action='store_true')
    p.add_argument('--physical',action='store_true')
    a = p.parse_args()
    if a.drive:
        USE_DRIVE = True
        OUTPUT = DEST/'with_drive'
        OUTPUT.mkdir(exist_ok=True)
        HISTORY['outward'] = [*HISTORY['outward'],'free_normal']
        HISTORY['tangent'] = [*HISTORY['tangent'],'free_side']
        GAP['outward'] = [*GAP['outward'],'free_hinge']
    if a.physical:
        PHYSICAL = True
        OUTPUT = DEST/'physical'/('with_drive' if USE_DRIVE else 'pair_history')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        SEEDS = (923701,)
    if a.native:
        native(a.native[0],a.native[1],int(a.native[2]))
    elif a.stage=='search':
        experiments = sorted(pd.read_csv(DEST/'audit.csv').experiment.unique())
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(launch,product(experiments,HISTORY,SEEDS)))
    elif a.stage=='score':
        selection()
    else:
        describe()
