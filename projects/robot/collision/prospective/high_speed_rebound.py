"""Score calibrated high-speed trial tracks using independently annotated rim contact."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from epde.operators.common.multisample import joint_subset_table


def measurements(trial, poses):
    frames = ['last_clear_before','first_touch','last_touch','first_clear_after']
    marks = np.asarray([getattr(trial,k) for k in frames],dtype=float)
    if not np.isfinite(marks).all() or not np.equal(marks,np.floor(marks)).all():
        raise ValueError('missing or noninteger annotation')
    b,c,last,a = marks.astype(int)
    if not b<c<=last<a:
        raise ValueError('invalid contact bracket')
    if (not np.isfinite(poses.frame).all() or not (poses.frame==np.floor(poses.frame)).all()
            or poses.frame.duplicated().any()):
        raise ValueError('duplicate or invalid frame index')
    data = poses.sort_values('frame').set_index('frame')
    if len(data)<6 or np.any(np.diff(data.time_s)<=0) or not np.isfinite(data.time_s).all():
        raise ValueError('invalid native timestamps')
    times=data.time_s
    rel=data[['x_b_mm','y_b_mm']].to_numpy()-data[['x_a_mm','y_a_mm']].to_numpy()
    if c not in data.index or a not in data.index or b not in data.index or last not in data.index:
        raise ValueError('annotation outside pose track')
    n=rel[data.index.get_loc(c)]
    if not np.isfinite(n).all() or np.linalg.norm(n)==0:
        raise ValueError('undefined contact normal')
    s=pd.Series(rel@(n/np.linalg.norm(n)),index=data.index)
    for w in (3,5):
        windows=[np.arange(b-w+1,b+1),np.arange(a,a+w)]
        if any(not np.isfinite(s.reindex(ix)).all() for ix in windows):
            if w==3:
                raise ValueError('missing body centers around contact')
            continue
        slopes=[]
        for ix in windows:
            t=times.loc[ix].to_numpy()
            t=t-t.mean()
            slopes.append(t@s.loc[ix].to_numpy()/(t@t))
        pre,post=slopes
        yield dict(trial_id=trial.trial_id,session=trial.session,robot_a=trial.robot_a,robot_b=trial.robot_b,
                   target_speed_mm_s=trial.target_speed_mm_s,spin_a=trial.spin_a,spin_b=trial.spin_b,
                   opposite=trial.spin_a!=trial.spin_b,positions=w,incoming_mm_s=pre,outgoing_mm_s=post,
                   effective_e=-post/pre if pre<0 else np.nan,
                   entry_bracket_ms=1000*(times[c]-times[b]),exit_bracket_ms=1000*(times[a]-times[last]),
                   pre_span_ms=1000*(times[b]-times[b-w+1]),post_span_ms=1000*(times[a+w-1]-times[a]))


def heldout(rows):
    scores=[]
    if rows.empty:
        return scores
    for w,group in rows[rows.incoming_mm_s<0].groupby('positions'):
        blocks=list(group.groupby('session'))
        if len(blocks)<2:
            continue
        for held,test in blocks:
            train=[]
            for name,g in blocks:
                if name!=held:
                    speed=-g.incoming_mm_s.to_numpy()
                    train.append((name,np.column_stack((speed,speed*g.opposite)),g.outgoing_mm_s,np.ones(len(g))))
            _,_,_,fits,ranks=joint_subset_table(train,subsets=[1,3,4],discrepancy='wape',weighting='samples')
            for model,coef,rank in zip(('shared_e','spin_specific_e','training_mean'),fits,ranks):
                if not rank:
                    continue
                pred=-test.incoming_mm_s*(coef[0]+coef[1]*test.opposite)+coef[2]
                y=test.outgoing_mm_s.to_numpy()
                mse=np.mean((pred-y)**2)
                scores.append(dict(held=held,model=model,positions=w,events=len(y),slope=coef[0],
                                   extra_opposite_slope=coef[1],intercept=coef[2],rmse_mm_s=np.sqrt(mse),
                                   r2=1-mse/np.var(y) if np.var(y)>0 else np.nan))
    return scores


def run(sheet, tracks, out):
    trials=pd.read_csv(sheet)
    if trials.trial_id.duplicated().any():
        raise ValueError('Trial IDs must be unique')
    poses=pd.read_csv(tracks)
    if not poses.trial_id.isin(trials.trial_id).all():
        raise ValueError('Pose table contains an unknown trial')
    results,audit=[],[]
    for t in trials.itertuples():
        label=t.observed_contact if pd.notna(t.observed_contact) else 'unrecorded'
        status=label
        if label=='single':
            try:
                if any(pd.isna(getattr(t,k)) for k in ['session','robot_a','robot_b','spin_a','spin_b']):
                    raise ValueError('missing session, robot IDs or spin assignment')
                if t.robot_a==t.robot_b:
                    raise ValueError('identical robot IDs')
                results.extend(list(measurements(t,poses[poses.trial_id==t.trial_id])))
                status='scored'
            except ValueError as err:
                status=str(err)
        audit.append(dict(trial_id=t.trial_id,observed_contact=label,status=status))
    out.mkdir(parents=True,exist_ok=True)
    velocity_fields=('trial_id session robot_a robot_b target_speed_mm_s spin_a spin_b opposite positions '
                     'incoming_mm_s outgoing_mm_s effective_e entry_bracket_ms exit_bracket_ms '
                     'pre_span_ms post_span_ms').split()
    score_fields=('held model positions events slope extra_opposite_slope intercept rmse_mm_s r2').split()
    data=pd.DataFrame(results,columns=velocity_fields)
    data.to_csv(out/'trial_velocities.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'trial_audit.csv',index=False)
    pd.DataFrame(heldout(data),columns=score_fields).to_csv(out/'session_holdouts.csv',index=False)
    print(f'{len(trials)} planned trials; {sum(r["status"]=="scored" for r in audit)} scored.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('sheet',type=Path)
    parser.add_argument('tracks',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    run(args.sheet,args.tracks,args.out)
