"""Compare common-gate forecasts and matched optical contrasts across recordings."""
import numpy as np
import pandas as pd
from scipy.stats import t as student

from video_gate_response import DEST, optimal_matches
from joint_rebound import errors
from epde.operators.common.multisample import joint_subset_table


def read_events():
    keys = ['event','recording','gate_frame','release']
    selected = pd.read_csv(DEST/'selection.csv').drop(columns=['closing','side'])
    labels = pd.read_csv(DEST/'optical_labels.csv')
    kinematics = pd.read_csv(DEST/'kinematics.csv')
    data = selected.merge(labels,on=keys,validate='one_to_one').merge(
        kinematics,on='event',validate='one_to_one').set_index('event')
    if len(data)!=len(selected) or not np.isfinite(data.registration_residual).all():
        raise ValueError('Incomplete registration or optical labels')
    data['abs_side'] = data.side.abs()
    return data[data.closing>0].copy()


def contrasts(data):
    rows=[]
    for batch,subset in [('first',data[data.batch==1]),('combined',data)]:
        for recording,group in subset.groupby('recording'):
            touch=group[group.optical_label=='apparent contact']
            miss=group[group.optical_label=='clear gap']
            for match_on,early in [('whole pass',touch),('by 40 ms',touch[touch.first_visible_touch_offset<=1]),
                                   ('by 80 ms',touch[touch.first_visible_touch_offset<=2])]:
                for i,j,cost in optimal_matches(early,miss):
                    a,b=group.loc[i],group.loc[j]
                    row=dict(batch=batch,match_on=match_on,recording=recording,contact_event=i,gap_event=j,
                             cost=cost,third_near=a.third_near,spin_group=a.spin_group,
                             same_robot_pair=(a.i==b.i and a.j==b.j),touch_offset=a.first_visible_touch_offset)
                    for col in ['closing','side','gate_gap','wall_gap','change_40','change_80','lead_change']:
                        row[col]=a[col]-b[col]
                    rows.append(row)
    return pd.DataFrame(rows)


def estimate(pairs):
    out=[]
    for (batch,mode),g in pairs.groupby(['batch','match_on']):
        for condition,sub in [('all',g),('without close third',g[~g.third_near])]:
            if sub.empty:
                continue
            columns=['change_40','change_80','lead_change']
            means=sub.groupby('recording')[columns].mean()
            for endpoint in columns:
                v=means[endpoint]
                margin=student.ppf(.975,len(v)-1)*v.sem() if len(v)>1 else np.nan
                out.append(dict(batch=batch,match_on=mode,condition=condition,endpoint=endpoint,
                                pairs=len(sub),recordings=len(v),mean_contact_minus_gap_px_s=v.mean(),
                                low95=v.mean()-margin,high95=v.mean()+margin))
    return pd.DataFrame(out)


def joint_scores(data,pairs):
    predictions=[]
    matched=pairs[(pairs.batch=='combined')&(pairs.match_on=='whole pass')]
    idx=pd.Index(np.r_[matched.contact_event,matched.gap_event])
    for horizon in (40,80):
        g=data.copy()
        g['clearance_term']=(g.closing-g.gate_gap.clip(lower=0)/(horizon/1000)).clip(lower=0)
        g['visible_by_endpoint']=(g.first_visible_touch_offset<=horizon/40).astype(float)
        for scope,cohort in [('all selected',g),('matched',g.loc[idx])]:
            columns=['closing','abs_side','clearance_term']
            models={'constant':8,'motion':11,'motion + clearance':15}
            if scope=='matched':
                columns.append('visible_by_endpoint')
                models={'constant':16,'motion':19,'motion + clearance':23,
                        'motion + observed contact':27}
            for held,test in cohort.groupby('recording'):
                train=[(name,d[columns].to_numpy(),d[f'change_{horizon}'].to_numpy(),np.ones(len(d)))
                       for name,d in cohort.groupby('recording') if name!=held]
                _,_,_,coefs,ranks=joint_subset_table(train,subsets=list(models.values()),
                                                      discrepancy='wape',weighting='samples')
                for (model,_),beta,rank in zip(models.items(),coefs,ranks):
                    if not rank:
                        motion=list(models).index('motion')
                        if model=='motion' or not ranks[motion]:
                            raise ValueError('The incoming-motion baseline is unidentifiable')
                        beta=coefs[motion]
                    pred=test[columns].to_numpy()@beta[:-1]+beta[-1]
                    predictions.extend(dict(event=i,horizon_ms=horizon,scope=scope,held=held,model=model,
                                            observed=y,predicted=p,fallback_to_motion=not rank,
                                            coefficients=' '.join(map(str,beta)))
                                       for i,y,p in zip(test.index,test[f'change_{horizon}'],pred))
    predictions=pd.DataFrame(predictions)
    scores=[]
    for (h,scope,model),g in predictions.groupby(['horizon_ms','scope','model']):
        for label,d in [('balanced',g),*g.groupby('held')]:
            w=1/d.groupby('held')['event'].transform('size')
            rmse,r2=errors(d.observed,d.predicted,w)
            scores.append(dict(horizon_ms=h,scope=scope,model=model,held=label,n=len(d),
                               recordings=d.held.nunique(),rmse_px_s=rmse,r2=r2))
    return predictions,pd.DataFrame(scores)


def pair_scores(data,pairs,forecasts):
    """Evaluate the change difference itself on each recording's held-out pairs."""
    sample=pairs[(pairs.batch=='combined')&(pairs.match_on=='whole pass')].copy()
    sample['delta_abs_side']=(data.loc[sample.contact_event,'abs_side'].to_numpy()
                              -data.loc[sample.gap_event,'abs_side'].to_numpy())
    rows=[]
    for ms in (40,80):
        for held,test in sample.groupby('recording'):
            cols=['closing','delta_abs_side']
            blocks=[(r,g[cols].to_numpy(),g[f'change_{ms}'].to_numpy(),np.ones(len(g)))
                    for r,g in sample.groupby('recording') if r!=held]
            _,_,_,fits,ranks=joint_subset_table(blocks,subsets=[4,7],discrepancy='wape',weighting='samples')
            if not ranks.all():
                raise ValueError('Rank-deficient matched difference fit')
            gr=forecasts[(forecasts.held==held)&(forecasts.scope=='matched')&
                         (forecasts.horizon_ms==ms)&(forecasts.model=='motion')].set_index('event')
            estimates={'zero difference':np.zeros(len(test)),
                       'difference of gate forecasts':(gr.loc[test.contact_event,'predicted'].to_numpy()
                                                      -gr.loc[test.gap_event,'predicted'].to_numpy())}
            for name,beta in zip(('conditional mean contrast','conditional joint contrast'),fits):
                estimates[name]=test[cols].to_numpy()@beta[:-1]+beta[-1]
            for name,pred in estimates.items():
                rows.extend(dict(horizon_ms=ms,held=held,contact_event=i,gap_event=j,model=name,
                                 observed=y,predicted=p) for i,j,y,p in zip(
                                     test.contact_event,test.gap_event,test[f'change_{ms}'],pred))
    pred=pd.DataFrame(rows)
    scores=[]
    for (ms,name),g in pred.groupby(['horizon_ms','model']):
        for label,sub in [('balanced',g),*g.groupby('held')]:
            weights=1/sub.groupby('held')['observed'].transform('size')
            rmse,r2=errors(sub.observed,sub.predicted,weights)
            scores.append(dict(horizon_ms=ms,model=name,held=label,pairs=len(sub),rmse_px_s=rmse,r2=r2))
    return pred,pd.DataFrame(scores)


if __name__=='__main__':
    events=read_events()
    pairs=contrasts(events)
    assert not pairs.duplicated(['batch','match_on','contact_event']).any()
    assert not pairs.duplicated(['batch','match_on','gap_event']).any()
    pairs.to_csv(DEST/'matched_pairs.csv',index=False)
    estimate(pairs).to_csv(DEST/'matched_effects.csv',index=False)
    preds,scores=joint_scores(events,pairs)
    preds.to_csv(DEST/'heldout_predictions.csv',index=False)
    scores.to_csv(DEST/'heldout_scores.csv',index=False)
    paired,paired_scores=pair_scores(events,pairs,preds)
    paired.to_csv(DEST/'pair_predictions.csv',index=False)
    paired_scores.to_csv(DEST/'pair_scores.csv',index=False)
