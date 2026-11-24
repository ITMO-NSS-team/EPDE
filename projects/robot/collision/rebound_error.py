"""Compare models of measured outgoing speed at entry and one camera interval later."""
import json

import numpy as np
import pandas as pd

from physics import CAP, FPS, OUT, SIGMA, heading, rotate
from projected_data import projected_tracks
from restitution_epde import DEST as PREVIOUS, score, weights


DEST = PREVIOUS/'error_reduction'
DRIVE = ['pwm20']
PRE = DRIVE+['closing_pre']
HISTORY = PRE+['normal_120ms','normal_400ms','abs_tangent_120ms','mean_abs_spin',
               'relative_heading','tangent_spin','nearby']
GAP = PRE+['depth_px']
CROSS = GAP+['crossing_speed']
LIBRARIES = {'pwm':DRIVE,'pre_speed':PRE,'history':HISTORY,'entry_depth':GAP,
             'entry_geometry':CROSS,'history_geometry':HISTORY+['depth_px','crossing_speed']}
KEY = ['recording','entry','i','j']


def features(track,df,flags,arenas):
    """Predictor columns end at entry. Secondary labels and their eligibility are separate."""
    ids = {rid:j for j,rid in enumerate(track.ids)}
    result = []
    for row in df.itertuples():
        s = row.entry
        i,j = ids[row.i],ids[row.j]
        p = track.pos
        # Truncate the optional longer history at the first missing or flagged older sample.
        start = max(0,s-11)
        bad = np.flatnonzero((flags[start:s,[i,j]]!=0).any(axis=1))
        if len(bad):
            start += bad[-1]+1
        rel = p[start:s+1,j]-p[start:s+1,i]
        if len(rel)<3:
            raise ValueError('The accepted event lacks its required two pre-entry frames.')
        nprev = rel[-2]/np.linalg.norm(rel[-2])
        n = rel[-1]/np.linalg.norm(rel[-1])
        tangent = rotate(nprev)
        def prior(frames):
            span = min(frames,len(rel)-2)
            return (rel[-2]-rel[-2-span])*FPS/span
        v1,v3,v10 = [prior(lag) for lag in (1,3,10)]
        span = min(3,len(rel)-2)
        w = (track.phi[s-1,[i,j]]-track.phi[s-1-span,[i,j]])*FPS/span
        hd = heading(track.phi[s-1,j])-heading(track.phi[s-1,i])
        others = np.ones(len(track.ids),bool)
        others[[i,j]] = False
        nearest = np.linalg.norm(p[s-1,[i,j],None]-p[s-1,others],axis=-1).min()
        size = np.linalg.norm(rel[-1])
        values = dict(pwm20=float(track.pwm==20),closing_pre=-v1@nprev,normal_120ms=v3@nprev,
            normal_400ms=v10@nprev,abs_tangent_120ms=abs(v3@tangent),
            tangent_spin=(v3@tangent)*w.sum(),mean_abs_spin=np.abs(w).mean(),relative_heading=hd@nprev,
            nearby=max(0.,1.2-nearest/CAP),depth_px=CAP-size,
            crossing_speed=(np.linalg.norm(rel[-2])-size)*FPS)
        # Keep the same initial normal. The delayed target shares no positions with predictors.
        f = s+2
        if f<len(p):
            delta = (p[f,j]-p[f,i])-(p[f-1,j]-p[f-1,i])
            values['delayed'] = delta@n*FPS
            third = np.linalg.norm(p[f,[i,j],None]-p[f,others],axis=-1).min()
            good = (flags[f,[i,j]]==0).all() and third>=1.03*CAP
        else:
            good = False
            values['delayed'] = np.nan
        for held,a in arenas.items():
            values['extended_'+held] = bool(good and
                    (np.linalg.norm(p[f,[i,j]]-a[:2],axis=1)<=a[2]-.25*SIGMA).all())
        result.append(values)
    return pd.concat([df.reset_index(drop=True),pd.DataFrame(result)],axis=1)


def fit(train,y,terms,penalty):
    x = train[terms].to_numpy()
    w = weights(train)
    w /= w.sum()
    center = w@x
    scale = np.sqrt(w@(x-center)**2)
    scale = np.where(scale>1e-12,scale,1.)
    z = (x-center)/scale
    mu = w@y
    design = np.vstack([np.sqrt(w)[:,None]*z,np.eye(len(terms))*np.sqrt(penalty)])
    rhs = np.r_[np.sqrt(w)*(y-mu),np.zeros(len(terms))]
    beta = np.linalg.lstsq(design,rhs,rcond=None)[0]/scale
    return float(mu-center@beta),beta


def evaluate(events):
    predictions,models = [],[]
    for held in sorted(events.recording.unique()):
        for window in ('entry','entry_matched','next'):
            df = events[events[held]].copy()
            if window!='entry':
                df = df[df['extended_'+held]]
            label = 'delayed' if window=='next' else 'outgoing'
            train = df[(df.recording!=held)&(df.hi+(window!='entry')<=df.split_frame)]
            valid = df[(df.recording!=held)&(df.lo>=df.split_frame)]
            test = df[df.recording==held]
            ty,vy = train[label].to_numpy(),valid[label].to_numpy()
            wt,wv = weights(train),weights(valid)
            table = test[KEY+['pwm20','closing_pre','depth_px','crossing_speed']].copy()
            table = table.assign(window=window,observed=test[label],constant=np.average(ty,weights=wt))
            e = max(0.,np.average(ty*train.closing,weights=wt)/np.average(train.closing**2,weights=wt))
            table['restitution'] = e*test.closing
            choices = []
            for name,terms in LIBRARIES.items():
                # Use training-prefix fits and only the other recordings' validation tails.
                options = []
                for penalty in ([0.,.03,.3] if 'history' in name else [0.]):
                    a,b = fit(train,ty,terms,penalty)
                    mse = np.average((vy-(a+valid[terms].to_numpy()@b))**2,weights=wv)
                    options.append((mse,penalty,a,b))
                mse,penalty,a,b = min(options,key=lambda q:q[:2])
                table[name] = a+test[terms].to_numpy()@b
                models.append(dict(held=held,window=window,name=name,penalty=penalty,train=len(train),
                    validation=len(valid),test=len(test),validation_rmse=np.sqrt(mse),
                    coefficients=json.dumps({'offset':a,**dict(zip(terms,b))})))
                choices.append((mse,len(terms),name))
            # Include the constant and physical baseline in the validation contest as well.
            for name,p in [('constant',np.average(ty,weights=wt)),('restitution',e*valid.closing)]:
                choices.append((np.average((vy-p)**2,weights=wv),1,name))
            selected = min(choices)[2]
            table['selected'] = table[selected]
            table['selection'] = selected
            predictions.append(table)
    return pd.concat(predictions,ignore_index=True),pd.DataFrame(models)


def summarize(pred):
    rows,uncertainty = [],[]
    methods = ['constant','restitution',*LIBRARIES,'selected']
    for (window,held),d in pred.groupby(['window','recording']):
        for method in methods:
            rows.append(dict(window=window,recording=held,events=len(d),model=method,
                **score(d.observed.to_numpy(),d[method].to_numpy())))
    by = pd.DataFrame(rows)
    summary = by.groupby(['window','model']).agg(events=('events','sum'),mean_rmse=('rmse','mean'),
                           mean_r2=('r2','mean'),mean_mae=('mae','mean'))
    for window,d in pred.groupby('window'):
        for method,baseline in [('pwm','constant'),('history','pwm'),('entry_depth','pre_speed'),
                                ('entry_geometry','pwm'),('selected','pwm'),('selected','restitution')]:
            rng = np.random.default_rng(25481)
            samples = []
            for _,q in d.groupby('recording'):
                frame_bin = q.entry.to_numpy()//int(10*FPS)
                z = np.zeros((frame_bin.max()+1,3))
                np.add.at(z,frame_bin,np.column_stack([np.ones(len(q)),(q[baseline]-q.observed)**2,
                                                      (q[method]-q.observed)**2]))
                draws = z[rng.integers(len(z),size=(2000,len(z)))].sum(1)
                samples.append(np.sqrt(draws[:,1]/draws[:,0])-np.sqrt(draws[:,2]/draws[:,0]))
            lo,hi = np.quantile(np.mean(samples,axis=0),[.025,.975])
            reduction = summary.loc[(window,baseline),'mean_rmse']-summary.loc[(window,method),'mean_rmse']
            uncertainty.append(dict(window=window,model=method,baseline=baseline,rmse_reduction=reduction,
                                    lower95=lo,upper95=hi))
    return by,summary,pd.DataFrame(uncertainty)


def write_report(pred,summary,intervals):
    names = {'constant':'One training mean','restitution':'Pure restitution','pwm':'Mean by PWM',
             'pre_speed':'PWM + incoming speed','history':'PWM + pre-entry history',
             'entry_depth':'PWM + incoming speed + entry depth','entry_geometry':'PWM + incoming speed + entry geometry',
             'history_geometry':'PWM + history + entry geometry','selected':'Selected on other recordings’ validation tails'}
    main = pred[pred.window=='entry']
    lines = ['# Predicting normal speed in the first 40 ms after a measured crossing','',
        'The primary comparison uses exactly the 1,888 held-out nominal entry events from the earlier restitution study. Raw measured center displacements supply both the target and its sign. Eligibility includes the original looser third-robot clearance, wall and tracking checks. All errors are in px/s.','',
        '| Model | Entry RMSE | Entry mean R² | Entry MAE | Matched entry RMSE | Next interval RMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for key,label in names.items():
        s = summary.loc['entry',key]
        lines.append(f'| {label} | {s.mean_rmse:.2f} | {s.mean_r2:.3f} | {s.mean_mae:.2f} | '
                     f'{summary.loc[("entry_matched",key),"mean_rmse"]:.2f} | {summary.loc[("next",key),"mean_rmse"]:.2f} |')
    lines += ['', '## Contributions to prediction error','',
        '| Compared model | Reference | RMSE reduction | Within-recording 95% interval |',
        '|---|---|---:|---:|']
    for r in intervals[intervals.window=='entry'].itertuples():
        lines.append(f'| {names[r.model]} | {names[r.baseline]} | {r.rmse_reduction:.2f} | {r.lower95:.2f} to {r.upper95:.2f} |')
    lines += ['', '## Variation across recordings','',
        '| Recording | Events | Mean observed speed | Observed standard deviation | History-model bias (prediction minus observation) | History-model RMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for tag,d in main.groupby('recording'):
        err = d.history-d.observed
        lines.append(f'| {tag} | {len(d)} | {d.observed.mean():.2f} | {d.observed.std(ddof=0):.2f} | '
                     f'{err.mean():+.2f} | {np.sqrt(np.mean(err**2)):.2f} |')
    lines += ['', '## Evaluation and interpretation','',
        'Each fold trains on the first three quarters of the other four recordings, with equal total weight per recording. Validation on their last quarter chooses ridge strength (0, 0.03 or 0.3 on standardized features) for each history model and selects the model for the last row. Every other row evaluates the named feature set. RMSE and R² are computed within each held-out session and averaged over the five. The paired intervals resample 10-second blocks in each recording 2,000 times, retaining equal recording weights and fixed fitted models.','',
        'PWM is the known motor setting (10 or 20). All history features end one frame before entry. Slopes use backwards differences over 40, 120 and 400 ms, shortened at the recording start or a tracking flag. The remaining predictors are absolute tangential speed, spin magnitude, signed tangential speed times summed spin, relative heading along the incoming pair axis and proximity to a third robot. The earlier restitution baseline uses the original entry axis and its published eligibility.','',
        'Entry geometry adds the depth below one diameter at the first threshold crossing and the speed of the crossing interval. The entry position is available at forecast time and also starts the measured velocity interval. The delayed check therefore refits all models to the next 40 ms interval, keeping the original normal fixed. Its position difference shares neither endpoint with the entry geometry. Both matched columns use the same events with available, quality-screened later measurements. Their physical timing differs.','',
        'The model predicts an observed event outcome conditional on a sampled crossing. A measured PWM association can include differences between sessions. Formal verification of instantaneous restitution requires independently timed physical contacts, enough samples on both sides of contact to estimate velocity uncertainty, and evaluation on new recordings.','',
        '[Per-recording scores](by_recording.csv) · [Fold models and fitted coefficients](fold_models.csv) · [Event predictions](predictions.csv.gz) · [Paired comparisons](paired_differences.csv)']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    data = pd.read_csv(PREVIOUS/'events.csv.gz').query("target=='entry' and width==2 and offset==0")
    raw,_ = projected_tracks()
    arenas = {t:json.loads((OUT/t/'training.json').read_text())['arenas'] for t in raw}
    tables = []
    for tag,tr in raw.items():
        with np.load(OUT/'preprocessed'/f'{tag}.npz') as z:
            tables.append(features(tr,data[data.recording==tag],z['quality_flags'],
                                   {t:np.array(a[tag[-2:]]) for t,a in arenas.items()}))
    pred,models = evaluate(pd.concat(tables,ignore_index=True))
    main = pred[pred.window=='entry']
    old = pd.read_csv(PREVIOUS/'predictions.csv.gz').query("target=='entry'")
    check = main.merge(old,on=KEY,validate='one_to_one',suffixes=('_new','_old'))
    assert len(check)==len(main)==len(old)
    for name in ['constant','restitution']:
        np.testing.assert_allclose(check[name+'_old'],check[name+'_new'])
    np.testing.assert_allclose(check.outgoing,check.observed)
    by,summary,uncertainty = summarize(pred)
    DEST.mkdir(exist_ok=True)
    pred.to_csv(DEST/'predictions.csv.gz',index=False)
    models.to_csv(DEST/'fold_models.csv',index=False)
    by.to_csv(DEST/'by_recording.csv',index=False)
    summary.to_csv(DEST/'scores.csv')
    uncertainty.to_csv(DEST/'paired_differences.csv',index=False)
    write_report(pred,summary,uncertainty)
    print(summary.round(3).to_string())


if __name__=='__main__':
    run()
