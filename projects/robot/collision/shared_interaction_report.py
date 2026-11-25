"""Held-out comparisons, paired uncertainty, physical controls and optical sensitivity."""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from shared_interaction_data import DEST, BASE, D_MM, DRIVE, MS

MAIN=['free','free_clamp','history','baseline','separate','log_time']
EXTRA=['without_clearance','without_spin_slide','without_history','without_third',
       'clearance_only','clearance_spin','clearance_spin_history','signed_slip','slip_reference']
KEY=['recording','stage','frame','i','j']


def negative_controls():
    matches=[]
    for path in sorted((DEST/'data').glob('*pairs.parquet')):
        rows=pd.read_parquet(path).query("mode=='native'")
        for (rec,stage,same),d in rows.groupby(['recording','stage','same']):
            near=d[d.kind=='near']
            far=d[d.kind=='distant']
            if near.empty or far.empty:
                continue
            def predictors(x):
                return x[['closing','side']].to_numpy()/[.15,.20]
            dist,idx=cKDTree(predictors(far)).query(predictors(near),k=min(12,len(far)),p=np.inf)
            dist,idx=dist.reshape(len(near),-1),idx.reshape(len(near),-1)
            a,b=np.where(dist<=1)
            ni,fi=set(),set()
            near_vals=near[KEY].to_numpy();far_vals=far[KEY].to_numpy()
            for j in np.argsort(dist[a,b],kind='stable'):
                k=a[j];l=idx[k,b[j]]
                if k in ni or l in fi:
                    continue
                nx,fx=near_vals[k],far_vals[l]
                if abs(nx[2]-fx[2])<=.84*d.hz.iloc[0]:
                    continue
                ni.add(k);fi.add(l)
                matches.append((*nx,*fx[2:],dist[k,b[j]]))
    m=pd.DataFrame(matches,columns=KEY+['far_frame','far_i','far_j','caliper_distance'])
    m.to_parquet(DEST/'matched_distant.parquet',index=False)
    return m


def metrics(d,models):
    rows=[]
    for name in models:
        use=d[name].notna()
        if not use.any():
            continue
        y=d.loc[use,'y'].to_numpy()*D_MM
        e=d.loc[use,name].to_numpy()*D_MM-y
        rows.append(dict(model=name,n=len(y),rmse=np.sqrt(np.mean(e**2)),mae=abs(e).mean(),
                         r2=1-np.mean(e**2)/y.var() if y.var()>0 else np.nan))
    return rows


def paired_intervals(main):
    rng=np.random.default_rng(731)
    output=[]
    for (test,fam,axis,h),part in main.groupby(['test','family','axis','horizon']):
        effects={}
        for a,b in [('baseline','separate'),('baseline','log_time'),('separate','log_time')]:
            boot=[];observed=[]
            for _,d in part.groupby('recording'):
                ea=((d[a]-d.y)*D_MM)**2; eb=((d[b]-d.y)*D_MM)**2
                block=pd.DataFrame(dict(stage=d.stage,block=d.seconds//10,a=ea,b=eb,n=1)).groupby(
                    ['stage','block'])[['a','b','n']].sum().to_numpy()
                draws=rng.integers(len(block),size=(1000,len(block)))
                s=block[draws].sum(1)
                boot.append(np.sqrt(s[:,0]/s[:,2])-np.sqrt(s[:,1]/s[:,2]))
                observed.append(np.sqrt(ea.mean())-np.sqrt(eb.mean()))
            q=np.quantile(np.mean(boot,axis=0),[.025,.975])
            output.append(dict(test=test,family=fam,axis=axis,horizon=h,comparison=a+' minus '+b,
                               gain_mm_s=np.mean(observed),lower=q[0],upper=q[1],improved=sum(v>0 for v in observed),
                               recordings=len(observed)))
    return pd.DataFrame(output)


def sampling_sensitivity(pred):
    near=pred[(pred.kind=='near')&(pred.family=='sweep')]
    keys=['recording','stage','i','j']
    origins=near[(near.horizon==40)&(near.axis=='n')]
    down=origins[origins.test=='downsample'][[*keys,'seconds','frame']].rename(columns={'frame':'frame25'})
    native=origins[origins.test=='recording'][[*keys,'seconds','frame']].rename(columns={'frame':'frame50'})
    pairs=pd.merge_asof(down.sort_values('seconds'),native.sort_values('seconds'),on='seconds',by=keys,
                        tolerance=.040001,direction='nearest').dropna(subset=['frame50'])
    pairs['frame50']=pairs.frame50.astype(int)
    pairs['offset_ms']=1000*(pairs.frame25/25-pairs.frame50/50)
    pairs['absolute_offset']=abs(pairs.offset_ms)
    pairs=pairs.sort_values('absolute_offset').drop_duplicates([*keys,'frame50'])
    rows=[]
    for _,g in pairs.groupby('recording'):
        for h in MS:
            for a in 'nt':
                cand=near[(near.axis==a)&(near.horizon==h)]
                stats=[]
                for test,col in [('recording','frame50'),('downsample','frame25')]:
                    chosen=g.merge(cand[cand.test==test],left_on=[*keys,col],right_on=[*keys,'frame'],
                                   validate='one_to_one')
                    assert len(chosen)==len(g)
                    stats.append(np.sqrt(np.mean((chosen.separate-chosen.y)**2))*D_MM)
                rows.append(dict(recording=g.recording.iloc[0],horizon=h,axis=a,n=len(g),
                                 median_time_offset_ms=g.absolute_offset.median(),rmse50=stats[0],rmse25=stats[1]))
    pd.DataFrame(rows).to_csv(DEST/'sampling_sensitivity.csv',index=False)


def optics():
    root=BASE/'video_gate_response'
    labels=pd.read_csv(root/'selection.csv').merge(pd.read_csv(root/'kinematics.csv'),on='event',suffixes=('_raw','_reg'))
    rim=pd.read_csv(DEST/'rim_centers.csv')
    ring=pd.read_csv(BASE/'video_rebound_verification/camera_motion.csv')
    result=[]
    for tag,g in labels.groupby('recording'):
        with np.load(BASE/f'{tag}.npz') as z:
            coords,ids,dia=z['raw_pos'],z['ids'],float(z['cap_px'])
        r=rim[rim.recording==tag].copy()
        steps=r.groupby(['event','id'])
        contiguous=steps.frame.diff()==1
        angle=np.angle(np.exp(4j*steps.marker_angle_mod_pi_2.diff()))/4
        da=steps.phi.diff()
        spin=contiguous & (abs(da)>.035)&(abs(da)<.65)
        offsets=r.set_index(['event','frame','id'])
        center_meas={ms:[] for ms in (40,80)}
        reg_meas={ms:[] for ms in (40,80)}
        for row in g.itertuples():
            t=row.gate_frame
            rel=coords[:,np.searchsorted(ids,row.j)]-coords[:,np.searchsorted(ids,row.i)]
            n=rel[t]/np.linalg.norm(rel[t])
            for ms in (40,80):
                u=25*ms//1000
                velocity=(rel[t+u]-rel[t])@n/(ms/1000)
                reg_meas[ms].append((getattr(row,f'change_{ms}')-row.closing_reg-velocity)*D_MM/dia)
                keys=[(row.event,t+dt,rid) for dt in (0,u) for rid in (row.i,row.j)]
                if all(k in offsets.index for k in keys):
                    di,dj,di1,dj1=offsets.loc[keys,['dx','dy']].to_numpy()
                    center_meas[ms].append((dj1-di1-dj+di)@n/(ms/1000)*D_MM/dia)
        cam=ring[ring.recording==tag]
        for ms in (40,80):
            result.append(dict(recording=tag,horizon=ms,registered_clips=len(g),
                rim_events=r.event.nunique(),rim_poses=len(r),paired_rim_windows=len(center_meas[ms]),
                median_marker_offset_px=np.median(np.hypot(r.dx,r.dy)),median_rim_fit_error_px=r.rim_residual.median(),
                registration_change_median_abs_mm_s=np.median(abs(np.array(reg_meas[ms]))),
                rim_change_median_abs_mm_s=np.median(abs(np.array(center_meas[ms]))) if center_meas[ms] else np.nan,
                median_ring_step_px=cam.center_step_px.median(),median_withheld_error_px=cam.withheld_median_error_px.median(),
                spin_checks=spin.sum(),sign_agreements=int((angle[spin]*da[spin]>0).sum())))
    pd.DataFrame(result).to_csv(DEST/'optical_audit.csv',index=False)


def drive_comparison():
    paths=sorted((DEST/'drive_sensitivity').glob('native_*'))
    if len(paths)!=11:
        return
    d=pd.concat([pd.read_parquet(p/'predictions.parquet') for p in paths if p.name!='native_pooled'],
                ignore_index=True).query("kind=='near'").copy()
    d['test']=np.where(d.fold.str.startswith('train_'),'cross-family','recording')
    d['family']=np.where(d.recording.str.startswith('N'),'swarm','sweep')
    rows=[]
    keys=['test','family','axis','horizon','recording']
    for key,g in d.groupby(keys):
        rows.extend(dict(**dict(zip(keys,key)),**r) for r in metrics(g,MAIN+EXTRA))
    scores=pd.DataFrame(rows)
    scores.to_csv(DEST/'fixed_drive_by_recording.csv',index=False)
    scores=scores.groupby(keys[:-1]+['model']).agg(n=('n','sum'),rmse=('rmse','mean'),
       worst_rmse=('rmse','max'),r2=('r2','mean'),worst_r2=('r2','min')).reset_index()
    scores.to_csv(DEST/'fixed_drive_scores.csv',index=False)
    paired_intervals(d).to_csv(DEST/'fixed_drive_uncertainty.csv',index=False)
    primary=pd.read_csv(DEST/'scores.csv').query("test=='cross-family' and cohort=='near'")
    fixed=scores.query("test=='cross-family'")
    shared=pd.concat([primary.assign(drive='learned'),fixed.assign(drive='fixed')],ignore_index=True)
    keys=['family','axis','horizon']
    baseline=shared[shared.model.isin(['free','free_clamp','history','baseline'])].sort_values('rmse').drop_duplicates(keys)
    compare=shared[shared.model.isin(['separate','log_time'])].merge(
        baseline[keys+['drive','model','rmse']].rename(columns={'drive':'benchmark_drive','model':'benchmark','rmse':'benchmark_rmse'}),
        on=keys,validate='many_to_one')
    compare['gain_mm_s']=compare.benchmark_rmse-compare.rmse
    compare.to_csv(DEST/'transfer_comparison.csv',index=False)
    supports=pd.concat([pd.read_parquet(p/'fixed_support.parquet') for p in paths if
                        p.name!='native_pooled' and not p.name.startswith('native_train')])
    keys=['axis','horizon','timing','mask']
    means=supports.groupby(keys).agg(fixed_support_folds=('held','size'),
        fixed_support_mean_rmse=('rmse_mm_s','mean'),fixed_support_mean_r2=('r2','mean'),front_folds=('on_front','sum'))
    fronts=pd.concat([pd.read_csv(p/'fronts.csv') for p in paths],ignore_index=True).join(means,on=keys)
    fronts.to_csv(DEST/'fixed_drive_fronts.csv',index=False)


def free_diagnostics():
    rows=[]
    for path in sorted((DEST/'data').glob('*_free.parquet')):
        d=pd.read_parquet(path).query("mode=='native'")
        if d.empty:
            continue
        tag=d.recording.iloc[0]
        fits=pd.read_csv(DEST/'folds'/('native_'+tag)/'free_fits.csv').set_index('horizon')
        for ms in MS:
            b=np.array(json.loads(fits.loc[ms,'coeff']))
            v=d[f'y{ms}_x'].to_numpy()+1j*d[f'y{ms}_y'].to_numpy()
            x=d[[n+'_x' for n in DRIVE]].to_numpy()+1j*d[[n+'_y' for n in DRIVE]].to_numpy()
            for model,e in [('zero',v),('persistence',v-x[:,0]),('selected',v-x@b)]:
                for prefix,mask in [('all',np.ones(len(d),bool)),('single',d.stage==1)]:
                    if not np.asarray(mask).any():
                        continue
                    a=e[mask]*D_MM
                    phi=d.loc[mask,'heading'].to_numpy()
                    for sign in (1,-1):
                        body=a*np.exp(-1j*sign*phi)
                        rows.append(dict(recording=tag,horizon=ms,cohort=prefix,model=model,
                            heading_sign=sign,n=len(a),vector_rmse=np.sqrt(np.mean(abs(a)**2)),
                            along_mean=body.real.mean(),across_mean=body.imag.mean(),
                            along_sd=body.real.std(),across_sd=body.imag.std(),
                            p50_mm_s=np.median(abs(a)),p90_mm_s=np.quantile(abs(a),.9)))
    pd.DataFrame(rows).to_csv(DEST/'isolated_motion.csv',index=False)


def main():
    folders=sorted((DEST/'folds').glob('*'))
    assert len(folders)==14
    pred=pd.concat([pd.read_parquet(p/'predictions.parquet') for p in folders if p.name!='native_pooled'],ignore_index=True)
    pred['test']=np.select([pred['mode']=='down25',pred.fold.str.startswith('train_')],['downsample','cross-family'],default='recording')
    pred['family']=np.where(pred.recording.str.startswith('N'),'swarm','sweep')
    assert not pred.duplicated(['test','axis','horizon','kind',*KEY]).any()
    m=negative_controls()
    near=m[KEY].copy()
    far=m[['recording','stage','far_frame','far_i','far_j']].copy()
    far.columns=KEY
    near['matched_near']=True;far['matched_far']=True
    pred=pred.merge(near,on=KEY,how='left',validate='many_to_one').merge(far,on=KEY,how='left',validate='many_to_one')
    near=pred.kind=='near'
    groups={'near':near,'matched near':near & pred.matched_near.eq(True),
            'matched distant':~near & pred.matched_far.eq(True),'exclude optical uncertain':near & pred.optical.ne('uncertain'),
            'already closing':near & (pred.closing>0)}
    for name in ['apparent contact','clear gap','uncertain']:
        groups[name]=near & (pred.optical==name)
    results=[]
    for cohort,mask in groups.items():
        for key,g in pred[mask].groupby(['test','family','recording','axis','horizon']):
            attrs=dict(zip(['test','family','recording','axis','horizon'],key))
            for row in metrics(g,MAIN+(EXTRA if cohort=='near' else [])):
                results.append(dict(**attrs,cohort=cohort,**row))
    scores=pd.DataFrame(results)
    scores.to_csv(DEST/'scores_by_recording.csv',index=False)
    group=['test','family','axis','horizon','cohort','model']
    summary=scores.groupby(group).agg(n=('n','sum'),recordings=('recording','size'),rmse=('rmse','mean'),
        worst_rmse=('rmse','max'),mae=('mae','mean'),r2=('r2','mean'),worst_r2=('r2','min')).reset_index()
    summary.to_csv(DEST/'scores.csv',index=False)
    paired_intervals(pred[near]).to_csv(DEST/'paired_uncertainty.csv',index=False)
    sampling_sensitivity(pred)
    checks=[]
    for (axis,h),block in summary.query("test=='cross-family' and cohort=='near'").groupby(['axis','horizon']):
        for model in ['separate','log_time']:
            gains={}
            for fam,g in block.groupby('family'):
                s=g.set_index('model').rmse
                gains[fam]=s[['free','free_clamp','history','baseline']].min()-s[model]
            checks.append(dict(axis=axis,horizon=h,model=model,**gains,improves_both=min(gains.values())>0))
    pd.DataFrame(checks).to_csv(DEST/'transfer_decisions.csv',index=False)
    fs=pd.concat([pd.read_parquet(p/'fixed_support.parquet') for p in folders if p.name.startswith('native_') and
                  not p.name.startswith('native_train') and p.name!='native_pooled'],ignore_index=True)
    means=fs.groupby(['axis','horizon','timing','mask']).agg(fixed_support_folds=('held','size'),
        fixed_support_mean_rmse=('rmse_mm_s','mean'),fixed_support_mean_r2=('r2','mean'),
        front_folds=('on_front','sum')).reset_index()
    fronts=pd.concat([pd.read_csv(p/'fronts.csv') for p in folders],ignore_index=True)
    fronts=fronts.merge(means,on=['axis','horizon','timing','mask'],how='left',validate='many_to_one')
    fronts.to_csv(DEST/'all_fronts.csv',index=False)
    pooled=fronts[fronts.fold=='pooled'].copy()
    pooled.to_csv(DEST/'pooled_front.csv',index=False)
    pd.concat([pd.read_csv(p) for p in (DEST/'data').glob('*_audit.csv')]).to_csv(DEST/'coverage.csv',index=False)
    optics()
    free_diagnostics()
    drive_comparison()
    print(pd.DataFrame(checks).round(3).to_string(index=False))


if __name__=='__main__':
    main()
