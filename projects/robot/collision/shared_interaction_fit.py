"""Joint EPDE subset search on pair motion beyond cross-fitted isolated motion."""
import os
for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[var]='1'
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from shared_interaction_data import DEST, BASE, D_MM, DRIVE, MS
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from epde.interface.search_config import active_config, set_active_config
from epde.operators.common.multisample import joint_subset_table, joint_instability_table

GROUPS=('clearance','clearance','spin_slide','spin_slide','history','third')
NAMES={'n':('hinge','approach','side_magnitude','spin_magnitude','recent_closing','third_normal'),
       't':('slide','slide_closing','same_slide','spin_modulated_slide','recent_slide','third_side')}
FORM={'hinge':r'[-g/h-f_n]_+', 'approach':'zc','side_magnitude':r'z|s|','spin_magnitude':r'zW',
      'recent_closing':'zLc','third_normal':'A_n','slide':'zs','slide_closing':'zsc',
      'same_slide':'zIs','spin_modulated_slide':'zsW','recent_slide':'zLs','third_side':'A_t'}


def family(name):
    return 'sweep' if name.startswith('Experiment') else 'swarm'


def periods(df):
    boundary=.75*df.segment_end
    return (df.seconds+.4<boundary).to_numpy(),(df.seconds-.44>boundary).to_numpy()


def balanced(df):
    return 1/df.groupby('recording',sort=False).recording.transform('size').to_numpy()


def rms(y,p,w):
    return float(np.sqrt(np.average((y-p)**2,weights=w)))


def mse_gram(g,b):
    return np.maximum(0,np.einsum('ij,jk,ik->i',b,g,b))


def free_basis(df,axis):
    return df[[v+'_'+axis for v in DRIVE]].to_numpy()


def learn_free(df,training,ms):
    src=df[df.recording.isin(training)]
    pre,val=periods(src)
    a,b=src[pre],src[val]
    zero=np.zeros(len(DRIVE))
    persistence=zero.copy()
    persistence[0]=1
    if not len(a) or not len(b):
        return persistence,'persistence fallback',{}
    w=balanced(a)
    x=np.vstack([free_basis(a,k) for k in 'xy'])
    y=np.concatenate([a[f'y{ms}_{k}'].to_numpy() for k in 'xy'])
    w=np.tile(w,2)
    gram=(x.T*w)@x
    reg=.001*np.diag(np.maximum(np.diag(gram),1e-8))
    fitted=np.linalg.solve(gram+reg,x.T@(w*y))
    models={'zero':zero,'persistence':persistence,'heading/history/PWM/spin':fitted}
    errors={name:np.sqrt(np.average(sum((b[f'y{ms}_{k}']-free_basis(b,k)@coef)**2 for k in 'xy'),
                                      weights=balanced(b))) for name,coef in models.items()}
    best=min(errors,key=errors.get)
    return models[best],best,errors


def library(df,h,axis,free):
    c,s=np.maximum(df.closing.to_numpy(),0),df.side.to_numpy()
    z=np.clip((1.30-df.r.to_numpy())/.30,0,1)
    w=abs(df.spin_sum.to_numpy())
    frac=df.recent_near.to_numpy()
    g=np.maximum(df.r.to_numpy()-1,0)
    if axis=='n':
        return np.column_stack([np.maximum(-g/h-free,0),z*c,z*abs(s),z*w,z*frac*c,df.third_n])
    return np.column_stack([z*s,z*s*c,z*s*df.same,z*s*w,z*frac*s,df.third_t])


def history_basis(df,axis):
    if axis=='n':
        return np.column_stack([np.ones(len(df)),df.pwm/10,df.recent_n,df.long_n,abs(df.side)])
    return np.column_stack([df.recent_t,df.long_t,df.side*df.pwm/10])


def front_mask(wape,dyad,valid):
    valid=valid & np.isfinite(wape)&np.isfinite(dyad)
    return np.array([ok and not np.any(valid&(wape<=x)&(dyad<=y)&((wape<x)|(dyad<y)))
                     for x,y,ok in zip(wape,dyad,valid)])


def equations(coeff,axis,timed):
    parts=[]
    for k,name in enumerate(NAMES[axis]):
        if timed:
            a,b=coeff[k],coeff[k+6]
            if a or b:
                parts.append(f'+ ({a:.5g}{b:+.5g} log(h/0.08)) {FORM[name]}')
        elif coeff[k]:
            parts.append(f'{coeff[k]:+.5g} {FORM[name]}')
    return 'f_'+axis+' '+' '.join(parts)


def attach_controls(df):
    df=df.copy()
    df['optical']='unreviewed'
    df['distant_match']=False
    labeled=pd.read_csv(BASE/'video_gate_response/selection.csv').merge(
        pd.read_csv(BASE/'video_gate_response/optical_labels.csv'),on=['event','recording','gate_frame','release'])
    for (tag,stage),g in df.groupby(['recording','stage']):
        close=g[g.kind=='near']
        far=g[g.kind=='distant']
        if len(close) and len(far):
            scale=np.array([.15,.20])
            dist,idx=cKDTree(close[['closing','side']].to_numpy()/scale).query(far[['closing','side']].to_numpy()/scale,p=np.inf)
            df.loc[far.index[dist<=1],'distant_match']=True
        if family(tag)=='swarm':
            for r in labeled[labeled.recording==tag].itertuples():
                q=close[(close.i==r.i)&(close.j==r.j)&(abs(close.frame-r.gate_frame)<=2)]
                if len(q)==1:
                    df.loc[q.index,'optical']=r.optical_label
    return df


def fit_fold(df,free_rows,train_names,test_names,label,mode,store,free_cache,fixed_drive=False):
    train_set=df.recording.isin(train_names).to_numpy()
    pre,tail=periods(df)
    near=df.kind.eq('near').to_numpy()
    tr=train_set&pre&near
    va=train_set&tail&near
    te=df.recording.isin(test_names).to_numpy()
    if not tr.any() or not va.any():
        raise ValueError('Empty training or validation')
    w=balanced(df[tr]);wv=balanced(df[va])
    # Fixed, chronological, balanced subset. Different fits see exactly the same events.
    blocks=[]
    for tag,g in df[tr].groupby('recording'):
        a=g.sort_values(['stage','frame','i','j']).index.to_numpy()
        take=np.round(np.linspace(0,len(a)-1,min(128,len(a)))).astype(int)
        blocks.append((tag,a[take]))
    fold=dict(mode=mode,fold=label)
    fvals={}
    for ms in MS:
        def get(names):
            if fixed_drive:
                return np.array([1.,0,0,0,0]),'fixed persistence',{}
            key=(tuple(sorted(names)),ms)
            if key not in free_cache:
                free_cache[key]=learn_free(free_rows,names,ms)
            return free_cache[key]
        coef,name,errs=get(train_names)
        store['free_fits'].append(dict(**fold,horizon=ms,model=name,coeff=json.dumps(coef.tolist()),
                                      validation_vector_rmse=json.dumps(errs),training=','.join(train_names)))
        for axis in ('n','t'):
            x=free_basis(df,axis)
            f=x@coef
            for tag in train_names:
                use=tr&(df.recording==tag).to_numpy()
                f[use]=x[use]@get([v for v in train_names if v!=tag])[0]
            fvals[ms,axis]=f
        if mode=='native' and len(test_names)==1 and label==test_names[0]:
            for tag,g in free_rows[free_rows.recording.isin(test_names)].groupby('recording'):
                e=(g[f'y{ms}_x']-free_basis(g,'x')@coef).to_numpy()+1j*(g[f'y{ms}_y']-free_basis(g,'y')@coef).to_numpy()
                for (_,rid),ix in g.groupby(['stage','robot']).indices.items():
                    b=g.iloc[ix]
                    residual=e[ix]*np.exp(-1j*b.heading.to_numpy())
                    d=dict(recording=tag,stage=int(b.stage.iloc[0]),robot=rid,horizon=ms,n=len(b),
                           along_bias=residual.real.mean(),across_bias=residual.imag.mean(),
                           along_sd=residual.real.std(),across_sd=residual.imag.std(),
                           median_mm_s=np.median(abs(residual))*D_MM,p90_mm_s=np.quantile(abs(residual),.9)*D_MM)
                    for lag in (.08,.40,.96):
                        samples=pd.Series(e[ix],index=b.frame)
                        earlier=samples.reindex(b.frame-round(lag*b.hz.iloc[0])).to_numpy()
                        ok=np.isfinite(earlier)
                        if ok.sum()>=5:
                            a=e[ix][ok];c=earlier[ok];a=a-a.mean();c=c-c.mean()
                            den=np.linalg.norm(a)*np.linalg.norm(c)
                            d[f'corr_{lag}']=float(np.vdot(a,c).real/den) if den>0 else np.nan
                    store['free_noise'].append(d)

    for axis in ('n','t'):
        data={}
        hist=history_basis(df,axis)
        for ms in MS:
            f=fvals[ms,axis]
            h=ms/1000
            x=library(df,h,axis,f)
            y=df[f'y{ms}_{axis}'].to_numpy()
            b=np.linalg.lstsq(hist[tr]*np.sqrt(w[:,None]),y[tr]*np.sqrt(w),rcond=None)[0]
            options={'free':f,'history':hist@b,'free_clamp':f+x[:,0] if axis=='n' else f}
            best=min(options,key=lambda k:rms(y[va],options[k][va],wv))
            data[ms]=(x,y-f)
            keep=['recording','stage','kind','frame','seconds','i','j','closing','optical','distant_match']
            pred=df.loc[te,keep].copy()
            pred=pred.assign(**fold,axis=axis,horizon=ms,y=y[te],baseline=options[best][te],reference=best,
                             **{k:v[te] for k,v in options.items()})
            data['pred',ms]=pred
        for targets in [(ms,) for ms in MS]+[MS]:
            timed=len(targets)>1
            # Group by event first, then horizon, for dyadic subdivisions.
            x=np.stack([data[ms][0] for ms in targets],axis=1)
            y=np.stack([data[ms][1] for ms in targets],axis=1)
            if timed:
                x=np.concatenate([x,x*np.log(np.asarray(targets)[None,:,None]/80.)],axis=2)
            size=x.shape[2]
            masks=list(range(1,64))
            addresses=[m|(m<<6) if timed else m for m in masks]
            group=[(k,x[ix].reshape(-1,size),y[ix].ravel(),np.ones(len(ix)*len(targets))) for k,ix in blocks]
            errs,_,_,screen_beta,rank=joint_subset_table(group,subsets=addresses,weighting='samples',discrepancy='wape')
            dyad=joint_instability_table(group,screen_beta,{k:v.shape for k,_,v,_ in group},main_var='u')
            train=x[tr].reshape(-1,size)
            train_y=y[tr].ravel()
            wt=np.repeat(w,len(targets))
            gram=(train.T*wt)@train
            cross=train.T@(wt*train_y)
            beta=np.zeros((len(masks),size))
            for idx,address in enumerate(addresses):
                cols=[k for k in range(size) if address&(1<<k)]
                a=gram[np.ix_(cols,cols)]
                norms=np.sqrt(np.maximum(a.diagonal(),1e-30))
                normalized=a/norms[:,None]/norms
                b,_,r,_=np.linalg.lstsq(normalized,cross[cols]/norms,rcond=1e-10)
                rank[idx] &= r==len(cols)
                beta[idx,cols]=b/norms
            front=front_mask(errs,dyad,rank)
            for k,mask in enumerate(masks):
                store['search_objectives'].append(dict(**fold,axis=axis,timing='log' if timed else 'separate',
                    horizon=0 if timed else targets[0],mask=mask,wape=errs[k],dyad=dyad[k],
                    full_rank=bool(rank[k]),on_front=bool(front[k])))
            if not front.any():
                raise ValueError(f'No valid front for {fold}, {axis}, {targets}')
            sv=np.column_stack([x[va].reshape(-1,size),y[va].ravel()])
            joint_weight=np.repeat(wv,len(targets))
            vg=(sv.T*joint_weight)@sv/joint_weight.sum()
            valid_loss=np.sqrt(mse_gram(vg,np.column_stack([beta,-np.ones(len(masks))])))
            def choice(allowed):
                cand=front_mask(errs,dyad,rank&allowed)
                return min(np.flatnonzero(cand),key=lambda k:(valid_loss[k],masks[k].bit_count(),k))
            chosen=choice(np.ones(len(masks),bool))
            picks={'log_time' if timed else 'separate':chosen}
            if not timed:
                for stage,limit in [('clearance_only',2),('clearance_spin',4),('clearance_spin_history',5)]:
                    picks[stage]=choice(np.array([m<(1<<limit) for m in masks]))
                for name in sorted(set(GROUPS)):
                    forbidden=sum(1<<i for i,g in enumerate(GROUPS) if g==name)
                    picks['without_'+name]=choice(np.array([not m&forbidden for m in masks]))
                if axis=='t' and mode=='native' and label.startswith('N') and len(test_names)==1:
                    # Direction of the optical marker in the old videos agrees with the archived headings.
                    # Only old training rows can calibrate a signed surface-slip sensitivity.
                    old=tr&df.recording.str.startswith('N').to_numpy()
                    z=np.clip((1.3-df.r.to_numpy())/.3,0,1)
                    terms=x[:,0, [k for k in range(6) if masks[chosen]&(1<<k)]]
                    xx=np.column_stack([terms,z*(df.side-df.spin_sum)])
                    ww=np.sqrt(balanced(df[old]))
                    b=np.linalg.lstsq(xx[old]*ww[:,None],y[old,0]*ww,rcond=None)[0]
                    b0=np.linalg.lstsq(terms[old]*ww[:,None],y[old,0]*ww,rcond=None)[0]
                    data['pred',targets[0]]['signed_slip']=fvals[targets[0],axis][te]+xx[te]@b
                    data['pred',targets[0]]['slip_reference']=fvals[targets[0],axis][te]+terms[te]@b0
            for j,ms in enumerate(targets):
                part=data['pred',ms]
                for name,k in picks.items():
                    part[name]=fvals[ms,axis][te]+x[te,j]@beta[k]
            # Fixed-support scores also allow comparison when the pooled front changes by fold.
            held_scores={}
            for j,ms in enumerate(targets):
                for tag in test_names:
                    test=(df.recording==tag).to_numpy()&near
                    mtx=np.column_stack([x[test,j],y[test,j]])
                    test_loss=mse_gram(mtx.T@mtx/len(mtx),np.column_stack([beta,-np.ones(len(masks))]))
                    obs=df.loc[test,f'y{ms}_{axis}'].to_numpy()
                    held_scores[ms,tag]=(D_MM*np.sqrt(test_loss),1-test_loss/np.var(obs))
                    if label!='pooled':
                        for k in np.flatnonzero(rank):
                            store['fixed_support'].append(dict(**fold,held=tag,axis=axis,horizon=ms,
                                timing='log' if timed else 'separate',mask=masks[k],on_front=front[k],
                                rmse_mm_s=held_scores[ms,tag][0][k],r2=held_scores[ms,tag][1][k]))
            for k in np.flatnonzero(front):
                for j,ms in enumerate(targets):
                    for tag in (test_names if label!='pooled' else ['pooled']):
                        rm,r2=held_scores[ms,tag] if label!='pooled' else (np.full(len(masks),np.nan),)*2
                        store['fronts'].append(dict(**fold,held=tag,axis=axis,horizon=ms,
                            timing='log' if timed else 'separate',mask=masks[k],terms='+'.join(n for i,n in enumerate(NAMES[axis]) if masks[k]&(1<<i)),
                            wape=errs[k],dyad=dyad[k],selected=k==chosen,validation_rmse_mm_s=D_MM*valid_loss[k],
                            rmse_mm_s=rm[k],r2=r2[k],
                            screen_coeff=json.dumps(screen_beta[k,:size].tolist()),coeff=json.dumps(beta[k].tolist()),
                            screen_equation=equations(screen_beta[k,:size],axis,timed),equation=equations(beta[k],axis,timed)))
        for ms in MS:
            store['predictions'].append(data['pred',ms])


def main(fixed_drive=False):
    cfg=active_config()
    set_active_config(replace(cfg,objectives=replace(cfg.objectives,instability_metric='dyad')))
    files=sorted((DEST/'data').glob('*_pairs.parquet'))
    assert len(files)==8
    pairs=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    free=pd.concat([pd.read_parquet(p) for p in sorted((DEST/'data').glob('*_free.parquet'))],ignore_index=True)
    tags=sorted(pairs.recording.unique())
    for mode in (('native',) if fixed_drive else ('native','down25')):
        def select(table):
            choose=table['mode'].eq('native' if mode=='native' else 'down25') | (
                table.recording.str.startswith('N') & table['mode'].eq('native'))
            return table[choose].reset_index(drop=True)
        d=attach_controls(select(pairs))
        f=select(free)
        cache={}
        splits=[(tag,[t for t in tags if t!=tag],[tag]) for tag in tags if mode=='native' or family(tag)=='sweep']
        if mode=='native':
            for fam in ('swarm','sweep'):
                splits.append(('train_'+fam,[t for t in tags if family(t)==fam],[t for t in tags if family(t)!=fam]))
            splits.append(('pooled',tags,[]))
        for label,tr,te in splits:
            store={k:[] for k in ('fronts','free_fits','free_noise','predictions','fixed_support','search_objectives')}
            fit_fold(d,f,tr,te,label,mode,store,cache,fixed_drive)
            folder=DEST/('drive_sensitivity' if fixed_drive else 'folds')/f'{mode}_{label}'
            folder.mkdir(exist_ok=True,parents=True)
            for name,rows in store.items():
                if name=='predictions' and te:
                    pd.concat(rows,ignore_index=True).to_parquet(folder/'predictions.parquet',index=False)
                elif name=='predictions':
                    continue
                elif name=='fixed_support' and rows:
                    pd.DataFrame(rows).to_parquet(folder/'fixed_support.parquet',index=False)
                elif rows:
                    pd.DataFrame(rows).to_csv(folder/(name+'.csv'),index=False)
            print(mode,label,len(store['fronts']),'front rows',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixed-drive',action='store_true')
    main(parser.parse_args().fixed_drive)
