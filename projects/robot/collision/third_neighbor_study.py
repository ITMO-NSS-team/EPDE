"""Matched signed contrasts, path forecasts and distance-resolved joint EPDE fits."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression

from shared_interaction_compact import development, fitted
from shared_interaction_data import DEST, D_MM, MS
from shared_interaction_fit import balanced, family, front_mask, periods
from interaction_paths import BASIS, KEY, OUT, PATHS
from epde.interface.search_config import active_config, set_active_config
from epde.operators.common.multisample import joint_subset_table, joint_instability_table

CALIPERS={'side':.25,'closing':.25,'abs_third':.04,'r':.035,'third_n':.12,'long_t':.40,'wall_gap':.50}
STRATA=['recording','stage','spin_class','crowd','block']


def mirror_matches(d,scale,calipers=CALIPERS,min_time_gap=.84):
    pairs=[]
    for _,g in d[(d.abs_third>=.03)&~d.third_flagged].groupby(STRATA):
        pos,neg=g[g.third_t>0],g[g.third_t<0]
        if pos.empty or neg.empty:
            continue
        def coords(f):
            return f[list(calipers)].to_numpy()/(scale*np.array(list(calipers.values())))
        a,b=coords(pos),coords(neg)
        adj=cKDTree(b).query_ball_point(a,1,p=np.inf)
        u=np.repeat(np.arange(len(a)),[len(s) for s in adj])
        if not len(u):
            continue
        v=np.concatenate(adj).astype(int)
        ok=abs(pos.seconds.to_numpy()[u]-neg.seconds.to_numpy()[v])>min_time_gap
        u,v=u[ok],v[ok]
        cost=np.mean((a[u]-b[v])**2,axis=1)+1e-9
        # One private dummy per positive event. The penalty prioritizes match count.
        ix=np.arange(len(a))
        graph=coo_matrix((np.r_[cost,np.full(len(a),len(a)+2)],
            (np.r_[u,ix],np.r_[v,len(b)+ix])),shape=(len(a),len(a)+len(b))).tocsr()
        l,r=min_weight_full_bipartite_matching(graph)
        used=r<len(b)
        pairs.extend(zip(pos.index.to_numpy()[l[used]],neg.index.to_numpy()[r[used]]))
    return np.asarray(pairs,dtype=int).reshape(-1,2)


def contrast_tables(d):
    summaries=[];by=[];details=[];balance=[]
    for design,scale in [('primary',1.),('tighter',.75)]:
        pairs=mirror_matches(d,scale)
        if not len(pairs):
            continue
        a=d.loc[pairs[:,0]].reset_index(drop=True)
        b=d.loc[pairs[:,1]].reset_index(drop=True)
        assert len(np.unique(pairs))==pairs.size
        assert (a[STRATA]==b[STRATA]).all().all()
        c=a[STRATA].copy()
        c['design']=design
        for col in ['frame','i','j','third_t']:
            c['positive_'+col]=a[col]
            c['negative_'+col]=b[col]
        c['lead']=D_MM*(a.lead_t-b.lead_t)
        c['incoming']=D_MM*(a.side-b.side)
        for ms in MS:
            c[f'slide_{ms}']=D_MM*(a[f'y{ms}_t']-b[f'y{ms}_t'])
            c[f'change_{ms}']=c[f'slide_{ms}']-c['lead']
            c[f'adjusted_{ms}']=c[f'slide_{ms}']-(a[f'base_{ms}']-b[f'base_{ms}'])*D_MM
            c[f'forecast_{ms}']=D_MM*(a[f'compact_{ms}']-b[f'compact_{ms}'])
        c['family']=c.recording.map(family)
        details.append(c)
        for fam,sub in c.groupby('family'):
            for name,cal in CALIPERS.items():
                aa=a.loc[sub.index,name];bb=b.loc[sub.index,name]
                sd=np.sqrt((aa.var()+bb.var())/2)
                diff=aa-bb
                assert abs(diff).max()<=scale*cal+1e-12
                balance.append(dict(design=design,family=fam,variable=name,caliper=scale*cal,
                    mean_difference=diff.mean(),smd=diff.mean()/sd if sd>0 else np.nan,median_abs=abs(diff).median()))
        for scope,g in [('all',c),('one third',c[c.crowd==1])]:
            for fam,sub in g.groupby('family'):
                for col in ['lead','incoming',*[f'{k}_{ms}' for ms in MS for k in ('slide','change','adjusted','forecast')]]:
                    boot=[];means=[];block_counts=[]
                    for tag,r in sub.dropna(subset=[col]).groupby('recording'):
                        x=r.groupby(['stage','block'])[col].agg(['sum','count']).to_numpy()
                        rng=np.random.default_rng(731+sorted(d.recording.unique()).index(tag))
                        samples=x[rng.integers(len(x),size=(2000,len(x)))].sum(1)
                        boot.append(samples[:,0]/samples[:,1]);means.append(r[col].mean())
                        block_counts.append(len(x))
                        by.append(dict(design=design,scope=scope,recording=tag,family=fam,measure=col,
                                       pairs=len(r),blocks=len(x),mean_mm_s=means[-1]))
                    if boot:
                        lo,hi=np.quantile(np.mean(boot,axis=0),[.025,.975])
                        if max(block_counts)<2:
                            lo,hi=np.nan,np.nan
                        summaries.append(dict(design=design,scope=scope,family=fam,measure=col,
                            pairs=sub[col].count(),records=len(means),positive_records=sum(x>0 for x in means),
                            mean_mm_s=np.mean(means),low=lo,high=hi))
    pd.concat(details,ignore_index=True).to_parquet(OUT/'matched_pairs.parquet',index=False)
    pd.DataFrame(balance).to_csv(OUT/'balance.csv',index=False)
    pd.DataFrame(by).to_csv(OUT/'contrasts_by_recording.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'contrasts.csv',index=False)


def path_inputs(g):
    return np.column_stack([g.r,g.closing,g.side,abs(g.side),g.long_n,g.long_t,g.recent_near,
        abs(g.spin_sum),g.same,g.crowd,g.third_n,g.third_t,g.wall_gap,g.pwm])


def normal_fit(x,y,w,prior):
    design=np.column_stack([np.ones(len(x)),x])
    strength=.01*np.diag(np.r_[.1,np.ones(x.shape[1])])
    w=w/w.sum()
    return np.linalg.solve((design.T*w)@design+strength,design.T@(y*w)+strength@prior)


def path_model(train,test,kind,label):
    x,xt=path_inputs(train),path_inputs(test)
    w=balanced(train);w=w/w.mean()
    mean=np.average(x,axis=0,weights=w)
    sd=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    sd=np.where(sd<1e-8,1.,sd)
    x,xt=(x-mean)/sd,(xt-mean)/sd
    classes=train.path.map(PATHS.index).to_numpy()
    idx=test.path.map(PATHS.index).to_numpy()
    fit=LogisticRegression(C=1.,max_iter=1000).fit(x,classes,sample_weight=w)
    prob=np.zeros((len(test),len(PATHS)))
    prob[:,fit.classes_]=fit.predict_proba(xt)
    np.testing.assert_allclose(prob.sum(1),1.)
    prevalence=np.bincount(classes,weights=w,minlength=len(PATHS))/w.sum()
    target=train.y400_n.to_numpy()
    prior=normal_fit(x,target,w,np.zeros(x.shape[1]+1))
    experts=[]
    for k in range(len(PATHS)):
        use=classes==k
        experts.append(normal_fit(x[use],target[use],w[use],prior) if use.any() else prior)
    features=np.column_stack([np.ones(len(test)),xt])
    conditional=features@np.array(experts).T
    y=test.y400_n.to_numpy()*D_MM
    hard=np.eye(len(PATHS))[idx]
    out=test[KEY+['path']].copy()
    out=out.assign(design=kind,fold=label,y=y,gate_linear=(features@prior)*D_MM,
            gate_mixture=(conditional*prob).sum(1)*D_MM,
            known_path=conditional[np.arange(len(test)),idx]*D_MM,
            brier=np.sum((hard-prob)**2,axis=1),prior_brier=np.sum((hard-prevalence)**2,axis=1),
            logloss=-np.log(np.maximum(prob[np.arange(len(idx)),idx],1e-15)),
            prior_logloss=-np.log(np.maximum(prevalence[idx],1e-15)))
    for k in range(len(PATHS)):
        out['p_'+str(k)]=prob[:,k]
    return out


def distance_model(train,valid,test,kind,label):
    """All subsets of four distance basis functions; pre-existing slide term mandatory."""
    def design(g):
        z=np.clip((1.30-g.r.to_numpy())/.30,0,1)
        return np.column_stack([z*g.side,g[BASIS]])
    x,xv,xt=map(design,(train,valid,test))
    w,wv=balanced(train),balanced(valid)
    masks=[1|(mask<<1) for mask in range(16)]
    front_rows=[];scores=[];curves=[]
    for ms in MS:
        target=f'y{ms}_t'
        y=(train[target]-train.side).to_numpy()
        blocks=[]
        for tag,ix in train.groupby('recording').indices.items():
            chrono=np.lexsort((train.j.to_numpy()[ix],train.i.to_numpy()[ix],train.frame.to_numpy()[ix],train.stage.to_numpy()[ix]))
            sel=ix[chrono[np.rint(np.linspace(0,len(ix)-1,min(128,len(ix)))).astype(int)]]
            blocks.append((tag,x[sel],y[sel],np.ones(len(sel))))
        err,_,_,coef,rank=joint_subset_table(blocks,subsets=masks,discrepancy='wape',weighting='samples')
        dyad=joint_instability_table(blocks,coef,{k:v.shape for k,_,v,_ in blocks},main_var='u')
        eligible=front_mask(err,dyad,rank)
        fits=[]; val=[]
        for mask in masks:
            active=[k for k in range(5) if mask&(1<<k)]
            b=np.zeros(5); b[active]=np.linalg.lstsq(x[:,active]*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
            fits.append(b)
            val.append(np.sqrt(np.average((valid[target]-valid.side-xv@b)**2,weights=wv))*D_MM)
        selected=min(np.flatnonzero(eligible),key=lambda k:(val[k],masks[k].bit_count(),k))
        groups=test.groupby('recording').indices.values()
        observed=test[target].to_numpy()
        for k,mask in enumerate(masks):
            residual=test.side.to_numpy()+xt@fits[k]-observed
            mse=np.array([np.mean(residual[ix]**2) for ix in groups])
            variance=np.array([observed[ix].var() for ix in groups])
            front_rows.append(dict(design=kind,fold=label,horizon=ms,mask=mask,wape=err[k],dyad=dyad[k],
                full_rank=rank[k],on_front=eligible[k],chosen=k==selected,validation_rmse=val[k],
                test_rmse_mm_s=D_MM*np.sqrt(mse).mean(),test_r2=np.mean(1-mse/variance),
                screen_beta=json.dumps(coef[k,:5].tolist()),refit_beta=json.dumps(fits[k].tolist())))
        ref=np.column_stack([x[:,0],train.third_t])
        br=np.linalg.lstsq(ref*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
        pred=test.side.to_numpy()+xt@fits[selected]
        pred_linear=test.side.to_numpy()+xt[:,0]*br[0]+test.third_t.to_numpy()*br[1]
        pred_none=test.side.to_numpy()+xt[:,0]*fits[0][0]
        curves.append(dict(design=kind,fold=label,horizon=ms,mask=masks[selected],slide=fits[selected][0],
                           **dict(zip(BASIS,fits[selected][1:])),fixed_triangle_weight=br[1]))
        for tag,ix in test.groupby('recording').indices.items():
            obs=test[target].to_numpy()[ix]
            row=dict(design=kind,fold=label,recording=tag,family=family(tag),horizon=ms,events=len(ix))
            for name,v in [('learned',pred),('triangle',pred_linear),('no_third',pred_none)]:
                mse=np.mean((v[ix]-obs)**2)
                row[name+'_rmse']=D_MM*np.sqrt(mse);row[name+'_r2']=1-mse/obs.var()
            scores.append(row)
    return front_rows,scores,curves


def main():
    cfg=active_config()
    set_active_config(replace(cfg,objectives=replace(cfg.objectives,instability_metric='dyad')))
    d=development().merge(pd.read_parquet(OUT/'event_details.parquet'),on=KEY,validate='one_to_one')
    assert len(d)==46467
    d['block']=(d.seconds//30).astype(int)
    tags=sorted(d.recording.unique())
    designs=[('recording',tag,[t for t in tags if tag!=t],[tag]) for tag in tags]
    for fam in ('swarm','sweep'):
        names=[t for t in tags if family(t)==fam];other=[t for t in tags if family(t)!=fam]
        label='train_'+fam
        designs.append(('cross-family',label,names,other))
        frozen=fitted(d,label,names)
        use=d.recording.isin(other);t=d[use];z=np.clip((1.3-t.r)/.3,0,1)
        for ms in MS:
            b=frozen['horizons'][str(ms)]['t']['compact']
            d.loc[use,f'base_{ms}']=t.side*(1+z*b[0])
            d.loc[use,f'compact_{ms}']=t.side*(1+z*b[0])+t.third_t*b[5]
    contrast_tables(d)
    print('Opposite-sign matches and recording-level uncertainty scored.',flush=True)
    pre,tail=periods(d)
    predictions=[];fronts=[];distance=[];weights=[]
    for kind,label,names,other in designs:
        source=d.recording.isin(names);train=d[source&pre];valid=d[source&tail];test=d[d.recording.isin(other)]
        predictions.append(path_model(train,test,kind,label))
        f,s,c=distance_model(train,valid,test,kind,label)
        fronts.extend(f);distance.extend(s);weights.extend(c)
    pred=pd.concat(predictions,ignore_index=True)
    normal=pd.concat([pd.read_parquet(p/'predictions.parquet').query("axis=='n' and horizon==400 and kind=='near'").assign(
        design='cross-family' if p.name.startswith('native_train_') else 'recording')
        for p in (DEST/'drive_sensitivity').glob('native_*') if p.name!='native_pooled'])
    pred=pred.merge(normal[KEY+['design','history','separate','y']].rename(columns={'y':'archive_y'}),on=KEY+['design'],
                    validate='one_to_one')
    assert len(pred)==2*len(d) and not pred.duplicated(KEY+['design']).any()
    np.testing.assert_allclose(pred.y,pred.archive_y*D_MM)
    pred=pred.drop(columns='archive_y');pred[['history','separate']]*=D_MM
    pred.to_parquet(OUT/'path_predictions.parquet',index=False)
    rows=[]
    for (kind,tag),g in pred.groupby(['design','recording']):
        for path,sub in [('all',g),*g.groupby('path')]:
            r=dict(design=kind,recording=tag,family=family(tag),path=path,events=len(sub),
                   mean_normal=sub.y.mean(),brier=sub.brier.mean(),prior_brier=sub.prior_brier.mean(),
                   logloss=sub.logloss.mean(),prior_logloss=sub.prior_logloss.mean())
            for name in ['history','separate','gate_linear','gate_mixture','known_path']:
                mse=np.mean((sub[name]-sub.y)**2)
                r[name+'_rmse']=np.sqrt(mse)
                r[name+'_r2']=1-mse/sub.y.var(ddof=0) if sub.y.var(ddof=0)>0 else np.nan
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/'path_scores.csv',index=False)
    cols=['path','path_1.20_10','path_1.40_10','path_1.30_5','path_1.30_15']
    d.melt(id_vars=['recording'],value_vars=cols,var_name='rule',value_name='class').groupby(
        ['recording','rule','class']).size().rename('events').to_csv(OUT/'path_counts.csv')
    pd.DataFrame(fronts).to_csv(OUT/'distance_fronts.csv',index=False)
    pd.DataFrame(distance).to_csv(OUT/'distance_scores.csv',index=False)
    pd.DataFrame(weights).to_csv(OUT/'distance_coefficients.csv',index=False)
    print('Scored path forecasts and distance kernels in all ten held-out designs.')


if __name__=='__main__':
    main()
