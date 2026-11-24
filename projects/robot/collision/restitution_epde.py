"""Event-level symbolic test of u_out = e * closing, using measured swarm tracks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from discovery import write_json
from impacts import episodes, pairs, slope
from physics import CAP, CUTOFF, FPS, OUT, SIGMA, heading, rotate
from projected_data import projected_tracks


DEST = OUT/'preprocessed'/'restitution_epde'
TERMS = ('closing','bias','abs_tangent','heading_n','spin_abs','closing_squared')
SCOPES = ('strict','loose')
TARGETS = ('entry','release')


def prefix(mask):
    return np.r_[0,np.cumsum(mask,dtype=np.int32)]


def extract(tr,flags,arenas):
    left,right,dist,counts,_ = pairs(tr,next(iter(arenas.values())))
    radial = {held:np.linalg.norm(tr.pos-a[:2],axis=-1)>a[2]-.25*SIGMA for held,a in arenas.items()}
    frames = []
    for k,(i,j) in enumerate(zip(left,right)):
        d = dist[:,k]
        rel = tr.pos[:,j]-tr.pos[:,i]
        walls = {key:prefix(mask[:,i]|mask[:,j]) for key,mask in radial.items()}
        quality = prefix((flags[:,i]!=0)|(flags[:,j]!=0))
        other = {label:prefix(np.maximum(counts[lim][:,i],counts[lim][:,j])>(d<lim))
                    for label,lim in [('strict',CUTOFF),('loose',1.03*CAP)]}
        for offset in (-2,0,2):
            threshold = CAP+offset
            entry,release = episodes(d,threshold,1.03*CAP)
            if not len(entry):
                continue
            for target in TARGETS:
                for w in (2,3,5):
                    post = entry if target=='entry' else release
                    lo,hi = entry-w,post+w
                    use = (lo>=0)&(hi<=len(d))
                    if target=='release':
                        use &= release-entry<=FPS
                    s,p,l,h = entry[use],post[use],lo[use],hi[use]
                    # Incoming slope uses positions entirely outside the entry threshold.
                    keep = ((quality[h]-quality[l])==0)&((other['loose'][h]-other['loose'][l])==0)
                    keep &= (d[l[:,None]+np.arange(w)]>=threshold).all(axis=1)
                    s,p,l,h = s[keep],p[keep],l[keep],h[keep]
                    if not len(s):
                        continue
                    n = rel[s]/d[s,None]
                    before = slope(rel,l,w)
                    closing = -np.sum(before*n,1)
                    vt = np.sum(before*rotate(n),1)
                    vn = np.sum(slope(rel,p,w)*n,1)
                    hd = heading(tr.phi[s-1,j])-heading(tr.phi[s-1,i])
                    spin = CAP/2*abs(slope(tr.phi[:,i]+tr.phi[:,j],l,w))
                    df = pd.DataFrame(dict(recording=tr.tag,i=tr.ids[i],j=tr.ids[j],entry=s,post=p,
                        lo=l,hi=h,target=target,width=w,offset=offset,strict=other['strict'][h]==other['strict'][l],
                        closing=closing,bias=1.,abs_tangent=abs(vt),heading_n=(hd*n).sum(1),
                        spin_abs=spin,closing_squared=closing**2/100.,outgoing=vn))
                    for key,cs in walls.items():
                        df[key] = cs[h]==cs[l]
                    frames.append(df[closing>0])
    return pd.concat(frames,ignore_index=True)


def prepare():
    raw,_ = projected_tracks()
    calibrations = {held:json.loads((OUT/held/'training.json').read_text())['arenas'] for held in raw}
    tables = []
    for tag,tr in sorted(raw.items()):
        arenas = {held:np.asarray(a[tag[-2:]]) for held,a in calibrations.items()}
        with np.load(OUT/'preprocessed'/f'{tag}.npz') as z:
            events = extract(tr,z['quality_flags'],arenas)
        events['split_frame'] = 3*len(tr.pos)//4
        tables.append(events)
        print(tag,len(events),flush=True)
    DEST.mkdir(exist_ok=True,parents=True)
    pd.concat(tables,ignore_index=True).sort_values(['recording','entry','i','j']).to_csv(
        DEST/'events.csv.gz',index=False)


def split(events,held,scope,target,width,offset):
    keep = events[held] & (events.target==target)&(events.width==width)&(events.offset==offset)
    if scope=='strict':
        keep &= events.strict
    df = events[keep]
    other = df.recording!=held
    return df[other&(df.hi<=df.split_frame)],df[other&(df.lo>=df.split_frame)],df[~other]


def weights(df):
    return 1/df.groupby('recording').entry.transform('count').to_numpy()


def score(y,p):
    error = y-p
    mse = np.mean(error**2)
    return dict(rmse=np.sqrt(mse),mae=np.abs(error).mean(),r2=1-mse/y.var() if y.var()>0 else np.nan)


def screen(train):
    # Equal counts per recording on the native screen, no replacement or replicated rows.
    blocks = []
    count = min(train.groupby('recording').size().min(),256)
    for ix,(_,d) in enumerate(train.groupby('recording')):
        idx = np.sort(np.random.default_rng(371+ix).choice(len(d),count,replace=False))
        blocks.append(d.iloc[idx])
    return pd.concat(blocks)


def native(scope,target,held):
    from expanded_native import search_screen
    import expanded_native as engine
    df = pd.read_csv(DEST/'events.csv.gz')
    train,_,_ = split(df,held,scope,target,2,0)
    d = screen(train)
    x = {k:d[k].to_numpy() for k in TERMS}
    with patch.object(engine,'RHS',4),patch.object(engine,'FACTORS',1):
        out = search_screen(d.outgoing.to_numpy(),x,{},(),x.__getitem__)
    write_json(DEST/f'{scope}_{target}_{held}.json',dict(held=held,scope=scope,target=target,
        width=2,offset=0,screen_events=len(d),**out))


def launch(job):
    scope,target,held = job
    path = DEST/f'{scope}_{target}_{held}.log'
    with path.open('w',encoding='utf-8') as log:
        proc = subprocess.run([sys.executable,'-u',__file__,'--native',*job],stdout=log,stderr=log,
                              env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    if proc.returncode:
        raise RuntimeError(path.read_text(errors='replace')[-2500:])
    print('Native search complete',*job,flush=True)


def search():
    events = pd.read_csv(DEST/'events.csv.gz')
    jobs = []
    for scope in SCOPES:
        for target in TARGETS:
            for held in sorted(events.recording.unique()):
                tr,va,te = split(events,held,scope,target,2,0)
                # A full five-recording transfer contest requires all four training recordings.
                # Fixed-form fits below also cover any partial folds with train and test events.
                if tr.recording.nunique()==4 and len(va) and len(te):
                    jobs.append((scope,target,held))
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(launch,jobs))


def evaluate():
    from expanded_search import pareto_flags, score_structures
    from pareto import configure
    configure(5)  # Four RHS terms plus target; also fixes the native dyadic partition sizes.
    events = pd.read_csv(DEST/'events.csv.gz')
    tags = sorted(events.recording.unique())
    counts,rows,candidates,predictions = [],[],[],[]
    subsets = [s for size in range(1,5) for s in combinations(range(len(TERMS)),size)]
    for scope in SCOPES:
        for target in TARGETS:
            for w in (2,3,5):
                for offset in (-2,0,2):
                    for held in tags:
                        tr,va,te = split(events,held,scope,target,w,offset)
                        key = dict(scope=scope,target=target,width=w,offset=offset,held=held)
                        counts.append(dict(**key,train=len(tr),validation=len(va),test=len(te),
                                training_recordings=tr.recording.nunique(),reversals=(te.outgoing>0).sum()))
                        if tr.empty or te.empty:
                            continue
                        closing,y,weight = tr.closing.to_numpy(),tr.outgoing.to_numpy(),weights(tr)
                        e = np.sum(weight*closing*y)/np.sum(weight*closing**2)
                        def record(name,p,coef=None,structure=None):
                            rows.append(dict(**key,model=name,events=len(te),coefficient=coef,terms=structure,
                                             **score(te.outgoing.to_numpy(),p)))
                        record('restitution',np.maximum(e,0.)*te.closing.to_numpy(),e,'closing')
                        mean = np.average(y,weights=weight)
                        record('constant',mean,mean,'bias')
                        record('zero',0.)
                        record('persistence',-te.closing.to_numpy())
                        x = np.column_stack((np.ones(len(tr)),closing))
                        a,b = np.linalg.lstsq(x*np.sqrt(weight)[:,None],y*np.sqrt(weight),rcond=None)[0]
                        record('affine',a+b*te.closing.to_numpy(),b,'bias+closing')
                        if w!=2 or offset!=0 or tr.recording.nunique()!=4 or va.empty:
                            continue
                        receipt = json.loads((DEST/f'{scope}_{target}_{held}.json').read_text())
                        def key_terms(t):
                            return tuple(i for i,k in enumerate(TERMS) if k in t)
                        native_all = {key_terms(a['terms']) for a in receipt['evaluated']}
                        native_objectives = {key_terms(a['terms']):(a['wape'],a['dyad']) for a in receipt['evaluated']}
                        native_front = {key_terms(a['terms']) for a in receipt['front']}
                        matrix = tr[[*TERMS,'outgoing']].to_numpy()
                        gram = (matrix*weight[:,None]).T@matrix/weight.sum()
                        sample = screen(tr)[[*TERMS,'outgoing']].to_numpy()
                        supports,betas,wape,dyad,eligible = score_structures(gram,sample,subsets)
                        front = pareto_flags(wape,dyad,eligible)
                        xv,xt = va[list(TERMS)].to_numpy(),te[list(TERMS)].to_numpy()
                        chosen = []
                        for ix,(s,beta) in enumerate(zip(supports,betas)):
                            names = [TERMS[k] for k in s]
                            if s in native_objectives:
                                np.testing.assert_allclose([wape[ix],dyad[ix]],native_objectives[s],
                                                           rtol=3e-5,atol=1e-6)
                            val = np.sqrt(np.average((va.outgoing-xv[:,s]@beta)**2,weights=weights(va)))
                            pred = xt[:,s]@beta
                            candidates.append(dict(**key,terms='+'.join(names),coefficients=json.dumps(dict(zip(names,beta))),
                                native_evaluated=s in native_all,native_final=s in native_front,wape=wape[ix],
                                dyad=dyad[ix],eligible=eligible[ix],exhaustive_front=front[ix],validation_rmse=val,
                                **score(te.outgoing.to_numpy(),pred)))
                            if s in native_all and eligible[ix]:
                                chosen.append((val,len(s),s,pred))
                        _,_,s,best = min(chosen,key=lambda item:item[:3])
                        record('epde',best,structure='+'.join(TERMS[k] for k in s))
                        table = te[['recording','entry','i','j','closing','outgoing']].copy()
                        table = table.assign(**key,restitution=max(0.,e)*te.closing,constant=mean,epde=best)
                        predictions.append(table)
    counts = pd.DataFrame(counts)
    scores = pd.DataFrame(rows)
    all_forms = pd.DataFrame(candidates)
    counts.to_csv(DEST/'coverage.csv',index=False)
    scores.to_csv(DEST/'scores.csv',index=False)
    all_forms.to_csv(DEST/'equations_by_fold.csv',index=False)
    group = all_forms.groupby(['scope','target','terms'],as_index=False)
    merged = group.agg(folds=('held','nunique'),eligible=('eligible','all'),wape=('wape','mean'),
             dyad=('dyad','mean'),validation_rmse=('validation_rmse','mean'),r2=('r2','mean'),rmse=('rmse','mean'),
             native_final_folds=('native_final','sum'),native_evaluated_folds=('native_evaluated','sum'))
    merged['merged_front'] = False
    for _,d in merged.groupby(['scope','target']):
        merged.loc[d.index,'merged_front'] = pareto_flags(d.wape.to_numpy(),d.dyad.to_numpy(),
                                                          (d.eligible&(d.folds==5)).to_numpy())
    merged.to_csv(DEST/'equations_merged.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(DEST/'predictions.csv.gz',index=False)
    report()
    print(scores[(scores.width==2)&(scores.offset==0)].groupby(['scope','target','model']).agg(
        folds=('held','nunique'),events=('events','sum'),rmse=('rmse','mean'),r2=('r2','mean')).round(3))


def report():
    counts = pd.read_csv(DEST/'coverage.csv')
    scores = pd.read_csv(DEST/'scores.csv')
    forms = pd.read_csv(DEST/'equations_merged.csv')
    pred = pd.read_csv(DEST/'predictions.csv.gz')
    main = scores.query("scope=='loose' and width==2 and offset==0")
    summary = main.groupby(['target','model']).agg(folds=('held','nunique'),events=('events','sum'),
                            rmse=('rmse','mean'),mae=('mae','mean'),r2=('r2','mean'))
    rng = np.random.default_rng(923701)
    intervals = []
    for target,df in pred.groupby('target'):
        # Preserve all simultaneous and nearby encounters in a ten-second block.
        sims = []
        for _,d in df.groupby('held'):
            rows = np.column_stack([np.ones(len(d)),(d.outgoing-d.restitution)**2,(d.outgoing-d.constant)**2])
            block = d.entry.to_numpy()//int(10*FPS)
            z = np.zeros((block.max()+1,3))
            np.add.at(z,block,rows)
            draw = z[rng.integers(len(z),size=(2000,len(z)))].sum(1)
            sims.append(np.sqrt(draw[:,1]/draw[:,0])-np.sqrt(draw[:,2]/draw[:,0]))
        lo,hi = np.quantile(np.mean(sims,axis=0),[.025,.975])
        delta = summary.loc[(target,'restitution'),'rmse']-summary.loc[(target,'constant'),'rmse']
        intervals.append(dict(target=target,restitution_minus_constant_rmse=delta,lower95=lo,upper95=hi))
    pd.DataFrame(intervals).to_csv(DEST/'paired_uncertainty.csv',index=False)
    # Direct comparison of the two timing conventions on the exact same impacts.
    entry = pred[pred.target=='entry']
    release = pred[pred.target=='release']
    paired = entry.merge(release,on=['held','i','j','entry'],validate='one_to_one',suffixes=('_entry','_release'))
    assert len(paired)==len(release)
    moment = []
    for held,d in paired.groupby('held'):
        np.testing.assert_allclose(d.closing_entry,d.closing_release)
        for target in TARGETS:
            x,y = d.closing_entry.to_numpy(),d['outgoing_'+target].to_numpy()
            moment.append(dict(held=held,target=target,events=len(d),mean_outgoing=y.mean(),
                               reversal_fraction=(y>0).mean(),descriptive_e=x@y/(x@x)))
    pd.DataFrame(moment).to_csv(DEST/'matched_timing.csv',index=False)
    labels = {'entry':'After measured entry','release':'After measured release','restitution':'Pure restitution',
              'constant':'Training mean','zero':'Zero speed','persistence':'Continue incoming motion',
              'affine':'Intercept + closing speed','epde':'Validation-selected native EPDE'}
    lines = ['# Event-level EPDE test of normal restitution','',
        'Tested hypothesis: positive incoming closing speed c = −u⁻ predicts outgoing signed speed as u⁺ = ec with zero intercept. The fitted coefficient measures finite camera windows. Robot mass is unnecessary for this velocity-ratio test.','',
        '## Main held-out results','',
        'Two positions per slope span 40 ms at 25 Hz. Each fold fits coefficients on the first 75% of the other four recordings, with equal weight per recording. The last 25% of those four recordings choose among eligible native EPDE proposals. Test RMSE and R² are first computed within recording and then averaged equally. Speeds and errors are in px/s.','',
        '| Outgoing window | Model | Events | RMSE | Mean R² |','|---|---|---:|---:|---:|']
    for (target,model),r in summary.iterrows():
        lines.append(f'| {labels[target]} | {labels[model]} | {int(r.events):,} | {r.rmse:.2f} | {r.r2:.3f} |')
    entry_result = intervals[0]
    assert entry_result['target']=='entry'
    lines += ['',f'For the entry-based measurement, pure restitution adds {entry_result["restitution_minus_constant_rmse"]:.2f} px/s of mean held-out RMSE relative to the training mean (paired 95% interval {entry_result["lower95"]:.2f} to {entry_result["upper95"]:.2f}). The recordings show frequent separation, but a shared instantaneous physical restitution coefficient remains unverified.']
    lines += ['', '## Learned coefficients and observed reversals','',
        'Both incoming and outgoing velocities are projected onto the same measured line of centers at entry. The pure model estimates e through the origin on training data. A negative estimate would be clipped to zero for the restitution prediction.','',
        '| Held-out recording | Entry events | Outgoing > 0 | Training e | Restitution R² | After-release events | Training e (release) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    coeff = main[main.model=='restitution'].set_index(['held','target'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(9.6,4.1),sharey=True,layout='constrained')
    for ax,target in zip(axes,TARGETS):
        for tag,d in pred[pred.target==target].groupby('held'):
            bins = np.array_split(np.argsort(d.closing.to_numpy()),6)
            ax.plot([d.closing.iloc[b].mean() for b in bins],[d.outgoing.iloc[b].mean() for b in bins],
                    marker='o',ms=3,lw=1,label=tag)
        xmax = ax.get_xlim()[1]
        rates = coeff.xs(target,level='target').coefficient
        ax.fill_between([0,xmax],[0,xmax*rates.min()],[0,xmax*rates.max()],color='0.7',alpha=.5,
                        label='Fold-fitted restitution slopes')
        ax.axhline(0,color='0.5',lw=.7)
        ax.set(xlim=(0,xmax),xlabel='Incoming closing speed (px/s)',title=labels[target])
    axes[0].set_ylabel('Mean outgoing normal speed (px/s)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=7)
    fig.savefig(DEST/'outgoing_by_closing.png',dpi=155)
    plt.close(fig)
    primary = counts.query("scope=='loose' and target=='entry' and width==2 and offset==0").set_index('held')
    for tag,c in primary.iterrows():
        a,b = coeff.loc[(tag,'entry')],coeff.loc[(tag,'release')]
        lines.append(f'| {tag} | {int(c.test)} | {c.reversals/c.test:.1%} | {a.coefficient:.3f} | {a.r2:.3f} | {int(b.events)} | {b.coefficient:.3f} |')
    lines += ['', 'Each point below averages one sixth of a recording’s held-out events, ordered by incoming speed. Shading spans the five zero-intercept slopes estimated on training recordings.','',
        '![Observed outgoing speed and the fitted proportional laws](outgoing_by_closing.png)','',
        '## Was the one-term law on the front?','',
        'Each balanced training screen contains the same number of distinct, time-ordered events per training recording. The native NSGA-II search used seed 923701, 12 epochs, up to four additive terms and one factor per term. EPDE removed the constant-valued bias token in its native search. A second enumeration used EPDE’s WAPE and dyadic coefficient-instability objectives on all 56 nonempty subsets of up to four of the six terms, explicitly including bias. The other terms are closing speed, its square divided by 100, absolute tangential speed, relative heading along the normal, and the magnitude of summed angular velocity times robot radius. All features precede entry except the axis used for projection. One response row represents one event, with no differentiation between rows.','',
        '| Outgoing window | Native evaluated (of 5) | Native final front (of 5) | Merged exhaustive front |',
        '|---|---:|---:|---|']
    for r in forms[forms.terms=='closing'].itertuples():
        lines.append(f'| {labels[r.target]} | {r.native_evaluated_folds} | {r.native_final_folds} | {"Yes" if r.merged_front else "No"} |')
    lines += ['', 'The merged front minimizes mean training-screen WAPE and mean dyadic instability among structures eligible in all five folds. Test R² does not determine front membership. The constant baseline predicts the training mean without access to incoming speed. Paired intervals for held-out RMSE differences use 2,000 within-recording ten-second block resamples, conditional on the trained models.','',
        '## Isolation and timing sensitivity','',
        'An entry is the first observed crossing below the calibrated 219.55 px diameter after clearing 1.03 diameters. All incoming slope samples must be outside the chosen entry threshold. Both robots must be free of tracking flags and wall proximity throughout the measurement window. Strict isolation excludes any third robot within 253 px of either member. The looser analysis excludes third centers within 1.03 diameters (226.14 px). The main entry analysis keeps inward and outward outcomes, including events without a subsequent observed release. Release-based outcomes require clearance within one second and are conditional on that event.','',
        '| Isolation | Outgoing window | Slope duration | Eligible events, nominal diameter | Recordings with events | Range of trained e across scorable folds |',
        '|---|---|---:|---:|---:|---|']
    for (scope,target,width),d in counts[counts.offset==0].groupby(['scope','target','width']):
        a = scores[(scores.scope==scope)&(scores.target==target)&(scores.width==width)&
                   (scores.offset==0)&(scores.model=='restitution')].coefficient
        erange = f'{a.min():.3f} to {a.max():.3f}' if len(a) else '—'
        lines.append(f'| {scope} | {labels[target]} | {1000*(width-1)/FPS:g} ms | {d.test.sum():,} | {(d.test>0).sum()} | {erange} |')
    lines += ['', 'The entry slope ends at the frame before the crossing. The first outgoing slope starts at the first frame inside the threshold, omitting the crossing interval from both velocity estimates. For release it starts at the first frame at or beyond 1.03 diameters. The contact timestamp is inferred from noisy center distances. Physical impact can occur between samples. Event-level conclusions concern the observed speed response at this resolution.','',
        '[All threshold and timing sensitivities](scores.csv) · [Eligibility and reversals](coverage.csv) · [Per-fold equations](equations_by_fold.csv) · [Merged fronts](equations_merged.csv) · [Timing on matched events](matched_timing.csv) · [Paired uncertainty](paired_uncertainty.csv)']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','prepare','search','evaluate','report'],default='all')
    parser.add_argument('--native',nargs=3,metavar=('SCOPE','TARGET','HELD'))
    args = parser.parse_args()
    if args.native:
        native(*args.native)
    elif args.stage=='report':
        report()
    else:
        for stage in (prepare,search,evaluate):
            if args.stage in ('all',stage.__name__):
                stage()
