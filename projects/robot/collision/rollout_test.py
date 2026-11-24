"""Multi-origin, free-running tests of the shared pair-normal law and motion baselines."""
import json

import numpy as np
import pandas as pd

from expanded_terms import IDS, columns
from pair_normal_data import PREVIOUS, STUDY, pair_inputs
from pair_normal_dynamics import NAMES, normal_dynamics
from pair_normal_report import reconcile
from physics import CAP, CUTOFF, DT, FPS, LAG, Track, robot_parameters, rotate
from projected_data import fold_calibrations, projected_tracks


DEST = STUDY/'rollouts'
HORIZONS = (1, 2, 5, 10, 20)  # 0.4-second forecast steps
MODELS = ('fixed', 'constant_velocity', 'constant_guarded', 'shared_free', 'shared_guarded')
HIST = 2*LAG+1
TOL = 1.  # camera pixels, for physical diagnostics


def fold_weights(path, terms):
    df = pd.read_csv(path, usecols=['terms','held','refit'])
    selected = df[df.terms=='+'.join(terms)].set_index('held')
    assert selected.index.is_unique and len(selected)==5
    return {tag:json.loads(value) for tag,value in selected.refit.items()}


def geometry_masks(p, left, right, arena):
    overlap = np.linalg.norm(p[:,left]-p[:,right],axis=-1)<CAP-TOL
    outside = np.linalg.norm(p-arena[:2],axis=-1)>arena[2]+TOL
    return overlap,outside


def settle(points, arena=None):
    """Project disjoint swarms in parallel. Recheck every pair before accepting convergence."""
    out = points.copy()
    count, robots = out.shape[:2]
    left,right = np.triu_indices(robots,1)
    valid = np.isfinite(out).all(axis=(1,2))
    active = np.flatnonzero(valid)
    refresh = True
    # Maintain a sparse candidate list, refreshing after ten sweeps or apparent convergence.
    for sweep in range(2000):
        if not len(active):
            break
        if sweep%10==0 or refresh:
            delta = out[active[:,None],left]-out[active[:,None],right]
            f,k = np.nonzero(np.sum(delta*delta,axis=2)<(CAP+1)**2)
            f,i,j = active[f],left[k],right[k]
        refresh = False
        diff = out[f,i]-out[f,j]
        dist = np.linalg.norm(diff,axis=1)
        bad = dist<CAP-1e-5
        if (dist[bad]<1e-10).any():
            valid[f[dist<1e-10]] = False
        b = bad & valid[f]
        shift = (.5*(CAP+1e-6-dist[b])/dist[b])[:,None]*diff[b]
        np.add.at(out,(f[b],i[b]),shift)
        np.add.at(out,(f[b],j[b]),-shift)
        moved = np.bincount(f[b],minlength=count)>0
        if arena is not None:
            rel = out[active]-arena[:2]
            r = np.linalg.norm(rel,axis=2)
            a,k = np.nonzero(r>arena[2]+1e-5)
            out[active[a],k] = arena[:2]+rel[a,k]*(arena[2]/r[a,k])[:,None]
            moved[active[a]] = True
        pending = active[valid[active] & ~moved[active]]
        if len(pending):
            delta = out[pending[:,None],left]-out[pending[:,None],right]
            moved[pending] = (np.sum(delta*delta,axis=2)<(CAP-1e-5)**2).any(axis=1)
            refresh = moved[pending].any()
        keep = valid[active] & moved[active]
        if not keep.all():
            active = active[keep]
            refresh = True
    else:
        valid[active] = False
    out[~valid] = np.nan
    return out


class SharedVelocity:
    """Evaluate the archived instantaneous supports on independent simulated histories."""
    def __init__(self, track, params, vector, correction, count):
        # These supports require 20 pose frames and use no event-memory columns. Refuse
        # silent substitution if a later equation selects a history-dependent term.
        if set(vector)!={'C006','C082','C108','C109','C156','C173','C180','C181','C183','C020*Gspin'}:
            raise ValueError('Review vector history requirements for this support.')
        if set(correction)!={'near2','limit2','turnLimit20','spinMagnitude','gapSpeed*Crowded',
                             'priorClosing','freeCapCorrection','individualIn','prior*Near','prior*Crowded'}:
            raise ValueError('Review pair history requirements for this support.')
        self.track,self.params = track,params
        self.calibration = robot_parameters(track,params[0])
        self.indices = [IDS.index(k) for k in vector]
        self.vector = list(vector.values())
        self.correction = correction
        n = len(track.ids)
        a,b = np.triu_indices(n,1)
        index = np.zeros((n,n),int)
        index[a,b] = index[b,a] = np.arange(len(a))
        blank = np.zeros((count,len(a)),np.float32)
        hist = dict.fromkeys(('active_age','fresh','c_entry','q_peak','unload','H_cap','H_c',
                              'repeat_count','X_slip','H_spin'),blank)
        self.memory = (hist,index,np.zeros((count,n,3)),np.zeros((count,len(a),2,3)),
                       np.zeros((count,n,2,2)),np.ones((count,n),bool))

    def __call__(self, raw, clean, angles):
        batch,n = raw.shape[0],raw.shape[2]
        # columns() also returns an unused target. Pad with the current pose so no field
        # reads an observation beyond its origin. Each block has its own 20-frame history.
        def packed(p):
            padded = np.concatenate((p,np.repeat(p[:,-1:],LAG,axis=1)),axis=1)
            return padded.reshape((-1,*p.shape[2:]))
        phi = packed(angles)
        def track(p):
            return Track(self.track.tag,self.track.pwm,self.track.ids,packed(p),phi)
        obs,tr = track(raw),track(clean)
        rows = np.arange(batch)*(HIST+LAG)+HIST-1
        slots = np.arange(batch)
        x,base,*_ = columns(tr,rows,slots,self.memory,self.params,obs)
        prior = base+x[...,self.indices]@self.vector
        pair,f,i,j,normal = pair_inputs(obs,tr,rows,self.memory,slots)
        dyn = normal_dynamics(tr,rows,f,i,j,normal,self.calibration,base,prior)
        features = {k:v.astype(np.float32) for k,v in pair.items()}
        features.update(dict(zip(NAMES,dyn.astype(np.float32).T)))
        terms = []
        for term in self.correction:
            factors = term.split('*')
            terms.append(np.prod([features[k] for k in factors],axis=0))
        desired = features['prior']+np.column_stack(terms)@list(self.correction.values())
        v = prior.copy()
        for b in np.unique(f):
            ix = f==b
            ids = np.column_stack([i[ix],j[ix]])
            dv,*_ = reconcile(ids,normal[ix],desired[ix]-dyn[ix,0])
            v[b,np.unique(ids)] += dv
        return v


def check_first_step(tag,starts,v,obs):
    with np.load(STUDY/f'{tag}_predictions.npz') as z, np.load(STUDY/f'{tag}__test.npz') as data:
        keep = np.isin(z['origins'][:,0],starts)
        origin = z['origins'][keep]
        b = np.searchsorted(starts,origin[:,0])
        ids = {rid:k for k,rid in enumerate(obs.ids)}
        i,j = ([ids[rid] for rid in origin[:,c]] for c in (1,2))
        n = data['vector_checks'][keep,:2]
        np.testing.assert_allclose(np.sum((v[b,i]-v[b,j])*n,axis=1),
                                   z['coherent_residual_shared'][keep],atol=3e-5,rtol=2e-6)
        return len(origin)


def start_frames(track):
    starts = np.arange(100,len(track.pos)-max(HORIZONS)*LAG,max(HORIZONS)*LAG)
    # Prove every contact history was resolved before the earliest origin, using only its past.
    i,j = np.triu_indices(len(track.ids),1)
    prefix = track.pos[LAG:starts[0]-3*LAG-3+1]
    assert (np.linalg.norm(prefix[:,i]-prefix[:,j],axis=2)>=1.03*CAP).any(axis=0).all()
    return starts


def simulate(model,obs,clean,starts,estimator,arena):
    ix = starts[:,None]+np.arange(1-HIST,1)
    raw,kin,phi = obs.pos[ix].copy(),clean.pos[ix].copy(),obs.phi[ix].copy()
    spin = (phi[:,-1]-phi[:,-1-LAG])/DT
    velocity = (raw[:,-1]-raw[:,-1-LAG])/DT
    path = [raw[:,-1].copy()]
    alive = np.ones(len(starts),bool)
    verified = 0
    for step in range(1,max(HORIZONS)+1):
        if model.startswith('shared'):
            # Failed origins remain failed and cannot interfere with other swarms.
            good = np.flatnonzero(alive)
            if len(good):
                velocity[good] = estimator(raw[good],kin[good],phi[good])
            if step==1:
                verified = check_first_step(obs.tag,starts,velocity,obs)
        if model=='fixed':
            velocity[:] = 0.
        p = raw[:,-1].copy()
        new = []
        for frame in range(LAG):
            p = p+velocity/FPS
            if model.endswith('guarded'):
                p = settle(p,arena)
            new.append(p)
        new = np.stack(new,axis=1)
        alive &= np.isfinite(new).all(axis=(1,2,3))
        new[~alive] = np.nan
        path.extend(new.transpose(1,0,2,3))
        raw = np.concatenate((raw[:,LAG:],new),axis=1)
        if model.startswith('shared') and step<max(HORIZONS):
            if model=='shared_free':
                # The ten selected correction terms need cleaned poses at step endpoints
                # and two frames before them. C180 also uses the last two endpoints.
                cleaned = new.copy()
                used = new[:,[-3,-1]].reshape(-1,*new.shape[2:])
                cleaned[:,[-3,-1]] = settle(used).reshape(len(new),2,*new.shape[2:])
                alive &= np.isfinite(cleaned).all(axis=(1,2,3))
            else:
                cleaned = new
            kin = np.concatenate((kin[:,LAG:],cleaned),axis=1)
            next_phi = phi[:,-1:]+np.arange(1,LAG+1)[None,:,None]*spin[:,None]/FPS
            phi = np.concatenate((phi[:,LAG:],next_phi),axis=1)
        velocity[~alive] = np.nan
    return np.stack(path,axis=1),verified


def evaluate(tag,model,starts,obs,path,arena):
    left,right = np.triu_indices(len(obs.ids),1)
    p0 = obs.pos[starts]
    rel = p0[:,left]-p0[:,right]
    dist = np.linalg.norm(rel,axis=2)
    normal = rel/dist[...,None]
    tangent = rotate(normal)
    incoming = rel-(obs.pos[starts-LAG][:,left]-obs.pos[starts-LAG][:,right])
    un = (incoming*normal).sum(2)
    close = dist<CUTOFF
    groups = {'all':close,'approaching':close&(un<0),'separating_or_stationary':close&(un>=0)}
    prev_overlap,prev_out = geometry_masks(p0,left,right,arena)
    new_overlap,new_out = np.zeros_like(prev_overlap),np.zeros_like(prev_out)
    ever_overlap,ever_out = np.zeros_like(prev_overlap),np.zeros_like(prev_out)
    stats,failures = [],[]
    for frame in range(1,path.shape[1]):
        p = path[:,frame]
        overlap,out = geometry_masks(p,left,right,arena)
        ever_overlap |= overlap
        ever_out |= out
        new_overlap |= overlap & ~prev_overlap
        new_out |= out & ~prev_out
        prev_overlap,prev_out = overlap,out
        if frame not in [k*LAG for k in HORIZONS]:
            continue
        measured = obs.pos[starts+frame]
        true_delta = (measured[:,left]-measured[:,right])-rel
        pred_delta = (p[:,left]-p[:,right])-rel
        ok = np.isfinite(p).all(axis=(1,2))
        for b,start in enumerate(starts):
            key = dict(recording=tag,model=model,start=int(start),seconds=frame/FPS)
            failures.append(dict(**key,failed=int(not ok[b]),wall_robots=ever_out[b].sum(),
                overlap_pairs=ever_overlap[b].sum(),new_wall_robots=new_out[b].sum(),
                new_overlap_pairs=new_overlap[b].sum(),robots=len(obs.ids),
                any_new_wall=int(new_out[b].any()),any_new_overlap=int(new_overlap[b].any()),
                wall_endpoint=out[b].sum(),overlap_endpoint=overlap[b].sum(),
                robot_sse=np.sum((p[b]-measured[b])**2) if ok[b] else np.nan))
            for label,mask in groups.items():
                take = mask[b]
                row = dict(**key,group=label,eligible=take.sum(),scored=take.sum() if ok[b] else 0)
                if ok[b]:
                    for name,axis in [('normal',normal),('tangent',tangent)]:
                        y = (true_delta[b,take]*axis[b,take]).sum(1)
                        pred = (pred_delta[b,take]*axis[b,take]).sum(1)
                        row.update({name+'_sum':y.sum(),name+'_sq':y@y,name+'_sse':np.sum((pred-y)**2)})
                stats.append(row)
    return stats,failures


def run():
    obs,clean = projected_tracks()
    params = fold_calibrations(clean)
    vector_terms = json.loads((PREVIOUS/'summary.json').read_text())['minimum_wape_terms']
    shared_terms = list(json.loads((STUDY/'correction'/'common_law.json').read_text())['shared'])
    vector = fold_weights(PREVIOUS/'cross_recording'/'projected_by_recording.csv.gz',vector_terms)
    correction = fold_weights(STUDY/'correction'/'candidate_errors.csv.gz',shared_terms)
    stats,failures,checks = [],[],[]
    for tag,tr in obs.items():
        starts = start_frames(tr)
        estimator = SharedVelocity(tr,params[tag],vector[tag],correction[tag],len(starts))
        arena = params[tag][2][int(tag[-2:])]
        for model in (*MODELS,'observed'):
            if model=='observed':
                path = tr.pos[starts[:,None]+np.arange(max(HORIZONS)*LAG+1)]
            else:
                path,verified = simulate(model,tr,clean[tag],starts,estimator,arena)
                if verified:
                    checks.append(dict(recording=tag,model=model,verified_first_step_pairs=verified))
            s,f = evaluate(tag,model,starts,tr,path,arena)
            stats.extend(s)
            failures.extend(f)
            print(tag,model,flush=True)
    DEST.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(stats).to_csv(DEST/'pair_errors_by_start.csv.gz',index=False)
    pd.DataFrame(failures).to_csv(DEST/'trajectory_checks_by_start.csv.gz',index=False)
    pd.DataFrame(checks).to_csv(DEST/'archive_checks.csv',index=False)
    summarize(pd.DataFrame(stats),pd.DataFrame(failures))


def summarize(stats,failures):
    by = stats.groupby(['model','seconds','group','recording']).sum(numeric_only=True)
    for name in ('normal','tangent'):
        by[name+'_rmse_px'] = np.sqrt(by[name+'_sse']/by.scored)
        ss = by[name+'_sq']-by[name+'_sum']**2/by.scored
        by[name+'_r2'] = 1-by[name+'_sse']/ss.where(ss>0)
    cols = ['normal_rmse_px','tangent_rmse_px','normal_r2','tangent_r2']
    res = by[cols].groupby(level=['model','seconds','group']).mean()
    res['eligible_pairs'] = by.eligible.groupby(level=['model','seconds','group']).sum()
    res['scored_pairs'] = by.scored.groupby(level=['model','seconds','group']).sum()
    by.drop(columns='start').to_csv(DEST/'pair_errors_by_recording.csv')
    res.to_csv(DEST/'pair_errors.csv')
    by = failures.groupby(['model','seconds','recording']).agg(starts=('start','count'),failed=('failed','sum'),
        wall_starts=('any_new_wall','sum'),overlap_starts=('any_new_overlap','sum'),
        wall_trajectories=('new_wall_robots','sum'),total_robots=('robots','sum'),
        robot_sse=('robot_sse','sum'))
    good = failures[failures.failed==0].groupby(['model','seconds','recording']).robots.sum()
    by['robot_rmse_px'] = np.sqrt(by.robot_sse/good)
    by.to_csv(DEST/'stability_by_recording.csv')
    out = by.drop(columns=['robot_sse','robot_rmse_px']).groupby(level=['model','seconds']).sum()
    out['robot_rmse_px'] = by.robot_rmse_px.groupby(level=['model','seconds']).mean()
    out.to_csv(DEST/'stability.csv')
    intervals = paired_intervals(stats)
    intervals.to_csv(DEST/'paired_differences.csv',index=False)
    write_report(res,out)
    print(res.xs('all',level='group').round(3).to_string(),flush=True)


def paired_intervals(stats):
    subset = stats[stats.group=='all']
    key = ['recording','seconds','start']
    rows = []
    for model,baseline in [('shared_free','fixed'),('shared_free','constant_velocity'),
                           ('shared_guarded','fixed'),('shared_guarded','constant_guarded'),
                           ('shared_guarded','shared_free')]:
        pair = subset[subset.model==model].merge(subset[subset.model==baseline],on=key,
                    suffixes=('_m','_b'),validate='one_to_one')
        for seconds,df in pair.groupby('seconds'):
            rng = np.random.default_rng(25481)
            samples,means = [],[]
            for _,rec in df.groupby('recording'):
                rec = rec.sort_values('start')
                values = rec[['scored_m','normal_sse_m','tangent_sse_m','normal_sse_b','tangent_sse_b']].to_numpy()
                valid = (rec.scored_m>0)&(rec.scored_m==rec.scored_b)
                values[~valid] = 0
                # Resample adjacent pairs of complete 8 s windows as a unit.
                blocks = np.add.reduceat(values,np.arange(0,len(values),2),axis=0)
                def difference(totals):
                    return (np.sqrt(totals[...,1:3]/totals[...,:1])
                            -np.sqrt(totals[...,3:5]/totals[...,:1]))
                samples.append(difference(blocks[rng.integers(len(blocks),size=(2000,len(blocks)))].sum(1)))
                means.append(difference(blocks.sum(0)))
            bounds = np.nanquantile(np.mean(samples,axis=0),[.025,.975],axis=0)
            for k,axis in enumerate(('normal','tangent')):
                rows.append(dict(model=model,baseline=baseline,seconds=seconds,axis=axis,
                    delta_rmse_px=np.mean(means,axis=0)[k],lower_95=bounds[0,k],upper_95=bounds[1,k]))
    return pd.DataFrame(rows)


def write_report(scores,stability):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    names = {'fixed':'Fixed position','constant_velocity':'Constant velocity',
             'constant_guarded':'Constant velocity + geometry','shared_free':'Shared law',
             'shared_guarded':'Shared law + geometry','observed':'Observed tracks'}
    fig,axes = plt.subplots(1,2,figsize=(9,4),layout='constrained',sharey=True)
    for ax,metric,title in zip(axes,['normal_rmse_px','tangent_rmse_px'],['Along the initial pair axis','Across the initial pair axis']):
        for name,color,marker,style in zip(MODELS,['#333333','#c25617','#c25617','#15638b','#15638b'],['o','s','s','^','^'],[':','--','-','--','-']):
            d = scores.xs((name,'all'),level=('model','group')).sort_index()
            ax.plot(d.index,d[metric],label=names[name],color=color,marker=marker,ls=style)
        ax.set(xscale='log',yscale='log',xlabel='Forecast horizon (s)',title=title)
        ax.set_xticks([.4,.8,2,4,8],['0.4','0.8','2','4','8'])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([20,50,100,200,500],['20','50','100','200','500'])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='y',which='major',alpha=.25)
    axes[0].set_ylabel('Relative displacement RMSE (px)')
    axes[0].legend(fontsize=8)
    fig.savefig(DEST/'rollout_errors.png',dpi=160)
    plt.close(fig)
    lines = ['# Free rollouts of the shared collision law','',
        'Five leave-one-recording-out coefficient fits, 40 nonoverlapping starting times in each recording, and up to eight seconds of autonomous evolution per start. The shared vector and normal equation structures were previously selected using the five recordings. This is a retrospective transfer test. Coefficients, propulsion and arena calibration for each held-out recording come from the other four.','',
        '![Relative displacement error versus horizon](rollout_errors.png)','',
        'The constrained shared law improves both pair directions against fixed position through four seconds. At eight seconds the normal displacement errors are effectively tied. Some tangential advantage persists. The geometry-only constant-velocity control has substantially larger errors.','',
        '## Errors along and across the line joining the robots','',
        'At each starting time, keep every pair initially within 253 px and freeze its observed normal and tangent. Endpoints use the measured relative displacement over the entire horizon. Positive normal displacement means separation along the initial axis. RMSE is in camera px (one robot diameter = 219.55 px). Entries are means of five recording-level RMSEs.','',
        '| Horizon (s) | Model | Pair-normal RMSE | Pair-tangent RMSE | Robot position RMSE | Normal R² | Tangent R² |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for sec in [k*LAG/FPS for k in HORIZONS]:
        for name in MODELS:
            s = scores.loc[name,sec,'all']
            v = stability.loc[name,sec]
            lines.append(f'| {sec:g} | {names[name]} | {s.normal_rmse_px:.2f} | {s.tangent_rmse_px:.2f} | {v.robot_rmse_px:.2f} | {s.normal_r2:.3f} | {s.tangent_r2:.3f} |')
    lines += ['', 'Robot position RMSE uses Euclidean displacement error per robot. Pair R² uses observed displacement variance within a recording, then weights recordings equally. Dividing displacement RMSE by its horizon converts it to interval-average velocity RMSE in px/s.','',
        '## Incoming motion','',
        'Classify pairs with their measured relative velocity over the 0.4 s before the origin. Displacements remain signed along the initial axes even if a pair turns or later separates.','',
        '| Horizon (s) | Initial state | Eligible pairs | Fixed normal RMSE | Constant normal RMSE | Shared normal RMSE | Shared + geometry normal RMSE |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for sec in [.4,2.,8.]:
        for group,label in [('approaching','Approaching'),('separating_or_stationary','Separating or stationary')]:
            values = [scores.loc[n,sec,group].normal_rmse_px for n in ('fixed','constant_velocity','shared_free','shared_guarded')]
            lines.append(f'| {sec:g} | {label} | {int(scores.loc["fixed",sec,group].eligible_pairs):,} | '+' | '.join(f'{n:.2f}' for n in values)+' |')
    lines += ['', '## Physical and numerical failures','',
        'Each count is a whole-swarm rollout out of 200. A wall crossing occurs when a center transitions from inside to more than 1 px beyond the fold-calibrated center envelope. A new overlap occurs when two centers cross below diameter minus 1 px. Count each rollout once per category by the given horizon. The observed tracks supply the reference because the fitted envelope and tracker both have error. Per-start files also count affected robots and pairs. Numerical failure means a nonfinite state or an unresolved projection. Errors use completed predictions and the CSVs give eligible and scored pair counts.','',
        '| Horizon (s) | Model | New wall crossing | New overlap | Numerical failure |',
        '|---:|---|---:|---:|---:|']
    for sec in [.4,2.,8.]:
        for name in (*MODELS,'observed'):
            a = stability.loc[name,sec]
            lines.append(f'| {sec:g} | {names[name]} | {int(a.wall_starts)} | {int(a.overlap_starts)} | {int(a.failed)} |')
    lines += ['', '## Implementation and checks','',
        'Starts are at frames 100, 300 and every 200 frames thereafter while eight seconds of outcomes remain. The first 20 frames of each pose history end at the starting frame. The full prior contact history is resolved at each origin. Fixed position holds the observed centers. Constant velocity extrapolates the last 0.4 s of raw center motion. The shared law uses the archived joint update that assigns one velocity to each robot across simultaneous contacts. It recalculates velocities and neighbors every 0.4 s. Between updates it advances centers at 25 Hz. Every subsequent heading advances at that robot’s signed angular rate measured in the 0.4 s before the origin. There is no noise injection or use of later observed poses.','',
        'All physical diagnostics inspect the path at 25 Hz. The free law uses frame-local disk-projected positions for its kinematic inputs and leaves predicted raw centers unconstrained. The geometry variants project centers into the fitted arena and separate overlapping disks after each 0.04 s advance. The constant-velocity geometry control applies the same correction. Final forecast outputs use raw observed positions as their reference.','',
        'The first model velocity, before geometric correction, is checked at every selected close-pair origin against the archived held-out coherent forecasts. Paired uncertainty intervals resample adjacent 16-second blocks within each recording 2,000 times and keep the five recording weights equal. Their scope is variation within these recordings.','',
        '[Errors by recording](pair_errors_by_recording.csv) · [Failures by recording](stability_by_recording.csv) · [Paired RMSE differences and intervals](paired_differences.csv) · [First-step verification](archive_checks.csv)']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    run()
