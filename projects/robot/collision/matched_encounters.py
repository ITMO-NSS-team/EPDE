"""Retrospective comparison of short, completed contact entries and near-miss passes."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from impacts import episodes, pairs
from physics import CAP, CUTOFF, FPS, LAG, OUT, arena_for, rotate
from projected_data import projected_tracks


DEST = OUT/'preprocessed'/'matched_encounters'
TOLERANCES = dict(closing=50.,angle=np.pi/6,spin_i=2.,spin_j=2.,wall_gap=.5*CAP)
OUTCOMES = ('normal_change','tangent_change','aligned_change','lead_normal','lead_tangent',
            'lead_aligned','fixed_normal_change','fixed_tangent_change','fixed_aligned_change')


def encounters(raw,flags,arena):
    left,right,dist,neighbors,wall = pairs(raw,arena)
    results = []
    for k,(i,j) in enumerate(zip(left,right)):
        d = dist[:,k]
        starts,ends = episodes(d,CAP+12,CUTOFF)
        rel = raw.pos[:,j]-raw.pos[:,i]
        # Checks including a time margin are applied identically to contact and miss events.
        usable = (starts>=11)&(ends-starts<=25)&(ends+LAG<len(d))
        last_stop = -1
        for start,end in zip(starts[usable],ends[usable]):
            closest = start+int(np.argmin(d[start:end]))
            stop = max(end,closest+LAG,start+LAG)
            if start-10<=last_stop:
                continue
            if (flags[start-10:stop+1,[i,j]]!=0).any() or (wall[start-10:stop+1,[i,j]]).any():
                continue
            normal = rel[start]/d[start]
            tangent = rotate(normal)
            def velocity(a,b):
                v = (rel[b]-rel[a])*FPS/(b-a)
                return np.array([v@normal,v@tangent])
            early = velocity(start-10,start-6)
            before = velocity(start-5,start-1)
            if before[0]>=0:
                continue
            last_stop = stop
            after = velocity(closest,closest+LAG)
            fixed = velocity(start,start+LAG)
            spin = (raw.phi[start-1,[i,j]]-raw.phi[start-5,[i,j]])*FPS/4
            si,sj = np.sign(spin).astype(int)
            orientation = -si if si==sj and si!=0 else np.nan
            window = slice(start-10,stop+1)
            def isolated(limit,interval):
                other = np.maximum(neighbors[limit][interval,i],neighbors[limit][interval,j])-(d[interval]<limit)
                return not np.any(other>0)
            strict = isolated(CUTOFF,window)
            looser = isolated(1.03*CAP,window)
            result = dict(recording=raw.tag,i=raw.ids[i],j=raw.ids[j],start=start,closest=closest,
                end=end,minimum_gap=d[closest]-CAP,strict=strict,physical_clearance=looser,
                isolated_at_entry=isolated(CUTOFF,slice(start-5,start+1)),
                closing=-before[0],angle=np.arctan2(before[1],-before[0]),spin_i=spin[0],spin_j=spin[1],
                sign_i=si,sign_j=sj,wall_gap=arena[2]-np.linalg.norm(raw.pos[start-1,[i,j]]-arena[:2],axis=1).max(),
                before_normal=before[0],before_tangent=before[1],normal_change=after[0]-before[0],
                tangent_change=after[1]-before[1],aligned_change=orientation*(after[1]-before[1]),
                lead_normal=before[0]-early[0],lead_tangent=before[1]-early[1],
                lead_aligned=orientation*(before[1]-early[1]),
                fixed_normal_change=fixed[0]-before[0],fixed_tangent_change=fixed[1]-before[1],
                fixed_aligned_change=orientation*(fixed[1]-before[1]))
            results.append(result)
    return pd.DataFrame(results)


def match(data,threshold):
    matches = []
    for keys,df in data.groupby(['recording','i','j','sign_i','sign_j']):
        a = df[df.minimum_gap<threshold]
        b = df[df.minimum_gap>=threshold+2]
        if a.empty or b.empty:
            continue
        cols = list(TOLERANCES)
        delta = (a[cols].to_numpy()[:,None,:]-b[cols].to_numpy()[None,:,:])/list(TOLERANCES.values())
        eligible = (abs(delta)<=1).all(2)
        cost = (delta**2).sum(2)
        # A disallowed assignment costs more than all allowed assignments combined.
        cost[~eligible] = 1+len(cols)*min(len(a),len(b))
        ii,jj = linear_sum_assignment(cost)
        for x,y in zip(ii,jj):
            if not eligible[x,y]:
                continue
            ca,mi = a.iloc[x],b.iloc[y]
            row = dict(recording=keys[0],i=keys[1],j=keys[2],sign_i=keys[3],sign_j=keys[4],
                       contact_frame=ca.start,miss_frame=mi.start,contact_closest=ca.closest,miss_closest=mi.closest,
                       cost=cost[x,y])
            row.update({key:ca[key]-mi[key] for key in (*OUTCOMES,*cols,'before_tangent')})
            matches.append(row)
    return pd.DataFrame(matches,columns=['recording','i','j','sign_i','sign_j','contact_frame','miss_frame',
                        'contact_closest','miss_closest','cost',*OUTCOMES,*TOLERANCES,'before_tangent'])


def write_report(summary):
    primary = summary[summary.threshold_px==0]
    crowded = primary[primary.group=='third_neighbor']
    lines = ['# Contact entries and matched near misses', '',
        'Five swarm recordings at 25 Hz. A contact entry is inferred from measured center separation below one calibrated diameter (219.55 px). A near miss remains at least 2 px outside that diameter. Video confirmation of physical contact was unavailable.', '',
        '| Third-neighbor criterion | Contact entries | Near misses | Matched comparisons | Co-rotating matches | Contributing recordings |',
        '|---|---:|---:|---:|---:|---:|']
    labels = {'isolated':'None within 253 px throughout', 'physical_clearance':'None within 1.03 diameters throughout',
              'isolated_at_entry':'None within 253 px immediately before and at entry', 'third_neighbor':'Within 253 px at some point'}
    for key,label in labels.items():
        d = primary[primary.group==key]
        lines.append(f'| {label} | {d.contacts.sum():,} | {d.misses.sum():,} | {d.matches.sum():,} | {d.corotating.sum():,} | {(d.matches>0).sum()} |')
    lines += ['', 'The strict and third-neighbor groups partition the encounters. The other two rows are overlapping sensitivity analyses. The strictly isolated events all came from the 30-robot recording. No matched estimate was available for continuous strict isolation.', '',
        '## Normal and tangential velocity', '',
        'Each entry below is the contact-minus-near-miss difference in velocity change, in px/s. Positive normal velocity points toward separation along the entry axis. The spin-aligned tangent uses minus the common spin sign times signed tangential velocity and includes co-rotating pairs only. Closest and entry columns average velocity over the 0.4 s following the named frame and subtract the same incoming velocity. Lead columns compare two incoming velocity windows before entry.', '',
        '| Recording with third neighbors | Matches | Normal, closest | Normal, entry | Normal, lead | Tangent, closest | Tangent, lead | Spin-aligned, closest | Spin-aligned, lead |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    cols = ['normal_change','fixed_normal_change','lead_normal','tangent_change','lead_tangent','aligned_change','lead_aligned']
    def effect_row(label, count, values):
        return f'| {label} | {count:,} | ' + ' | '.join(f'{v:+.2f}' for v in values) + ' |'
    for row in crowded.itertuples():
        lines.append(effect_row(row.recording,row.matches,[getattr(row,k) for k in cols]))
    lines += [effect_row('Equal weight per recording',crowded.matches.sum(),crowded[cols].mean()),
        '', 'Normal contrasts switch sign with the timing reference in every recording. Closest approach is selected retrospectively from each trajectory. Matching also leaves differences in pre-entry tangential motion. These observations limit attribution to contact or spin.', '',
        '## Sensitivity to the measured contact boundary', '',
        'Offsets shift both thresholds. The 2 px gap between contact and near miss is retained. Effects are equal-weight means over contributing recordings in the third-neighbor group.', '',
        '| Boundary offset (px) | Matches | Normal, closest | Normal, entry | Spin-aligned, closest |',
        '|---:|---:|---:|---:|---:|']
    for offset,d in summary[summary.group=='third_neighbor'].groupby('threshold_px'):
        lines.append(effect_row(f'{offset:+d}',d.matches.sum(),d[['normal_change','fixed_normal_change','aligned_change']].mean()))
    lines += ['', '## Selection and matching', '',
        'An encounter starts at the first inward crossing of diameter + 12 px following clearance of 253 px. It must clear 253 px again within 25 frames. Closest approach is the minimum measured separation before that clearance. The analysis excludes nonapproaching pairs, wall encounters and flagged tracks. Overlapping observation windows for the same pair are screened in chronological order. This restricts inference to short, completed passes.', '',
        'We use the axis from the first robot to the second at entry. Relative velocity is the second robot’s velocity minus the first’s. Incoming velocity runs from frame −5 to −1 relative to entry. The earlier window runs from −10 to −6. Continuous isolation and quality checks span frame −10 through the later of release or 10 frames after closest approach. The entry-only isolation check spans frames −5 through 0.', '',
        'Matches require the same recording, robot identities and individual spin signs. Maximum differences are 50 px/s in incoming closing speed, 30 degrees in signed approach angle, 2 rad/s for each angular velocity and half a diameter in wall clearance. We maximize the number of matches without reuse, then minimize the sum of squared differences scaled by these tolerances. Repeated encounters can share robots. Equal recording weights prevent the largest recording from dominating the reported contrasts.', '',
        'The [per-recording results](by_recording.csv) include covariate differences and all outcomes. The [matches](matches.csv.gz) give frame indices for each contact and near miss. The [encounters](encounters.csv.gz) include eligible events before matching.']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    raw,clean = projected_tracks()
    frames = []
    for tag,tr in raw.items():
        params = arena_for([track.part(0,3*len(track.pos)//4) for t,track in clean.items() if t!=tag])
        with np.load(OUT/'preprocessed'/f'{tag}.npz') as z:
            frames.append(encounters(tr,z['quality_flags'],params[int(tag[-2:])]))
    DEST.mkdir(parents=True,exist_ok=True)
    all_events = pd.concat(frames,ignore_index=True)
    all_events.to_csv(DEST/'encounters.csv.gz',index=False)
    all_matches,summary = [],[]
    for label,df in [('isolated',all_events[all_events.strict]),('physical_clearance',all_events[all_events.physical_clearance]),
                     ('isolated_at_entry',all_events[all_events.isolated_at_entry]),
                     ('third_neighbor',all_events[~all_events.strict])]:
        for threshold in (-2,0,2):
            m = match(df,threshold)
            all_matches.append(m.assign(group=label,threshold_px=threshold))
            for tag in raw:
                events = df[df.recording==tag]
                sub = m[m.recording==tag]
                row = dict(group=label,threshold_px=threshold,recording=tag,contacts=(events.minimum_gap<threshold).sum(),
                        misses=(events.minimum_gap>=threshold+2).sum(),matches=len(sub),corotating=sub.aligned_change.count(),
                        robot_pairs=len(sub[['i','j']].drop_duplicates()))
                for key in (*OUTCOMES,*TOLERANCES,'before_tangent'):
                    row[key] = pd.to_numeric(sub[key]).mean()
                summary.append(row)
    pd.concat(all_matches,ignore_index=True).to_csv(DEST/'matches.csv.gz',index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(DEST/'by_recording.csv',index=False)
    preview = summary[summary.threshold_px==0].groupby('group').agg(
        contacts=('contacts','sum'),misses=('misses','sum'),matches=('matches','sum'),corotating=('corotating','sum'),
        contributing=('normal_change','count'),normal=('normal_change','mean'),tangent=('tangent_change','mean'),
        aligned=('aligned_change','mean'),lead_n=('lead_normal','mean'),lead_t=('lead_tangent','mean'))
    preview.to_csv(DEST/'summary.csv')
    write_report(summary)
    print(preview.round(2).to_string())


if __name__=='__main__':
    run()
