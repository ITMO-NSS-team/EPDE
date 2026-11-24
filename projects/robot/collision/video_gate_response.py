"""Select completed inward crossings and match video-reviewed robot encounters."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from impacts import episodes, pairs
from physics import CAP, CUTOFF, FPS, OUT, SIGMA, Track


BASE = OUT / 'preprocessed'
DEST = BASE / 'video_gate_response'
GATE = CAP + 20
CALIPERS = {'closing': 75., 'side': 100., 'gate_gap': 10., 'wall_gap': .5 * CAP}
STRATA = ['spin_group', 'third_near']


def candidates(path):
    tag = path.stem
    arena = np.array(json.loads((OUT / tag / 'training.json').read_text())['arenas'][tag[-2:]])
    with np.load(path) as data:
        track = Track(tag, int(data['pwm']), data['ids'], data['raw_pos'], data['phi'])
        flags = data['quality_flags']
    i_all, j_all, distance, counts, wall = pairs(track, arena)
    old = pd.read_csv(BASE / 'video_rebound_verification/reviewed_encounters.csv')
    pilot = pd.read_csv(BASE / 'matched_encounters/video_review_pilot.csv').rename(columns={
        'robot_id_a': 'i', 'robot_id_b': 'j', 'track_start_frame_0based': 'lo',
        'track_end_frame_0based': 'hi'})
    old = pd.concat([old, pilot]).query('recording == @tag')
    frames = []
    for k, (i,j) in enumerate(zip(i_all, j_all)):
        sep = distance[:, k]
        start, end = episodes(sep, GATE, CUTOFF)
        usable = (start >= 6) & (end - start <= 8) & (end + 2 < len(sep))
        start, end = start[usable], end[usable]
        if not len(start):
            continue
        previous = old[(old.i == track.ids[i]) & (old.j == track.ids[j])]
        valid = np.ones(len(start), dtype=bool)
        for r in previous.itertuples():
            valid &= (end + 5 < r.lo) | (start - 5 > r.hi)
        for f in range(-5, 3):
            ix = start + f
            valid &= ~(wall[ix,i] | wall[ix,j] | (flags[ix,i] != 0) | (flags[ix,j] != 0))
        start, end = start[valid], end[valid]
        if not len(start):
            continue
        rel = track.pos[:,j] - track.pos[:,i]
        n = rel[start] / sep[start,None]
        tan = np.column_stack((-n[:,1], n[:,0]))
        v = (rel[start-1] - rel[start-3]) * FPS / 2
        spin = np.sign(track.phi[start-1][:,[i,j]] - track.phi[start-3][:,[i,j]])
        spin_group = np.where(spin.prod(1) < 0, 'opposite',
                              np.where(spin.sum(1) == 2, '++', np.where(spin.sum(1) == -2, '--','unknown')))
        third = np.maximum(counts[1.03*CAP][:,i], counts[1.03*CAP][:,j]) > (sep < 1.03*CAP)
        region = start[:,None] + np.arange(-3,3)
        df = pd.DataFrame(dict(recording=tag, i=track.ids[i], j=track.ids[j],
            gate_frame=start, release=end, closing=-(n*v).sum(1), side=(tan*v).sum(1),
            spin_group=spin_group, third_near=third[region].any(1), gate_gap=sep[start]-CAP,
            wall_gap=arena[2]-np.linalg.norm(track.pos[start-1][:,[i,j]]-arena[:2],axis=2).max(1),
            minimum_gap=[sep[a:b].min()-CAP for a,b in zip(start,end)]))
        frames.append(df[(df.closing > 0) & (df.spin_group != 'unknown')])
    return pd.concat(frames, ignore_index=True)


def optimal_matches(contact, miss):
    """Largest number of disjoint caliper matches, then smallest standardized distance."""
    matches = []
    for key, a in contact.groupby(STRATA):
        b = miss[(miss.spin_group == key[0]) & (miss.third_near == key[1])]
        if b.empty:
            continue
        cols = list(CALIPERS)
        delta = (a[cols].to_numpy()[:,None] - b[cols].to_numpy()[None,:]) / list(CALIPERS.values())
        allowed = (abs(delta) <= 1).all(2)
        # Concurrent encounters could share motion or a third participant.
        allowed &= abs(a.gate_frame.to_numpy()[:,None] - b.gate_frame.to_numpy()) > 12
        distance = (delta**2).sum(2)
        left, right = linear_sum_assignment(np.where(allowed, distance, 5*(1+min(len(a),len(b)))))
        matches.extend((a.index[i], b.index[j], distance[i,j]) for i,j in zip(left,right) if allowed[i,j])
    return matches


def select():
    selected = []
    for seed, path in enumerate(sorted(BASE.glob('N*_PWM*_D*.npz')), 9042):
        df = candidates(path)
        inner = df[df.minimum_gap <= 0]
        outer = df[df.minimum_gap >= 10]
        pairs_found = optimal_matches(inner, outer)
        rng = np.random.default_rng(seed)
        # Prefer pairs with no close third neighbor. Sample within strata without viewing responses.
        order = sorted(rng.permutation(len(pairs_found)), key=lambda k: bool(df.loc[pairs_found[k][0],'third_near']))
        taken = []
        for k in order:
            a,b,_ = pairs_found[k]
            overlaps = any(abs(df.at[t,'gate_frame'] - df.at[u,'gate_frame']) <= 15
                           for t in (a,b) for u in taken)
            if not overlaps:
                taken.extend((a,b))
            if len(taken) == 32:
                break
        for stratum in (inner, outer):
            count = len(set(taken).intersection(stratum.index))
            for t in rng.permutation(stratum.index):
                if count >= 16:
                    break
                if all(abs(df.at[t,'gate_frame']-df.at[u,'gate_frame'])>15 for u in taken):
                    taken.append(t)
                    count += 1
        selected.append(df.loc[taken].sort_values('gate_frame'))
    selected = pd.concat(selected,ignore_index=True)
    selected.insert(0,'event',np.arange(1,len(selected)+1))
    selected['batch'] = 1
    DEST.mkdir(parents=True,exist_ok=True)
    selected.to_csv(DEST/'selection.csv',index=False)
    print(selected.groupby(['recording','third_near']).size().to_string())


def expand():
    """Recruit additional optical controls using existing labels and gate covariates."""
    first = pd.read_csv(DEST/'selection.csv')
    first = first[first.batch==1] if 'batch' in first else first.assign(batch=1)
    labels = pd.read_csv(DEST/'optical_labels.csv')[['event','optical_label','first_visible_touch_offset']]
    reviewed = first.merge(labels,on='event',validate='one_to_one')
    additions = []
    for path in sorted(BASE.glob('N*_PWM*_D*.npz')):
        df = candidates(path)
        original = reviewed[reviewed.recording==path.stem]
        used = list(original.gate_frame)
        chosen = []
        early = original[original.first_visible_touch_offset==1]
        gaps = original[original.optical_label=='clear gap'].sample(frac=1,random_state=326)
        for anchors,pool,per_anchor,limit in [
                (early,df[df.minimum_gap>=8],3,3*len(early)),
                (gaps,df[df.minimum_gap<=0],1,7)]:
            n = 0
            for anchor in anchors.itertuples():
                eligible = pool[(pool.spin_group==anchor.spin_group)&(pool.third_near==anchor.third_near)]
                delta = np.column_stack([(eligible[c]-getattr(anchor,c))/tol for c,tol in CALIPERS.items()])
                cost = (delta**2).sum(1)
                for_increment = 0
                for ix in np.argsort(cost):
                    if (abs(delta[ix])>1).any():
                        continue
                    event = eligible.iloc[ix]
                    if any(abs(t-event.gate_frame)<=15 for t in used):
                        continue
                    chosen.append(event.name)
                    used.append(event.gate_frame)
                    for_increment += 1
                    n += 1
                    if for_increment>=per_anchor or n>=limit:
                        break
                if n>=limit:
                    break
        additions.append(df.loc[chosen].sort_values('gate_frame'))
    new = pd.concat(additions,ignore_index=True).assign(batch=2)
    new['event'] = np.arange(len(first)+1,len(first)+len(new)+1)
    all_rows = pd.concat([first,new],ignore_index=True)
    all_rows.to_csv(DEST/'selection.csv',index=False)
    print(all_rows.groupby(['recording','batch']).size().to_string())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--expand',action='store_true')
    expand() if parser.parse_args().expand else select()
