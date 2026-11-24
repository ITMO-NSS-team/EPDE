"""Audit observation rows and export frame-local, hard-disk center estimates at 25 Hz."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from physics import CAP, FPS, OUT, Track, arena_for, swarm_files, swarm_meta

DEST = OUT/'preprocessed'
REPEAT, FULL_REPEAT, POSITION_JUMP, HEADING_JUMP = 1, 2, 4, 8
FLAGS = {REPEAT:'identical_pose_run', FULL_REPEAT:'whole_frame_repeat',
         POSITION_JUMP:'position_jump_endpoints', HEADING_JUMP:'heading_jump_endpoints'}


def load_observations(path):
    meta = swarm_meta(path)
    with open(path,'rb') as stream:
        frames = pickle.load(stream)
    ids = np.array(sorted({int(row[0]) for row in frames[0]}))
    if len(ids)!=meta['n']:
        raise ValueError(f'{path}: initial IDs disagree with robot count')
    loc = {key:i for i,key in enumerate(ids)}
    pos = np.full((len(frames),len(ids),2),np.nan)
    angles = np.full(pos.shape[:2],np.nan)
    for f,frame in enumerate(frames):
        seen = set()
        for rid, angle, xy in frame:
            if rid not in loc or rid in seen:
                raise ValueError(f'{path}, frame {f}: unexpected or duplicate ID {rid}')
            seen.add(rid)
            pos[f,loc[rid]] = xy
            angles[f,loc[rid]] = angle
        if len(seen)!=len(ids):
            raise ValueError(f'{path}, frame {f}: missing IDs {set(ids)-seen}')
    if not np.isfinite(pos).all() or not np.isfinite(angles).all():
        raise ValueError(f'{path}: nonfinite pose')
    return Track(meta['tag'],meta['pwm'],ids,pos,np.unwrap(np.deg2rad(angles),axis=0))


def segments(mask):
    edges = np.diff(np.r_[False,mask,False].astype(np.int8))
    return zip(np.flatnonzero(edges>0),np.flatnonzero(edges<0))


def quality(track):
    step = np.linalg.norm(np.diff(track.pos,axis=0),axis=-1)
    turn = np.abs(np.rad2deg(np.diff(track.phi,axis=0)))
    same = (step==0)&(turn==0)
    flags = np.zeros(track.phi.shape,dtype=np.uint8)
    for j in range(len(track.ids)):
        for first,stop in segments(same[:,j]):
            if stop-first>=2:
                flags[first:stop+1,j] |= REPEAT
    whole = same.all(axis=1)
    for bad,bit in [(step>.2*CAP,POSITION_JUMP), (turn>60,HEADING_JUMP),
                    (np.broadcast_to(whole[:,None],step.shape),FULL_REPEAT)]:
        flags[:-1][bad] |= bit
        flags[1:][bad] |= bit
    xc,yc,rad = arena_for([track.part(0,3*len(track.pos)//4)])[int(track.tag[-2:])]
    excess = np.linalg.norm(track.pos-[xc,yc],axis=-1)-rad
    return flags, excess>.1*CAP, dict(whole_frame_repeats=int(whole.sum()), position_jumps=int((step>.2*CAP).sum()),
        heading_jumps=int((turn>60).sum()), max_step_px=step.max(), max_heading_step_deg=turn.max(),
        flagged_robot_frames=int(np.count_nonzero(flags)), flagged_pct=100*np.mean(flags!=0),
        outside_prefix_envelope_by_0p1D=int((excess>.1*CAP).sum()), max_envelope_excess_px=excess.max())


def project_centers(pos):
    """Split each correction equally. Recheck every pair after each simultaneous update."""
    left,right = np.triu_indices(pos.shape[1],1)
    corrected = pos.copy()
    penetrations, involved = [], 0
    raw_min, clean_min, frames_with_overlap = np.inf,np.inf,0
    for start in range(0,len(pos),96):
        block = corrected[start:start+96]
        active = np.arange(len(block))
        for iteration in range(1001):
            diff = block[active[:,None],left]-block[active[:,None],right]
            dist = np.linalg.norm(diff,axis=-1)
            if (dist==0).any():
                raise ValueError('Coincident centers have an undefined correction direction')
            bad = dist<CAP
            f,pair = np.nonzero(bad)
            if iteration==0:
                raw_min = min(raw_min,dist.min())
                penetrations.extend((CAP-dist[bad]).tolist())
                frames_with_overlap += bad.any(axis=1).sum()
                touched = np.zeros(block.shape[:2],bool)
                touched[f,left[pair]] = touched[f,right[pair]] = True
                involved += touched.sum()
            done = ~bad.any(axis=1)
            if np.any(done):
                clean_min = min(clean_min,dist[done].min())
            if not len(f):
                break
            if iteration==1000:
                raise RuntimeError(f'Unresolved overlap near frame {start+active[f[0]]}')
            # A sub-micropixel margin prevents a roundoff-induced residual overlap.
            correction = .5*((CAP+1e-7-dist[f,pair])/dist[f,pair])[:,None]*diff[f,pair]
            np.add.at(block,(active[f],left[pair]),correction)
            np.add.at(block,(active[f],right[pair]),-correction)
            active = active[~done]
    np.testing.assert_allclose(corrected.mean(axis=1),pos.mean(axis=1),rtol=0,atol=1e-9)
    assert clean_min>=CAP
    shifts = np.linalg.norm(corrected-pos,axis=-1)
    positive = shifts[shifts>0]
    depth = np.asarray(penetrations)
    return corrected,dict(raw_overlapping_pair_frames=len(depth),raw_frames_with_overlap=int(frames_with_overlap),
                 raw_involved_robot_frames=int(involved),raw_min_separation_px=raw_min,
                 overlap_gt_1px=int((depth>1).sum()),overlap_gt_5px=int((depth>5).sum()),
                 overlap_gt_10px=int((depth>10).sum()),
                 median_overlap_px=float(np.median(depth)) if len(depth) else 0.,max_overlap_px=max(0.,CAP-raw_min),
                 clean_min_separation_px=clean_min, remaining_overlapping_pair_frames=0,
                 moved_robot_frames=len(positive),median_shift_when_moved_px=np.median(positive) if len(positive) else 0.,
                 max_shift_px=shifts.max())


def load_product(path):
    """Return observed and projected tracks with the per-pose review flags."""
    with np.load(path) as z:
        tag,pwm,ids,phi = str(z['recording']),int(z['pwm']),z['ids'],z['phi']
        return Track(tag,pwm,ids,z['raw_pos'],phi), Track(tag,pwm,ids,z['pos'],phi), z['quality_flags']


def write_manifest():
    files = [f'{swarm_meta(path)["tag"]}.npz' for path in swarm_files()]
    data = dict(files=files, fps=FPS, cap_diameter_px=CAP, cap_diameter_mm=48.7,
                fields=dict(pos='Projected centers [frame, robot, x/y] in camera pixels',
                            raw_pos='Observed centers in the same order and units',
                            phi='Observed unwrapped headings in radians',ids='Physical robot IDs in column order',
                            frame='Original zero-based frame index',quality_flags='Per-pose bit mask',
                            outside_reference_envelope='Review only, beyond the training-prefix center envelope by 0.1 diameters',
                            raw_sha256='SHA-256 of the corresponding source pickle'),
                quality_flag_bits=FLAGS,
                review_thresholds=dict(identical_pose_run='At least three identical consecutive position-heading samples',
                                       position_jump='One frame exceeds 0.2 cap diameters',
                                       heading_jump='One frame exceeds 60 degrees after angle unwrapping'),
                geometry='Equal and opposite center shifts for violating pairs, repeated until all separations meet the diameter. Each frame uses only its recorded centers.',
                history='Use raw_pos for existing strict-radius crossing and apparent-overlap terms. Re-establish contact labels before fitting event-dependent terms to projected data.',
                evaluation='heldout_raw_target.csv compares all pipelines against original velocity targets and raw pair normals. The qc test subset also requires reliable input windows for neighbors up to two steps away.')
    (DEST/'manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')


def run():
    DEST.mkdir(exist_ok=True,parents=True)
    summary, events = [],[]
    for path in swarm_files():
        tr = load_observations(path)
        flags, boundary_review, audit = quality(tr)
        clean, geometry = project_centers(tr.pos)
        for j,rid in enumerate(tr.ids):
            for bit,name in FLAGS.items():
                for lo,hi in segments((flags[:,j]&bit)!=0):
                    events.append(dict(recording=tr.tag,robot_id=rid,first_frame=lo,last_frame=hi-1,flag=name))
            for lo,hi in segments(boundary_review[:,j]):
                events.append(dict(recording=tr.tag,robot_id=rid,first_frame=lo,last_frame=hi-1,flag='outside_reference_envelope'))
        summary.append(dict(recording=tr.tag, frames=len(tr.pos), robots=len(tr.ids), cap_px=CAP,
                       missing_ids=0,duplicate_ids=0,nonfinite_poses=0,**audit,**geometry))
        tmp = DEST/f'{tr.tag}.tmp.npz'
        np.savez_compressed(tmp,recording=tr.tag,pwm=tr.pwm,ids=tr.ids,fps=FPS,cap_px=CAP,
                frame=np.arange(len(clean)),pos=clean,raw_pos=tr.pos,phi=tr.phi,quality_flags=flags,
                outside_reference_envelope=boundary_review,flag_bits=json.dumps(FLAGS),
                raw_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        tmp.replace(DEST/f'{tr.tag}.npz')
        print(tr.tag,geometry['raw_overlapping_pair_frames'],'overlaps projected;',audit['flagged_robot_frames'],'flagged poses',flush=True)
    pd.DataFrame(events).to_csv(DEST/'flagged_intervals.csv',index=False)
    pd.DataFrame(summary).to_csv(DEST/'audit.csv',index=False)
    write_manifest()


if __name__=='__main__':
    run()
