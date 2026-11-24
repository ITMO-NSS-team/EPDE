"""Use the 400 mm arena and 48.7 mm disc to register distances without camera orientation."""
import json
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from new_pair_interactions import DEST, GATES, H, HZ, OUTER, ROBOT, clip_rows, episodes, read_stage

DIAMETER_MM = 48.7
ARENA_MM = 400.
IMAGE_DIAMETER_PX = 212.
HERE = DEST/'physical'


def envelope(cloud, quantile):
    """Robust circle through sector-wise outer center positions, with equal weights by angle."""
    cx = np.quantile(cloud.real,[.01,.99]).mean()+1j*np.quantile(cloud.imag,[.01,.99]).mean()
    for _ in range(3):
        rel = cloud-cx
        sector = np.minimum(71, ((np.angle(rel)+np.pi)*72/(2*np.pi)).astype(int))
        edge = []
        for k in range(72):
            pts = rel[sector==k]
            if len(pts)>=20:
                r = np.quantile(abs(pts),quantile)
                theta = np.angle(pts[abs(pts)>=r].mean())
                edge.append(cx+r*np.exp(1j*theta))
        assert len(edge)>=65
        edge = np.asarray(edge)
        p = least_squares(lambda x:abs(edge-(x[0]+1j*x[1]))-x[2],
                          [cx.real,cx.imag,np.median(abs(edge-cx))],loss='soft_l1',f_scale=.4).x
        cx = p[0]+1j*p[1]
    return p


def main():
    HERE.mkdir(exist_ok=True)
    events,regular,calibration = [],[],[]
    for archive in sorted(ROBOT.glob('Experiment_[123].zip')):
        with zipfile.ZipFile(archive) as z:
            tracks = {int(s.split('/data_N=')[1].split('_')[0]):read_stage(z,s,int(s.split('/data_N=')[1].split('_')[0]))
                      for s in z.namelist() if '/data_N=' in s}
        # Estimate scale from separate high-occupancy stages. Positions alone enter calibration.
        clouds = {size:tr[2][:12*HZ:5].ravel() for size,tr in tracks.items() if size>=20}
        for size,(times,ids,raw,phi,missing) in sorted(tracks.items()):
            if size==1:
                continue
            cloud = np.concatenate([p for k,p in clouds.items() if k!=size])
            cx,cy,rad = envelope(cloud,.99)
            mm_per_unit = (ARENA_MM-DIAMETER_MM)/(2*rad)
            diameter = DIAMETER_MM/mm_per_unit
            # Sensitivity of the inferred diameter to the tracked wall envelope.
            alt = [2*envelope(cloud,q)[2]*DIAMETER_MM/(ARENA_MM-DIAMETER_MM) for q in (.97,.995)]
            calibration.append(dict(experiment=archive.stem,N=size,center_x=cx,center_y=cy,radius=rad,
                mm_per_archive_unit=mm_per_unit,diameter_archive_units=diameter,
                diameter_q97=alt[0],diameter_q995=alt[1],reference_stages=sum(k!=size for k in clouds)))
            pos = raw/diameter
            jumps = (abs(np.diff(pos,axis=0))>.25)|(abs(np.diff(phi,axis=0))>np.pi/3)
            flags = missing.copy()
            flags[1:] |= jumps
            flags[:-1] |= jumps
            prefix = np.vstack([np.zeros(size,int),flags.cumsum(axis=0)])
            left,right = np.triu_indices(size,1)
            rel = pos[:,right]-pos[:,left]
            dist = abs(rel)
            grid = np.arange(22,len(raw)-H,H)
            tt,k = np.where(dist[grid]<OUTER)
            reg = clip_rows(grid[tt],k,dist,rel,left,right,ids,pos,phi,prefix,0.)
            regular.append(reg.assign(N=size,experiment=archive.stem,end=times[-1]))
            for gate in (.98,1.,1.02,1.05,1.1):
                entries = [episodes(d,gate,OUTER)[0] for d in dist.T]
                t = np.concatenate(entries)
                pairs = np.repeat(np.arange(len(entries)),[len(x) for x in entries])
                e = clip_rows(t,pairs,dist,rel,left,right,ids,pos,phi,prefix,gate)
                i,j = np.searchsorted(ids,e.id_i),np.searchsorted(ids,e.id_j)
                f = e.frame.to_numpy()
                r = abs(pos[f[:,None]+np.arange(H+1),j[:,None]]-pos[f[:,None]+np.arange(H+1),i[:,None]])
                e['range_increased'] = r[:,-1]>r[:,0]
                e['cleared_any'] = (r>=OUTER).any(1)
                events.append(e.assign(experiment=archive.stem,N=size))
        print('Calibrated and measured',archive.stem,flush=True)
    cal = pd.DataFrame(calibration)
    cal.to_csv(HERE/'scale.csv',index=False)
    ev = pd.concat(events,ignore_index=True)
    ev.to_csv(HERE/'events.csv.gz',index=False,float_format='%.9g')
    reg = pd.concat(regular,ignore_index=True)
    reg.to_csv(HERE/'origins.csv.gz',index=False,float_format='%.9g')
    rows,rest = [],[]
    for gate,d in ev.groupby('gate'):
        approach = d[d.incoming<0]
        for exp,part in [('all',approach),*approach.groupby('experiment')]:
            for group,s in [('all',part),('third',part[part.third]),('no third',part[~part.third]),
                            ('same spin',part[part.same]),('opposite spin',part[part.spin_i*part.spin_j<0]),
                            ('two robots',part[part.N==2])]:
                if len(s):
                    rows.append(dict(gate=gate,entry_mm=gate*DIAMETER_MM,experiment=exp,group=group,n=len(s),
                        in_median_mm_s=DIAMETER_MM*s.incoming.median(),out_median_mm_s=DIAMETER_MM*s.outward.median(),
                        opening=(s.outward>0).mean(),increased_distance=s.range_increased.mean(),
                        cleared=s.cleared_any.mean()))
        close = d[d.incoming_short<0]
        for held in sorted(close.experiment.unique()):
            tr,te = close[close.experiment!=held],close[close.experiment==held]
            if tr.empty or len(te)<2:
                continue
            w = 1/tr.groupby('experiment').frame.transform('count').to_numpy()
            c,y = -tr.incoming_short.to_numpy(),tr.immediate.to_numpy()
            e = max(0.,np.sum(w*c*y)/np.sum(w*c*c))
            obs = te.immediate.to_numpy()
            for method,pred in [('restitution',-e*te.incoming_short),('training mean',np.average(y,weights=w))]:
                mse = np.mean((obs-pred)**2)
                rest.append(dict(held=held,gate=gate,model=method,events=len(te),e=e if method=='restitution' else np.nan,
                                 r2=1-mse/obs.var(),rmse_mm_s=DIAMETER_MM*np.sqrt(mse)))
    pd.DataFrame(rows).to_csv(HERE/'encounters.csv',index=False)
    pd.DataFrame(rest).to_csv(HERE/'restitution.csv',index=False)
    (HERE/'dimensions.json').write_text(json.dumps(dict(diameter_mm=DIAMETER_MM,radius_mm=24.35,height_mm=19.2,
        nominal_diameter_original_px=IMAGE_DIAMETER_PX,mm_per_original_px=DIAMETER_MM/IMAGE_DIAMETER_PX,
        arena_mm=ARENA_MM,body_shape='circular'),indent=1),encoding='utf-8')


if __name__=='__main__':
    main()
