"""Learn each robot's free motion on other experiments and project it onto held-out pairs."""
import json
import zipfile

import numpy as np
import pandas as pd

from new_pair_interactions import DEST, HZ, H, ROBOT, read_stage


def robots():
    inputs = json.loads((DEST/'inputs.json').read_text())
    tables = []
    for archive in sorted(ROBOT.glob('Experiment_[123].zip')):
        scale = inputs[archive.stem]['reference_spacing']
        with zipfile.ZipFile(archive) as z:
            for path in z.namelist():
                if '/data_N=' not in path:
                    continue
                size = int(path.split('/data_N=')[1].split('_')[0])
                times,ids,raw,phi,missing = read_stage(z,path,size)
                pos = raw/scale
                jump = (abs(np.diff(pos,axis=0))>.25)|(abs(np.diff(phi,axis=0))>np.pi/3)
                flags = missing.copy()
                flags[1:] |= jump
                flags[:-1] |= jump
                nearest = np.full(pos.shape,np.inf)
                if size>1:
                    for j in range(0,len(pos),100):
                        p = pos[j:j+100]
                        dist = abs(p[:,:,None]-p[:,None,:])
                        dist[:,np.arange(size),np.arange(size)] = np.inf
                        nearest[j:j+100] = np.min(dist,axis=2)
                bad = np.vstack([np.zeros(size,int),flags.cumsum(0)])
                near = np.vstack([np.zeros(size,int),(nearest<1.5).cumsum(0)])
                rows = np.arange(22,len(times)-H,H)
                frames,idx = np.where(bad[rows+H+1]==bad[rows-22])
                t = rows[frames]
                free = near[t+H+1,idx]==near[t-22,idx]
                v = (pos[t-1,idx]-pos[t-11,idx])*HZ/10
                old = (pos[t-12,idx]-pos[t-22,idx])*HZ/10
                actual = (pos[t+H,idx]-pos[t,idx])*HZ/H
                tables.append(pd.DataFrame(dict(experiment=archive.stem,N=size,frame=t,id=ids[idx],end=times[-1],
                    free=free,x=pos[t,idx].real,y=pos[t,idx].imag,theta=phi[t,idx],vx=v.real,vy=v.imag,
                    old_x=old.real,old_y=old.imag,out_x=actual.real,out_y=actual.imag)))
    return pd.concat(tables,ignore_index=True)


def design(rows):
    return np.column_stack([np.ones(len(rows)),np.cos(rows.theta),np.sin(rows.theta),
                            rows.vx+1j*rows.vy,rows.old_x+1j*rows.old_y])


def main():
    df = robots()
    origin = pd.read_csv(DEST/'origins.csv.gz',usecols=['experiment','N','frame','id_i','id_j','outward','tangent'])
    key = ['experiment','N','frame','id']
    calibration = []
    for held in sorted(df.experiment.unique()):
        train = df[df.free & (df.experiment!=held) & (df.frame+H<.75*HZ*df.end)]
        w = 1/np.sqrt(train.groupby('experiment').frame.transform('count').to_numpy())
        pooled_x = design(train)*w[:,None]
        pooled_y = (train.out_x+1j*train.out_y).to_numpy()*w
        common = np.linalg.lstsq(pooled_x,pooled_y,rcond=None)[0]
        # Twelve free-motion observations of prior weight keep coefficients defined
        # for robots introduced only at high occupancy and rarely observed alone.
        prior_gram = 12*(pooled_x.conj().T@pooled_x)/np.sum(w*w)
        forecasts = []
        for rid,rows in df.groupby('id'):
            fit = train[train.id==rid]
            coef = common
            if len(fit):
                counts = fit.groupby('experiment').frame.transform('count').to_numpy()
                weights = np.sqrt(len(fit)/(fit.experiment.nunique()*counts))
                x = design(fit)*weights[:,None]
                y = (fit.out_x+1j*fit.out_y).to_numpy()*weights
                coef = np.linalg.solve(x.conj().T@x+prior_gram,x.conj().T@y+prior_gram@common)
            pred = design(rows)@coef
            forecasts.append(rows[key+['x','y','out_x','out_y']].assign(px=pred.real,py=pred.imag))
            calibration.append(dict(held=held,id=rid,free_training_origins=len(fit),sources=fit.experiment.nunique(),
                                    coefficients=json.dumps([[z.real,z.imag] for z in coef])))
        all_robots = pd.concat(forecasts)
        both = origin
        for who in ('i','j'):
            columns = {'id':f'id_{who}',**{c:f'{c}_{who}' for c in ('x','y','px','py','out_x','out_y')}}
            both = both.merge(all_robots.rename(columns=columns),on=[*key[:-1],f'id_{who}'],validate='many_to_one')
        assert len(both)==len(origin)
        q = both.x_j-both.x_i+1j*(both.y_j-both.y_i)
        n = q/abs(q)
        v = both.px_j-both.px_i+1j*(both.py_j-both.py_i)
        actual = both.out_x_j-both.out_x_i+1j*(both.out_y_j-both.out_y_i)
        np.testing.assert_allclose(actual*np.conj(n),both.outward+1j*both.tangent,atol=2e-7)
        both['free_normal'] = (v*np.conj(n)).to_numpy().real
        both['free_side'] = (v*np.conj(n)).to_numpy().imag
        both[['experiment','N','frame','id_i','id_j','free_normal','free_side']].to_csv(
            DEST/f'{held}_drive.csv.gz',index=False,float_format='%.9g')
    pd.DataFrame(calibration).to_csv(DEST/'free_motion_calibration.csv',index=False)
    print('Free-motion calibration complete',len(df),'robot origins',int(df.free.sum()),'clear origins',flush=True)


if __name__=='__main__':
    main()
