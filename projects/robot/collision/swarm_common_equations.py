"""Fit single shared coefficient sets after evaluating their structures across recordings."""
import json

from swarm_term_fields import COMBINED, DEST, LIB, summarize
from swarm_term_search import fit
from physics import LAG, OUT, Track, arena_for, fit_self, robot_parameters
import numpy as np
import pandas as pd


def main():
    tracks = []
    for path in sorted((OUT/'preprocessed').glob('N*.npz')):
        with np.load(path) as z:
            tr = Track(path.stem,int(z['pwm']),z['ids'],z['raw_pos'],z['phi'])
            tracks.append(tr.part(0,3*len(tr.pos)//4))
    arenas = arena_for(tracks)
    lookup, _ = fit_self(tracks,arenas)
    rows = []
    for target,folder,library,history,compact in [
            ('density',DEST,LIB,1,('Drive','Diffusion','Contact','Wall','Memory','BlockedPair')),
            ('polarization',DEST/'combined',COMBINED,31,('IndividualTurn','Damping','SpinMemory','DriveTransport','SlowTurn'))]:
        names = library[target]
        merged = pd.read_csv(folder/'merged_front.csv').query('target == @target').sort_values('wape')
        scores = pd.read_csv(folder/'independent_subsets.csv.gz').query('target == @target')
        grams = []
        for index, tr in enumerate(tracks):
            origins = np.arange(history,len(tr.pos[::LAG])-1)
            result = summarize(tr,robot_parameters(tr,lookup),arenas[int(tr.tag[-2:])],
                               origins,index,False,combined=history>1)
            grams.append(result[target+'_gram'])
        gram = np.mean(grams,axis=0)
        for choice,terms in [('fixed physical model',compact),
                             ('lowest mean screen WAPE on merged front',merged.iloc[0].terms.split(' + '))]:
            idx = sorted(names.index(t) for t in terms)
            ordered = [names[i] for i in idx]
            key = ' + '.join(ordered)
            coef = fit(gram,idx)
            s = scores[scores.terms == key]
            assert len(s)==len(tracks)
            rows.append(dict(target=target,choice=choice,terms=key,pooled_prefix_coefficients=json.dumps(dict(zip(ordered,coef))),
                             mean_fold_refit_r2=s.r2.mean(),min_fold_refit_r2=s.r2.min(),max_fold_refit_r2=s.r2.max(),
                             mean_fold_refit_rmse=s.rmse.mean(),exact_front_folds=int(s.exact_front.sum())))
    pd.DataFrame(rows).to_csv(DEST/'common_equations.csv',index=False)
    metadata = dict(length_unit='robot diameter',time_unit='seconds',step_s=0.4,
                    recordings={tr.tag:len(tr.pos) for tr in tracks},arena_px={k:v.tolist() for k,v in arenas.items()},
                    lookup={f'{k[0]}/{k[1]}':v.tolist() for k,v in lookup.items()},
                    calibration_columns=['drive_body_x_px_per_s','drive_body_y_px_per_s',
                                         'signed_spin_rad_per_s','spin_increment_sd_rad'])
    (DEST/'common_calibration.json').write_text(json.dumps(metadata,indent=1),encoding='utf-8')


if __name__=='__main__':
    main()
