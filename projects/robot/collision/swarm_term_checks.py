"""Matched checks on contact radius and on the age of the angular-velocity observation."""
import json

from swarm_term_fields import DEST, SCREEN_ROWS, complex_xy, dot, groups
from swarm_term_search import fit
import numpy as np
import pandas as pd

from physics import CAP, CUTOFF, DT, LAG, OUT, Track, geometry, robot_parameters, wall_vectors
from swarm_fields import BOUNDARY, NGRID

MODELS = {
    'density': {'base': [0,1,2,3,4], 'blocking 1.03 diameters': [0,1,2,3,4,5],
                'blocking 1.10 diameters': [0,1,2,3,4,6], 'blocking original radius': [0,1,2,3,4,7]},
    'polarization': {'base': [0,1], 'recent 0-0.4 s': [0,1,2], 'delayed 0.4-0.8 s': [0,1,3],
                     'earlier 12 s mean': [0,1,4], 'earlier mean and recent': [0,1,2,4]},
}


def time_moments(x):
    a = x.reshape(len(x), -1, x.shape[-1])
    return (a.conj().transpose(0,2,1) @ a).real / a.shape[1]


def calculate(tr, params, arena, screen_seed=None):
    pos, phi = tr.pos[::LAG], tr.phi[::LAG]
    rates = params[:, 2]
    lab = groups(rates, 2)
    half = arena[2] / (np.sqrt(2) * CAP)
    g = np.linspace(-half, half, NGRID)[BOUNDARY:-BOUNDARY]
    grid = (g[:,None] + 1j*g[None,:]).ravel()
    state = (complex_xy(pos)-complex_xy(arena[:2])) / CAP
    p = np.exp(1j * (phi + np.angle(complex_xy(params[:, :2]))))
    grams = {k: [] for k in MODELS}
    means = {k: [] for k in MODELS}
    samples = []
    if screen_seed is not None:
        length = (len(pos)-32)*len(grid)*8
        draws = np.sort(np.random.default_rng(screen_seed).choice(length,SCREEN_ROWS,replace=False))
    for j in np.array_split(np.arange(1,len(pos)-1), int(np.ceil((len(pos)-2)/40))):
        d = grid-state[j,:,None]
        kernel = np.exp(-.5 * abs(d)**2)/8
        future_kernel = np.exp(-.5 * abs(grid-state[j+1,:,None])**2)/8
        def flux(velocity):
            return DT * (kernel * dot(d,velocity[:,:,None])).sum(1)

        dist, normal = geometry(pos[j])
        n = complex_xy(normal)
        drive = p[j] * abs(complex_xy(params[:,:2])) / CAP
        closing = np.maximum(-dot(drive[:,:,None]-drive[:,None,:], n), 0)
        rho_delta = (future_kernel-kernel).sum(1)
        columns = [flux(drive), DT*(kernel*(abs(d)**2-2)).sum(1),
                   flux((n*(dist<CUTOFF)).sum(2)), flux(complex_xy(wall_vectors(pos[j],arena))),
                   flux((state[j]-state[j-1])/DT)]
        for cutoff in (1.03*CAP, 1.10*CAP, CUTOFF):
            columns.append(flux(.5*(n*closing*(dist<cutoff)).sum(2)))
        grams['density'].append(time_moments(np.stack([*columns,rho_delta],axis=-1)))
        means['density'].append(rho_delta.mean(axis=1))

        use = j >= 31
        t = j[use]
        if not len(t):
            continue
        k, kn = kernel[use], future_kernel[use]
        advanced = p[t]*np.exp(1j*DT*rates)
        recent = phi[t] - phi[t-1] - DT*rates
        delayed = phi[t-1] - phi[t-2] - DT*rates
        slow = (phi[t-1]-phi[t-31])/30 - DT*rates
        group_cols = []
        for b in range(4):
            mask = lab == b
            def total(v, w=k):
                return (w[:,mask]*v[:,mask,None]).sum(axis=1)

            present, shifted = total(p[t]), total(advanced)
            group_cols.append(np.stack([shifted-present, -shifted, total(1j*recent*advanced),
                total(1j*delayed*advanced), total(1j*slow*advanced), total(p[t+1],kn)-present],axis=-1))
        polar = np.stack(group_cols,axis=2)
        grams['polarization'].append(time_moments(polar))
        means['polarization'].append(polar[..., -1].mean(axis=1))
        if screen_seed is not None:
            offset = (t[0]-31)*len(grid)*8
            flat = polar.reshape(-1,polar.shape[-1])
            idx = draws[(draws>=offset)&(draws<offset+2*len(flat))]-offset
            selected = flat[idx//2]
            samples.append(np.where((idx%2)[:,None],selected.imag,selected.real))
    result = {k:(np.concatenate(grams[k]),np.concatenate(means[k])) for k in MODELS}
    if samples:
        result['screen'] = np.concatenate(samples)
        assert len(result['screen'])==SCREEN_ROWS
    return result


def main():
    directory = DEST/'spin_history'
    directory.mkdir(exist_ok=True)
    tracks = {}
    for path in sorted((OUT/'preprocessed').glob('N*.npz')):
        with np.load(path) as z:
            tracks[path.stem] = Track(path.stem,int(z['pwm']),z['ids'],z['raw_pos'],z['phi'])
    rows, errors = [], {}
    for held in tracks:
        spec = json.loads((OUT/held/'training.json').read_text())
        lookup = {tuple(map(int,k.split('/'))): np.array(v) for k,v in spec['lookup'].items()}
        datasets = {}
        history = {}
        for j, (tag, tr) in enumerate(tracks.items()):
            part = tr if tag==held else tr.part(0,spec['split_frames'][tag])
            params, arena = robot_parameters(tr,lookup), np.array(spec['arenas'][tag[-2:]])
            result = calculate(part, params, arena, None if tag==held else 4301+j)
            datasets[tag] = result
            roles = [('test' if tag==held else 'train',result)]
            if tag!=held:
                cut = spec['split_frames'][tag]
                tail = tr.part(((cut+LAG-1)//LAG)*LAG,None)
                roles.append(('valid',calculate(tail,params,arena)))
                history.setdefault('train_polarization_screen', []).append(result['screen'])
            for role, content in roles:
                grams, ymean = content['polarization']
                gram = grams.mean(axis=0)
                variance = gram[-1,-1]-np.mean(abs(ymean.mean(axis=0))**2)
                history.setdefault(role+'_polarization_gram',[]).append(gram)
                history.setdefault(role+'_polarization_variance',[]).append(variance)
        np.savez_compressed(directory/f'{held}_moments.npz',**{k:np.stack(v) for k,v in history.items()})
        for target, models in MODELS.items():
            train = np.mean([datasets[t][target][0].mean(axis=0) for t in spec['train']],axis=0)
            test, ymeans = datasets[held][target]
            var = test[:,-1,-1].mean() - np.mean(abs(ymeans.mean(axis=0))**2)
            for name, idx in models.items():
                coef = fit(train,idx)
                ids = [*idx,train.shape[-1]-1]
                weights = np.r_[coef,-1.]
                loss = weights @ test[:,ids][:,:,ids] @ weights
                errors[target,name,held] = loss
                rows.append(dict(held=held,target=target,model=name,origins=len(loss),rmse=np.sqrt(loss.mean()),
                                 r2=1-loss.mean()/var,coefficients=json.dumps(coef.tolist())))
    pd.DataFrame(rows).to_csv(DEST/'sensitivity.csv',index=False)
    intervals = []
    for target, models in MODELS.items():
        for name in models:
            if name=='base':
                continue
            rng = np.random.default_rng(75391)
            draws, improvements = [], []
            for held in tracks:
                baseline, candidate = errors[target,'base',held], errors[target,name,held]
                starts = np.arange(0,len(baseline),25)
                counts = np.diff(np.r_[starts,len(baseline)])
                pick = rng.integers(0,len(starts),size=(2000,len(starts)))
                def bootstrap(loss):
                    return np.sqrt(np.add.reduceat(loss,starts)[pick].sum(1)/counts[pick].sum(1))

                draws.append(bootstrap(baseline)-bootstrap(candidate))
                improvements.append(np.sqrt(baseline.mean())-np.sqrt(candidate.mean()))
            lo,hi = np.quantile(np.mean(draws,axis=0),[.025,.975])
            intervals.append(dict(target=target,model=name,rmse_reduction=np.mean(improvements),
                                  lower=lo,upper=hi,improved_recordings=sum(np.array(improvements)>0)))
    pd.DataFrame(intervals).to_csv(DEST/'paired_intervals.csv',index=False)
    print(pd.DataFrame(intervals).round(6).to_string(index=False))


if __name__=='__main__':
    main()
