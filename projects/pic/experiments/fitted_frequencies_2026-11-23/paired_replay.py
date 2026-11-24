"""Refit frequencies on the paired full records and on archived oscillator competitors."""
from contextlib import redirect_stdout
from ast import literal_eval
import json
import os
from unittest.mock import patch

from check_support import check, fit_terms
from frequency_study import ROOT, directory, live, np, read_archive
from dyad_discrepancy import cross_validate
from epoch_settings import SEEDS
from oscillator_dominance import compare
from epde.operators.common import frequency_fit
from epde.operators.common.frequency_fit import fitted_factors


def replay(problem, raw, plan, labels, point):
    """Preserve the archived target, intercept, products and observations."""
    target = point[1]
    features = [labels[j] for j, w in point[2] if j != target and w != 0.]
    intercept = any(not term for term in features)
    equation, rows = fit_terms(problem, raw, plan, labels[target], features)
    assert equation is not None
    columns = [rows.columns(term, 'u')[0] for term in equation.structure]
    if intercept:
        columns.append(np.ones(len(columns[0])))
    matrix = np.column_stack(columns)
    target_index = equation.target_idx
    y = matrix[:,target_index]
    x = np.delete(matrix, target_index, axis=1)
    weights = np.linalg.lstsq(x,y,rcond=None)[0]
    residual = y-x@weights
    train = np.mean(abs(residual))/np.mean(abs(y))
    # One descriptor gives the folds the original support. The frequencies stay fixed.
    desc = ('u',target_index,[(j,1.) for j in range(len(columns))],0.,0.,1)
    _, cv, _ = cross_validate(matrix, plan['shape'], 7, [desc])
    tones = {f"{f.label}@{int(f.param('dim'))}":float(f.param('freq'))
             for term in equation.structure for f in fitted_factors(term)}
    return dict(fitted=tones, train_wape=float(train), multiscale_wape=float(cv[0]),
                residual_sum_squares=float(residual @ residual))


def oscillator_starts(seed, noise):
    problem = live.load_data('ode')
    raw = problem.noisy(noise,seed)
    dest = directory('ode',seed,noise)
    archive = read_archive(dest)
    assert archive['train_hashes'] == {k:live.R.digest(v) for k,v in raw.items()}
    target, features, _ = live.R.parse(problem.truth[0])
    labels = [target,*features]
    point = ('u',0,[(i,1.) for i in range(len(labels))])
    baseline = replay(problem,raw,archive['plan'],labels,point)
    message = 'Frequencies of trigonometric found in the data:'
    peaks, = [literal_eval(line.partition(message)[2].strip()) for line in
              (dest/'search.log').read_text().splitlines() if message in line]
    fits = []
    # Repeat the same one-frequency fit with each data-derived peak supplied as its spectral start.
    for peak in peaks:
        with patch.object(frequency_fit,'residual_modes',return_value=[peak]):
            fits.append(replay(problem,raw,archive['plan'],labels,point))
    return dict(seed=seed,noise=noise,data_peaks=peaks,default=baseline,
                best=min([baseline,*fits], key=lambda r:r['residual_sum_squares']))


def competitors(seed):
    problem = live.load_data('ode')
    raw = problem.noisy(0,seed)
    # compare verifies the old rival dominates all accepted candidates in all three archives.
    archive, rival, _ = compare(seed)
    assert archive['train_hashes'] == {k: live.R.digest(v) for k,v in raw.items()}
    labels = archive['labels']
    remove = {i for i,term in enumerate(labels) if not term or any(n=='cos' for n,_ in term)}
    physical = list(rival)
    physical[2] = [[j,w] for j,w in rival[2] if j not in remove]
    assert len(physical[2])==4 and len(rival[2])==6
    return {name:replay(problem,raw,archive['plan'],labels,point)
            for name,point in [('physical',physical),('expanded',rival)]}


def duffing_handover():
    name, seed, noise = 'duffing', 923703, 5
    problem = live.load_data(name)
    raw = problem.noisy(noise,seed)
    a = read_archive(directory(name,seed,noise))
    assert a['train_hashes'] == {k:live.R.digest(v) for k,v in raw.items()}
    labels = a['labels']
    def signature(term):
        return tuple(sorted((n,tuple(sorted((k,v) for k,v in pars if k!='freq'))) for n,pars in term))
    target, features, _ = live.R.parse(problem.truth[0])
    form = sorted(signature(t) for t in [target,*features])
    found = []
    for point_id, in a['systems']:
        point = a['points'][point_id]
        if sorted(signature(labels[j]) for j,w in point[2]) != form:
            continue
        before = [v for j,w in point[2] for n,pars in labels[j] for k,v in pars if k=='freq']
        result = replay(problem,raw,a['plan'],labels,point)
        found.append(dict(seed=seed,noise=noise,point_id=point_id,first_call=point[5],
                          target=labels[point[1]],old_frequency=before[0],old_wape=point[3],**result))
    assert found
    return found


if __name__ == '__main__':
    with open(os.devnull,'w') as quiet, redirect_stdout(quiet):
        fits = [check(n,s,q) for n in ('duffing','ode','kdv_cossin') for s in SEEDS for q in (0,5)]
        oscillator = {str(s):competitors(s) for s in SEEDS}
        duffing = duffing_handover()
        starts = [oscillator_starts(s,q) for s in SEEDS for q in (0,5)]
    results = dict(conditional=fits, clean_oscillator=oscillator, discovered_duffing=duffing,
                   oscillator_spectral_starts=starts)
    path = ROOT/'paired_replay.json'
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(path)
    print(path)
