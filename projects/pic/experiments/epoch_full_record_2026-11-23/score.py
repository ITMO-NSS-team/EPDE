"""Score whole Pareto fronts at the final epoch on each verified full-record run."""
import argparse
import json
import re
from statistics import median
from time import sleep

from epoch_settings import ROOT, OPTIONS, SEEDS, NOISE, directory, epochs, live, np
from compare import read_archive, structural_matches, system_front, system_objectives, write_csv
from dyad_discrepancy import native_row_scores
from recovery import catalog_diagnostics, lorenz_identity


def assess(job, *, dest=None, observation=None):
    name, seed, noise = job
    dest = directory(*job) if dest is None else dest
    archive = read_archive(dest)
    assert (archive['system'], archive['seed'], archive['noise_percent']) == job
    assert archive['epochs_completed'] == archive['settings']['training_epochs'] == epochs(name)
    assert archive['budget_unit'] == 'epochs' and archive['settings']['early_stopping_patience'] is None
    problem = live.load_data(name, **OPTIONS.get(name, {}))
    if observation is None:
        raw = problem.noisy(noise, seed)
    else:
        raw, measurement = observation(problem, noise, seed)
        assert archive['measurement'] == measurement
    assert archive['input_shape'] == list(problem.shape)
    assert archive['train_hashes'] == {k:live.R.digest(v) for k,v in raw.items()}
    combos = np.asarray(archive['systems'], dtype=int)
    returned = np.asarray(archive['native_front'], dtype=int)
    points = archive['points']
    train = np.array([p[3] for p in points])
    dyad = np.array([p[4] for p in points])
    valid = (train < live.LOSS_NAN_VAL) & (dyad < live.LOSS_NAN_VAL)
    _, native_literal, native_hit = structural_matches(name, problem, archive, returned)
    _, literal, hit = structural_matches(name, problem, archive, combos)
    generated = hit(range(len(combos)))
    native = dict(hit=native_hit(range(len(returned))), size=len(returned),
                  literal=int(native_literal.any()))
    assert not native['hit'] or generated
    base = dict(dataset=name, seed=seed, noise_percent=noise, epochs=epochs(name),
                input_shape='x'.join(map(str, problem.shape)), fitness_calls=archive['calls'],
                profile_fits=archive['operator']['extra_fits'], generated=generated,
                candidates=len(combos), search_seconds=archive['seconds'])
    log = (dest/'search.log').read_text(errors='replace')
    population_ids = re.findall(r'^Assigned weight \d+ for (\d+)\s*$', log, re.M)
    assert sorted(map(int,population_ids)) == list(range(len(population_ids))) and population_ids
    base['population_size'] = len(population_ids)
    if observation is not None:
        base['noise_family'] = measurement['family']
        base['field_rms'] = measurement['field_rms']
    def row(method, front, fits=0):
        return dict(**base, method=method, front_hit=front['hit'], front_size=front['size'],
                    literal_front_hit=front['literal'], validation_refits=fits)
    rows = [row('native', native)]
    identity = lorenz_identity(archive) if name == 'lorenz' else None
    catalog = dict(generated=generated)
    def credit(record, front):
        catalog[record['method']] = record['front_hit']
        if identity is not None and not record['front_hit']:
            record['front_hit'] = int(identity(front))
    credit(rows[0], returned)
    (half, multi), geometry = native_row_scores(problem, raw, archive, combos)
    for method, score in [('ordinary',train), ('halves',half), ('multiscale',multi)]:
        pool, objectives = system_objectives(score, dyad, valid, combos)
        selected = pool[system_front(objectives)]
        refits = sum(g['refits'] for g in geometry if method=='multiscale'
                     or (method=='halves' and g['slabs']==2))
        rows.append(row(method, dict(hit=hit(selected), size=len(selected),
                                     literal=int(literal[selected].any())), refits))
        credit(rows[-1], combos[selected])
    if identity is not None and not generated:
        generated = int(any(r['front_hit'] for r in rows) or identity(combos))
        for r in rows:
            r['generated'] = generated
    temp = dest/'fronts.pending'
    temp.write_text(json.dumps(rows, allow_nan=False))
    temp.replace(dest/'fronts.json')
    if identity is not None:
        temp = dest/'catalog_matches.pending'
        temp.write_text(json.dumps(catalog))
        temp.replace(dest/'catalog_matches.json')


def report(jobs):
    data = [r for j in jobs for r in catalog_diagnostics(directory(*j), json.loads((directory(*j)/'fronts.json').read_text()))]
    names = list(live.DATASETS)
    assert len(data)==len(names)*len(SEEDS)*len(NOISE)*4
    write_csv(ROOT/'runs.csv', data)
    groups = []
    for name in ['suite', *names]:
        subset = [r for r in data if r['dataset']==name or (name=='suite' and r['dataset']!='ode_simple')]
        for method in ('native','ordinary','halves','multiscale'):
            record = dict(dataset=name, method=method)
            for q, label in ((0,'clean'),(5,'noisy')):
                select = [r for r in subset if r['noise_percent']==q and r['method']==method]
                record.update({label+'_runs':len(select), label+'_generated':sum(r['generated'] for r in select),
                               label+'_front':sum(r['front_hit'] for r in select),
                               label+'_literal_front':sum(r['literal_front_hit'] for r in select),
                               label+'_median_front_size':median(r['front_size'] for r in select)})
            groups.append(record)
    write_csv(ROOT/'by_dataset.csv',groups)
    for r in groups[:4]:
        print(r)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    jobs = [(n,s,q) for s in SEEDS for q in NOISE for n in live.DATASETS]
    while True:
        pending = []
        for j in jobs:
            dest = directory(*j)
            result_path = dest/'fronts.json'
            if result_path.is_file():
                first = json.loads(result_path.read_text())[0]
                if 'population_size' in first and (j[0] != 'lorenz' or (dest/'catalog_matches.json').is_file()):
                    continue
            if (dest/'archive.json.gz').is_file():
                assess(j)
                print('Scored',*j,flush=True)
            else:
                pending.append(j)
        if not pending:
            report(jobs)
            break
        if not args.watch:
            break
        sleep(10)
