"""Apply the November 7 Lorenz identity credit to the current full-record CV fronts."""
import argparse
import json
from pathlib import Path
import sys
import time

from compare import previous, read_archive, structural_matches, system_front, system_objectives, write_csv
from dyad_discrepancy import native_row_scores
from experiment import ROOT, live, np, trial
from fronts_by_noise import archive_dir, assessment_path


def historical_label(term):
    names = []
    for label, entries in term:
        params = dict(entries)
        if params['power'] != 1:
            label += '^%g' % params['power']
        if 'freq' in params:
            freq = params['freq']
            label += '~1' if abs(freq-1) < .01 else '~%.4g' % freq
        if 'dim' in params:
            label += '@%d' % int(round(params['dim']))
        names.append(label)
    return '*'.join(sorted(names))


def extract(job):
    name, seed, _, noise = job
    archive = read_archive(archive_dir(job))
    combos = np.asarray(archive['systems'])[np.asarray(archive['operator']['system_cost']) <= live.BUDGET]
    problem = live.load_data(name)
    raw = problem.noisy(noise, seed)
    (_, scores), _ = native_row_scores(problem, raw, archive, combos)
    points = archive['points']
    dyad = np.array([p[4] for p in points])
    valid = (np.array([p[3] for p in points]) < previous.LOSS_NAN_VAL) & (dyad < previous.LOSS_NAN_VAL)
    pool, values = system_objectives(scores, dyad, valid, combos)
    front = pool[system_front(values)]
    _, _, hit = structural_matches(name, problem, archive, combos)
    recorded = next(r for r in json.loads(assessment_path(job).read_text()) if r['rule'] == 'multiscale')
    assert (len(front), hit(front)) == (recorded['front_size'], recorded['front_hit'])
    labels = [historical_label(term) for term in archive['labels']]
    def key(i):
        _, target, pairs, *_ = points[i]
        terms = [j for j, w in pairs if j != target and w != 0.]
        return labels[target], tuple(sorted(labels[j] for j in terms if j != 0)), 0 in terms
    return recorded, [tuple(key(i) for i in combo) for combo in combos[front]]


def score_table(auto):
    sys.path.insert(0, str(auto))
    import identity_hits as historical
    classifier = historical.Classifier('lorenz', use_cache=False, progress=False)
    jobs = [('lorenz', s, 'full', q) for q in (5,0) for s in trial.CONFIRMATION]
    rows = []
    for job in jobs:
        while not assessment_path(job).is_file():
            time.sleep(5)
        recorded, systems = extract(job)
        positives, unknown = 0, 0
        for system in systems:
            verdicts = [classifier.classify(eq) for eq in system]
            positives += classifier.full(system)['full']
            unknown += any(v['kind']=='unparsed' for v in verdicts) and all(
                v['kind'] in ('law','identity','unparsed') for v in verdicts)
        rows.append(dict(noise_percent=job[3], seed=job[1], front_size=len(systems),
                         catalog_hit=recorded['front_hit'], history_identity_hit=int(positives>0),
                         credited_systems=positives, unresolved_systems=unknown))
        print(job[1], job[3], positives, unknown, flush=True)
    write_csv(ROOT/'historical_lorenz.csv', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('historical_auto', type=Path, help='Original universal-search auto directory')
    score_table(parser.parse_args().historical_auto)
