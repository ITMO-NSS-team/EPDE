import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import numpy as np
from resolution import solve

ROOT = Path(__file__).resolve().parent
COHORTS = {
    'core': 'full_pipeline_uncertainty_2026-11-17',
    'k2': 'identifiability_design_2026-11-17',
    'k4': 'nuisance_direction_2026-11-17',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configurations(cohort, data, archive):
    names = ['local', 'broad'] if cohort == 'core' else ['local', 'broad', 'selected']
    stream = {'core': 'refit_pilot_', 'k2': '', 'k4': 'studentized_'}[cohort]
    scores = {n: archive[stream + n + '_scores'] for n in names}
    assert all(x.shape == (128,) for x in scores.values())

    def cutoff(banks):
        return float(np.sort(np.maximum.reduce([scores[b] for b in banks]))[122])

    primary = data['joint']
    if cohort == 'core':
        stored = primary['refit_pilot']['cutoff']
    else:
        if cohort == 'k4':
            primary = primary['studentized']
        stored = primary['original']['cutoff']
    assert np.isclose(cutoff(names[:2]), stored, rtol=1e-12, atol=0.)
    configs = [('original', names[:2], stored)]
    if cohort != 'core':
        replacement = ['local', 'selected']
        paired_cut = primary['replacement']['cutoff']
        assert np.isclose(cutoff(replacement), paired_cut, rtol=1e-12, atol=0.)
        common = cutoff(names)
        configs += [('replacement', replacement, paired_cut), ('all_three', names, common)]
        configs += [(b + '_at_common_cut', [b], common) for b in names]
    return configs


def run(job):
    cohort, path = job
    data = json.loads(path.read_text(encoding='utf-8'))
    array_path = path.with_suffix('.npz')
    array_hash = digest(array_path)
    assert array_hash == data['array_sha256']
    rows = []
    with np.load(array_path, allow_pickle=False) as archive:
        for method, banks, cutoff in configurations(cohort, data, archive):
            suffix = '_anchor_covariance' if cohort == 'core' else '_covariance'
            moments = [archive[b + '_observed'] for b in banks]
            covs = [archive[b + suffix] for b in banks]
            for candidate in data['candidates']:
                if candidate['label'] != 'reference':
                    continue
                features = candidate['xi']
                target = candidate['yi']
                assert len(set([*features, target])) == len(features) + 1
                for omitted in [None, *features]:
                    free = [i for i in features if i != omitted]
                    result = solve(moments, covs, [*free, target], cutoff)
                    rows.append(dict(method=method, banks=banks, cutoff=cutoff,
                                     candidate=candidate['id'], target=target, free=free,
                                     omitted=omitted,
                                     term=data['labels'][omitted] if omitted is not None else None,
                                     **result))
    return dict(cohort=cohort, file=path.name, system=data['system'], seed=data['seed'],
                profile=data['profile'], json_sha256=digest(path), array_sha256=array_hash, rows=rows)


def main():
    jobs = []
    for cohort, directory in COHORTS.items():
        pattern = '*.json' if cohort == 'core' else 'kdv_cossin_*.json'
        paths = sorted((ROOT.parent / directory / 'results').glob(pattern))
        assert len(paths) == (69 if cohort == 'core' else 9)
        jobs.extend((cohort, path) for path in paths)
    with ProcessPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(run, jobs))
    result = dict(protocol_sha256=digest(ROOT / 'protocol.txt'),
                  source_sha256={p: digest(ROOT / p) for p in ('study.py', 'resolution.py')},
                  records=records)
    (ROOT / 'certificates.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    print('Completed', len(records), 'fields and', sum(len(r['rows']) for r in records), 'tests.')


if __name__ == '__main__':
    main()
