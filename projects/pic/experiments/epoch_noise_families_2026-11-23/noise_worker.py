"""Discover and score one family, with a durable verification receipt for resumption."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
from importlib import import_module
import json
import sys
from unittest.mock import patch

from filelock import FileLock
import psutil
from cpu_budget import await_capacity, restrict

if __name__ == '__main__':
    restrict(psutil.Process())

from noise_suite import (BASELINE, FAMILIES, LEVEL, OPTIONS, SEEDS, atomic_text, directory, epochs,
                         live, measurement, read_json, trial, whole_record)

sys.path.insert(0, str(BASELINE.parent))
assess = import_module(f'{BASELINE.name}.score').assess
score_lorenz = import_module(f'{BASELINE.name}.report').score_lorenz


def checksums(dest, name):
    files = ['archive.json.gz', 'fronts.json']
    if name == 'lorenz':
        files.append('lorenz_identity.json')
    result = {}
    for filename in files:
        with (dest / filename).open('rb') as stream:
            result[filename] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def verified(job):
    name, seed, family = job
    dest = directory(*job)
    receipt = dest / 'verified.json'
    if not receipt.exists():
        return False
    record = read_json(receipt)
    if record['job'] != list(job) or record['sha256'] != checksums(dest, name):
        raise ValueError(f'Verification receipt mismatch at {dest}')
    return True


def run(job):
    name, seed, family = job
    dest = directory(*job)
    dest.mkdir(parents=True, exist_ok=True)
    with FileLock(dest / 'worker.lock'):
        if verified(job):
            return
        observation = measurement(family)
        if not (dest / 'archive.json.gz').exists():
            await_capacity()
            with (dest / 'search.log').open('w', encoding='utf8', buffering=1) as log:
                with redirect_stdout(log), redirect_stderr(log), patch.object(live, 'split', whole_record), \
                        patch.object(trial, 'folder', return_value=dest):
                    trial.worker(name, seed, 'candidate', noise_level=LEVEL, epoch_budget=epochs(name),
                                 loader_options=OPTIONS.get(name), observation=observation)
        assess((name, seed, LEVEL), dest=dest, observation=observation)
        rows = read_json(dest / 'fronts.json')
        assert {r['method'] for r in rows} == {'native', 'ordinary', 'halves', 'multiscale'} and len(rows) == 4
        if name == 'lorenz':
            sys.path.insert(0, str(live.operators.HISTORICAL))
            from identity_hits import Classifier
            classifier = Classifier('lorenz', use_cache=False, progress=False)
            score_lorenz(seed, LEVEL, classifier, rows, dest=dest, observation=observation)
        atomic_text(dest / 'verified.json', json.dumps(dict(job=job, sha256=checksums(dest, name))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=live.DATASETS)
    parser.add_argument('seed', type=int, choices=SEEDS)
    parser.add_argument('family', choices=FAMILIES)
    args = parser.parse_args()
    run((args.name, args.seed, args.family))
