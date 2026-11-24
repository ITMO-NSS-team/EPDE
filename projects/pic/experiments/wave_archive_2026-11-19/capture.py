"""Capture actual scored coefficients and terminal membership without influencing EPDE."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'live_preserved_fit_2026-11-18'
SEEDS = tuple(range(919101, 919107))
sys.path.insert(0, str(PRIOR))


class Archive:
    def __init__(self, output_dir=None):
        self.output_dir = ROOT/'results' if output_dir is None else output_dir
        self.term_ids = {(): 0}
        self.labels = ['1']
        self.rows = {}
        self.calls = 0

    @lru_cache(maxsize=None)
    def term_id(self, name):
        from recheck import labels_of
        label, = labels_of('0 = ' + name)
        if label not in self.term_ids:
            self.term_ids[label] = len(self.labels)
            self.labels.append(name)
        return self.term_ids[label]

    def point(self, eq):
        pivot = eq.target_idx
        target = self.term_id(eq.structure[pivot].name)
        terms = [(target, -1.)]
        for j, term in enumerate(eq.structure):
            if j != pivot:
                weight = float(eq.weights_final[eq.weight_index(j,pivot)])
                if weight != 0:
                    terms.append((self.term_id(term.name), weight))
        if eq.weights_final[-1] != 0:
            terms.append((0, float(eq.weights_final[-1])))
        terms.sort()
        return (target, tuple(i for i,w in terms), tuple(w for i,w in terms),
                float(eq.fitness_value), float(eq.coefficients_stability))

    def evaluated(self, eq):
        self.calls += 1
        key = self.point(eq)
        if key not in self.rows:
            self.rows[key] = dict(first_call=self.calls, calls=0, terminal=False, returned=False)
        self.rows[key]['calls'] += 1

    def finish(self, search, record):
        assert self.calls == record['calls']
        levels = search.equations(only_print=False, num=sys.maxsize)
        for level, solutions in enumerate(levels):
            for solution in solutions:
                key = self.point(solution.vals['u'])
                self.rows[key]['terminal'] = True
                if level == 0:
                    self.rows[key]['returned'] = True
        rows = [dict(pivot=p, ids=ids, weights=w, wape=a, dyad=b, **details)
                for (p, ids, w, a, b), details in self.rows.items()]
        archive = dict(seed=record['seed'], labels=self.labels, rows=rows, calls=self.calls,
                       terminal_levels=[len(s) for s in levels])
        path = self.output_dir / f"archive_{record['seed']}.json.gz"
        tmp = path.with_suffix('.pending')
        with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=4) as f:
            json.dump(archive, f, allow_nan=False, separators=(',',':'))
        tmp.replace(path)


def launch(seed):
    folder = ROOT/'results'
    folder.mkdir(exist_ok=True)
    if (folder/f'wave_{seed}_wape.json').is_file() and (folder/f'archive_{seed}.json.gz').is_file():
        return 0
    with (folder/f'search_{seed}.log').open('w') as log:
        proc = subprocess.run([sys.executable, '-u', __file__, str(seed)], stdout=log, stderr=subprocess.STDOUT,
                              env=dict(os.environ, PYTHONHASHSEED='0'))
    print(seed, proc.returncode, flush=True)
    return proc.returncode


if __name__=='__main__':
    if len(sys.argv)==2:
        from search import run_job
        assert os.environ['PYTHONHASHSEED']=='0'
        run_job('wave',int(sys.argv[1]),'wape',observer=Archive(),output_dir=ROOT/'results')
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            raise SystemExit(int(any(list(pool.map(launch, SEEDS)))))
