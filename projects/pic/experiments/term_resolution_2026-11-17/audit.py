"""Exact rational replay on the archived binary64 observations and cutoffs."""
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import json
import numpy as np
from study import ROOT, COHORTS, digest


def exact_column(column):
    parts = [float(x).as_integer_ratio() for x in column]
    scale = max(d for _, d in parts)
    return [n * (scale // d) for n, d in parts], scale


class Bank:
    def __init__(self, observed, covariance):
        self.observed = observed
        self.covariance = covariance
        self.columns = {}
        self.entries = {}

    def entry(self, i, j):
        if i > j:
            return self.entry(j, i)
        if (i, j) not in self.entries:
            for k in (i, j):
                if k not in self.columns:
                    self.columns[k] = exact_column(self.observed[:, k])
            a, x = self.columns[i]
            b, y = self.columns[j]
            gram = Fraction(sum(u*v for u, v in zip(a, b)), x*y)
            cov = (Fraction(float(self.covariance[i, j])) + Fraction(float(self.covariance[j, i]))) / 2
            self.entries[i, j] = gram, cov
        return self.entries[i, j]


def pd(matrix):
    # Positive pivots in rational symmetric elimination are equivalent to PD.
    while matrix:
        pivot = matrix[0][0]
        if pivot <= 0:
            return False
        matrix = [[matrix[i][j] - matrix[i][0]*matrix[0][j]/pivot
                   for j in range(1, len(matrix))] for i in range(1, len(matrix))]
    return True


def quadratic(matrix, vector):
    return sum(a*b*q for a, row in zip(vector, matrix) for b, q in zip(vector, row))


def main():
    result = json.loads((ROOT/'certificates.json').read_text())
    assert result['protocol_sha256'] == digest(ROOT/'protocol.txt')
    for name, expected in result['source_sha256'].items():
        assert digest(ROOT/name) == expected
    checked = Counter()
    for record in result['records']:
        path = ROOT.parent/COHORTS[record['cohort']]/'results'/record['file']
        assert digest(path) == record['json_sha256']
        assert digest(path.with_suffix('.npz')) == record['array_sha256']
        with np.load(path.with_suffix('.npz')) as archive:
            suffix = '_anchor_covariance' if record['cohort'] == 'core' else '_covariance'
            banks = {b: Bank(archive[b+'_observed'], archive[b+suffix])
                     for b in {b for r in record['rows'] for b in r['banks']}}
        for row in record['rows']:
            indices = [*row['free'], row['target']]
            cutoff = Fraction(row['cutoff'])
            matrices = []
            for name in row['banks']:
                entries = [[banks[name].entry(i, j) for j in indices] for i in indices]
                matrices.append([[g - cutoff*c for g,c in line] for line in entries])
            if row['status'] == 'compatible':
                vector = [-Fraction(x) for x in row['coefficients']] + [Fraction(1)]
                assert all(quadratic(m, vector) < 0 for m in matrices), (record['file'], row)
            elif row['status'] == 'excluded':
                weights = [Fraction(w) for w in row['weights']]
                assert min(weights) >= 0 and max(weights) > 0
                combo = [[sum(w*m[i][j] for w,m in zip(weights, matrices))
                          for j in range(len(indices))] for i in range(len(indices))]
                assert pd(combo), (record['file'], row)
            checked[row['status']] += 1
    out = dict(status='PASS', exact_binary64_certificates=dict(checked),
               certificates_sha256=digest(ROOT/'certificates.json'), audit_sha256=digest(ROOT/'audit.py'))
    (ROOT/'audit.json').write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == '__main__':
    main()
