"""Weak residual replay of archived coefficients on a supplied field."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'heldout_dominators_2026-11-23'
sys.path.insert(0, str(PRIOR))
from audit import folder, live, previous, read_archive, write_csv
from live import np
from scipy.sparse import csr_matrix


def read_rows(path):
    with path.open(encoding='utf8') as stream:
        return list(csv.DictReader(stream))


class FrozenScores:
    def __init__(self, problem, archive, point_ids, side, plan):
        raw_labels = archive['labels']
        points = [archive['points'][i] for i in point_ids]
        columns = sorted({j for point in points for j, _ in point[2]})
        position = {column: i for i, column in enumerate(columns)}
        terms, extra = live.R.symbols(problem, [raw_labels[j] for j in columns])
        self.slices, axes = side
        self.bank = live.R.LocalBank(axes, terms, {k: v[self.slices] for k, v in extra.items()}, plan)
        self.constant = next((position[j] for j in columns if not raw_labels[j]), None)
        self.pivots = np.array([position[p[1]] for p in points])
        rows, cols, values = [], [], []
        for row, p in enumerate(points):
            for j, w in p[2]:
                rows.append(row)
                cols.append(position[j])
                values.append(w)
        self.coef = csr_matrix((values, (rows, cols)), shape=(len(points), len(columns)))

    def __call__(self, field):
        matrix = self.bank.matrix({'u': field[self.slices]})
        if self.constant is not None:
            np.testing.assert_allclose(matrix[:, self.constant], 1., atol=1e-12)
            matrix[:, self.constant] = 1.
        scale = np.mean(np.abs(matrix), axis=0)[self.pivots]
        residual = np.mean(np.abs(self.coef @ matrix.T), axis=1)
        return np.divide(residual, scale, out=np.full_like(scale, np.inf), where=scale > 0)
