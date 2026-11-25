"""Optional final scoring of a full-record archive against aligned measured trajectories."""
import argparse
import json
from pathlib import Path
from settings import limit_threads

if __name__ == '__main__':
    limit_threads()

from noise_suite import OPTIONS, live
from compare import read_archive, system_front, system_objectives
from dyad_discrepancy import designs_by_width
from slabs import refit, wape
import numpy as np
from scipy.sparse import csr_matrix


class ArchiveScorer:
    def __init__(self, problem, archive):
        self.problem = problem
        self.plan = {k: tuple(v) for k, v in archive['plan'].items()}
        self.axes = live.R.axes_of(problem)
        if (problem.name != archive['system'] or list(problem.shape) != archive['input_shape']
                or list(problem.variables) != archive['variables']):
            raise ValueError('The problem name, fields or grid shape differ from the discovery archive.')
        self.labels = archive['labels']
        self.width = max(1, 24_000_000 // int(np.prod(problem.shape)))
        self.small_bank = self._bank(self.labels) if len(self.labels) <= self.width else None
        original = np.asarray(archive['systems'], dtype=int)
        ids, inverse = np.unique(original.ravel(), return_inverse=True)
        self.systems = inverse.reshape(original.shape)  # indices into the compact point arrays
        self.points = [archive['points'][i] for i in ids]
        self.train = np.array([p[3] for p in self.points])
        self.stability = np.array([p[4] for p in self.points])
        self.valid = (self.train < live.LOSS_NAN_VAL) & (self.stability < live.LOSS_NAN_VAL)
        self.targets = np.array([p[1] for p in self.points])
        indices, weights, pointers = [], [], [0]
        for p in self.points:
            for column, weight in p[2]:
                indices.append(column)
                weights.append(weight)
            pointers.append(len(indices))
        self.coefficients = csr_matrix((weights, indices, pointers), shape=(len(ids), len(self.labels)))
        self.constant = self.labels.index([])

    def _bank(self, labels):
        terms, opaque = live.R.symbols(self.problem, labels)
        return live.R.LocalBank(self.axes, terms, opaque, self.plan)

    def matrix(self, fields):
        if set(fields) != set(self.problem.variables):
            raise ValueError(f'Expected measured fields {self.problem.variables}.')
        for name, values in fields.items():
            if np.shape(values) != self.problem.shape or not np.isfinite(values).all():
                raise ValueError(f'Invalid shape or nonfinite values in field {name}.')
        matrix = np.empty((int(np.prod(self.plan['shape'])), self.coefficients.shape[1]))
        for start in range(0, len(self.labels), self.width):
            columns = slice(start, min(start+self.width, len(self.labels)))
            bank = self.small_bank or self._bank(self.labels[columns])
            matrix[:, columns] = bank.matrix(fields)
        np.testing.assert_allclose(matrix[:, self.constant], 1., atol=1e-12)
        matrix[:, self.constant] = 1.
        return matrix

    def recording_errors(self, recordings):
        rows = [wape(self.matrix(fields), self.coefficients, self.targets) for fields in recordings]
        if not rows:
            raise ValueError('Supply at least one validation recording.')
        return np.stack(rows)

    def trajectory_error(self, recordings):
        return self.recording_errors(recordings).mean(axis=0)

    def blocked_error(self, matrix, gap):
        if len(self.axes) != 1 or gap < 0:
            raise ValueError('The buffered comparison requires a time series and a nonnegative gap.')
        n = self.problem.shape[0]
        # cut also covers the largest residual pointwise FD stencil used by the archive.
        radius = self.plan['ms'][0] + self.plan['cut'][0]
        first = np.arange(len(matrix)) * self.plan['ss'][0]
        last = first + 2 * radius
        left = matrix[last < (n-gap)//2]
        right = matrix[first >= (n-gap)//2 + gap]
        width = max(len(p[2]) - 1 for p in self.points)
        if min(len(left), len(right)) <= width:
            return None
        groups, pivots, mapping = designs_by_width(self.points)
        errors = []
        for train, held in ((left, right), (right, left)):
            coeff, invalid = refit(groups, train, train.T @ train, len(pivots))
            residual = wape(held, coeff, pivots)
            residual[invalid] = np.inf
            errors.append(residual[mapping])
        return np.mean(errors, axis=0)

    def select(self, errors):
        pool, objectives = system_objectives(errors, self.stability, self.valid, self.systems)
        return dict(front=pool[system_front(objectives)].tolist())

    def select_multisample(self, errors):
        """Keep nondominated systems across recording-by-equation WAPE and training instability."""
        errors = np.asarray(errors)
        if errors.ndim != 2 or errors.shape[0] == 0 or errors.shape[1] != len(self.valid):
            raise ValueError('Expected a nonempty recording-by-equation-point score matrix.')
        mean_errors = errors.mean(axis=0)
        pool, _ = system_objectives(mean_errors, self.stability, self.valid, self.systems)
        systems = self.systems[pool]
        objectives = np.column_stack([row[systems] for row in errors] + [self.stability[systems]])
        return dict(front=pool[system_front(objectives)].tolist())


def load_recording(path, scorer):
    with np.load(path, allow_pickle=False) as data:
        required = set(scorer.problem.variables) | {f'axis{i}' for i in range(len(scorer.axes))}
        if not required.issubset(data.files):
            raise ValueError(f'{path}: missing {sorted(required.difference(data.files))}.')
        for i, axis in enumerate(scorer.axes):
            if not np.array_equal(axis, data[f'axis{i}']):
                raise ValueError(f'{path}: axis{i} differs from the training coordinates.')
        return {name: data[name] for name in scorer.problem.variables}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive_directory', type=Path)
    parser.add_argument('recordings', nargs='+', type=Path,
                        help='NPZ files with axis0, axis1, etc. and one array per measured variable.')
    args = parser.parse_args()
    archive = read_archive(args.archive_directory)
    dataset = archive['system']
    problem = live.load_data(dataset, **archive.get('loader_options', OPTIONS.get(dataset, {})))
    scorer = ArchiveScorer(problem, archive)
    by_record = scorer.recording_errors(load_recording(path, scorer) for path in args.recordings)
    errors = by_record.mean(axis=0)
    multisample = scorer.select_multisample(by_record)
    selection = dict(front=multisample['front'], multisample=multisample,
                     mean_error_front=scorer.select(errors)['front'], recordings=[str(path) for path in args.recordings])
    selection['members'] = [dict(system_index=i, points=archive['systems'][i],
                                 validation_wape=errors[scorer.systems[i]].tolist(),
                                 wape_by_recording=by_record[:, scorer.systems[i]].tolist(),
                                 training_instability=scorer.stability[scorer.systems[i]].tolist())
                            for i in multisample['front']]
    print(json.dumps(selection, allow_nan=False))
