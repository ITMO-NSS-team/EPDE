"""Cross-validate weak discrepancy on the exact slabs used by coefficient stability."""
from collections import defaultdict

from experiment import live, np
from epde.operators.common.survival import _axis_layouts, _block_rows, _dyadic_slab_edges
from slabs import refit, wape


def designs_by_width(points):
    unique = {}
    mapping = []
    for _, target, pairs, *_ in points:
        predictors = tuple(sorted(j for j, _ in pairs if j != target))
        key = (target, predictors)
        mapping.append(unique.setdefault(key, len(unique)))
    targets = np.array([target for target, _ in unique])
    widths = defaultdict(list)
    for (target, predictors), i in unique.items():
        widths[len(predictors)].append((i, target, predictors))
    groups = {w: tuple(np.asarray(a, dtype=int) for a in zip(*items)) for w, items in widths.items()}
    return groups, targets, np.asarray(mapping)


def cross_validate(matrix, shape, reference_width, points):
    groups, targets, mapping = designs_by_width(points)
    assert max(groups) <= reference_width
    axis_means, halves, geometry = [], [], []
    for axis, order, moved_shape in _axis_layouts(len(matrix), shape):
        ordered = matrix if order is None else matrix[order]
        levels = []
        edges = _dyadic_slab_edges(len(matrix), moved_shape, _block_rows(reference_width))
        for boundaries in edges:
            blocks = [ordered[i:j] for i, j in zip(boundaries, boundaries[1:])]
            grams = [b.T @ b for b in blocks]
            fold_scores = []
            for k, held in enumerate(blocks):
                training = np.concatenate(blocks[:k] + blocks[k+1:])
                gram = np.add.reduce(grams[:k] + grams[k+1:])
                weights, invalid = refit(groups, training, gram, len(targets))
                scores = wape(held, weights, targets)
                scores[invalid] = np.inf
                fold_scores.append(scores)
            levels.append(np.mean(fold_scores, axis=0))
            geometry.append(dict(axis=axis, slabs=len(blocks), min_rows=min(map(len, blocks)),
                                 refits=len(blocks)*len(targets)))
        halves.append(levels[0])
        axis_means.append(np.mean(levels, axis=0))
    return np.mean(halves, axis=0)[mapping], np.mean(axis_means, axis=0)[mapping], geometry


def native_row_scores(problem, raw, archive, combos):
    plan = {k: tuple(v) for k, v in archive['plan'].items()}
    labels = archive['labels']
    axes = live.R.axes_of(problem)
    # The weak integrator caches intermediate products at full field resolution.
    # Assemble wide libraries in column blocks without splitting observations or windows.
    width = max(1, min(len(labels), 24_000_000 // int(np.prod(problem.shape))))
    matrix = np.empty((int(np.prod(plan['shape'])), len(labels)))
    for start in range(0, len(labels), width):
        block = slice(start, start+width)
        terms, opaque = live.R.symbols(problem, labels[block])
        matrix[:,block] = live.R.LocalBank(axes, terms, opaque, plan).matrix(raw)
    assert len(matrix) == np.prod(plan['shape']) and not archive['labels'][0]
    np.testing.assert_allclose(matrix[:, 0], 1., atol=1e-12)
    matrix[:, 0] = 1.
    ids = np.unique(combos)
    settings = archive['settings']
    assert settings['system_rps_mode'] == 'plain' and settings['instability_metric'] == 'dyad'
    halves, multiscale, geometry = cross_validate(matrix, plan['shape'], settings['equation_terms_max_number'],
                                                   [archive['points'][i] for i in ids])
    scores = np.full((2, len(archive['points'])), np.inf)
    scores[0, ids], scores[1, ids] = halves, multiscale
    return scores, geometry
