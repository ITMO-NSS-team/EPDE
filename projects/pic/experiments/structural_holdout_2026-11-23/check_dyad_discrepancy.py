"""Compare the batched scorer with independent per-fold direct least squares."""
from dyad_discrepancy import cross_validate
from experiment import full_directory, live, np
from assess_swap import read_archive


def check():
    rng = np.random.default_rng(297)
    shape = (8, 32)
    n = np.prod(shape)
    x, z = rng.normal(size=(2, n))
    drift = np.repeat(np.linspace(-.5, .5, shape[0]), shape[1])
    mat = np.column_stack((np.ones(n), x, z, 2*x-z, (2+drift)*x-z+.1*rng.normal(size=n)))
    targets = [3, 4, 4, 4]
    features = [(1, 2), (1, 2), (0, 1), (2, 1)]
    points = [('', y, [(y, -1.)]+[(j, 0.) for j in predictors], 0., 0., 0)
              for y, predictors in zip(targets, features)]
    halves, all_scales, geom = cross_validate(mat, shape, 7, points)
    direct_axes, direct_halves = [], []
    grid = np.arange(n).reshape(shape)
    for axis in range(len(shape)):
        levels = []
        for count in (2, 4, 8):
            slabs = [fold.ravel() for fold in np.split(grid, count, axis=axis)]
            np.testing.assert_array_equal(np.sort(np.concatenate(slabs)), np.arange(n))
            scores = []
            for k, held in enumerate(slabs):
                train = np.concatenate(slabs[:k]+slabs[k+1:])
                assert np.intersect1d(train, held).size == 0
                fold = []
                for y, predictors in zip(targets, features):
                    design = mat[:, predictors]
                    w = np.linalg.lstsq(design[train], mat[train, y], rcond=None)[0]
                    fold.append(abs(design[held]@w-mat[held, y]).sum()/abs(mat[held, y]).sum())
                scores.append(fold)
            levels.append(np.mean(scores, axis=0))
        direct_halves.append(levels[0])
        direct_axes.append(np.mean(levels, axis=0))
    np.testing.assert_allclose(halves, np.mean(direct_halves, axis=0), atol=1e-12)
    np.testing.assert_allclose(all_scales, np.mean(direct_axes, axis=0), atol=1e-12)
    assert [(g['axis'], g['slabs']) for g in geom] == [(a, c) for a in (0, 1) for c in (2, 4, 8)]
    assert abs(all_scales[0]) < 1e-12 and all_scales[1] > .05
    assert abs(halves[1]-all_scales[1]) > .001
    # Uneven native edges and a design with duplicated predictors in different columns.
    x = np.linspace(.1, 3, 139)
    singular = np.column_stack((x, x, np.zeros_like(x), 2*x))
    pts = [('', 3, [(3, -1.), (0, 1.), (1, 1.)]), ('', 2, [(2, -1.), (0, 0.)])]
    h, m, info = cross_validate(singular, (139,), 7, pts)
    assert np.isinf(h).all() and np.isinf(m).all()
    assert [s['min_rows'] for s in info] == [69, 34]
    print('Direct least squares agrees across two axes and three scales. Singular and zero-target checks passed.')


def check_archive():
    """Cross-check real weak moments using coordinate masks and direct least squares."""
    name, seed = 'ks', 923701
    archive = read_archive(full_directory(name, seed, 'candidate'))
    problem = live.load_data(name)
    plan = {k: tuple(v) for k, v in archive['plan'].items()}
    terms, opaque = live.R.symbols(problem, archive['labels'])
    matrix = live.R.LocalBank(live.R.axes_of(problem), terms, opaque, plan).matrix(problem.noisy(5, seed))
    matrix[:, 0] = 1.
    points = [archive['points'][i] for i in np.unique(archive['systems'])[:12]]
    _, result, geometry = cross_validate(matrix, plan['shape'], 7, points)
    coords = np.indices(plan['shape']).reshape(len(plan['shape']), -1)
    axes = []
    for axis, size in enumerate(plan['shape']):
        levels = []
        for g in (item for item in geometry if item['axis'] == axis):
            edges = np.rint(np.linspace(0, size, g['slabs']+1)).astype(int)
            folds = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                held = (lo <= coords[axis]) & (coords[axis] < hi)
                errors = []
                for _, target, pairs, *_ in points:
                    predictors = [j for j, _ in pairs if j != target]
                    x, y = matrix[:, predictors], matrix[:, target]
                    scale = np.linalg.norm(x[~held], axis=0)
                    scale[scale == 0.] = 1.
                    w, _, rank, _ = np.linalg.lstsq(x[~held]/scale, y[~held], rcond=None)
                    err = abs(x[held] @ (w/scale)-y[held]).sum()/abs(y[held]).sum()
                    errors.append(err if rank == len(predictors) else np.inf)
                folds.append(errors)
            levels.append(np.mean(folds, axis=0))
        axes.append(np.mean(levels, axis=0))
    np.testing.assert_allclose(result, np.mean(axes, axis=0), rtol=3e-7, atol=1e-10)
    print(f'Direct coordinate-mask predictions agree for {len(points)} archived {name} equations.')


if __name__ == '__main__':
    check()
    check_archive()
