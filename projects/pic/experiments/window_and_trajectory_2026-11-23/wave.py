"""Training-only time window selection, followed by equal-size archive replay."""
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import rankdata

from frozen import ROOT, PRIOR, FrozenScores, folder, live, np, previous, read_archive, read_rows, write_csv


def choose_window(training, axes, plan):
    wf = live.R.WF
    steps = [a[1] - a[0] for a in axes]
    delta = np.diff(training, n=2, axis=0)
    sigma = np.median(np.abs(delta - np.median(delta))) / (.6744897501960817 * np.sqrt(6))
    cut = plan['cut']
    trimmed = training[tuple(slice(c, n-c) for c,n in zip(cut,training.shape))]
    original = plan['ms'][0]
    radii = [original, *range(2*(original//2+1), trimmed.shape[0]//2, 2)]
    choices = []
    for radius in radii:
        ms = (radius, plan['ms'][1])
        shape = wf.lattice_shape(trimmed.shape, ms, plan['ss'])
        if radius != original and shape[0] < 4:
            continue
        kernels = wf.window_kernels(ms, steps, plan['q'], (2, 0))
        derivative = wf.wint_s(trimmed, (2, 0), ms, plan['ss'], kernels)
        noise_power = sigma**2 * np.sum(kernels[0][2]**2) * np.sum(kernels[1][0]**2)
        fraction = noise_power / max(np.mean(derivative**2), np.finfo(float).tiny)
        choices.append((radius, float(fraction), dict(plan, ms=ms, shape=shape)))
    selected = next((choice for choice in choices if choice[1] <= .10), min(choices,key=lambda x:x[1]))
    return selected[2], [(r, f) for r, f, _ in choices]


def lists(score, dyad, valid):
    pool = np.flatnonzero(valid & np.isfinite(score) & (score <= 1.))
    ranks = np.full(len(score), np.inf)
    ranks[pool] = rankdata(score[pool]) + rankdata(dyad[pool])
    first = previous.nondominated(pool, score, dyad)
    layer = first
    fixed = []
    while len(fixed) < 6:
        if not len(layer):
            raise ValueError('Fewer than six eligible candidate handovers')
        ordered = sorted(layer, key=lambda i: (ranks[i], i))
        fixed.extend(ordered[:6-len(fixed)])
        if len(fixed) < 6:
            pool = np.setdiff1d(pool, layer)
            layer = previous.nondominated(pool, score, dyad)
    return first, fixed


def run(reference):
    seed = int(reference['seed'])
    problem = live.load_data('wave')
    archive = read_archive(folder('wave',seed,'candidate'))
    clean = problem.data['u']
    original = problem.noisy(5, seed)['u']
    _, train, held = live.split(live.R.axes_of(problem))
    assert live.R.digest(original[train[0]]) == archive['train_hashes']['u']
    plan = {k: tuple(v) for k,v in archive['plan'].items()}
    selected, choices = choose_window(original[train[0]], train[1], plan)
    ids = sorted({i for (i,), cost in zip(archive['systems'], archive['operator']['system_cost'], strict=True)
                  if cost <= live.BUDGET})
    _, base_wape, all_dyad, all_valid, _ = previous.replay(problem, {'u': original}, archive)
    dyad, valid = all_dyad[ids], all_valid[ids]
    allowed, frequencies = previous.registry('wave', problem)
    forms = {system[0] for system in allowed}
    labels = [previous.label_key(term, frequencies) for term in archive['labels']]
    correct = np.array([frozenset(labels[j] for j,_ in archive['points'][i][2]) in forms for i in ids])
    target, predictors, weights = live.R.parse(problem.truth[0])
    physical_terms, _ = live.R.symbols(problem, [target, *predictors])
    records = []
    for arm, window in [('native', plan), ('selected', selected)]:
        scorer = FrozenScores(problem, archive, ids, held, window)
        physical = live.R.LocalBank(held[1], physical_terms, {}, window)
        def physical_error(field):
            matrix = physical.matrix({'u':field[held[0]]})
            return np.abs(matrix[:,1:] @ weights - matrix[:,0]).sum() / np.abs(matrix[:,0]).sum()
        clean_error = physical_error(clean)
        for draw in [0, *range(927101,927117)]:
            field = original if draw == 0 else problem.noisy(5, draw)['u']
            score = scorer(field)
            if arm == 'native' and draw == 0:
                np.testing.assert_allclose(score, base_wape[ids], rtol=1e-11, atol=1e-13)
            front, fixed = lists(score, dyad, valid)
            if arm == 'native' and draw == 0:
                assert len(front) == int(reference['held_front_size']) and not correct[front].any()
            matching = np.flatnonzero(correct & valid & np.isfinite(score))
            best = min(matching, key=lambda i: (score[i], dyad[i], ids[i]), default=None)
            rivals = [] if best is None else [i for i in front if score[i] <= score[best] and dyad[i] <= dyad[best]
                                               and (score[i] < score[best] or dyad[i] < dyad[best])]
            rival = min(rivals, key=lambda i: (score[i], dyad[i], ids[i]), default=None)
            records.append(dict(seed=seed, draw=draw, arm=arm, radius=window['ms'][0], rows=np.prod(window['shape']),
                                front_size=len(front), front_hit=int(correct[front].any()), fixed_size=len(fixed),
                                fixed_hit=int(correct[fixed].any()), fixed_ids=' '.join(str(ids[i]) for i in fixed),
                                best_actual_wape=score[best] if best is not None else None,
                                rival_wape=score[rival] if rival is not None else None,
                                rival_target=labels[archive['points'][ids[rival]][1]] if rival is not None else None,
                                physical_wape=physical_error(field), physical_clean_wape=clean_error))
    return records, [dict(seed=seed,radius=r,estimated_noise_fraction=f,selected=int(r==selected['ms'][0]))
                     for r,f in choices]


if __name__ == '__main__':
    references = [r for r in read_rows(PRIOR/'runs.csv') if r['dataset']=='wave']
    with ProcessPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(run, references))
    write_csv(ROOT/'wave_results.csv', [r for rows,_ in outputs for r in rows])
    write_csv(ROOT/'wave_selection.csv', [r for _,rows in outputs for r in rows])
