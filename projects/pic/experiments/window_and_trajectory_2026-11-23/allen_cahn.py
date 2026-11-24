"""Test frozen Allen–Cahn equations and original rivals on new periodic profiles."""
from collections import defaultdict

from frozen import ROOT, PRIOR, FrozenScores, folder, live, np, read_archive, read_rows, write_csv


def react(u, duration):
    return u / np.sqrt(u*u + (1-u*u)*np.exp(-10*duration))


def trajectory(mode, diffusivity, times, positions, factor, substeps):
    n = len(positions)*factor
    dx = (positions[1]-positions[0])/factor
    x = positions[0]+np.arange(n)*dx
    np.testing.assert_allclose(x[::factor], positions)
    initial = .45*np.sin(np.pi*mode*x) + .20*np.cos(np.pi*(mode+1)*x+.3)
    step = (times[1]-times[0])/substeps
    np.testing.assert_allclose(np.diff(times), step*substeps)
    multiplier = np.exp(-diffusivity*(2*np.pi*np.fft.rfftfreq(n,dx))**2*step)
    u = initial.copy()
    frames = [u[::factor].copy()]
    for _ in times[1:]:
        for _ in range(substeps):
            u = react(u, step/2)
            u = np.fft.irfft(np.fft.rfft(u)*multiplier, n=n)
            u = react(u, step/2)
        frames.append(u[::factor].copy())
    result = np.stack(frames)
    if diffusivity == 0:
        exact = react(initial[None, ::factor], times[:, None])
        np.testing.assert_allclose(result, exact, atol=1e-10)
        return exact
    return result


def frozen_pairs():
    points = defaultdict(list)
    for row in read_rows(PRIOR/'correct_points.csv'):
        if row['dataset']=='ac' and row['eligible']=='1':
            points[int(row['seed'])].append(row)
    original_winners = {(seed, min(rows,key=lambda r:(float(r['held_wape']),float(r['dyad']),int(r['point'])))['point'])
                        for seed,rows in points.items()}
    witnesses = read_rows(PRIOR/'witnesses.csv')
    problem = live.load_data('ac')
    _, _, held = live.split(live.R.axes_of(problem))
    pairs = []
    for rival in witnesses:
        seed = int(rival['seed'])
        if (rival['dataset']!='ac' or (seed,rival['correct_point']) not in original_winners
                or rival['selection'] not in ('held_front','simple_same_target')):
            continue
        correct = next(row for row in points[seed] if row['point']==rival['correct_point'])
        archive = read_archive(folder('ac',seed,'candidate'))
        plan = {k: tuple(v) for k,v in archive['plan'].items()}
        ids = [int(correct['point']), int(rival['point'])]
        scorer = FrozenScores(problem, archive, ids, held, plan)
        native = scorer(problem.noisy(5,seed)['u'])
        np.testing.assert_allclose(native, [float(r['held_wape']) for r in (correct,rival)], rtol=1e-11)
        assert native[0] >= native[1]
        pairs.append((dict(seed=seed, comparison=rival['selection'], correct=ids[0], rival=ids[1],
                           same_target=int(correct['target']==rival['target'])), scorer))
    assert len(original_winners)==11 and len(pairs)==16
    return problem, pairs


def run():
    problem, pairs = frozen_pairs()
    times, positions = live.R.axes_of(problem)
    results, checks = [], []
    for mode in (1,3,6):
        fields = {}
        for diffusivity in (.0001, 0.):
            field = trajectory(mode, diffusivity, times, positions, factor=16, substeps=80)
            coarse = trajectory(mode, diffusivity, times, positions, factor=8, substeps=40)
            max_error = float(np.max(np.abs(coarse-field)))
            assert max_error < 1e-4
            fields[diffusivity] = field
            checks.append(dict(mode=mode,diffusivity=diffusivity,max_refinement_difference=max_error))
        assert np.max(np.abs(fields[0.]-fields[.0001])) > 1e-4
        sigma = .05*fields[.0001].std()
        for draw in [0,*range(927201,927217)]:
            noise = 0. if draw==0 else sigma*np.random.default_rng(draw).standard_normal(problem.shape)
            for diffusivity, clean in fields.items():
                observed = clean + noise
                for pair, scorer in pairs:
                    full, rival = scorer(observed)
                    results.append(dict(**pair,mode=mode,diffusivity=diffusivity,draw=draw,
                                        full_wape=full,rival_wape=rival,margin=rival-full))
    write_csv(ROOT/'ac_transfer.csv', results)
    write_csv(ROOT/'ac_convergence.csv', checks)


if __name__ == '__main__':
    run()
