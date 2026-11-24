"""Replay the clean forced oscillator's archived competitors on identical weak rows."""
from scipy.integrate import solve_ivp

from epoch_settings import SEEDS, directory, live, np
from compare import previous, read_archive
from dyad_discrepancy import cross_validate, native_row_scores


def compare(seed):
    archive = read_archive(directory('ode', seed, 0))
    problem = live.load_data('ode')
    raw = problem.noisy(0, seed)
    combos = np.asarray(archive['systems'])
    ids = combos[:, 0]
    points = archive['points']
    forms, frequencies = previous.registry('ode', problem)
    labels = [previous.label_key(term, frequencies) for term in archive['labels']]
    supports = [frozenset(labels[j] for j, _ in p[2]) for p in points]
    accepted = {form[0] for form in forms}
    correct = [i for i in ids if supports[i] in accepted]
    assert any(supports[i] in accepted for i, in archive['native_front'])
    train, instability = np.array([p[3:5] for p in points]).T
    (_, cv), _ = native_row_scores(problem, raw, archive, combos)
    profiled = {c[0] for row in archive['operator']['profiles'] for c in row['children']}
    for metric, scores in [('ordinary', train), ('slab CV', cv)]:
        pool = ids[(scores[ids] <= 1.) & (train[ids] < live.LOSS_NAN_VAL) &
                   (instability[ids] < live.LOSS_NAN_VAL)]
        assert set(correct) <= set(pool)
        best = min(correct, key=scores.__getitem__)
        better = ((scores[pool] <= scores[best]) & (instability[pool] <= instability[best])
                  & ((scores[pool] < scores[best]) | (instability[pool] < instability[best])))
        witness, = pool[better]
        assert witness in profiled
        assert all(scores[witness] <= scores[i] and instability[witness] <= instability[i] and
                   (scores[witness] < scores[i] or instability[witness] < instability[i]) for i in correct)
        print(seed, metric, 'physical / competitor discrepancy', scores[best], scores[witness],
              'instability', instability[best], instability[witness])
    return archive, points[witness], problem


def reference_check(archive, point, problem):
    t, = live.R.axes_of(problem)
    y = problem.data['u']
    # The response to initial velocity is linear. Fit that one nuisance parameter on raw displacement.
    def rhs(t, state):
        u, v, response, derivative = state
        return v, 1.5*t - 4*u - np.sin(2*t)*v, derivative, -4*response - np.sin(2*t)*derivative
    solution = solve_ivp(rhs, (t[0],t[-1]), (y[0],0.,0.,1.), t_eval=t,
                         method='DOP853', rtol=2e-11, atol=1e-12)
    assert solution.success
    base, response = solution.y[[0,2]]
    reference = base + response * (response @ (y-base) / (response @ response))
    indices = [j for j, _ in point[2]]
    target = indices.index(point[1])
    labels = [archive['labels'][j] for j in indices]
    terms, opaque = live.R.symbols(problem, labels)
    plan = {k:tuple(v) for k,v in archive['plan'].items()}
    bank = live.R.LocalBank((t,),terms,opaque,plan)
    physical = [i for i,label in enumerate(labels) if label and all(name != 'cos' for name,_ in label)]
    assert len(physical)==4 and len(labels)==6
    for name, field in [('stored',y), ('reference integration',reference)]:
        matrix = bank.matrix({'u':field})
        for model, keep in [('physical',physical), ('expanded',range(len(labels)))]:
            # cross_validate reads support and target and refits weights for each held-out slab.
            descriptor = ('u', target, [(i,1.) for i in keep], 0., 0., 1)
            _, score, _ = cross_validate(matrix, plan['shape'], 7, [descriptor])
            print(name, model, 'CV WAPE at the competitor frequency', score[0])


if __name__=='__main__':
    for seed in SEEDS:
        archive, point, problem = compare(seed)
        if seed == SEEDS[0]:
            reference_check(archive, point, problem)
