"""Replay the eight original clean nontrigonometric misses on the saved epoch archives."""
import csv
import json
from dataclasses import replace

from epoch_settings import OPTIONS, ROOT, SEEDS, directory, live, np
from compare import previous, read_archive, structural_matches


def term_keys(archive, frequencies):
    return [previous.label_key(label, frequencies) for label in archive['labels']]


def coefficients(point, keys):
    return {keys[j]: value for j, value in point[2] if j != point[1]}


def scores(point):
    return dict(wape=point[3], instability=point[4])


def burgers(problem, archive):
    t, x = problem.grids
    field_error = float(np.max(np.abs(problem.data['u'] - x/(t-2))))
    assert field_error < 1e-11
    ut = -x/(t-2)**2
    utt = 2*x/(t-2)**3
    u = x/(t-2)
    np.testing.assert_allclose(ut, .5*u + (.5*t-.25*t*t)*utt, rtol=1e-14)
    keys = term_keys(archive, ())
    ids = np.asarray(archive['native_front'])
    accepted, physical, _ = structural_matches(problem.name, problem, archive, ids)
    assert accepted.any() and not physical.any()
    earlier = replace(problem, truth_alternatives=[alt for alt in problem.truth_alternatives
                                                   if not any('d^2u/dx0^2' in eq for eq in alt)])
    original, _, _ = structural_matches(problem.name, earlier, archive, ids)
    assert not original.any()
    exact = {'u': .5, 'd^2u/dx0^2*x@0': .5, 'd^2u/dx0^2*x^2@0': -.25}
    for i, in ids[accepted]:
        point = archive['points'][i]
        weights = coefficients(point, keys)
        if keys[point[1]] == 'du/dx0' and weights.keys() == exact.keys():
            np.testing.assert_allclose([weights[k] for k in exact], list(exact.values()), rtol=1e-6)
            return dict(point=int(i), first_call=point[5], measured_field_max_error=field_error,
                        coefficients=weights, **scores(point))
    raise AssertionError('No front member with the verified identity and coefficients')


def divide(problem, archive):
    # Use the same target to compare the published support with the simplest
    # surviving extension. Both tests use the original archived scores.
    keys = term_keys(archive, ())
    ids = np.asarray(archive['systems'])
    _, physical, _ = structural_matches(problem.name, problem, archive, ids)
    points = archive['points']
    correct_ids = [int(i) for i, in ids[physical]]
    winner, = min(archive['native_front'], key=lambda row: len(points[row[0]][2]))
    rival = points[winner]
    best = min((i for i in correct_ids if points[i][1] == rival[1]), key=lambda i: points[i][3])
    law = points[best]
    assert keys[rival[1]] == 'du/dx1'
    assert rival[3] < law[3] and rival[4] < law[4]
    w = coefficients(rival, keys)
    dx = np.diff(live.R.axes_of(problem)[1])[0]
    # Solve for u_x. The centered first derivative contributes -h^2/6,
    # and the centered second derivative contributes +h^2/48 after rearranging.
    predicted = {'d^3u/dx1^3': -dx**2/6, 'd^4u/dx1^4*x@1': dx**2/48}
    assert w.keys() == coefficients(law, keys).keys() | predicted.keys()
    errors = {k: abs(w[k]/v-1) for k, v in predicted.items()}
    assert max(errors.values()) < .001
    return dict(physical_point=best, physical=scores(law), front_point=winner, front=scores(rival),
                coefficients=w, centered_stencil_prediction=predicted, relative_coefficient_errors=errors)


def stencil_replay(problem):
    """Compare two spatial derivative orders on identical interior measurements."""
    t, x = live.R.axes_of(problem)
    dt, dx = t[1]-t[0], x[1]-x[0]
    u = problem.data['u']
    ut = (u[:-4] - 8*u[1:-3] + 8*u[3:-1] - u[4:])[:, 2:-2]/(12*dt)
    left2, left, mid, right, right2 = (a[2:-2] for a in
        (u[:, :-4], u[:, 1:-3], u[:, 2:-2], u[:, 3:-1], u[:, 4:]))
    slopes = [(right-left)/(2*dx), (left2-8*left+8*right-right2)/(12*dx)]
    curvatures = [(left-2*mid+right)/dx**2, (-left2+16*left-30*mid+16*right-right2)/(12*dx**2)]
    return {order: float(np.sum(abs(ut + 2*slope/x[2:-2] - .5*curvature))/np.sum(abs(ut)))
            for order, slope, curvature in zip(('second_order', 'fourth_order'), slopes, curvatures)}


def navier_stokes(problem, archive):
    forms, frequencies = previous.registry('ns', problem)
    labels = term_keys(archive, frequencies)
    names = ('x_momentum', 'y_momentum', 'continuity')
    laws = {s: name for form in forms for s, name in zip(form, names)}
    supports = [frozenset(labels[j] for j, _ in p[2]) for p in archive['points']]
    tags = [laws.get(s) for s in supports]
    combos = archive['systems']
    slots = {v: sorted({tags[c[k]] for c in combos if tags[c[k]] is not None})
             for k, v in enumerate(archive['variables'])}
    coverage = [len({tags[i] for i in c} - {None}) for c in combos]
    first = {name: min((archive['points'][i][5] for i, tag in enumerate(tags) if tag == name),
                       default=None) for name in names}
    return dict(first_scored_call=first, accepted_laws_by_slot=slots,
                most_laws_together=max(coverage), complete_candidates=coverage.count(3))


def main():
    with (ROOT/'runs.csv').open(newline='') as f:
        native = [r for r in csv.DictReader(f) if r['dataset'] != 'ode_simple'
                  and r['noise_percent'] == '0' and r['method'] == 'native']
    totals = {}
    for r in native:
        totals[r['dataset']] = totals.get(r['dataset'], 0) + int(r['front_hit'])
    other_hits = sum(totals.values()) - totals['burgers_inviscid']
    result = dict(original_clean_hits=other_hits, clean_core_runs=len(native), runs={})
    for name, audit in [('burgers_inviscid', burgers), ('pde_divide', divide), ('ns', navier_stokes)]:
        problem = live.load_data(name, **OPTIONS.get(name, {}))
        result['runs'][name] = {}
        for seed in SEEDS:
            archive = read_archive(directory(name, seed, 0))
            assert (archive['system'], archive['seed'], archive['noise_percent']) == (name, seed, 0)
            assert archive['train_hashes'] == {k: live.R.digest(v) for k,v in problem.data.items()}
            result['runs'][name][seed] = audit(problem, archive)
        if name == 'pde_divide':
            result['divide_raw_grid_wape'] = stencil_replay(problem)
    result['identity_inclusive_clean_hits'] = other_hits + len(result['runs']['burgers_inviscid'])
    output = ROOT/'clean_misses.json'
    tmp = output.with_suffix('.pending')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(output)
    print(output)


if __name__ == '__main__':
    main()
