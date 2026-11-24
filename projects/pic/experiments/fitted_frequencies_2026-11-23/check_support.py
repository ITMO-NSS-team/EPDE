"""Condition on the registered terms to test frequency estimation separately from discovery."""
import json

from frequency_study import ROOT, blind, directory, live, np, read_archive
from epde.operators.common.frequency_fit import fit_support, fitted_factors
from epde.preprocessing import weak_form as wf
from epde.structure.main_structures import Equation, Term
from epde.structure.weak_rows import WeakRows


def fit_terms(problem, raw, window, target, features):
    axes = live.R.axes_of(problem)
    search = live.U.EpdeSearch(use_solver=False, system_rps_mode='plain', engine='nsga2',
                               evaluation_form='weak', verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={'accuracy': 4})
    domain = search.createDomain(list(problem.grids), ID=0)[1]
    trajectory = search.createTrajectory(raw, domain, cache_id=0)[1]
    coordinates = ['t','x','y'][:len(axes)]
    families = [live.U.TrigonometricTokens(fit_frequencies=True),
                live.U.GridTokens(labels=coordinates, max_power=2)]
    search.create_pool(data=[trajectory], additional_tokens=families,
                       max_deriv_order=(2,) + (4,)*(len(axes)-1), data_fun_pow=3, deriv_fun_pow=2)
    def term(label):
        factors = []
        for name, pars in label:
            params = dict(pars)
            if 'freq' in params:
                params['freq'] = 0.
            if name == 'x' and 'dim' in params:
                name = coordinates[int(params['dim'])]
            factors.append(search.pool.create(label=name, **params)[1])
        return Term(search.pool, passed_term=factors, max_factors_in_term=2)
    structure = [term(label) for label in [target, *features] if label]
    eq = Equation(search.pool, structure, 'u', {'max_terms_number': {'value': len(structure)},
                                                'max_factors_in_term': {'value': 2}})
    eq.target_idx = 0
    p = window
    plan = wf.WindowPlan(0, p['q'], p['ms'], p['ss'], p['cut'], p['shape'], 0.)
    rows = WeakRows({0: wf.Fields(raw, axes)}, {(0, 'u'): plan}, [0])
    rows.fit_frequencies = True
    fitted = fit_support(eq, list(range(1, len(structure))), any(not label for label in features), rows)
    return fitted, rows


def check(name, seed, noise):
    problem = blind(live.load_data(name))
    raw = problem.noisy(noise, seed)
    archive = read_archive(directory(name, seed, noise))
    assert archive['train_hashes'] == {k: live.R.digest(v) for k,v in raw.items()}
    target, features, _ = live.R.parse(problem.truth[0])
    fitted, rows = fit_terms(problem, raw, archive['plan'], target, features)
    values = {f"{f.label}@{int(f.param('dim'))}": f.param('freq')
              for t in fitted.structure for f in fitted_factors(t)}
    registered = {f"{label}@{int(dict(pars)['dim'])}": dict(pars)['freq']
                  for feature in features for label, pars in feature if 'freq' in dict(pars)}
    assert values and values.keys() == registered.keys()
    error = {label: abs(value/registered[label]-1) for label, value in values.items()}
    return dict(dataset=name, seed=seed, noise=noise, fitted=values, registered=registered,
                within_one_percent=all(e <= .01 for e in error.values()), solves=rows.frequency_solves)


if __name__ == '__main__':
    records = [check(name, 923701, noise) for name in ('duffing','kdv_cossin') for noise in (0,5)]
    (ROOT/'conditional_frequencies.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))
    assert all(r['within_one_percent'] for r in records if r['noise'] == 0)
