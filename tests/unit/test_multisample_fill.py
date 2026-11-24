"""Joint fill-and-prune proposals use every record with the fit's weights."""
import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.operators.common import fill_prune
from epde.operators.common.multisample import fit_row_weights
from epde.operators.common.plain_fit import target_gene
from epde.structure.weak_rows import term_columns


@pytest.mark.parametrize('form', ['strong', 'weak'])
@pytest.mark.parametrize('weighting', ['samples', 'points'])
def test_joint_proposal_gram_matches_independent_block_sum(form, weighting):
    search = epde.EpdeSearch(system_rps_mode='plain', engine='nsga2',
                             multisample_mode='joint', multisample_weighting=weighting,
                             default_preprocessor_type='FD', boundary_width=4,
                             evaluation_form=form, equation_terms_max_number=3,
                             data_fun_pow=2, verbose_params={'show_iter_idx': False})
    records = []
    for key, n, amplitude in [(7, 401, 1.0), (19, 503, 2.0)]:
        t = np.linspace(0, 15, n)
        u = amplitude * np.sin(t) + 0.3 * np.cos(2 * t)
        du = amplitude * np.cos(t) - 0.6 * np.sin(2 * t)
        domain = search.createDomain(t, ID=key)[1]
        records.append(search.createTrajectory({'u': u}, domain, cache_id=key,
                                                derivs=du[:, None])[1])
    search.create_pool(records, max_deriv_order=1)
    search._prepare_weak_rows()
    system = epde.translate_equation('1.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}',
                                     search.pool, all_vars=['u'])
    equation = system.vals['u']
    fill_prune.reset()
    keys = fill_prune._trajectory_keys()
    assert keys == (7, 19)
    raw_weights = global_var.samples_manager.row_weights('u', 'dm')
    weights = np.concatenate([fit_row_weights(raw_weights[key], weighting) for key in keys])
    slot = 'u' if form == 'weak' else None
    store = fill_prune.gram_store(equation, keys, weights, slot)
    blocks = []
    for key in keys:
        values = np.column_stack([term_columns(term, 'u')[key] for term in store['lib']])
        w = fit_row_weights(raw_weights[key], weighting)
        blocks.append(values.T @ (w[:, None] * values))
    expected = sum(blocks)
    scale = np.sqrt(np.diag(expected))
    expected = expected / scale[:, None] / scale[None, :]
    np.testing.assert_allclose(store['Gn'], expected, atol=2e-14, rtol=2e-13)
    assert store['ok'].all()


def test_fill_accepts_a_tuple_as_the_id_of_one_record():
    search = epde.EpdeSearch(system_rps_mode='plain', engine='nsga2',
                             boundary_width=0, equation_terms_max_number=3, data_fun_pow=2,
                             system_rps_params={'fill_probability': 1.0, 'fill_levels': 'max'})
    try:
        key = ('experiment', 1)
        t = np.linspace(0, 3, 80)
        u = np.exp(-t)
        domain = search.createDomain(t, ID=7)[1]
        _, record = search.createTrajectory({'u': u}, domain, cache_id=key,
                                             derivs=(-u)[:, None])
        search.create_pool([record], max_deriv_order=1)
        system = epde.translate_equation('1.0 * u{power: 2.0} + 0.0 = du/dx0{power: 1.0}',
                                         search.pool, all_vars=['u'])
        equation = system.vals['u']
        target = equation.target
        equation._plain_genes = target_gene(target)
        features = [term for term in equation.structure if term is not target]
        assert fill_prune.fill(equation, target, features, system,
                               search.config.search_space.system_rps_params)
        assert any(term.name == 'u{power: 1.0}' for term in equation.structure)
    finally:
        search.close()
