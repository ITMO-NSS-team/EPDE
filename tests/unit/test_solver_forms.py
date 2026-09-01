"""``solver_forms`` converts a discovered system into the solver's form.

It was unreachable: three stacked bugs, every one of which raised before a
caller saw anything back.

1. ``EpdeSearch.solver_forms`` called ``SystemSolverInterface.form(grids=...)``
   without ``domain_key``, which the signature requires -- an unconditional
   ``TypeError``. A search may hold several trajectories, each with its own
   grid, so there is no implicit "the" grid; the choice is now a ``sample_key``
   argument defaulting to the first registered trajectory.
2. ``SystemSolverInterface.use_grids`` validated caller-supplied grids against
   ``samples_manager.sampleIDs``, a property that has never existed (it is
   spelled ``trajecatoryIDs``) -- ``AttributeError`` on the explicit-grids path.
3. The default branch of ``use_grids`` took the trajectory's grids as numpy and
   never converted them, while the explicit branch did. Everything downstream
   is torch (``_term_solver_form`` opens with ``torch.ones_like``), so the
   default path -- the one ``solver_forms()`` takes -- fed numpy into torch.
"""

import inspect

import numpy as np
import pytest
import torch

import epde
import epde.globals as global_var
from epde.integrate.interface import SystemSolverInterface


def _fitted(multiobjective, samples=1):
    search = epde.EpdeSearch(multiobjective_mode=multiobjective,
                             verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    if multiobjective:
        search.set_moeadd_params(population_size=6, training_epochs=2)
    else:
        search.set_singleobjective_params(population_size=4, training_epochs=2)

    grid = np.linspace(0, 4 * np.pi, 120)
    trajectories = []
    for idx in range(samples):
        data = np.sin(grid + 0.3 * idx) + 1.3 * np.cos(grid + 0.3 * idx)
        _, domain = search.createDomain(grid, boundary_width=10, ID=idx)
        trajectories.append(
            search.createTrajectory({'u': data}, domain, cache_id=idx)[1])
    search.fit(data=trajectories, max_deriv_order=(2,), data_fun_pow=1,
               equation_terms_max_number=3, equation_factors_max_number=1)
    return search, grid


class TestItRunsAtAll:

    @pytest.mark.parametrize('multiobjective', [False, True])
    def test_the_default_call_returns_forms(self, multiobjective):
        search, _ = _fitted(multiobjective)
        forms = search.solver_forms()
        assert forms
        entry = forms[0][0] if multiobjective else forms[0]
        variable, terms = entry[0]
        assert variable == 'u'
        assert terms

    @pytest.mark.parametrize('multiobjective', [False, True])
    def test_explicit_grids_are_accepted(self, multiobjective):
        """The branch that read the misspelled ``sampleIDs``."""
        search, grid = _fitted(multiobjective)
        assert search.solver_forms(grids=[grid])

    def test_mismatched_grid_count_is_still_rejected(self):
        """The validation that branch exists for must survive the fix."""
        search, grid = _fitted(False)
        with pytest.raises(ValueError, match='does not match'):
            search.solver_forms(grids=[grid, grid])


class TestTheGridChoiceIsExplicit:

    def test_sample_key_is_a_parameter(self):
        params = inspect.signature(epde.EpdeSearch.solver_forms).parameters
        assert 'sample_key' in params
        assert params['sample_key'].default is None

    def test_each_trajectory_can_be_selected(self):
        search, _ = _fitted(False, samples=2)
        ids = global_var.samples_manager.trajecatoryIDs
        assert len(ids) == 2
        for key in ids:
            assert search.solver_forms(sample_key=key)

    def test_an_unknown_sample_key_fails_loudly(self):
        search, _ = _fitted(False)
        with pytest.raises(KeyError):
            search.solver_forms(sample_key=99)


class TestGridsReachTorch:

    def test_the_default_branch_converts(self):
        """``use_grids`` used to leave the trajectory's numpy grids alone."""
        _fitted(False)
        interface = SystemSolverInterface.__new__(SystemSolverInterface)
        interface.grids = None
        interface._device = 'cpu'
        key = global_var.samples_manager.trajecatoryIDs[0]
        interface.use_grids(domain_key=key)
        assert interface.grids
        assert all(isinstance(g, torch.Tensor) for g in interface.grids)

    def test_the_two_branches_agree_on_type(self):
        _, grid = _fitted(False)
        key = global_var.samples_manager.trajecatoryIDs[0]

        default = SystemSolverInterface.__new__(SystemSolverInterface)
        default.grids, default._device = None, 'cpu'
        default.use_grids(domain_key=key)

        explicit = SystemSolverInterface.__new__(SystemSolverInterface)
        explicit.grids, explicit._device = None, 'cpu'
        explicit.use_grids(domain_key=key, grids=[grid])

        assert type(default.grids[0]) is type(explicit.grids[0])


def _fitted_with_grid_tokens():
    """Two trajectories on DIFFERENT domains, and a token pool that can put a
    non-derivative factor into a term.

    The tests above use ``equation_factors_max_number=1`` and no additional
    families, so every factor they ever build is a derivative -- which is
    exactly why the branch exercised below went unnoticed.
    """
    # The truth is translated and scored AFTER fit returns, and the post-fit
    # release empties the evaluated-term cache -- ``Cache.get`` then raises
    # KeyError on the missing subcache rather than recomputing.
    search = epde.EpdeSearch(multiobjective_mode=False,
                             free_tensor_cache_after_fit=False,
                             verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    search.set_singleobjective_params(population_size=4, training_epochs=1)

    coarse = np.linspace(0, 4 * np.pi, 120)
    fine = np.linspace(0, 4 * np.pi, 90)
    trajectories = []
    for idx, grid in enumerate((coarse, fine)):
        _, domain = search.createDomain(grid, boundary_width=10, ID=idx)
        trajectories.append(search.createTrajectory(
            {'u': np.sin(grid)}, domain, cache_id=idx)[1])

    grid_tokens = epde.GridTokens(['x_0'], dimensionality=0, max_power=2)
    search.fit(data=trajectories, max_deriv_order=(2,), data_fun_pow=1,
               equation_terms_max_number=3,
               additional_tokens=[grid_tokens],
               equation_factors_max_number={'factors_num': [1, 2],
                                            'probas': [0.5, 0.5]})
    return search


def _translated_with_grid_factor(search):
    """``-1 * u * x_0 = u_xx``, fitted.

    Built explicitly rather than taken from the search: whether a random
    population happens to place a grid token inside a term is not something a
    regression test should depend on, and a CONSTANT coefficient is collapsed
    to a scalar by ``_term_solver_form`` anyway. ``x_0`` guarantees a
    non-derivative factor whose values vary along the grid.
    """
    from epde.interface.equation_translator import translate_equation
    from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
    from epde.operators.common.sparsity import build_sparsity_operator
    from epde.interface.search_config import active_config

    system = translate_equation(
        '-1.0 * u{power: 1.0} * x_0{power: 1.0, dim: 0.0} + 0.0 = '
        'd^2u/dx0^2{power: 1.0}', search.pool, all_vars=['u'])
    system.vals['u'].main_var_to_explain = 'u'
    system.use_default_singleobjective_function()

    # A translated equation carries no fit; EqRightPartSelector's two
    # suboperators are what supply one.
    cfg = active_config().objectives
    sparsity = build_sparsity_operator(cfg.sparsity_cls, cfg.sparsity_kwargs)
    coeffs = LinRegBasedCoeffsEquation()
    for equation in system.vals:
        sparsity.apply(equation, {})
        coeffs.apply(equation, {})
    return system


def _grid_factor_coeff(system, sample_key):
    """The tensor coefficient of the term carrying ``x_0``, for one sample."""
    interface = SystemSolverInterface(system_to_adapt=system)
    _, terms = interface.form(domain_key=sample_key, mode='NN')[0]
    for name, term in terms.items():
        if 'x_0' in name:
            return term['coeff'].reshape(-1)
    raise AssertionError(f'no x_0 term in the solver form: {sorted(terms)}')


class TestNonDerivativeFactorsCarryTheirSample:
    """``_term_solver_form`` multiplies the coefficient by every NON-derivative
    factor's evaluated values. Those come back from ``Factor.evaluate`` as a
    per-sample dict keyed by trajectory ID, but the code read ``[-1]``.

    ``-1`` is the sentinel key ``Factor.evaluate`` assigns when it is handed an
    explicit grid LIST, so the literal was right for that path and wrong for
    the cached one -- every trig, grid or custom token in a term died with
    ``KeyError: -1`` the moment the solver was switched on.
    """

    def test_forms_build_for_every_trajectory(self):
        search = _fitted_with_grid_tokens()
        system = _translated_with_grid_factor(search)
        for key in global_var.samples_manager.trajecatoryIDs:
            assert _grid_factor_coeff(system, key) is not None

    def test_each_sample_gets_its_own_grid(self):
        """The point of keying by sample: the two trajectories live on domains
        of different size, so a coefficient built from a grid token cannot be
        the same tensor for both. 120 - 2*10 = 100 against 90 - 2*10 = 70."""
        search = _fitted_with_grid_tokens()
        system = _translated_with_grid_factor(search)
        first, second = global_var.samples_manager.trajecatoryIDs
        assert _grid_factor_coeff(system, first).shape[0] == 100
        assert _grid_factor_coeff(system, second).shape[0] == 70

    def test_the_sentinel_key_is_not_used_on_the_cached_path(self):
        source = inspect.getsource(SystemSolverInterface._term_solver_form)
        code = [line.split('#')[0] for line in source.splitlines()]
        assert not [line for line in code if 'evaluate(grids=grid_arg)[-1]' in line]
        assert 'sample_key' in inspect.signature(
            SystemSolverInterface._term_solver_form).parameters
