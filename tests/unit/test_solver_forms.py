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
