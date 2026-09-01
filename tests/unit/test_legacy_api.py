"""The pre-``domain_refactor`` calling convention still runs, under warning.

37 scripts in ``projects/`` build a search with ``coordinate_tensors=`` /
``boundary=`` and call ``fit`` with raw arrays plus ``variable_names``. The
domain refactor removed all of those parameters, so every one of them raised
``TypeError`` on its first line. The shim maps them onto
``createDomain``/``createTrajectory`` so they keep working until migrated.

Two things matter beyond "it runs": the warning has to be *visible* to the
script's author (DeprecationWarning is suppressed outside ``__main__``, so the
stacklevel has to land on their line), and arguments that were removed outright
must raise rather than be silently guessed at.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

import epde

from epde import TrigonometricTokens
from epde.interface.interface import EpdeSearch
from epde.interface.legacy_api import (LEGACY_DATA_KEYS, LEGACY_INIT_KEYS,
                                       REMOVED_KEYS, reject_removed,
                                       split_legacy, warn_legacy)
from epde.structure.domain import Trajectory


@pytest.fixture
def grid():
    return np.linspace(0, 20, 120)


@pytest.fixture
def legacy_search(grid):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        search = EpdeSearch(boundary=10, coordinate_tensors=(grid,),
                            verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    return search


# ---------------------------------------------------------------------------
# removed outright
# ---------------------------------------------------------------------------

class TestRemovedArguments:
    """No silent translation: guessing an equivalent would change what the
    search optimizes."""

    def test_use_pic_is_refused_with_a_directed_message(self):
        with pytest.raises(TypeError, match='second_objective'):
            EpdeSearch(use_pic=True, verbose_params={'show_iter_idx': False})

    def test_use_pic_false_is_refused_too(self):
        with pytest.raises(TypeError, match='use_pic has been removed'):
            EpdeSearch(use_pic=False, verbose_params={'show_iter_idx': False})

    def test_use_default_strategy_is_refused(self):
        with pytest.raises(TypeError, match='director'):
            EpdeSearch(use_default_strategy=True,
                       verbose_params={'show_iter_idx': False})

    def test_every_removed_key_names_its_replacement(self):
        for key, message in REMOVED_KEYS.items():
            assert key in message
            assert len(message) > 60, key

    def test_reject_removed_passes_clean_kwargs(self):
        reject_removed({'boundary': 5, 'device': 'cpu'})

    def test_unknown_argument_still_fails(self):
        with pytest.raises(ValueError, match='Unknown search-config parameter'):
            EpdeSearch(bogus_setting=1, verbose_params={'show_iter_idx': False})


# ---------------------------------------------------------------------------
# the constructor
# ---------------------------------------------------------------------------

class TestLegacyConstructor:

    def test_old_domain_arguments_are_accepted(self, grid):
        with pytest.warns(DeprecationWarning, match='coordinate_tensors'):
            search = EpdeSearch(boundary=10, coordinate_tensors=(grid,),
                                verbose_params={'show_iter_idx': False})
        assert search._legacy_domain is not None
        assert list(search._legacy_domain.inner_shape) == [100]

    def test_boundary_maps_onto_boundary_width(self, grid):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            search = EpdeSearch(boundary=25, coordinate_tensors=(grid,),
                                verbose_params={'show_iter_idx': False})
        assert list(search._legacy_domain.inner_shape) == [70]

    def test_device_cuda_no_longer_explodes(self, grid):
        """It used to raise NotImplementedError from Cache.__init__, because
        init_caches passed the search device into a numpy-backed cache."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            search = EpdeSearch(boundary=10, coordinate_tensors=(grid,),
                                device='cuda',
                                verbose_params={'show_iter_idx': False})
        assert search.config.solver.device == 'cuda'

    def test_inert_arguments_are_named_in_the_warning(self, grid):
        with pytest.warns(DeprecationWarning, match='ignored'):
            EpdeSearch(boundary=10, coordinate_tensors=(grid,),
                       dimensionality=0, prune_domain=False,
                       verbose_params={'show_iter_idx': False})

    def test_no_warning_for_the_current_api(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            EpdeSearch(boundary_width=10, verbose_params={'show_iter_idx': False})


# ---------------------------------------------------------------------------
# the data path
# ---------------------------------------------------------------------------

class TestLegacyData:

    def test_raw_arrays_become_trajectories(self, legacy_search):
        x = 1.5 + 0.5 * np.sin(np.linspace(0, 20, 120))
        with pytest.warns(DeprecationWarning, match='createTrajectory'):
            trajectories = legacy_search._as_trajectories(
                [x], {'variable_names': ['u']}, 'fit')
        assert len(trajectories) == 1
        assert isinstance(trajectories[0], Trajectory)
        assert trajectories[0].variable_names == ['u']

    def test_multiple_variables(self, legacy_search):
        t = np.linspace(0, 20, 120)
        x, y = 1.5 + 0.5 * np.sin(t), 1.0 + 0.4 * np.cos(t)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            trajectories = legacy_search._as_trajectories(
                [x, y], {'variable_names': ['u', 'v']}, 'fit')
        assert trajectories[0].variable_names == ['u', 'v']

    def test_bare_array_defaults_to_u(self, legacy_search):
        x = 1.5 + 0.5 * np.sin(np.linspace(0, 20, 120))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            trajectories = legacy_search._as_trajectories(x, {}, 'fit')
        assert trajectories[0].variable_names == ['u']

    def test_max_deriv_order_is_not_translated_at_all(self, legacy_search):
        """It never became a legacy argument.

        ``max_deriv_order`` and the two ``*_fun_pow`` describe the token POOL,
        not a data sample, so they stayed real ``create_pool``/``fit``
        parameters through the domain refactor. An old call passes them by the
        same name to the same method and they are still read -- the arrays are
        the only part of that call that needs translating.
        """
        assert not ({'max_deriv_order', 'data_fun_pow', 'deriv_fun_pow'}
                    & LEGACY_DATA_KEYS)

        x = 1.5 + 0.5 * np.sin(np.linspace(0, 20, 120))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            legacy_search.create_pool(data=[x], variable_names=['u'],
                                      max_deriv_order=(2,))
        assert legacy_search.pool_params['max_deriv_order'] == (2,)
        assert legacy_search.pool.families_cardinality()[1] > 0

    def test_name_count_mismatch_is_reported(self, legacy_search):
        x = np.linspace(0, 1, 120)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            with pytest.raises(ValueError, match='Mismatching numbers'):
                legacy_search._as_trajectories(
                    [x], {'variable_names': ['u', 'v']}, 'fit')

    def test_raw_arrays_without_a_domain_explain_both_options(self):
        search = EpdeSearch(verbose_params={'show_iter_idx': False})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            with pytest.raises(ValueError, match='no domain to attach them to'):
                search._as_trajectories(np.linspace(0, 1, 50), {}, 'fit')

    def test_trajectories_pass_through_untouched(self, legacy_search):
        x = 1.5 + 0.5 * np.sin(np.linspace(0, 20, 120))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            built = legacy_search._as_trajectories(x, {}, 'fit')
            again = legacy_search._as_trajectories(built, {}, 'fit')
        assert again is built

    def test_unknown_data_kwarg_is_rejected(self, legacy_search):
        x = np.linspace(0, 1, 120)
        with pytest.raises(TypeError, match='unexpected keyword'):
            legacy_search._as_trajectories(x, {'nonsense': 1}, 'fit')

    def test_search_level_setting_passed_to_fit_is_rejected(self, legacy_search):
        """population_size configures the search, not the data; accepting it
        silently on fit would make it look effective when it is not."""
        x = np.linspace(0, 1, 120)
        with pytest.raises(TypeError, match='population_size'):
            legacy_search._as_trajectories(x, {'population_size': 4}, 'fit')


class TestWarningVisibility:
    """DeprecationWarning is hidden outside ``__main__``, so a warning blamed
    on interface.py is invisible to exactly the person who needs it."""

    def test_constructor_warning_points_at_the_caller(self, grid):
        with pytest.warns(DeprecationWarning) as caught:
            EpdeSearch(boundary=10, coordinate_tensors=(grid,),
                       verbose_params={'show_iter_idx': False})
        assert caught[0].filename == __file__

    def test_helper_path_warning_points_at_the_caller(self, legacy_search):
        """Through create_pool -> _as_trajectories -> warn_legacy, i.e. with
        the extra frame the stacklevel has to account for. (``fit`` takes the
        identical path; this one does not also run the optimizer.)"""
        t = np.linspace(0, 20, 120)
        x = 1.5 + 0.5 * np.sin(t)
        with pytest.warns(DeprecationWarning) as caught:
            legacy_search.create_pool(
                data=[x], variable_names=['u'], max_deriv_order=(1,),
                additional_tokens=[TrigonometricTokens(freq=(0.9, 1.1),
                                                       dimensionality=0)])
        legacy = [w for w in caught if 'pre-domain_refactor' in str(w.message)]
        assert legacy, 'no legacy warning emitted'
        assert legacy[0].filename == __file__, legacy[0].filename

    def test_both_public_entry_points_warn(self, legacy_search):
        x = 1.5 + 0.5 * np.sin(np.linspace(0, 20, 120))
        with pytest.warns(DeprecationWarning, match=r'create_pool\(\.\.\.\)'):
            legacy_search.create_pool(data=[x], variable_names=['u'],
                                      max_deriv_order=(1,))
        with pytest.warns(DeprecationWarning, match=r'fit\(\.\.\.\)'):
            legacy_search._as_trajectories(x, {'variable_names': ['u']}, 'fit')


    def test_the_warning_survives_the_process_wide_ignore(self):
        """``globals.init_verbose`` installs ``filterwarnings('ignore')``
        unless show_warnings=True, so merely CONSTRUCTING a search used to
        silence this whole channel for the rest of the process -- invisibly,
        because pytest.warns resets the filters and the tests above stayed
        green. Run it in a real interpreter, the way a script does.
        """
        import subprocess
        import sys

        script = '; '.join((
            'import numpy as np, epde',
            'epde.EpdeSearch(verbose_params={"show_iter_idx": False})',
            'epde.EpdeSearch(boundary=10, '
            'coordinate_tensors=(np.linspace(0, 1, 50),), '
            'verbose_params={"show_iter_idx": False})'))
        done = subprocess.run([sys.executable, '-c', script],
                              capture_output=True, text=True,
                              cwd=str(Path(epde.__file__).parent.parent))
        assert done.returncode == 0, done.stderr[-2000:]
        assert 'pre-domain_refactor' in done.stderr, done.stderr[-2000:]


class TestHelpers:

    def test_split_is_exhaustive(self):
        legacy, rest = split_legacy({'boundary': 1, 'device': 'cpu'},
                                    LEGACY_INIT_KEYS)
        assert legacy == {'boundary': 1}
        assert rest == {'device': 'cpu'}

    def test_no_key_is_both_removed_and_translatable(self):
        assert not set(REMOVED_KEYS) & (LEGACY_INIT_KEYS | LEGACY_DATA_KEYS)

    def test_empty_legacy_emits_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            warn_legacy('fit(...)', {}, 'anything')
