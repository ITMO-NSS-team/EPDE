"""``epde.globals`` holds runtime state; settings come from the config.

The module used to hold both. Seven objective settings lived there as mutable
scalars, each with a ``set_*`` writer and a ``resolve_*`` reader -- a second
declaration of values that ``default_search_config.json`` already carried, with
its own validation, its own documentation and its own default. Two of the
``getattr`` fallbacks reading them had drifted out of agreement with the value
they fell back from (``rps_amplification_cap`` -> ``None``, i.e. guard
disabled, against a declared ``100.0``; ``vc_mode_decouple`` -> ``False``
against a declared ``True``), which is what a second source of truth buys.

These tests pin the split in both directions: the settings are gone from the
module, and nothing in the package reads them any more. They follow the
inverted-test precedent set by ``test_gram_mode.TestNotConfigurable``.
"""

import os
from pathlib import Path

import pytest

import epde
import epde.globals as global_var
from conftest import using_config
from epde.interface.search_config import (active_config, load_search_config,
                                          reset_active_config,
                                          set_active_config)

EPDE_ROOT = Path(epde.__file__).parent

#: Every name the config layer took over.
REMOVED = (
    # the settings themselves
    'discrepancy_metric', 'complexity_metric', 'instability_metric',
    'second_objective', 'single_objective_metric', 'anchor_on_residual',
    'rps_amplification_cap', 'time_axis',
    # their writers
    'set_discrepancy_metric', 'set_complexity_metric',
    'set_instability_metric', 'set_second_objective',
    'set_single_objective_metric', 'set_anchor_on_residual',
    'set_rps_amplification_cap', 'set_time_axis',
    # their readers
    'resolve_discrepancy_metric', 'resolve_complexity_metric',
    'resolve_instability_metric', 'resolve_second_objective',
    'resolve_gram_mode',
    # menus, now owned by the loader
    '_GRAM_BY_INSTABILITY', '_DISCREPANCY_MENU', '_DISCREPANCY_ALIASES',
    # vcoef estimator knobs, now owned by VaryingCoefSetup
    'vc_k_max', 'vc_freq_coef', 'vc_mode_decouple',
    # dead everywhere
    'noise_seed',
)

#: What the module is FOR: the objects that exist while a search runs.
RUNTIME_STATE = ('init_caches', 'delete_cache', 'release_tensor_cache',
                 'init_verbose', 'VerboseManager', 'reset_hist',
                 'TrainHistory', 'reset_control_nn', 'reset_data_repr_nn',
                 'vc_modes_cache', 'EPDEDeprecationWarning',
                 'EPDEUsageWarning')


class TestTheModuleIsRuntimeStateOnly:

    @pytest.mark.parametrize('name', REMOVED)
    def test_the_setting_is_gone(self, name):
        assert not hasattr(global_var, name)

    @pytest.mark.parametrize('name', RUNTIME_STATE)
    def test_the_runtime_state_stayed(self, name):
        assert hasattr(global_var, name)

    def test_the_module_still_exists(self):
        """``test_use_pic_removal`` locates the package root through
        ``inspect.getfile(global_var)``, and ~40 sites import it for the
        caches."""
        assert Path(global_var.__file__).is_file()


class TestNothingReadsTheOldNames:
    """A source scan, the same instrument as
    ``test_gram_mode::test_nothing_reads_a_stored_mode``.

    A read of a name that no longer exists is an ``AttributeError`` at runtime
    -- but only on the branch that reads it, which for several of these is a
    non-default estimator that the fast test suite never touches.
    """

    def _offenders(self, needles):
        found = []
        for path in EPDE_ROOT.rglob('*.py'):
            for lineno, line in enumerate(
                    path.read_text(encoding='utf-8', errors='ignore')
                        .splitlines(), 1):
                code = line.split('#')[0]
                for needle in needles:
                    if needle in code:
                        found.append('%s:%s %s' % (path.name, lineno, needle))
        return found

    @pytest.mark.parametrize('name', REMOVED)
    def test_no_module_attribute_access_survives(self, name):
        assert not self._offenders(['global_var.%s' % name,
                                    'globals.%s' % name])

    def test_no_getattr_fallback_survives(self):
        """``getattr(global_var, 'x', default)`` is how the two divergent
        defaults hid: the fallback duplicated the declaration and disagreed
        with it. The config has no fallback to disagree with."""
        assert not self._offenders(
            ["getattr(global_var, '%s'" % n for n in REMOVED] +
            ["getattr(_gv, '%s'" % n for n in REMOVED])


class TestTheConfigIsTheSourceOfTruth:

    def test_an_unconfigured_process_still_resolves_everything(self):
        """An operator driven directly -- unit tests, offline tools -- must
        still get its settings from the config file rather than from a
        Python-level duplicate."""
        reset_active_config()
        objectives = active_config().objectives
        assert objectives.instability_metric == 'chi2'
        assert objectives.discrepancy_metric == 'wape'
        assert objectives.second_objective == 'instability'

    def test_the_search_publishes_its_config(self):
        search = epde.EpdeSearch(instability_metric='cv',
                                 verbose_params={'show_iter_idx': False})
        assert active_config() is search.config
        assert active_config().objectives.instability_metric == 'cv'
        assert active_config().objectives.gram_mode == 'axis'

    def test_the_hoisted_reads_observe_the_config(self):
        """``anchor_on_residual`` and the amplification cap are resolved once
        per call now, not per inner iteration -- they must still track the
        configuration rather than a value captured at import."""
        from epde.operators.common import fitness as fitness_mod

        with using_config(rps_amplification_cap=7.5):
            assert (active_config().search_space.rps_amplification_cap == 7.5)
            # the guard reads it through the config, so it sees the override
            assert fitness_mod.active_config().search_space \
                .rps_amplification_cap == 7.5

    def test_setting_is_rejected_at_the_loader_not_silently_absorbed(self):
        with pytest.raises(ValueError, match='instability_metric'):
            load_search_config(overrides={'instability_metric': 'nope'})


class TestVcoefKnobsBelongToTheEstimator:

    def test_they_are_class_attributes(self):
        from epde.operators.common.stability import VaryingCoefSetup
        assert VaryingCoefSetup.K_MAX == 6
        assert VaryingCoefSetup.FREQ_COEF == 1.0
        assert VaryingCoefSetup.MODE_DECOUPLE is True

    def test_mode_decouple_default_matches_the_old_effective_value(self):
        """``globals`` declared True while the ``getattr`` fallback said False.
        The declaration won, because the attribute always existed -- so True is
        the behaviour that must be preserved."""
        from epde.operators.common.stability import VaryingCoefSetup
        assert VaryingCoefSetup.MODE_DECOUPLE is True

    def test_the_super_gram_path_carries_mode_decouple(self):
        """``from_full`` builds instances via ``__new__``, bypassing
        ``__init__``, so every attribute ``_solve_gammas`` reads has to be set
        by hand there. A missing one raises only on the RPS sweep."""
        import inspect

        from epde.operators.common.stability import VaryingCoefSetup
        source = inspect.getsource(VaryingCoefSetup.from_full)
        assert 'mode_decouple' in source
        assert 'mode_decouple' in inspect.signature(
            VaryingCoefSetup.precompute_super).parameters


class TestSingleObjectiveHonoursTheConfig:
    """``discrepancy_metric`` was the one objective setting that never reached
    single-objective mode: ``BaselineDirector.use_baseline`` hardcoded
    a separate WAPE-only filler, so the kwarg was a silent no-op there. It now
    builds a bare ``Discrepancy()``, exactly as MOEA/D does, which resolves the
    option from the configuration at compute time. That WAPE-only class has
    since been retired, leaving one discrepancy family.
    """

    @staticmethod
    def _record(monkeypatch, built):
        """Capture the filler list at assembly time.

        Only RESET the operator-param singleton here -- it now reads the
        active config, and the single-objective config is entered by the
        caller below. Constructing it at this point would capture the
        multiobjective operator table, which has no
        ``PopulationLevelCrossover`` block. First use inside the
        ``using_config`` block builds it against the right one.
        """
        from epde.operators.utils.default_parameter_loader import EvolutionaryParams
        from epde.optimizers.single_criterion import strategy as so_strategy

        EvolutionaryParams.reset()
        original = so_strategy.SolverFreeFitness

        def _recording(*args, **kwargs):
            built.append(list(kwargs.get('objectives', ())))
            return original(*args, **kwargs)

        monkeypatch.setattr(so_strategy, 'SolverFreeFitness', _recording)
        return so_strategy

    def test_the_director_builds_a_config_resolving_filler(self, monkeypatch):
        from epde.operators.common.objectives import Discrepancy

        built = []
        so_strategy = self._record(monkeypatch, built)
        director = so_strategy.BaselineDirector()
        with using_config(multiobjective_mode=False,
                          discrepancy_metric='l2_relative'):
            director.use_baseline(params={})
            objectives = built[0]
            assert any(isinstance(o, Discrepancy) for o in objectives)
            discrepancy = next(o for o in objectives
                               if isinstance(o, Discrepancy))
            # bare: no instance override, so it follows the config
            assert discrepancy.metric is None
            assert discrepancy._resolved_metric() == 'l2_relative'

    def test_the_instability_filler_follows_the_configured_metric(self,
                                                                 monkeypatch):
        from epde.operators.common.objectives import Instability

        built = []
        so_strategy = self._record(monkeypatch, built)
        for metric, expected in (('discrepancy', False), ('instability', True)):
            built.clear()
            with using_config(multiobjective_mode=False,
                              single_objective_metric=metric):
                so_strategy.BaselineDirector().use_baseline(params={})
            present = any(isinstance(o, Instability) for o in built[0])
            assert present is expected, metric

    def test_the_wape_only_filler_is_retired(self):
        """One discrepancy family. ``WAPEDiscrepancy`` was a second
        ``EquationObjective`` reachable only from the hardcoded
        single-objective branch; with that branch reading the config, nothing
        selected it. Its 'wape' option lives on as ``Discrepancy``'s default.
        """
        from epde.operators.common import objectives as objectives_mod

        assert not hasattr(objectives_mod, 'WAPEDiscrepancy')
        assert objectives_mod.OBJECTIVE_REGISTRY['discrepancy'] is             objectives_mod.Discrepancy
        assert 'wape' in objectives_mod.Discrepancy.OPTIONS
