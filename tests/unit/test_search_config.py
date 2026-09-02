"""Contract tests for the grouped search configuration.

The loader is the single place that decides what an unconfigured EPDE run
does, so these pin four things: the shipped defaults still equal the
pre-config interface's defaults, the precedence order holds, the flat-kwarg
bridge covers every public parameter, and malformed configs fail loudly rather
than silently doing something else.
"""

import json
import os
import tempfile

import pytest

from epde.interface.search_config import (
    GROUP_CLASSES, KEY_GROUP, MULTI_OBJECTIVE_OPERATORS, SPARSITY_REGISTRY,
    default_device,
    TOKEN_REGISTRY, UNSET, FromConfig, SearchConfig, build_tokens,
    collect_overrides, load_search_config, resolve_sparsity)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _leaf_keys(payload, depth=0):
    """Every key name in a nested JSON payload, to two levels."""
    if not isinstance(payload, dict) or depth > 2:
        return
    for key, value in payload.items():
        yield key
        yield from _leaf_keys(value, depth + 1)


def _write(tmp_path, payload, suffix='.json'):
    path = os.path.join(str(tmp_path), 'cfg' + suffix)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle)
    return path


# ---------------------------------------------------------------------------
# the default pin
# ---------------------------------------------------------------------------

class TestDefaultPin:
    """The shipped JSON must reproduce the pre-config interface's behaviour.

    Every value here was the default of a real signature or the value a
    ``globals.resolve_*`` fallback returned, so a diff in this test is a
    behaviour change, not a cosmetic one.
    """

    def test_domain_and_preprocessing(self):
        cfg = load_search_config()
        assert cfg.domain.boundary_width == 5        # createDomain
        assert cfg.domain.time_axis == 0
        assert cfg.preprocessing.default_preprocessor_type == 'poly'
        assert cfg.preprocessing.preprocessor_kwargs == {}
        assert cfg.preprocessing.max_deriv_order == 1  # createTrajectory

    def test_search_space(self):
        cfg = load_search_config().search_space
        assert cfg.data_fun_pow == 1
        assert cfg.deriv_fun_pow == 1
        assert cfg.equation_terms_max_number == 6      # fit
        assert cfg.equation_factors_max_number == 1
        assert cfg.rps_amplification_cap == 100.0
        assert list(cfg.tokens) == []

    def test_objectives_are_the_pre_config_defaults(self):
        """These used to be compared against ``globals.resolve_*``. Those
        resolvers were the second declaration of the same values -- the
        duplication this config removed -- so the pin is now against the
        literals they returned."""
        cfg = load_search_config().objectives
        assert cfg.discrepancy_metric == 'wape'
        assert cfg.complexity_metric == 'factors'
        assert cfg.instability_metric == 'chi2'
        assert cfg.single_objective_metric == 'discrepancy'
        assert cfg.anchor_on_residual is False
        assert cfg.multiobjective_mode is True

    def test_gram_mode_is_derived_from_the_instability_metric(self):
        assert load_search_config().objectives.gram_mode is None   # chi2

    def test_second_objective_default_is_what_use_pic_true_meant(self):
        assert load_search_config().objectives.second_objective == 'instability'

    def test_solver_defaults(self):
        cfg = load_search_config().solver
        assert cfg.use_solver is False
        assert cfg.solver_backend == 'autograd'
        # The GPU is the default when one is usable; a machine without one
        # resolves to cpu through the same call.
        assert cfg.device == default_device()
        assert cfg.mode == 'NN'
        assert cfg.use_cache is False

    def test_device_follows_gpu_availability(self, monkeypatch):
        """cuda when torch reports a usable GPU, cpu when it does not."""
        import epde.interface.search_config as sc
        import torch
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
        assert sc.default_device() == 'cuda'
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        assert sc.default_device() == 'cpu'

    def test_an_explicit_device_still_wins(self):
        """Forcing the CPU on a GPU machine -- a bit-identical A/B, say."""
        assert load_search_config(overrides={'device': 'cpu'}).solver.device == 'cpu'

    def test_solver_keys_reach_the_operator_mapping(self):
        """pinn_loss_mult / error_metric / deepxde_config are parameters of
        SolverBasedFitness AND settings of the solver group. They are declared
        once, in the group, and referenced from the operator block, so they
        cannot disagree -- they used to be written in two JSON files and did."""
        cfg = load_search_config()
        params = cfg.evolution.operators['SolverBasedFitness']
        assert cfg.solver.pinn_loss_mult == params['pinn_loss_mult']
        assert cfg.solver.error_metric == params['error_metric']
        assert cfg.solver.deepxde_config == params['deepxde_config']

    def test_moving_the_setting_moves_the_operator_parameter(self):
        """The property that makes one declaration real."""
        cfg = load_search_config(overrides={'pinn_loss_mult': 7.5})
        assert cfg.solver.pinn_loss_mult == 7.5
        assert cfg.evolution.operators['SolverBasedFitness']['pinn_loss_mult'] == 7.5
        assert cfg.evolution.operators['PIC']['pinn_loss_mult'] == 7.5

    def test_an_explicit_operator_override_still_wins(self):
        """The reference is a default, not a lock."""
        cfg = load_search_config(overrides={
            'operators': {'SolverBasedFitness': {'pinn_loss_mult': 2.0}}})
        assert cfg.solver.pinn_loss_mult == 0.0
        assert cfg.evolution.operators['SolverBasedFitness']['pinn_loss_mult'] == 2.0

    def test_evolution_defaults_are_the_multiobjective_ones(self):
        cfg = load_search_config().evolution
        assert cfg.population_size == 6      # set_moeadd_params
        assert cfg.training_epochs == 100
        assert cfg.neighbors_number == 3
        assert cfg.PBI_penalty == 5.0
        assert cfg.subregion_mating_limitation == 0.9

    def test_runtime_defaults(self):
        cfg = load_search_config().runtime
        assert cfg.memory_for_cache == 15
        assert cfg.verbose_params == {'show_iter_idx': True}
        assert cfg.free_tensor_cache_after_fit is True


class TestSingleObjectiveDefaults:
    """``set_singleobjective_params`` used 4 / 50 where MOEA/D used 6 / 100.
    Flipping the mode must not silently change them."""

    def test_mode_flip_restores_the_single_objective_values(self):
        cfg = load_search_config(
            overrides=collect_overrides(multiobjective_mode=False))
        assert cfg.evolution.population_size == 4
        assert cfg.evolution.training_epochs == 50

    def test_explicit_kwarg_still_wins(self):
        cfg = load_search_config(overrides=collect_overrides(
            multiobjective_mode=False, population_size=11))
        assert cfg.evolution.population_size == 11
        assert cfg.evolution.training_epochs == 50

    def test_explicit_file_value_still_wins(self, tmp_path):
        path = _write(tmp_path, {'evolution': {'population_size': 9}})
        cfg = load_search_config(
            path, collect_overrides(multiobjective_mode=False))
        assert cfg.evolution.population_size == 9


# ---------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------

class TestPrecedence:

    def test_file_beats_builtin(self, tmp_path):
        path = _write(tmp_path, {'search_space': {'equation_terms_max_number': 12}})
        assert load_search_config(path).search_space.equation_terms_max_number == 12

    def test_kwarg_beats_file(self, tmp_path):
        path = _write(tmp_path, {'search_space': {'equation_terms_max_number': 12}})
        cfg = load_search_config(
            path, collect_overrides(equation_terms_max_number=3))
        assert cfg.search_space.equation_terms_max_number == 3

    def test_kwarg_wins_regardless_of_group(self):
        """EpdeSearch(use_solver=True) must work without the caller knowing
        that use_solver is filed under `solver`."""
        cfg = load_search_config(overrides=collect_overrides(use_solver=True))
        assert cfg.solver.use_solver is True

    def test_explicit_none_beats_a_file_value(self, tmp_path):
        """A ``None`` kwarg must still override the file -- that is why UNSET
        exists as a third state. It resolves to the built-in default rather
        than staying None: ``null`` used to mean "leave the process global
        alone", and with the settings no longer held in mutable globals there
        is nothing to leave alone. The file's 'l2' is what must not survive.
        """
        path = _write(tmp_path, {'objectives': {'discrepancy_metric': 'l2'}})
        cfg = load_search_config(path, collect_overrides(discrepancy_metric=None))
        assert cfg.objectives.discrepancy_metric == 'wape'

    def test_null_resolves_to_the_default_not_to_none(self):
        """Consumers read the config directly, so they must never be handed a
        None they would have to fall back from."""
        for key in ('discrepancy_metric', 'complexity_metric',
                    'instability_metric', 'single_objective_metric',
                    'second_objective'):
            cfg = load_search_config({'objectives': {key: None}})
            assert getattr(cfg.objectives, key) is not None, key

    def test_unset_is_not_an_override(self, tmp_path):
        path = _write(tmp_path, {'objectives': {'discrepancy_metric': 'l2'}})
        cfg = load_search_config(path, collect_overrides(discrepancy_metric=UNSET))
        assert cfg.objectives.discrepancy_metric == 'l2'

    def test_collect_overrides_drops_only_unset(self):
        assert collect_overrides(a=UNSET, b=None, c=0) == {'b': None, 'c': 0}

    def test_dict_config_is_accepted(self):
        cfg = load_search_config({'domain': {'boundary_width': 20}})
        assert cfg.domain.boundary_width == 20

    def test_no_unset_can_survive(self):
        cfg = load_search_config()
        for group in GROUP_CLASSES:
            for value in cfg.as_dict()[group].values():
                assert value is not UNSET


class TestMergePolicy:
    """Asymmetric on purpose: a file merges dict-valued keys one level deep so
    it can set one sub-key, while a kwarg replaces wholesale because that is
    what passing ``verbose_params={...}`` has always done."""

    def test_file_merges_one_level_deep(self, tmp_path):
        path = _write(tmp_path, {'runtime': {'verbose_params': {'show_iter_stats': True}}})
        verbose = load_search_config(path).runtime.verbose_params
        assert verbose == {'show_iter_idx': True, 'show_iter_stats': True}

    def test_kwarg_replaces_wholesale(self):
        cfg = load_search_config(overrides=collect_overrides(
            verbose_params={'show_iter_stats': True}))
        assert cfg.runtime.verbose_params == {'show_iter_stats': True}

    def test_nested_solver_dict_merges_too(self, tmp_path):
        path = _write(tmp_path, {'solver': {'deepxde_config': {'epochs': 10}}})
        cfg = load_search_config(path).solver.deepxde_config
        assert cfg['epochs'] == 10
        assert cfg['activation'] == 'tanh'      # untouched sub-key survives


class TestIsolation:

    def test_two_resolutions_share_no_mutable_state(self):
        first, second = load_search_config(), load_search_config()
        first.runtime.verbose_params['show_iter_idx'] = 'poisoned'
        assert second.runtime.verbose_params['show_iter_idx'] is True

    def test_mutating_a_resolved_config_does_not_touch_the_defaults(self):
        load_search_config().solver.deepxde_config['epochs'] = -1
        assert load_search_config().solver.deepxde_config['epochs'] == 2000

    def test_config_is_frozen(self):
        cfg = load_search_config()
        with pytest.raises(Exception):
            cfg.solver.use_solver = True

    def test_attribute_typo_raises(self):
        with pytest.raises(AttributeError):
            load_search_config().solver.devcie


# ---------------------------------------------------------------------------
# the flat-kwarg bridge
# ---------------------------------------------------------------------------

class TestKeyGroup:

    def test_every_leaf_maps_to_exactly_one_group(self):
        from dataclasses import fields
        raw = {group: [f.name for f in fields(cls)]
               for group, cls in GROUP_CLASSES.items()}
        seen = {}
        for group, values in raw.items():
            for key in values:
                assert key not in seen, (
                    'key %r declared in both %r and %r' % (key, seen[key], group))
                seen[key] = group
        assert seen == KEY_GROUP

    def test_a_field_default_is_what_a_search_resolves_to(self):
        """The property the old key-set comparison did NOT check.

        ``pinn_loss_mult`` was 0.0 on the dataclass and 1e4 in the shipped
        JSON that actually supplied it, and every test passed. Compare VALUES,
        for every field, so a default that is written but not used cannot
        exist again.
        """
        from dataclasses import MISSING, fields
        cfg = load_search_config()
        for group, cls in GROUP_CLASSES.items():
            for spec in fields(cls):
                if spec.default is not MISSING:
                    declared = spec.default
                elif spec.default_factory is not MISSING:
                    declared = spec.default_factory()
                else:
                    raise AssertionError('%s.%s has no default' % (group, spec.name))
                if group == 'evolution' and spec.name == 'operators':
                    continue    # resolved from the operator tables, not a leaf
                if group == 'objectives' and spec.name == 'sparsity_cls':
                    continue    # the name is resolved to the operator class
                assert getattr(getattr(cfg, group), spec.name) == declared,                     '%s.%s: declared %r, resolved %r' % (
                        group, spec.name, declared,
                        getattr(getattr(cfg, group), spec.name))

    def test_no_shipped_json_declares_a_search_setting(self):
        """Shadowing made impossible by construction, not by vigilance.

        The bug was a second declaration in a data file that quietly won. Any
        JSON shipped inside the package that names a config key would be one
        again.
        """
        import epde
        root = os.path.dirname(os.path.abspath(epde.__file__))
        offenders = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith('.json'):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, encoding='utf-8') as handle:
                        payload = json.load(handle)
                except (ValueError, OSError):
                    continue
                for key in _leaf_keys(payload):
                    if key in KEY_GROUP:
                        offenders.append((os.path.relpath(path, root), key))
        assert not offenders, offenders

    def test_with_overrides_round_trips(self):
        cfg = load_search_config().with_overrides(use_solver=True,
                                                  boundary_width=7,
                                                  population_size=UNSET)
        assert cfg.solver.use_solver is True
        assert cfg.domain.boundary_width == 7
        assert cfg.evolution.population_size == 6


class TestCoverageTotality:
    """Every config-scope parameter of every public entry point must live in
    exactly one group, so a parameter cannot be added later without being
    placed. Kwarg-only parameters are listed explicitly, which forces a
    deliberate decision rather than an accidental omission."""

    KWARG_ONLY = {
        # Catch-alls, not unplaced parameters: their contents are routed
        # through the loader (unknown names raise) or through the legacy shim.
        'self', 'config', 'kwargs', 'args', 'solver_kwargs', 'legacy_kwargs',
        # objects / tensors / callables -- cannot be serialised
        'director', 'gfunction', 'function_form', 'nds_method',
        'ndl_update_method', 'sorting_method', 'early_stopping_callback',
        'preprocessor_pipeline', 'preprocessor', 'derivs', 'pool',
        'population', 'optimizer', 'net', 'additional_tokens',
        'cached_token_tensors', 'system', 'boundary_conditions', 'system_file',
        # data
        'grids', 'entries', 'domain', 'data', 'ID', 'cache_id', 'grid',
        'variable_names', 'example_tensor', 'mem_for_cache_frac',
        'mem_for_cache_abs',
        # reporting, not search configuration
        'only_print', 'only_str', 'num', 'dimensions',
        # owned by the operator that reads it: the default comes from
        # LASSOSparsity.initial_sparsity_interval and this is a per-run
        # override of it, not a description of the search space
        # (see test_sparsity_interval.py).
        'eq_sparsity_interval',
    }

    def test_no_public_parameter_is_unplaced(self):
        import inspect
        from epde.interface.interface import EpdeSearch

        methods = ['__init__', 'createDomain', 'createTrajectory',
                   'set_preprocessor', 'create_pool', 'fit',
                   'set_moeadd_params', 'set_singleobjective_params',
                   'predict']
        unplaced = []
        for name in methods:
            method = getattr(EpdeSearch, name, None)
            if method is None:
                continue
            for param in inspect.signature(method).parameters:
                if param in self.KWARG_ONLY or param in KEY_GROUP:
                    continue
                unplaced.append('%s.%s' % (name, param))
        assert not unplaced, (
            'parameters belonging to no config group and not declared '
            'kwarg-only: %s' % unplaced)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_unknown_group_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown config group'):
            load_search_config({'evolutionary': {'population_size': 2}})

    def test_unknown_leaf_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown key'):
            load_search_config({'evolution': {'populaton_size': 2}})

    def test_unknown_kwarg_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown search-config parameter'):
            load_search_config(overrides={'use_picc': True})

    def test_removed_use_pic_is_rejected_as_a_config_key(self):
        with pytest.raises(ValueError, match='Unknown search-config parameter'):
            load_search_config(overrides={'use_pic': True})

    def test_comment_keys_are_ignored(self):
        cfg = load_search_config({'_comment': 'hi',
                                  'domain': {'_note': 'x', 'boundary_width': 3}})
        assert cfg.domain.boundary_width == 3

    def test_group_must_be_a_mapping(self):
        with pytest.raises(ValueError, match='must be a mapping'):
            load_search_config({'domain': 5})

    def test_bool_keys_are_type_checked(self):
        with pytest.raises(ValueError, match='must be a bool'):
            load_search_config(overrides={'use_solver': 'yes'})

    def test_bad_backend_is_rejected(self):
        with pytest.raises(ValueError, match='solver_backend'):
            load_search_config(overrides={'solver_backend': 'jax'})

    def test_bad_second_objective_is_rejected(self):
        with pytest.raises(ValueError, match='second_objective'):
            load_search_config(overrides={'second_objective': 'stability'})

    def test_solver_with_single_objective_is_rejected(self):
        """BaselineDirector.use_baseline ignores use_solver entirely, so this
        combination would silently run solver-free."""
        with pytest.raises(ValueError, match='multiobjective_mode=False'):
            load_search_config(overrides={'use_solver': True,
                                          'multiobjective_mode': False})

    def test_missing_file_is_reported(self):
        with pytest.raises(FileNotFoundError):
            load_search_config('no_such_config_file.json')


class TestSparsityRegistry:

    def test_names_resolve_to_classes(self):
        from epde.operators.common.sparsity import LASSOSparsity, VWSRSparsity
        assert resolve_sparsity('vwsr') is VWSRSparsity
        assert resolve_sparsity('lasso') is LASSOSparsity

    def test_default_is_vwsr(self):
        from epde.operators.common.sparsity import VWSRSparsity
        assert load_search_config().objectives.sparsity_cls is VWSRSparsity

    def test_class_passes_through(self):
        from epde.operators.common.sparsity import LASSOSparsity
        cfg = load_search_config(overrides={'sparsity_cls': LASSOSparsity})
        assert cfg.objectives.sparsity_cls is LASSOSparsity

    def test_none_passes_through(self):
        assert load_search_config(overrides={'sparsity_cls': None}
                                  ).objectives.sparsity_cls is None

    def test_bogus_name_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown sparsity'):
            resolve_sparsity('elasticnet')


class TestTokenRegistry:

    def test_declared_family_matches_a_direct_call(self):
        from epde.interface.prepared_tokens import GridTokens
        built = build_tokens([{'family': 'grid', 'labels': ['x_0'],
                               'dimensionality': 0, 'max_power': 2}])
        direct = GridTokens(labels=['x_0'], dimensionality=0, max_power=2)
        assert len(built) == 1
        assert type(built[0]) is GridTokens
        assert built[0].token_family.ftype == direct.token_family.ftype

    def test_trigonometric_family(self):
        from epde.interface.prepared_tokens import TrigonometricTokens
        built = build_tokens([{'family': 'trigonometric',
                               'freq': (0.999, 1.001), 'dimensionality': 0}])
        assert type(built[0]) is TrigonometricTokens

    def test_empty_spec_list(self):
        assert build_tokens([]) == []
        assert build_tokens(None) == []

    def test_unknown_family_names_the_alternatives(self):
        with pytest.raises(ValueError, match='Unknown token family'):
            build_tokens([{'family': 'cache_stored', 'token_labels': ['t']}])

    def test_data_bound_families_are_not_declarable(self):
        """CacheStoredTokens & co need tensors, so they must stay objects."""
        for name in ('cache_stored', 'custom', 'external_derivatives',
                     'control_var'):
            assert name not in TOKEN_REGISTRY

    def test_missing_family_key(self):
        with pytest.raises(ValueError, match="no 'family' key"):
            build_tokens([{'labels': ['x_0']}])

    def test_bad_constructor_args_are_reported(self):
        with pytest.raises(ValueError, match='Could not build token family'):
            build_tokens([{'family': 'grid', 'nonsense': 1}])

    def test_tokens_from_a_config_file(self, tmp_path):
        path = _write(tmp_path, {'search_space': {'tokens': [
            {'family': 'grid', 'labels': ['x_0'], 'dimensionality': 0}]}})
        specs = load_search_config(path).search_space.tokens
        assert len(specs) == 1
        assert build_tokens(specs)[0].__class__.__name__ == 'GridTokens'
