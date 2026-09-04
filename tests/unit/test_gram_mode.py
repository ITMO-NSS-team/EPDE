"""The Gram mode is derived from the instability metric, not configured.

``gram_mode`` selects what the sparsity keep-rule builds: the
varying-coefficient ``VaryingCoefSetup``, the axis-aligned ``GramSetup``, or
NOTHING. Which one is not a free choice: the keep-rule's L1 threshold is scaled
by the instability statistic (``PhysicsInformedLasso._keep_rule_scores``), the
same one the Pareto axis reads, so what gets built is whatever that estimator
is computed from. ``'vcoef'`` needs its mode solve, ``'cv'`` needs the
sliding-window stack, and the basis-free estimators ('chi2' -- the default --
'het', 'tile', 'survival') need neither and build neither.

So the mode is not an independent setting. It follows
``objectives.instability_metric``, and there is nothing to set out of step.
It is exposed as a derived PROPERTY of ``ObjectivesConfig`` -- not a field, so
it never reaches ``KEY_GROUP``, ``as_dict()`` or the JSON, and cannot be set.
"""

from pathlib import Path

import pytest

import epde
import epde.globals as global_var
from epde.interface.search_config import KEY_GROUP, load_search_config

EPDE_ROOT = Path(epde.__file__).parent


def _gram_mode(metric):
    return load_search_config(
        overrides={'instability_metric': metric}).objectives.gram_mode


class TestDerivation:

    @pytest.mark.parametrize('metric, expected', [
        (None, None),             # resolves to chi2, the shipped default
        ('chi2', None),           # basis-free: no Gram machinery at all
        ('het', None),
        ('tile', None),
        ('survival', None),
        ('vcoef', 'vcoef'),       # needs its own mode solve
        ('cv', 'axis'),           # needs the sliding-window stack
    ])
    def test_mode_follows_the_metric(self, metric, expected):
        assert _gram_mode(metric) == expected

    def test_the_default_builds_nothing(self):
        """chi2 scores from the active columns, so the default search builds
        no Gram machinery -- 27% off the ODE benchmark's wall time."""
        assert _gram_mode(None) is None

    def test_every_valid_metric_resolves(self):
        for metric in (None, 'vcoef', 'cv', 'survival', 'tile', 'het', 'chi2'):
            assert _gram_mode(metric) in ('vcoef', 'axis', None)

    def test_a_new_search_clears_the_basis_cache(self):
        """The vc basis resolution is per (grid_shape, main_var) AND per mode.
        Clearing it used to hang off set_instability_metric; it now hangs off
        init_caches, beside the other per-search cache resets -- the cache is
        runtime state, not a setting."""
        global_var.vc_modes_cache['stale'] = object()
        global_var.init_caches(set_grids=True)
        assert not global_var.vc_modes_cache


class TestNotConfigurable:

    def test_the_setter_is_gone(self):
        assert not hasattr(global_var, 'set_gram_config')

    def test_there_is_no_module_level_mode(self):
        assert not hasattr(global_var, 'gram_mode')

    def test_it_is_not_a_config_field(self):
        """A property, so it cannot be constructed, dumped or overridden."""
        from dataclasses import fields
        from epde.interface.search_config import ObjectivesConfig
        assert 'gram_mode' not in {f.name for f in fields(ObjectivesConfig)}
        assert 'gram_mode' not in load_search_config().as_dict()['objectives']

    def test_it_is_not_a_config_key(self):
        assert 'gram_mode' not in KEY_GROUP

    def test_setting_it_in_a_config_is_refused(self):
        with pytest.raises(ValueError, match='Unknown key'):
            load_search_config({'objectives': {'gram_mode': 'axis'}})

    def test_passing_it_to_the_constructor_is_refused(self):
        with pytest.raises(ValueError, match='Unknown search-config parameter'):
            epde.EpdeSearch(gram_mode='axis',
                            verbose_params={'show_iter_idx': False})

    def test_nothing_reads_a_stored_mode(self):
        """Both consumers must derive it from the config; a stored module
        attribute would reintroduce the drift the derivation removes."""
        offenders = []
        for path in EPDE_ROOT.rglob('*.py'):
            for lineno, line in enumerate(
                    path.read_text(encoding='utf-8', errors='ignore')
                        .splitlines(), 1):
                code = line.split('#')[0]
                if ('global_var.gram_mode' in code
                        or 'globals.gram_mode' in code):
                    offenders.append('%s:%s' % (path.name, lineno))
        assert not offenders, offenders


class TestConsumersAgree:
    """The pairing is only worth deriving if the two estimators really do read
    what their Gram writes -- pin the caches those branches depend on."""

    def test_vcoef_reads_the_varying_coefficient_cache(self):
        import inspect
        from epde.operators.common.objectives import Instability

        source = inspect.getsource(Instability.compute)
        assert '_cached_vc_score' in source

    def test_cv_reads_the_sliding_window_cache(self):
        import inspect
        from epde.operators.common.objectives import Instability

        source = inspect.getsource(Instability.compute)
        assert '_cached_sw_weights' in source

    def test_the_keep_rule_and_the_objective_share_the_estimator(self):
        """The whole point of the derivation: one statistic, both sides.

        This used to grep ``inspect.getsource(Instability.compute)`` for each
        estimator's ``__name__``. That could only ever see that the name was
        MENTIONED -- it passed for a dispatch that called the estimator with
        different arguments, and it broke as soon as the inline dict became a
        shared table even though the sharing had just become stronger. The
        two sides are now one object, so assert that.
        """
        from epde.operators.common.objectives import _BASIS_FREE_METRICS
        from epde.operators.common.sparsity import _KEEP_RULE_ESTIMATORS

        assert _KEEP_RULE_ESTIMATORS is _BASIS_FREE_METRICS

    def test_every_menu_metric_has_exactly_one_home(self):
        """No metric may be listed without an implementation, or implemented
        without being selectable -- the failure mode a per-side dispatch
        invites (a ``KeyError`` deep inside the sparsity step for a name the
        config happily accepted)."""
        from epde.interface.search_config import METRIC_MENUS
        from epde.operators.common.objectives import _BASIS_FREE_METRICS

        # vcoef and cv score from their own Gram setup, not the shared table.
        gram_backed = {'vcoef', 'cv'}
        menu = set(METRIC_MENUS['instability_metric'])
        assert gram_backed < menu
        assert menu - gram_backed == set(_BASIS_FREE_METRICS)
