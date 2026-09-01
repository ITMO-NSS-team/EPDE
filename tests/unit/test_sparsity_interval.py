"""The sparsity interval belongs to the operator that reads it.

``eq_sparsity_interval`` seeds the per-equation ``('sparsity', var)``
metaparameter, which is the ``alpha`` of the legacy LASSO estimator. It is an
*initial* range only -- metaparameter mutation and crossover move alpha outside
it during the search -- and ``LASSOSparsity.apply`` is the only reader of the
value in the tree. So it is a parameter of that estimator, not a description of
the space of equations being searched, and it is no longer a ``search_space``
config key: the default lives on the class.

These tests pin the three claims that justify the placement: the class owns the
historical default, the default (VWSR) pipeline declares that it ignores the
value, and nothing else reads it.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import epde
from epde.interface.search_config import (KEY_GROUP, SearchSpaceConfig,
                                          load_search_config,
                                          sparsity_settings)
from epde.operators.common.sparsity import (LASSOSparsity, VWSRSparsity,
                                            build_sparsity_operator,
                                            initial_sparsity_interval)

EPDE_ROOT = Path(epde.__file__).parent


class TestOwnership:

    def test_lasso_owns_the_historical_default(self):
        """(1e-4, 2.5) -- what ``fit``'s signature used to hardcode."""
        assert LASSOSparsity.initial_sparsity_interval == (1e-4, 2.5)

    def test_vwsr_declares_that_it_ignores_the_value(self):
        lo, hi = VWSRSparsity.initial_sparsity_interval
        assert lo == hi == 1.0

    def test_helper_reads_the_class(self):
        assert initial_sparsity_interval(LASSOSparsity) == (1e-4, 2.5)
        assert initial_sparsity_interval(VWSRSparsity) == (1.0, 1.0)

    def test_unknown_operator_gets_the_neutral_interval(self):
        class CustomSparsity:
            pass

        assert initial_sparsity_interval(CustomSparsity) == (1.0, 1.0)
        assert initial_sparsity_interval(None) == (1.0, 1.0)

    def test_the_neutral_value_matches_the_metaparameter_default(self):
        """1.0 is what an SoEq carries when nobody seeds it."""
        from epde.structure.main_structures import             _DEFAULT_EQUATION_METAPARAMETERS as defaults

        assert defaults['sparsity']['value'] == 1.0


class TestNotASearchSpaceSetting:

    def test_it_is_not_a_config_key(self):
        assert 'eq_sparsity_interval' not in KEY_GROUP

    def test_it_is_not_a_search_space_field(self):
        from dataclasses import fields

        assert 'eq_sparsity_interval' not in [f.name for f in
                                              fields(SearchSpaceConfig)]

    def test_setting_it_in_a_config_is_refused(self):
        with pytest.raises(ValueError, match='Unknown key'):
            load_search_config({'search_space':
                                {'eq_sparsity_interval': [1e-3, 1.0]}})

    def test_passing_it_to_the_constructor_is_refused(self):
        with pytest.raises(ValueError, match='Unknown search-config parameter'):
            epde.EpdeSearch(eq_sparsity_interval=(1e-3, 1.0),
                            verbose_params={'show_iter_idx': False})


class TestSoleReader:
    """If a second consumer of the metaparameter ever appears, the value stops
    being one operator's parameter and this placement stops being right."""

    def _metaparameter_lines(self):
        pattern = re.compile(r"metaparameters\s*\[\s*\(\s*['\"]sparsity['\"]")
        for path in EPDE_ROOT.rglob('*.py'):
            for lineno, line in enumerate(
                    path.read_text(encoding='utf-8', errors='ignore')
                        .splitlines(), 1):
                if pattern.search(line) and not line.lstrip().startswith('#'):
                    yield path, lineno, line

    def test_only_the_lasso_operator_reads_the_value(self):
        readers = [(path, lineno) for path, lineno, line
                   in self._metaparameter_lines()
                   if "['value']" in line and '=' not in line.split(']')[-1]]
        assert readers, 'the reader disappeared -- did the seeding change?'
        for path, lineno in readers:
            assert path.name == 'sparsity.py', '%s:%s' % (path, lineno)

    def test_the_writers_are_the_population_constructors(self):
        writers = {path.name for path, _, line in self._metaparameter_lines()
                   if line.lstrip().startswith('metaparameters[')}
        assert writers <= {'population_constr.py', 'equation_translator.py'}


class TestSparsityKwargs:
    """``objectives.sparsity_kwargs`` is the config home for it: a map that
    travels with ``sparsity_cls`` and configures that operator."""

    def test_the_group_carries_an_empty_map_by_default(self):
        assert load_search_config().objectives.sparsity_kwargs == {}

    def test_it_configures_the_chosen_operator(self):
        cfg = load_search_config({'objectives': {
            'sparsity_cls': 'lasso',
            'sparsity_kwargs': {'initial_sparsity_interval': [1e-3, 1.0]}}})
        assert cfg.objectives.sparsity_kwargs == {
            'initial_sparsity_interval': (1e-3, 1.0)}      # list -> tuple

    def test_a_kwarg_overrides_a_file(self):
        cfg = load_search_config(
            {'objectives': {'sparsity_cls': 'lasso',
                            'sparsity_kwargs': {
                                'initial_sparsity_interval': [1e-3, 1.0]}}},
            overrides={'sparsity_kwargs': {
                'initial_sparsity_interval': (1e-6, 9.0)}})
        assert cfg.objectives.sparsity_kwargs['initial_sparsity_interval']             == (1e-6, 9.0)

    def test_an_unknown_setting_is_refused(self):
        """A typo would otherwise sit on the operator doing nothing -- the
        failure this config layer exists to remove."""
        with pytest.raises(ValueError, match='initial_sparsity_interval'):
            load_search_config({'objectives': {
                'sparsity_kwargs': {'initial_sparsity_intervals': [1, 2]}}})

    def test_the_error_names_the_operator(self):
        with pytest.raises(ValueError, match='LASSOSparsity'):
            load_search_config({'objectives': {
                'sparsity_cls': 'lasso',
                'sparsity_kwargs': {'alpha': 0.1}}})

    def test_the_settable_names_come_from_the_class(self):
        assert 'initial_sparsity_interval' in sparsity_settings(LASSOSparsity)
        assert 'key' not in sparsity_settings(LASSOSparsity)

    def test_the_builder_applies_them_to_the_instance(self):
        operator = build_sparsity_operator(
            LASSOSparsity, {'initial_sparsity_interval': (1e-3, 1.0)})
        assert isinstance(operator, LASSOSparsity)
        assert operator.initial_sparsity_interval == (1e-3, 1.0)
        assert LASSOSparsity.initial_sparsity_interval == (1e-4, 2.5)

    def test_the_builder_defaults_to_vwsr(self):
        assert isinstance(build_sparsity_operator(), VWSRSparsity)

    @pytest.mark.parametrize('director_cls, extra', [
        ('epde.optimizers.moeadd.strategy:MOEADDDirector', {}),
        ('epde.optimizers.single_criterion.strategy:BaselineDirector', {}),
    ])
    def test_both_directors_accept_the_map(self, director_cls, extra):
        """The single-objective branch used to hardcode VWSRSparsity, so
        sparsity_cls was silently ignored there."""
        import importlib
        import inspect

        module, name = director_cls.split(':')
        cls = getattr(importlib.import_module(module), name)
        params = inspect.signature(cls.use_baseline).parameters
        assert 'sparsity_cls' in params
        assert 'sparsity_kwargs' in params


def _search(**kwargs):
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, **kwargs)
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    return search


def _trajectory(search, cache_id):
    grid = np.linspace(0, 4 * np.pi, 60)
    domain = search.createDomain(grid, boundary_width=5, ID=cache_id)[1]
    return search.createTrajectory({'u': np.sin(grid)}, domain,
                                   cache_id=cache_id)[1]


class TestFitResolution:
    """``fit`` takes its default from the sparsity operator in use, and says so
    when the operator in use cannot act on an explicitly passed one."""

    def _captured_interval(self, monkeypatch, search, **fit_kwargs):
        captured = {}

        def recorder(self, terms, factors, interval, optimizer=None,
                     population=None):
            captured['interval'] = interval

        monkeypatch.setattr(type(search), '_run_optimization', recorder)
        search.fit(data=[_trajectory(search, 70)], max_deriv_order=(2,),
                   **fit_kwargs)
        return captured['interval']

    def test_default_comes_from_the_lasso_class(self, monkeypatch):
        search = _search(sparsity_cls='lasso')
        assert self._captured_interval(monkeypatch, search) == (1e-4, 2.5)

    def test_a_configured_interval_is_used(self, monkeypatch):
        search = _search(sparsity_cls='lasso',
                         sparsity_kwargs={'initial_sparsity_interval':
                                          (1e-3, 1.0)})
        assert self._captured_interval(monkeypatch, search) == (1e-3, 1.0)

    def test_the_fit_argument_still_wins_over_the_config(self, monkeypatch):
        search = _search(sparsity_cls='lasso',
                         sparsity_kwargs={'initial_sparsity_interval':
                                          (1e-3, 1.0)})
        got = self._captured_interval(monkeypatch, search,
                                      eq_sparsity_interval=(1e-6, 9.0))
        assert got == (1e-6, 9.0)

    def test_default_is_neutral_under_vwsr(self, monkeypatch):
        search = _search()
        assert self._captured_interval(monkeypatch, search) == (1.0, 1.0)

    def test_an_explicit_interval_still_wins_for_lasso(self, monkeypatch):
        search = _search(sparsity_cls='lasso')
        got = self._captured_interval(monkeypatch, search,
                                      eq_sparsity_interval=(1e-3, 1.0))
        assert got == (1e-3, 1.0)

    def test_an_explicit_interval_warns_under_vwsr(self):
        search = _search()
        with pytest.warns(UserWarning, match='VWSRSparsity never reads it'):
            with pytest.raises(ValueError):      # no data, no pool
                search.fit(eq_sparsity_interval=(1e-3, 1.0))

    def test_the_warning_points_at_the_caller(self):
        """``fit`` is wrapped by @_loop_stats.timed, so the naive stacklevel
        blames _loop_stats.py -- and a diagnostic filed against a file the
        reader does not own is not a diagnostic."""
        search = _search()
        with pytest.warns(UserWarning) as caught:
            with pytest.raises(ValueError):
                search.fit(eq_sparsity_interval=(1e-3, 1.0))
        mine = [w for w in caught if 'eq_sparsity_interval' in str(w.message)]
        assert mine, 'no warning emitted'
        assert mine[0].filename == __file__, mine[0].filename

    def test_no_warning_when_the_operator_reads_it(self):
        import warnings

        search = _search(sparsity_cls='lasso')
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            with pytest.raises(ValueError):
                search.fit(eq_sparsity_interval=(1e-3, 1.0))
