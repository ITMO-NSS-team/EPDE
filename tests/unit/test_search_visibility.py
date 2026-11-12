"""States of the plain-mode / NSGA-II search that EPDE reports as ``EPDEUsageWarning`` (the category that
``init_verbose`` leaves enabled) or as a loop-stats record, instead of passing over them silently.

* the system-redundancy test that is off (no axis ranges, a token it cannot differentiate) warns once per
  reason and fit, and ``plain_fit.release`` re-arms it;
* the fill-and-prune move that falls back for a reason that holds for the whole equation warns once per fit,
  and every fallback is a loop-stats record;
* end to end, a pool too small for its population: slots that hold copies, an equation shorter than its
  term budget;
* the search-space collapse of the initial population;
* an ``early_stopping_callback`` that raises.
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde import _loop_stats
from epde.operators.common import fill_prune as FP
from epde.operators.common import plain_fit as PF
from epde.operators.common import system_redundancy as SR


def _recorded(call):
    """The EPDEUsageWarning messages ``call()`` raises (every warning shown, none filtered)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        call()
    return [str(w.message) for w in caught if issubclass(w.category, global_var.EPDEUsageWarning)]


# ---------------------------------------------------------------------------
# The redundancy test that is off.
# ---------------------------------------------------------------------------

class TestRedundancyOff:

    def test_warns_once_per_fit_and_release_rearms(self, monkeypatch):
        monkeypatch.setattr(SR, '_axis_ranges', lambda: None)
        SR.reset()
        try:
            equations = [object(), object()]
            first = _recorded(lambda: SR.redundant_equations(equations))
            again = _recorded(lambda: SR.redundant_equations(equations))
            PF.release()
            after_release = _recorded(lambda: SR.redundant_equations(equations))
        finally:
            SR.reset()
        assert len(first) == 1 and 'axis ranges' in first[0]
        assert again == []
        assert len(after_release) == 1

    def test_the_result_stays_empty(self, monkeypatch):
        monkeypatch.setattr(SR, '_axis_ranges', lambda: None)
        SR.reset()
        try:
            with pytest.warns(global_var.EPDEUsageWarning, match='follows from the other equations'):
                assert SR.redundant_equations([object(), object()]) == []
        finally:
            SR.reset()

    def test_a_single_equation_is_silent(self, monkeypatch):
        monkeypatch.setattr(SR, '_axis_ranges', lambda: None)
        SR.reset()
        assert _recorded(lambda: SR.redundant_equations([object()])) == []


# ---------------------------------------------------------------------------
# The fill-and-prune fallbacks.
# ---------------------------------------------------------------------------

def _raising(reason):
    def attempt(*args, **kwargs):
        raise FP._Fallback(reason)
    return attempt


def _fill(equation):
    np.random.seed(0)
    return FP.fill(equation, None, [], None, {'fill_probability': 1.0})


class TestFillFallback:

    def test_a_whole_equation_reason_warns_once_per_fit(self, monkeypatch):
        monkeypatch.setattr(FP, '_attempt', _raising('target_not_in_library'))
        equation = SimpleNamespace(main_var_to_explain='u')
        FP.reset()
        try:
            results = []
            first = _recorded(lambda: results.append(_fill(equation)))
            again = _recorded(lambda: results.append(_fill(equation)))
            FP.reset()
            after_reset = _recorded(lambda: results.append(_fill(equation)))
        finally:
            FP.reset()
        assert results == [False, False, False]
        assert len(first) == 1 and "'u'" in first[0] and 'not in the library' in first[0]
        assert again == []
        assert len(after_reset) == 1

    def test_the_reasons_are_told_apart_per_variable(self, monkeypatch):
        monkeypatch.setattr(FP, '_attempt', _raising('no_candidates'))
        FP.reset()
        try:
            messages = _recorded(lambda: [_fill(SimpleNamespace(main_var_to_explain=var)) for var in ('u', 'v', 'u')])
        finally:
            FP.reset()
        assert len(messages) == 2

    def test_a_per_call_reason_is_silent(self, monkeypatch):
        monkeypatch.setattr(FP, '_attempt', _raising('same_structure'))
        FP.reset()
        try:
            results = []
            messages = _recorded(lambda: results.append(_fill(SimpleNamespace(main_var_to_explain='u'))))
        finally:
            FP.reset()
        assert results == [False] and messages == []

    def test_every_fallback_is_a_loop_stats_record(self, monkeypatch):
        monkeypatch.setattr(_loop_stats, '_ENABLED', True)
        _loop_stats.reset()
        monkeypatch.setattr(FP, '_attempt', _raising('target_not_in_library'))
        FP.reset()
        try:
            _recorded(lambda: _fill(SimpleNamespace(main_var_to_explain='u')))
            monkeypatch.setattr(FP, '_attempt', _raising('same_structure'))
            _recorded(lambda: _fill(SimpleNamespace(main_var_to_explain='u')))
            entries = {site: bucket['entries'] for site, bucket in _loop_stats._stats.items()}
        finally:
            FP.reset()
            _loop_stats.reset()
        assert entries['PlainMutation.fill_fallback.target_not_in_library'] == 1
        assert entries['PlainMutation.fill_fallback.same_structure'] == 1

    def test_every_whole_equation_reason_has_a_text(self):
        assert set(FP._FALLBACK_TEXTS) == {'target_not_in_library', 'target_column_unusable', 'no_candidates',
                                           'k_below_1', 'beam_linalg'}


# ---------------------------------------------------------------------------
# End to end.
# ---------------------------------------------------------------------------

def _search(population_size=8, epochs=2, terms=4, **extra):
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, engine='nsga2', system_rps_mode='plain',
                             discrepancy_metric='wape', instability_metric='dyad',
                             equation_terms_max_number=terms, equation_factors_max_number=1,
                             data_fun_pow=1, population_size=population_size, training_epochs=epochs, **extra)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _trajectory(search):
    t = np.linspace(0, 4 * np.pi, 300)
    domain = search.createDomain(t, ID=0)[1]
    return search.createTrajectory({'u': np.sin(t)}, domain, cache_id=0)[1]


def _fit_recorded(search):
    """The EPDEUsageWarning messages of one fit of ``search`` on a one-variable trajectory."""
    trajectory = _trajectory(search)
    np.random.seed(2)
    return _recorded(lambda: search.fit(data=[trajectory], max_deriv_order=(1,)))


class TestTinyPool:
    """One variable, first derivatives, one factor per term: a pool of a handful of terms, fewer distinct
    equations than the population and fewer distinct terms than the term budget."""

    def test_copies_and_a_shorter_equation_are_reported(self):
        search = _search(population_size=12, terms=6)
        try:
            messages = _fit_recorded(search)
        finally:
            global_var.samples_manager.reset()
        assert any('distinct' in m and 'slots hold copies' in m for m in messages), messages
        assert any('shorter equation' in m for m in messages), messages


class TestCollapse:

    def test_a_pool_of_degenerate_candidates_warns(self, monkeypatch):
        from epde.operators.multiobjective import moeadd_specific
        monkeypatch.setattr(moeadd_specific, 'has_degenerate_equation', lambda system: True)
        search = _search(operators={'InitialParetoLevelSorting': {'uniqueness_attempt_limit': 2}})
        try:
            messages = _fit_recorded(search)
        finally:
            global_var.samples_manager.reset()
        assert any('collapse' in m for m in messages), messages


class TestCallback:

    def test_a_raising_callback_warns_and_the_fit_goes_on(self):
        calls = []

        def callback(snapshot, epoch):
            calls.append(epoch)
            raise RuntimeError('boom')

        search = _search(epochs=2)
        search.set_moeadd_params(early_stopping_callback=callback)
        try:
            messages = _fit_recorded(search)
            history = len(search.pareto_history)
        finally:
            global_var.samples_manager.reset()
        assert any('early_stopping_callback' in m and 'boom' in m for m in messages), messages
        assert calls == [0, 1] and history == 2
