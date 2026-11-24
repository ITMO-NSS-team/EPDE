"""Own the runtime bindings used by EPDE's procedural operator API per search.

Legacy operators and user hooks access the currently bound modules directly.
A public search operation activates its owner's bindings. Switching costs a
fixed number of references and never copies a tensor or a population.
"""
import random
import threading
import warnings
import weakref
from collections import OrderedDict
from functools import wraps

import numpy as np
import torch

_active = None
_lock = threading.RLock()
_calls = []
_MISSING = object()


def random_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def restore_random_state(states):
    python, numpy, torch_cpu = states
    random.setstate(python)
    np.random.set_state(numpy)
    torch.set_rng_state(torch_cpu)


def _bindings():
    import epde.globals as g
    from epde.interface import search_config
    from epde.operators.common import (fill_prune, invariant_terms, plain_fit, product_rule,
                                       right_part_selection, survival, system_redundancy)
    from epde.operators.utils.default_parameter_loader import EvolutionaryParams
    # A fresh binding for each mutable module-level run object. Initializers
    # already in EpdeSearch create the heavy objects once the run is active.
    return [
        (g, {**dict.fromkeys(('tensor_cache', 'grid_cache', 'initial_data_cache', 'samples_manager',
                              'history', 'verbose')), 'vc_modes_cache': dict,
             'control_nn': lambda: _MISSING, 'solution_guess_nn': lambda: _MISSING}),
        (search_config, {'_ACTIVE': None}), (EvolutionaryParams, {'_container_instances': dict}),
        (plain_fit, {'_SIBLINGS': dict, '_PLAIN_TOKENS': dict, '_ORDER': lambda: {'rng': None}}),
        (fill_prune, {'_LIBRARY': dict, '_STORES': dict, '_WARNED': set}),
        (product_rule, {'_GROUPS': OrderedDict}), (invariant_terms, {'_RULES': dict, '_REPORTED': set}),
        (system_redundancy, {'_REPORTED': set}), (survival, {'_PAIRED_GRAMS': list}),
        (right_part_selection, {'_MATING_SECTOR': lambda: {'value': None},
                                '_SYSTEM_HISTORY': list, '_RESAMPLE_SCOPE': list}),
    ]


class SearchRuntime:
    def __init__(self):
        self.bindings = {(obj, name): factory() if factory else None
                         for obj, specs in _bindings() for name, factory in specs.items()}
        self.remember_rng()

    def remember_rng(self):
        self.rng = random_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None

    def activate(self):
        global _active
        previous = _active() if _active else None
        if self is previous:
            return
        if previous is not None:
            if _calls and previous is _calls[-1]:
                previous.remember_rng()
            for obj, name in previous.bindings:
                previous.bindings[obj, name] = getattr(obj, name, _MISSING)
        for (obj, name), value in self.bindings.items():
            if value is _MISSING:
                if hasattr(obj, name):
                    delattr(obj, name)
            else:
                setattr(obj, name, value)
            self.bindings[obj, name] = None  # the live modules own these references until suspended
        restore_random_state(self.rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        _active = weakref.ref(self)


def search_operation(method):
    """Keep procedural calls within this public operation on the owning run."""
    @wraps(method)
    def bound(self, *args, **kwargs):
        with _lock:
            runtime = getattr(self, '_runtime', None)
            if runtime is None:
                return method(self, *args, **kwargs)
            if _calls and runtime is _calls[-1]:
                return method(self, *args, **kwargs)
            runtime.activate()
            _calls.append(runtime)
            try:
                with warnings.catch_warnings():
                    import epde.globals as g
                    if g.verbose is not None:
                        g.configure_warning_filters(g.verbose.show_warnings)
                    return method(self, *args, **kwargs)
            finally:
                runtime.remember_rng()
                _calls.pop()
                if _calls:
                    _calls[-1].activate()
    return bound


def search_constructor(method):
    initialize = search_operation(method)

    @wraps(method)
    def construct(self, *args, **kwargs):
        with _lock:
            previous = _active() if _active else None
            self._runtime = SearchRuntime()
            try:
                initialize(self, *args, **kwargs)
            except BaseException:
                if previous is not None:
                    previous.activate()
                raise
    return construct
