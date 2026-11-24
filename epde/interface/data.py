"""Data accepted by discovery, independently of a search or tensor cache."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from math import prod
from typing import Mapping

import numpy as np


def derivative_orders(order, ndim):
    values = [order] * ndim if np.isscalar(order) else list(order)
    if len(values) != ndim or any(not isinstance(n, (int, np.integer)) or
                                  isinstance(n, (bool, np.bool_)) or n < 0 for n in values):
        raise ValueError(f'max_deriv_order must contain {ndim} nonnegative integers: {values}')
    return tuple(map(int, values))


def check_supplied_orders(declared, requested, ndim):
    if declared is None or tuple(declared) != derivative_orders(requested, ndim):
        raise ValueError(f'Supplied derivative orders {declared} do not match requested orders {requested}')


def _array(name, values):
    values = np.asarray(values)
    if values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
        raise ValueError(f'{name} must be finite real values (NaN and infinity are invalid)')
    return values


@dataclass
class TrajectoryData:
    """One sample. Coordinates are axis vectors or full coordinate grids in axis order.

    Derivative columns follow EPDE's axis-major order. External inputs are
    aligned observed arrays, bound by label to declared stored token families.
    """
    coordinates: Mapping
    fields: Mapping
    derivatives: Mapping = field(default_factory=dict)
    derivative_orders: tuple | None = None
    external_inputs: Mapping = field(default_factory=dict)
    trajectory_id: str | int | None = None

    def __post_init__(self):
        if isinstance(self.trajectory_id, np.integer):
            self.trajectory_id = int(self.trajectory_id)
        if self.trajectory_id is not None and not isinstance(self.trajectory_id, (int, str)):
            raise TypeError('trajectory_id must be an integer or string')
        if not self.coordinates or not self.fields:
            raise ValueError('A trajectory needs named coordinates and observed fields')
        for name in ('coordinates', 'fields', 'derivatives', 'external_inputs'):
            values = getattr(self, name)
            if any(not isinstance(key, str) or not key for key in values):
                raise ValueError(f'{name} needs nonempty string names')
            label = ' derivatives' if name == 'derivatives' else ''
            setattr(self, name, {k: _array(k + label, v) for k, v in values.items()})
        shape = next(iter(self.fields.values())).shape
        if len(shape) != len(self.coordinates) or not all(shape):
            raise ValueError('Field dimensions must match the declared coordinate axes')
        if self.fields.keys() & self.external_inputs.keys():
            raise ValueError('Field and external input names must be distinct')
        for name, array in (*self.fields.items(), *self.external_inputs.items()):
            if array.shape != shape:
                raise ValueError(f'{name}: expected shape {shape}, got {array.shape}')
        for axis, (name, grid) in enumerate(self.coordinates.items()):
            if grid.shape not in ((shape[axis],), shape):
                raise ValueError(f'{name}: expected an axis vector or grid of shape {shape}')
        if self.derivatives or self.derivative_orders is not None:
            if self.derivatives.keys() != self.fields.keys() or self.derivative_orders is None:
                raise ValueError('Supply derivative orders and one derivative stack for every field')
            self.derivative_orders = derivative_orders(self.derivative_orders, len(shape))
            expected = (prod(shape), sum(self.derivative_orders))
            for name, stack in self.derivatives.items():
                if stack.shape != expected:
                    raise ValueError(f'{name}: derivatives must be finite with shape {expected}, got {stack.shape}')

    @property
    def grids(self):
        shape = next(iter(self.fields.values())).shape
        grids = []
        for axis, grid in enumerate(self.coordinates.values()):
            if grid.shape != shape:
                view = [1] * len(shape)
                view[axis] = shape[axis]
                grid = np.broadcast_to(grid.reshape(view), shape)
            grids.append(grid)
        return grids

    def check_orders(self, order):
        if self.derivatives:
            check_supplied_orders(self.derivative_orders, order, len(self.coordinates))


def read_trajectories_csv(path, *, coordinates, fields, trajectory_id=None,
                          derivatives=None, derivative_orders=None, external_inputs=None):
    """Read long-format CSV. Mappings associate EPDE names with header names.

    A list of names is shorthand for an identity mapping. For derivatives,
    map each observed field to its ordered list of column names. With multiple
    trajectory IDs, samples are returned in lexical ID order.
    """
    def columns(names):
        if isinstance(names, str):
            names = [names]
        result = dict(names) if isinstance(names, Mapping) else {key: key for key in names}
        if len(names) != len(result) or len(set(result.values())) != len(result):
            raise ValueError('Column mappings must have distinct names within each group')
        return result

    coordinates, fields = columns(coordinates), columns(fields)
    external = columns(external_inputs or {})
    derivs = {k: [v] if isinstance(v, str) else list(v) for k, v in (derivatives or {}).items()}
    required = set(coordinates.values()) | set(fields.values()) | set(external.values())
    required.update(c for names in derivs.values() for c in names)
    if not coordinates or not fields:
        raise ValueError('Declare coordinate and field columns')
    groups = {}
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        missing = (required | ({trajectory_id} if trajectory_id else set())) - set(headers)
        if missing or len(set(headers)) != len(headers):
            raise ValueError(f'Invalid CSV headers; missing {sorted(missing)} or duplicate column names')
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f'Unexpected extra columns on CSV line {line}')
            key = row[trajectory_id] if trajectory_id else None
            try:
                numeric = {c: float(row[c]) for c in required}
                if not all(np.isfinite(list(numeric.values()))) or (trajectory_id and not key):
                    raise ValueError()
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Invalid data on CSV line {line}') from exc
            groups.setdefault(key, []).append(numeric)
    if not groups:
        raise ValueError('CSV contains no observations')
    result = []
    for key in sorted(groups):
        rows = groups[key]
        points = np.array([[r[c] for c in coordinates.values()] for r in rows])
        axes = [np.unique(v) for v in points.T]
        shape = tuple(map(len, axes))
        if len(np.unique(points, axis=0)) != len(rows):
            raise ValueError(f'Duplicate coordinate tuple in trajectory {key!r}')
        if prod(shape) != len(rows):
            raise ValueError(f'Incomplete Cartesian grid in trajectory {key!r}')
        order = np.lexsort(points.T[::-1])

        def values(column):
            return np.array([row[column] for row in rows])[order]

        result.append(TrajectoryData(dict(zip(coordinates, axes)),
                      {k: values(c).reshape(shape) for k, c in fields.items()},
                      {k: np.column_stack([values(c) for c in cols]) if cols else np.empty((len(rows), 0))
                       for k, cols in derivs.items()},
                      derivative_orders, {k: values(c).reshape(shape) for k, c in external.items()}, key))
    return result


def attach_inputs(search, data, families, order):
    """Adapt independent observations to the existing trajectory evaluator."""
    from epde.cache.cache_refactored import Cache
    from epde.interface.prepared_tokens import CacheStoredTokens
    from epde.structure.domain import Trajectory
    import epde.globals as runtime

    samples = (data,) if isinstance(data, Trajectory) else data
    if isinstance(samples, (tuple, list)) and samples and all(isinstance(d, Trajectory) for d in samples):
        if all(d._cache is runtime.tensor_cache for d in samples):
            return list(samples)
        # A legacy trajectory can be handed to a second search. Its prepared
        # tensors and cached coordinate grids belong to the first search.
        from epde.interface.search_config import active_config
        search._check_evaluation_lattice(samples, active_config().search_space)
        runtime.grid_cache = Cache()
        runtime.tensor_cache = Cache()
        search._cache_mem_set = False
        domains, rebound = {}, []
        for sample in samples:
            source = sample._domain
            if source not in domains:
                domain = search.createDomain(sample.grids(), time_axis=source._time_axis, ID=source.ID,
                                             boundary_width='auto' if source.boundary_auto else source.boundary_width,
                                             gfunction=source.g_func_form)[1]
                domains[source] = domain
            fields = {k: sample.raw_field(k) for k in sample.variable_names}
            rebound.append(search.createTrajectory(fields, domains[source], cache_id=sample.ID,
                           derivs=sample._derivs, cached_token_tensors=sample._additional_cached_tokens,
                           preprocessor=sample._preprocessor_pipeline)[1])
        return rebound
    if isinstance(data, TrajectoryData):
        data = [data]
    if not isinstance(data, (tuple, list)) or not any(isinstance(d, TrajectoryData) for d in data):
        return data
    if not all(isinstance(d, TrajectoryData) for d in data):
        raise TypeError('Use TrajectoryData for every trajectory in a discovery request')
    first = data[0]
    ids = [i if d.trajectory_id is None else d.trajectory_id for i, d in enumerate(data)]
    if len(ids) != len(set(ids)):
        raise ValueError('Trajectory IDs must be unique within a run')
    for d in data:
        d.check_orders(order)
        if (list(d.fields) != list(first.fields) or list(d.coordinates) != list(first.coordinates)
                or d.external_inputs.keys() != first.external_inputs.keys()):
            raise ValueError('Trajectories must share ordered field and axis names and external input labels')
    stored = {}
    for family in families:
        if isinstance(family, CacheStoredTokens):
            for label in family.token_family.declared_tokens:
                if label in first.external_inputs:
                    if family.token_tensors or label in stored:
                        raise ValueError(f'Ambiguous external input binding for {label!r}')
                    stored[label] = family
    for label in first.external_inputs:
        if label in stored:
            continue
        family = CacheStoredTokens(label, [label], params_ranges={'power': (1, 1)}, meaningful=True)
        families.append(family)
        stored[label] = family
    # Old trajectories keep their own grid cache. Reusing an ID in a new
    # request must not replace their coordinates or read stale ones.
    runtime.grid_cache = Cache()
    runtime.tensor_cache = Cache()
    search._cache_mem_set = False
    trajectories = []
    for key, d in zip(ids, data):
        tensors = {}
        for label, values in d.external_inputs.items():
            tensors.setdefault(stored[label], {})[label] = values
        domain = search.createDomain(d.grids, ID=key)[1]
        _, trajectory = search.createTrajectory(d.fields, domain, cache_id=key,
                                                derivs=d.derivatives or None, cached_token_tensors=tensors)
        trajectory.axis_names = tuple(d.coordinates)
        trajectories.append(trajectory)
    return trajectories
