"""EPDE's preprocessing pipeline shared by discovery and signal checks.

Supplied derivative stacks follow EPDE's axis-major column order and are
used verbatim. Numeric preprocessing also returns the processed field:
when smoothing is enabled, powers of the field must use that same field.
"""


def derivative_orders(search, ndim):
    from epde.interface.search_config import load_search_config
    from epde.interface.data import derivative_orders as expand_orders
    return list(expand_orders(load_search_config(search).preprocessing.max_deriv_order, ndim))


def make_preprocessor(search):
    from epde.interface.search_config import load_search_config
    from epde.preprocessing.factory import make_preprocessor as build
    cfg = load_search_config(search)
    return build(cfg.preprocessing.default_preprocessor_type, cfg.preprocessing.preprocessor_kwargs,
                 cfg.domain.time_axis)


def validate_data(problem, data):
    """Fail before building a search when a loader or override is unusable."""
    problem.trajectory_data(data)


def validate_derivatives(problem, data, search):
    problem.trajectory_data(data, derivative_orders(search, len(problem.grids)))


def prepare_data(problem, data, search):
    """Return (processed fields, derivative stacks) without an EPDE search."""
    from epde.preprocessing.preprocessor import prepare_field
    orders = derivative_orders(search, len(problem.grids))
    sample = problem.trajectory_data(data, orders)
    pipeline = None if sample.derivatives else make_preprocessor(search)
    fields, stacks = {}, {}
    grids = sample.grids
    for var, observed in sample.fields.items():
        fields[var], stacks[var] = prepare_field(var, observed, grids, orders, pipeline,
                                                 supplied=sample.derivatives.get(var))
    return fields, stacks
