"""EPDE's preprocessing pipeline shared by discovery and signal checks.

Supplied derivative stacks follow EPDE's axis-major column order and are
used verbatim. Numeric preprocessing also returns the processed field:
when smoothing is enabled, powers of the field must use that same field.
"""
import numpy as np


def derivative_orders(search, ndim):
    from epde.interface.search_config import load_search_config
    orders = load_search_config(search).preprocessing.max_deriv_order
    if np.isscalar(orders):
        orders = [orders] * ndim
    else:
        orders = list(orders)
    if len(orders) != ndim or any(int(k) != k or k < 0 for k in orders):
        raise ValueError(f'max_deriv_order must contain {ndim} nonnegative integers: {orders}')
    return [int(k) for k in orders]


def make_preprocessor(search):
    """Build the same native EPDE pipeline as EpdeSearch.set_preprocessor."""
    from epde.interface.search_config import load_search_config
    from epde.preprocessing.preprocessor import ConcretePrepBuilder
    from epde.preprocessing.preprocessor_setups import PreprocessorSetup
    prep = load_search_config(search).preprocessing
    setup = PreprocessorSetup()
    setup.builder = ConcretePrepBuilder()
    builders = {'FD': setup.build_FD_preprocessing,
                'poly': setup.build_poly_diff_preprocessing,
                'spectral': setup.build_spectral_preprocessing,
                'ANN': setup.build_ANN_preprocessing}
    if prep.default_preprocessor_type not in builders:
        raise ValueError(f'Unsupported EPDE preprocessor: {prep.default_preprocessor_type}')
    builders[prep.default_preprocessor_type](**(prep.preprocessor_kwargs or {}))
    return setup.builder.prep_pipeline


def validate_data(problem, data):
    """Fail before building a search when a loader or override is unusable."""
    if list(data) != problem.variables:
        raise ValueError(f'data variables must be {problem.variables}, got {list(data)}')
    if any(np.shape(grid) != problem.shape for grid in problem.grids):
        raise ValueError('grid shapes must match the data shape')
    for var, values in data.items():
        if np.shape(values) != problem.shape:
            raise ValueError(f'{var}: shape must be {problem.shape}, got {np.shape(values)}')
        if problem.derivs is not None and not np.array_equal(values, problem.data[var]):
            raise ValueError(f'{var}: replaced fields cannot reuse pre-computed derivatives; '
                             'provide a new consistent Problem')
        if not np.isfinite(values).all():
            raise ValueError(f'{var}: data contains NaN or infinity; crop a finite region before discovery')


def validate_derivatives(problem, data, search):
    """Validate the axis-major schema before either checking or searching."""
    orders = derivative_orders(search, len(problem.grids))
    if problem.derivs is None:
        return
    if problem.deriv_orders is None or list(problem.deriv_orders) != orders:
        raise ValueError(f'pre-computed derivative orders {problem.deriv_orders} '
                         f'do not match configured orders {orders}')
    if len(problem.derivs) != len(data):
        raise ValueError('one pre-computed derivative stack is required per variable')
    for (var, values), stack in zip(data.items(), problem.derivs):
        stack = np.asarray(stack)
        expected = (values.size, sum(orders))
        if stack.shape != expected or not np.isfinite(stack).all():
            raise ValueError(f'{var}: derivatives must be finite with shape {expected}, got {stack.shape}')


def prepare_data(problem, data, search):
    """Return (processed fields, derivative stacks) without an EPDE search."""
    validate_data(problem, data)
    validate_derivatives(problem, data, search)
    orders = derivative_orders(search, len(problem.grids))
    supplied = problem.derivs
    if supplied is not None and len(supplied) != len(data):
        raise ValueError('one pre-computed derivative stack is required per variable')
    pipeline = make_preprocessor(search) if supplied is None else None
    fields, stacks = {}, {}
    for i, (var, values) in enumerate(data.items()):
        if supplied is None:
            field, stack = pipeline.run(values.copy(), grid=list(problem.grids), max_order=orders)
        else:
            field, stack = values, np.asarray(supplied[i])
        stack = np.asarray(stack)
        expected = (values.size, sum(orders))
        if stack.shape != expected or not np.isfinite(stack).all():
            raise ValueError(f'{var}: derivatives must be finite with shape {expected}, got {stack.shape}')
        fields[var], stacks[var] = field, stack
    return fields, stacks
