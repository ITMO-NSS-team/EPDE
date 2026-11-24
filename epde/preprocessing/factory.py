"""Build the configured EPDE preprocessor for discovery and diagnostics."""
import warnings

import epde.globals as global_var
from .preprocessor import ConcretePrepBuilder
from .preprocessor_setups import PreprocessorSetup


def make_preprocessor(default_preprocessor_type, preprocessor_kwargs, time_axis):
    setup = PreprocessorSetup()
    builder = ConcretePrepBuilder()
    setup.builder = builder

    if default_preprocessor_type == 'ANN':
        setup.build_ANN_preprocessing(**preprocessor_kwargs)
    elif default_preprocessor_type == 'poly':
        setup.build_poly_diff_preprocessing(**preprocessor_kwargs)
    elif default_preprocessor_type == 'spectral':
        setup.build_spectral_preprocessing(**preprocessor_kwargs)
    elif default_preprocessor_type == 'FD':
        setup.build_FD_preprocessing(
            **{'time_axis': time_axis, **preprocessor_kwargs})
    elif default_preprocessor_type == 'forward':
        warnings.warn("default_preprocessor_type='forward' is 'FD' with scheme='forward': use "
                      "default_preprocessor_type='FD', preprocessor_kwargs={'scheme': 'forward', ...}.",
                      global_var.EPDEDeprecationWarning, stacklevel=2)
        setup.build_FD_preprocessing(
            **{'time_axis': time_axis, **preprocessor_kwargs, 'scheme': 'forward'})
    else:
        raise NotImplementedError(
            f'Incorrect default preprocessor type {default_preprocessor_type!r}. '
            "Allowed: 'poly', 'ANN', 'spectral', 'FD', 'forward'.")
    pipeline = setup.builder.prep_pipeline
    pipeline.specification = (default_preprocessor_type, dict(preprocessor_kwargs), time_axis)
    pipeline.defer_for_forward = default_preprocessor_type in ('FD', 'poly', 'spectral')
    return pipeline
