import numpy as np


def simple_sorting(elements):
    if any([not all([eq.fitness_calculated for eq in candidate.vals]) for candidate in elements]):
        raise ValueError('Somehow not all of candidates in the population have their fitness evaluated.')
    # Sort by the system's scalar objective (``SoEq.obj_fun``), which the
    # registered objective reader fills -- discrepancy by default, or
    # instability when single_objective_metric == 'instability'. This
    # replaces the former hardcoded ``x.vals['u'].fitness_value`` so the
    # single-objective search follows whichever objective is configured
    # and works for any variable name (not only 'u').
    return sorted(elements, key=lambda x: float(np.sum(x.obj_fun)))
