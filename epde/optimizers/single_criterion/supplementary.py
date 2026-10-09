import numpy as np


def is_stamped(candidate) -> bool:
    """True when the fitness host stamped ``candidate`` as failed.

    The stamp ``LOSS_NAN_VAL`` (1e7) is written on every stamped objective,
    but it is "worst" only for bounded metrics: vcoef, cv, chi2, chi2_centered
    and het_raw can score a legitimate wrong form above it. So a stamped form
    is recognised by the rule MOEA/D drops it by -- some equation's discrepancy
    ``fitness_value >= LOSS_NAN_VAL`` -- never by its objective value. Imported
    lazily: the predicate lives with the multiobjective operators.
    """
    from epde.operators.multiobjective.moeadd_specific import has_degenerate_equation
    return has_degenerate_equation(candidate)


def simple_sorting(elements):
    if any([not all([eq.fitness_calculated for eq in candidate.vals]) for candidate in elements]):
        raise ValueError('Somehow not all of candidates in the population have their fitness evaluated.')
    # Sort by the system's scalar objective (``SoEq.obj_fun``), which the
    # registered objective reader fills -- discrepancy by default, or
    # instability when single_objective_metric == 'instability'. This
    # replaces the former hardcoded ``x.vals['u'].fitness_value`` so the
    # single-objective search follows whichever objective is configured
    # and works for any variable name (not only 'u').
    # Stamped forms go after every real one whatever their value, since the
    # 1e7 stamp can undercut a legitimate objective (see ``is_stamped``).
    return sorted(elements, key=lambda x: (is_stamped(x), float(np.sum(x.obj_fun))))
