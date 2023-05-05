def simple_sorting(elements):
    if any([not all([eq.fitness_calculated for eq in candidate.vals]) for candidate in elements]):
        raise ValueError('Somehow not all of candidates in the population have their fitness evaluated.')
    return sorted(elements, key = lambda x: x.vals['u'].fitness_value) # TODO: replace['u']
