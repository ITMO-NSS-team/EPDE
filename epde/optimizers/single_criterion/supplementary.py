def simple_sorting(elements):
    # TODO: thorough testing
    if any([not all([eq.fitness_calculated for eq in candidate.chromosome]) for candidate in elements]):
        raise ValueError('Somehow not all of candidates in the population have their fitness evaluated.')
    return sorted(elements, key = lambda x: x.fitness)