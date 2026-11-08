"""The ``evolution.engine='nsga2'`` strategy: MOEA/DD's chain with its mating
selection and survival replaced (``operators``)."""
from epde.optimizers.moeadd.strategy import MOEADDDirector
from epde.optimizers.nsga.operators import AppendUpdater, NSGASelection


class NSGADirector(MOEADDDirector):
    """Builds the MOEA/DD chain -- initial sorting, selection, variation,
    offspring update -- with :class:`NSGASelection` as the selection and
    :class:`AppendUpdater` as the offspring update's level updater."""

    def use_baseline(self, *args, **kwargs):
        kwargs['selection'] = NSGASelection(['parents_fraction'])
        kwargs['level_updater'] = AppendUpdater()
        super().use_baseline(*args, **kwargs)
