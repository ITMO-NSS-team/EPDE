"""``evolution.engine='nsga2'``: NSGA-II, an alternative to MOEA/DD on the same
fitness, right-part selection and variation operators."""
from epde.optimizers.nsga.optimizer import NSGAOptimizer, moeadd_parents_per_sector
from epde.optimizers.nsga.strategy import NSGADirector
