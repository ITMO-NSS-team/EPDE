import unittest
import logging

import numpy as np

from epde.optimizers.strategy import Strategy
from epde.optimizers.builder import StrategyBuilder, OptimizationPatternDirector
from epde.optimizers.blocks import Block, EvolutionaryBlock, InputBlock, LinkedBlocks

level = logging.critical
logging.getLogger("base_moeadd").setLevel(level)

class TestStrategy(unittest.TestCase):
    def setUp(self):
        pass
    
    # def test_strat_con