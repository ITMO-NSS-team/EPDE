from epde.solver.base import BaseSolverAdapter
from epde.integrate.deepxde_integration import DeepXDEAdapter as _DeepXDEAdapter
import numpy as np
from epde.structure.main_structures import Equation, SoEq
from typing import List, Union

class UnifiedDeepXDEAdapter(BaseSolverAdapter):
    def __init__(self, **config):
        self._adapter = _DeepXDEAdapter(**config)
        self.config = config

    def solve(self, equation_or_system: Union[Equation, SoEq],
              grids: List[np.ndarray],
              data: Union[np.ndarray, List[np.ndarray]]) -> (List[np.ndarray], float):
        return self._adapter.solve(equation_or_system, grids, data)

    def get_requirements(self):
        return {
            'form': 'pde_residual',
            'needs_initial_conditions': True,
            'needs_boundary_conditions': True,
            'supports_multisample': False
        }