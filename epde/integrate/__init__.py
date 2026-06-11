from .interface import SystemSolverInterface
from .bop import BOPElement, BoundaryConditions
from .pinn_integration import SolverAdapter
from .numeric_integration import OdeintAdapter


# ``deepxde_integration`` does ``import deepxde``, which prints a backend
# banner on first load. Defer that until the DeepXDE adapter is actually
# requested so plain ``import epde`` / ``from epde.integrate import
# SolverAdapter`` stays quiet.
def __getattr__(name):
    if name == 'DeepXDEAdapter':
        from .deepxde_integration import DeepXDEAdapter
        return DeepXDEAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
