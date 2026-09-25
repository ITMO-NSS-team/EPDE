from epde.solver.base import BaseSolverAdapter
from epde.solver.unified_deepxde_adapter import UnifiedDeepXDEAdapter
from epde.solver.classical_ode_adapter import ClassicalODEAdapter

class SolverFactory:
    _registry = {
        'deepxde': UnifiedDeepXDEAdapter,
        'classical_ode': ClassicalODEAdapter,
    }

    @classmethod
    def register(cls, name: str, adapter_class):
        """Регистрирует новый тип солвера."""
        cls._registry[name] = adapter_class

    @classmethod
    def create(cls, solver_type: str, **config) -> BaseSolverAdapter:
        """Создаёт экземпляр адаптера по имени."""
        if solver_type not in cls._registry:
            raise ValueError(f"Неизвестный тип солвера: {solver_type}. Доступны: {list(cls._registry.keys())}")
        return cls._registry[solver_type](**config)