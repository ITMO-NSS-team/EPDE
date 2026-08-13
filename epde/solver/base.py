from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Dict, Any
import numpy as np
from epde.structure.main_structures import Equation, SoEq

class BaseSolverAdapter(ABC):
    """Базовый класс для всех адаптеров солверов."""

    @abstractmethod
    def solve(self, equation_or_system: Union[Equation, SoEq],
              grids: List[np.ndarray],
              data: Union[np.ndarray, List[np.ndarray]]) -> Tuple[List[np.ndarray], float]:
        """
        Решает уравнение/систему на заданных сетках.

        Parameters
        ----------
        equation_or_system : Equation or SoEq
            Одиночное уравнение или система.
        grids : list of np.ndarray
            Сетки координат (каждая размерность – отдельный массив).
        data : np.ndarray or list of np.ndarray
            Эталонные значения (для каждой переменной) для расчёта ошибки.

        Returns
        -------
        solutions : list of np.ndarray
            Список решений (по одному массиву на переменную).
        loss : float
            Числовая метрика ошибки (например, RMSE по всей сетке).
        """
        pass

    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """
        Возвращает требования солвера к представлению уравнений.

        Returns
        -------
        dict
            Ключи:
            - 'form': 'explicit_ode', 'pde_residual', 'system_of_odes'
            - 'needs_initial_conditions': bool
            - 'needs_boundary_conditions': bool
            - 'supports_multisample': bool
        """
        pass

    def supports_multisample(self) -> bool:
        return self.get_requirements().get('supports_multisample', False)