""" Module for construct a domain with complex or irregular geometry"""
import torch
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def contains(self, pts: torch.Tensor) -> torch.BoolTensor:
        """
        Returns a mask (N,) indicating which pts are inside the shape.
        """
        ...

    @abstractmethod
    def boundary(self, pts: torch.Tensor, rtol: float = 1e-4, atol: float = 0.0) -> torch.BoolTensor:
        """
        Returns a mask (N,) indicating which pts lie on the boundary of the shape.
        """
        ...

class Rectangle(Shape):
    def __init__(self, lower, upper, dims=None):
        """
        lower: sequence of lower bounds for each spatial dimension
        upper: sequence of upper bounds for each spatial dimension
        dims: indices of dimensions in the grid to apply the rectangle (default first len(lower))
        """
        self.lower = torch.as_tensor(lower, dtype=torch.float32)
        self.upper = torch.as_tensor(upper, dtype=torch.float32)
        self.dims = dims if dims is not None else list(range(self.lower.numel()))

    def _select(self, pts: torch.Tensor) -> torch.Tensor:
        return pts[:, self.dims]

    def contains(self, pts: torch.Tensor) -> torch.BoolTensor:
        sub = self._select(pts)
        return ((sub >= self.lower) & (sub <= self.upper)).all(dim=1)

    def boundary(self, pts: torch.Tensor, rtol: float = 1e-4, atol: float = 0.0) -> torch.BoolTensor:
        sub = self._select(pts)
        inside_or_bound = self.contains(pts)
        on_lower = torch.isclose(sub, self.lower.unsqueeze(0), rtol=rtol, atol=atol).any(dim=1)
        on_upper = torch.isclose(sub, self.upper.unsqueeze(0), rtol=rtol, atol=atol).any(dim=1)
        return inside_or_bound & (on_lower | on_upper)

class Circle(Shape):
    def __init__(self, center, radius, dims=None):
        """
        center: sequence of center coordinates for each spatial dimension
        radius: scalar radius
        dims: indices of dimensions in the grid to apply the circle (default first len(center))
        """
        self.center = torch.as_tensor(center, dtype=torch.float32)
        self.radius_sq = float(radius) ** 2
        self.dims = dims if dims is not None else list(range(self.center.numel()))

    def _select(self, pts: torch.Tensor) -> torch.Tensor:
        return pts[:, self.dims]

    def contains(self, pts: torch.Tensor) -> torch.BoolTensor:
        sub = self._select(pts)
        sqd = ((sub - self.center) ** 2).sum(dim=1)
        return sqd <= self.radius_sq

    def boundary(self, pts: torch.Tensor, rtol: float = 1e-4, atol: float = 0.0) -> torch.BoolTensor:
        sub = self._select(pts)
        sqd = ((sub - self.center) ** 2).sum(dim=1)
        return torch.isclose(sqd, torch.tensor(self.radius_sq, dtype=pts.dtype), rtol=rtol, atol=atol)

# CSG operations

def csg_difference(grid: torch.Tensor, shape: Shape) -> torch.Tensor:
    """
    Returns points of `grid` outside the given `shape`.

    Args:
        grid: (N, D) tensor of coordinates.
        shape: a Shape instance.
    """
    mask_outside = ~shape.contains(grid)
    return grid[mask_outside]


def csg_boundary(grid: torch.Tensor, shape: Shape, rtol: float = 1e-4, atol: float = 0.0) -> torch.Tensor:
    """
    Returns points of `grid` that lie on the boundary of the given `shape`.

    Args:
        grid: (N, D) tensor of coordinates.
        shape: a Shape instance.
        rtol: relative tolerance for boundary detection.
        atol: absolute tolerance for boundary detection.
    """
    mask_bnd = shape.boundary(grid, rtol=rtol, atol=atol)
    return grid[mask_bnd]




