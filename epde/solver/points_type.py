"""Module for determine types of grid points. Only for *NN* mode."""


from typing import Union
from scipy.spatial import Delaunay
import numpy as np
import torch

class Points_type():
    """
    Discretizing the grid and allocating subsets for Finite Difference method.
    """

    def __init__(self, grid: torch.Tensor):
        """
        Initializes the `Points_type` object with a grid of discretization points.
        
                This grid represents the spatial or temporal domain over which the differential equation is defined.
                It's crucial for evaluating equation terms and calculating residuals, which are then used to assess the fitness of candidate equations.
        
                Args:
                    grid (torch.Tensor): Discretization points of the domain.
        
                Returns:
                    None
        """

        self.grid = grid

    @staticmethod
    def shift_points(grid: torch.Tensor, axis: int, shift: float) -> torch.Tensor:
        """
        Shifts the grid points along a specified axis by a given amount.
        
                This function modifies the grid to explore alternative configurations in the search space.
                By shifting points, the algorithm can evaluate different arrangements and identify solutions that better fit the observed data.
        
                Args:
                    grid (torch.Tensor): Discretization of the computational domain.
                    axis (int): The axis along which to apply the shift.
                    shift (float): The amount by which to shift the points.
        
                Returns:
                    torch.Tensor: A new tensor with the shifted grid points.
        """

        grid_shift = grid.clone()
        grid_shift[:, axis] = grid[:, axis] + shift
        return grid_shift

    @staticmethod
    def _in_hull(p: torch.Tensor, hull: torch.Tensor) -> np.ndarray:
        """
        Test if points in `p` are within the convex hull defined by `hull`.
        
        This function determines whether a given set of points lies within the
        convex hull formed by another set of points. It leverages Delaunay
        triangulation for efficient spatial partitioning and point location
        in multi-dimensional spaces. This check is crucial for validating
        the physical plausibility of discovered equations by ensuring that
        solutions remain within the bounds defined by the observed data.
        
        Args:
            p (torch.Tensor): A `NxK` tensor representing the coordinates of `N` points in `K` dimensions. These are the points to be tested for inclusion in the hull.
            hull (torch.Tensor): A `MxK` tensor representing the coordinates of `M` points in `K` dimensions. These points define the convex hull. Can also be a precomputed `scipy.spatial.Delaunay` object.
        
        Returns:
            np.ndarray: A boolean array of length `N`. Each element indicates whether the corresponding point in `p` is inside the convex hull defined by `hull`. `True` indicates that the point is inside, `False` otherwise.
        """

        if p.shape[1] > 1:
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull.cpu())

            return hull.find_simplex(p.cpu()) >= 0
        elif p.shape[1] == 1:
            # this one is not a snippet from a stackexchange it does the same
            # but for a 1-D case, which is not covered in a code above
            upbound = torch.max(hull).cpu()
            lowbound = torch.min(hull).cpu()
            return np.array(((p.cpu() <= upbound) & (p.cpu() >= lowbound)).reshape(-1))

    def point_typization(self) -> dict:
        """
        Identifies the type of each point in the grid based on its proximity to the boundary.
        
                This method classifies each point in the grid as either 'central' (lying strictly inside the domain) or
                assigns it a string of 'f' and 'b' characters representing the directions in which small perturbations
                would keep the point within the domain. This information is crucial for understanding the point's
                location relative to the domain's boundaries and is used in subsequent calculations.
        
                Args:
                    self: An instance of the Points_type class containing the grid of points.
        
                Returns:
                    dict: A dictionary where keys are the points in the grid (represented as NumPy arrays) and values
                          are either 'central' or a string of 'f' and 'b' characters. 'f' indicates that moving the
                          point slightly in the positive direction along the corresponding axis keeps it inside the domain,
                          while 'b' indicates the same for the negative direction.
        """

        direction_list = []
        for axis in range(self.grid.shape[1]):
            for direction in range(2):
                direction_list.append(
                    Points_type._in_hull(Points_type.shift_points(
                     self.grid, axis, (-1) ** direction * 0.0001), self.grid))

        direction_list = np.array(direction_list)
        direction_list = np.transpose(direction_list)

        point_type = {}

        for i, point in enumerate(self.grid):
            if np.all(direction_list[i]):
                point_type[point] = 'central'
            else:
                p_type = ''
                j = 0
                while j < len(direction_list[i]):
                    if (j % 2 == 0 and direction_list[i, j]) or (
                            j % 2 == 0 and direction_list[i, j] and
                            direction_list[i, j + 1]):
                        p_type += 'f'
                    else:
                        p_type += 'b'
                    j += 2
                if self.grid.shape[-1] == 1:
                    point_type[point] = 'central'
                else:
                    point_type[point] = p_type
        return point_type

    def grid_sort(self) -> dict:
        """
        Sorts grid points into subsets based on their typization, preparing them for equation discovery.
        
        The method groups points based on a prior typization, organizing them into subsets 
        associated with distinct types. This organization facilitates subsequent processing 
        steps in identifying governing differential equations.
        
        Args:
            self: An instance of the Points_type class containing the grid points and typization information.
        
        Returns:
            dict: A dictionary where keys are point types and values are tensors containing the 
                  corresponding grid points, enabling type-specific analysis.
        """

        point_type = self.point_typization()
        point_types = set(point_type.values())
        grid_dict = {}
        for p_type in point_types:
            grid_dict[p_type] = []
        for point in list(point_type.keys()):
            p_type = point_type[point]
            grid_dict[p_type].append(point)
        for p_type in point_types:
            grid_dict[p_type] = torch.stack(grid_dict[p_type])
        return grid_dict

    def bnd_sort(self, grid_dict: dict, b_coord: Union[torch.Tensor, list]) -> list:
        """
        Sorts boundary points into dictionaries based on their correspondence to grid partitions.
        
                This function organizes boundary points by associating them with specific grid partitions.
                This is useful for applying boundary conditions or constraints specific to each partition when solving differential equations.
        
                Args:
                    grid_dict (dict): A dictionary where keys represent grid partition identifiers and values are tensors representing the coordinates of points within that partition.
                    b_coord (Union[torch.Tensor, list]): A tensor or list of tensors containing the coordinates of boundary points. If a list is provided, it implies a periodic boundary condition with multiple boundary segments.
        
                Returns:
                    list: A list of dictionaries. Each dictionary corresponds to a boundary segment (if `b_coord` is a list) and contains keys matching `grid_dict`. The values are tensors of boundary points that fall within the corresponding grid partition. If `b_coord` is a single tensor, a single dictionary is returned instead of a list.
        """

        def bnd_to_dict(grid_dict, b_coord):
            bnd_dict = {}
            for k, v in grid_dict.items():
                bnd_dict[k] = []
                for bnd in b_coord:
                    if ((bnd == v).all(axis=1)).any():
                        bnd_dict[k].append(bnd)
                if bnd_dict[k] == []:
                    del bnd_dict[k]
                else:
                    bnd_dict[k] = torch.stack(bnd_dict[k])
            return bnd_dict

        if isinstance(b_coord, list):
            bnd_dict_list = [bnd_to_dict(grid_dict, bnd) for bnd in b_coord]
            return bnd_dict_list
        else:
            return bnd_to_dict(grid_dict, b_coord)
