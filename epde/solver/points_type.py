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
        Args:
            grid (torch.Tensor): discretization points of comp-l domain.
        """

        self.grid = grid

    @staticmethod
    def shift_points(grid: torch.Tensor, axis: int, shift: float) -> torch.Tensor:
        """ Shifts all values of an array 'grid' on a value 'shift' in a direction of
        axis 'axis', somewhat is equivalent to a np.roll.

        Args:
            grid (torch.Tensor): discretization of comp-l domain.
            axis (int): axis to which the shift is applied.
            shift (float): shift value.

        Returns:
            torch.Tensor: shifted array of a n-D points.
        """

        grid_shift = grid.clone()
        grid_shift[:, axis] = grid[:, axis] + shift
        return grid_shift

    @staticmethod
    def _in_hull(p: torch.Tensor, hull: torch.Tensor) -> np.ndarray:
        """ Test if points in `p` are in `hull`
        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed.
        Args:
            p (torch.Tensor): shifted array of a n-D points.
            hull (torch.Tensor): initial array of a n-D points.
        Returns:
            np.ndarray: array of a n-D boolean type points.
            True - if 'p' in 'hull', False - otherwise.
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
        """ Allocating subsets for FD (i.e., 'f', 'b', 'central').

        Returns:
            dict: type with a points in a 'grid' above. Type may be 'central' - inner point
            and string of 'f' and 'b', where the length of the string is a dimension n. 'f' means that if we add
            small number to a position of corresponding coordinate we stay in the 'hull'. 'b' means that if we
            subtract small number from o a position of corresponding coordinate we stay in the 'hull'.
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
        """ Sorting grid points for each subset from result Points_type.point_typization.

        Returns:
            dict: sorted grid in each subset (see Points_type.point_typization).
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
        """ Sorting boundary points

        Args:
            grid_dict (dict): _description_
            b_coord (Union[torch.Tensor, list]): boundary points of grid.
            It will be list if periodic condition is.
        
        Returns:
            list: bnd_dict is similar to grid_dict but with b_coord values. It
            will be list of 'bnd_dict's if 'b_coord' is list too.
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
