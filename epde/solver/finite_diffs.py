"""Module for subgrid creation corresponding to numerical scheme. It's used only *NN* method."""

from copy import  copy
import numpy as np

flatten_list = lambda t: [item for sublist in t for item in sublist]


class First_order_scheme():
    """Class for numerical scheme construction. Central o(h^2) difference scheme
    is used for 'central' points, forward ('f') and backward ('b') o(h) schemes
    are used for boundary points. 'central', and combination 'f','b' are
    corresponding to points_type.

    """

    def __init__(self, term: list, nvars: int, axes_scheme_type: str):
        """
        Args:
            term (list): differentiation direction. Example: [0,0]->d2u/dx2
            if x is first direction in the grid.
            nvars (int): task parameters. Example: if grid(x,t) -> nvars = 2.
            axes_scheme_type (str): scheme type: 'central' or combination of 'f' and 'b'
        """

        self.term = term
        self.nvars = nvars
        if axes_scheme_type == 'central':
            self.direction_list = ['central' for _ in self.term]
        else:
            self.direction_list = [axes_scheme_type[i] for i in self.term]

    # the idea is simple - central difference changes
    # [0]->([1]-[-1])/(2h) (in terms of grid nodes position)
    @staticmethod
    def _finite_diff_shift(diff: list, axis: int, mode: str) ->  list:
        """ 1st order points shift for the corresponding finite difference mode.

        Args:
            diff (list): values of finite differences.
            axis (int): axis.
            mode (str): the finite difference mode (i.e., forward, backward, central).

        Returns:
            list: list with shifted points.
        """
 
        diff_p = copy(diff)
        diff_m = copy(diff)
        if mode == 'central':
            diff_p[axis] = diff_p[axis] + 1
            diff_m[axis] = diff_m[axis] - 1
        elif mode == 'f':
            diff_p[axis] = diff_p[axis] + 1
        elif mode == 'b':
            diff_m[axis] = diff_m[axis] - 1
        return [diff_p, diff_m]

    def scheme_build(self) -> list:
        """ Building first order (in terms of accuracy) finite-difference scheme.
        Start from list of zeros where them numbers equal nvars. After that we
        move value in that axis which corresponding to term. [0,0]->[[1,0],[-1,0]]
        it means that term was [0] (d/dx) and mode (scheme_type) is 'central'.

        Returns:
            list: numerical scheme.
        """

        order = len(self.term)
        finite_diff = [[0 for _ in range(self.nvars)]]
        for i in range(order):
            diff_list = []
            for diff in finite_diff:
                # we use [0,0]->[[1,0],[-1,0]] rule for the axis
                f_diff = self._finite_diff_shift(
                    diff, self.term[i], self.direction_list[i])

                if len(diff_list) == 0:
                    # and put it to the pool of differentials if it is empty
                    diff_list = f_diff
                else:
                    # or add to the existing pool
                    for diffs in f_diff:
                        diff_list.append(diffs)
            # then we go to the next differential if needed
            finite_diff = diff_list
        return finite_diff

    def sign_order(self, h: float = 1 / 2) -> list :
        """ Determines the sign of the derivative for the corresponding transformation
        from Finite_diffs.scheme_build().

        From transformations above, we always start from +1 (1)
        Every +1 changes to ->[+1,-1] when order of differential rises
        [0,0] (+1) ->([1,0]-[-1,0]) ([+1,-1])
        Every -1 changes to [-1,+1]
        [[1,0],[-1,0]] ([+1,-1])->[[1,1],[1,-1],[-1,1],[-1,-1]] ([+1,-1,-1,+1])

        Args:
            h (float, optional): discretizing parameter in finite-
            difference method. Defaults to 1/2.

        Returns:
            list: list, with signs for corresponding points.
        """

        sign_list = [1]
        for _ in range(len(self.term)):
            start_list = []
            for sign in sign_list:
                if np.unique(self.direction_list)[0] == 'central':
                    start_list.append([sign * (1 / (2 * h)),
                                       -sign * (1 / (2 * h))])
                else:
                    start_list.append([sign / h, -sign / h])
            sign_list = flatten_list(start_list)
        return sign_list


class Second_order_scheme():
    """
    Crank–Nicolson method. This realization only for boundary points.
    """
    def __init__(self, term: list, nvars: int, axes_scheme_type: str):
        """
        Args:
            term (list): differentiation direction. Example: [0,0]->d2u/dx2 if x is first
                    direction in the grid.
            nvars (int): task parameters. Example: if grid(x,t) -> nvars = 2.
            axes_scheme_type (str): scheme type: 'central' or combination of 'f' and 'b'

        Raises:
            ValueError: _description_
        """
        
        self.term = term
        self.nvars = nvars
        try:
            axes_scheme_type == 'central'
        except:
            print('These scheme only for "f" and "b" points')
            raise ValueError
        self.direction_list = [axes_scheme_type[i] for i in self.term]

    @staticmethod
    def _second_order_shift(diff, axis, mode) -> list:
        """ 2st order points shift for the corresponding finite difference mode.

        Args:
            diff (list): values of finite differences.
            axis (int): axis.
            mode (str): the finite difference mode (i.e., forward, backward).

        Returns:
            list: list with shifted points.
        """
        diff_1 = copy(diff)
        diff_2 = copy(diff)
        diff_3 = copy(diff)
        if mode == 'f':
            diff_3[axis] = diff_3[axis] + 2
            diff_2[axis] = diff_2[axis] + 1
        elif mode == 'b':
            diff_3[axis] = diff_3[axis] - 2
            diff_2[axis] = diff_2[axis] - 1
        else:
            print('Wrong mode')
        return [diff_3, diff_2, diff_1]

    def scheme_build(self) -> list:
        """Scheme building for Crank-Nicolson variant, it's identical to
        'scheme_build' in first order method, but value is shifted by
        'second_order_shift'.

        Returns:
            list: numerical scheme list.
        """

        order = len(self.term)
        finite_diff = [[0 for _ in range(self.nvars)]]
        # when we increase differential order
        for i in range(order):
            diff_list = []
            for diff in finite_diff:
                # we use [0,0]->[[1,0],[-1,0]] rule for the axis
                f_diff = self._second_order_shift(
                    diff, self.term[i], self.direction_list[i])
                if len(diff_list) == 0:
                    # and put it to the pool of differentials if it is empty
                    diff_list = f_diff
                else:
                    # or add to the existing pool
                    for diffs in f_diff:
                        diff_list.append(diffs)
            # then we go to the next differential if needed
            finite_diff = diff_list
        return finite_diff

    def sign_order(self, h: float = 1/2) -> list:
        """ Signs definition for second order schemes.

        Args:
            h (float, optional): discretizing parameter in finite-
            difference method (i.e., grid resolution for scheme). Defaults to 1/2.

        Returns:
            list: list, with signs for corresponding points.
        """

        sign_list = [1]
        for i in range(len(self.term)):
            start_list = []
            for sign in sign_list:
                if self.direction_list[i] == 'f':
                    start_list.append([3 * (1 / (2 * h)) * sign,
                                       -4 * (1 / (2 * h)) * sign,
                                       (1 / (2 * h)) * sign])
                elif self.direction_list[i] == 'b':
                    start_list.append([-3 * (1 / (2 * h)) * sign,
                                       4 * (1 / (2 * h)) * sign,
                                       -(1 / (2 * h)) * sign])
            sign_list = flatten_list(start_list)
        return sign_list


class Finite_diffs():
    """
    Class for numerical scheme choosing.
    """

    def __init__(self, term: list, nvars: int, axes_scheme_type: str):
        """
        Args:
            term (list): differentiation direction. Example: [0,0]->d2u/dx2 if x is first
                    direction in the grid.
            nvars (int): task parameters. Example: if grid(x,t) -> nvars = 2.
            axes_scheme_type (str): scheme type: 'central' or combination of 'f' and 'b'
        """

        self.term = term
        self.nvars = nvars
        self.axes_scheme_type = axes_scheme_type

    def scheme_choose(self, scheme_label: str, h:float = 1 / 2) -> list:
        """ Method for numerical scheme choosing via realized above.

        Args:
            scheme_label (str): '2'- for second order scheme (only boundaries points),
                '1' - for first order scheme.
            h (float, optional): discretizing parameter in finite-
            difference method (i.e., grid resolution for scheme). Defaults to 1/2.

        Returns:
            list: list where list[0] is numerical scheme and list[1] is signs.
        """

        if self.term == [None]:
            return [[None], [1]]
        elif scheme_label == '2':
            cl_scheme = Second_order_scheme(self.term, self.nvars,
                                                        self.axes_scheme_type)
        elif scheme_label == '1':
            cl_scheme = First_order_scheme(self.term, self.nvars,
                                                        self.axes_scheme_type)

        scheme = cl_scheme.scheme_build()
        sign = cl_scheme.sign_order(h=h)
        return [scheme, sign]
