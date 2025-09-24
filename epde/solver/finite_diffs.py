"""Module for subgrid creation corresponding to numerical scheme. It's used only *NN* method."""

from copy import  copy
import numpy as np

flatten_list = lambda t: [item for sublist in t for item in sublist]


class First_order_scheme():
    """
    Class for constructing numerical schemes. It employs a central difference scheme (second-order accurate) for interior points and first-order accurate forward or backward difference schemes for boundary points. The `points_type` attribute determines whether central differences are used, or a combination of forward and backward differences.
    """


    def __init__(self, term: list, nvars: int, axes_scheme_type: str):
        """
        Initializes a first-order finite difference scheme for a specific term in a partial differential equation.
        
                Args:
                    term (list): Specifies the differentiation order with respect to each independent variable.
                                 For example, [1, 0] represents a first-order derivative with respect to the first variable.
                    nvars (int): The number of independent variables in the partial differential equation.
                                 For example, if the equation is defined on a grid (x, t), then nvars = 2.
                    axes_scheme_type (str): Specifies the finite difference scheme to use along each axis.
                                             It can be 'central' for a central difference scheme or a combination of 'f' (forward) and 'b' (backward)
                                             to define a one-sided scheme.
        
                Returns:
                    None
        
                Why:
                    This initialization sets up the necessary information for approximating derivatives numerically.
                    The 'term' specifies which derivative is being approximated, 'nvars' defines the dimensionality of the problem,
                    and 'axes_scheme_type' determines the type of finite difference approximation used along each axis.
                    This is a crucial step in discretizing the partial differential equation for numerical solution.
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
        """
        Generates shifted coordinate points based on the specified finite difference scheme.
        
        This function creates two new coordinate points by shifting the given point
        along the specified axis, according to the chosen finite difference mode
        (forward, backward, or central). These shifted points are used to approximate
        derivatives at the original point.
        
        Args:
            diff (list): A list representing the coordinate point.
            axis (int): The axis along which the shift is applied.
            mode (str): The finite difference mode ('f' for forward, 'b' for backward, 'central' for central difference).
        
        Returns:
            list: A list containing two shifted coordinate points (diff_p, diff_m).
                  diff_p is shifted in the positive direction, and diff_m in the negative direction.
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
        """
        Builds a first-order finite-difference scheme to approximate derivatives.
        
                This method constructs a numerical scheme by iteratively applying finite difference approximations
                to each term in the differential equation. It starts with a base scheme (list of zeros) and
                perturbs it along the axes corresponding to each term, generating a list of finite difference
                approximations. This process effectively discretizes the differential equation, enabling its
                numerical solution.
        
                Args:
                    self: The instance of the `First_order_scheme` class, containing information about the
                        differential equation, number of variables, and desired scheme type.
        
                Returns:
                    list: A list of lists representing the numerical scheme. Each inner list corresponds to a
                        finite difference approximation of a term in the equation.
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
        """
        Determines the sign of the derivative for each term in the finite difference scheme.
        
        The method iteratively refines the sign list based on the order of the differential operator.
        Each positive sign (+1) is transformed into [+1, -1], while each negative sign (-1) becomes [-1, +1].
        This process ensures that the signs alternate correctly to approximate the derivative.
        
        Args:
            h (float, optional): Discretization parameter used in the finite difference method. Defaults to 1/2.
        
        Returns:
            list: A list containing the signs (+1 or -1) corresponding to each point in the finite difference scheme.
                  These signs are crucial for correctly weighting the terms in the approximation of the derivative.
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
        Initializes a second-order finite difference scheme for a specific term in the differential equation.
        
                This scheme approximates second-order derivatives using finite differences
                based on the specified differentiation direction and scheme type. This
                initializer prepares the scheme for calculating the contribution of this
                term to the overall equation's residual.
        
                Args:
                    term (list): Differentiation direction represented as a list.
                                 For example, [0, 0] corresponds to d^2u/dx^2 if x is the
                                 first direction in the grid.
                    nvars (int): The number of independent variables in the problem.
                                 For example, if the grid is defined by grid(x, t), then nvars = 2.
                    axes_scheme_type (str): The type of finite difference scheme to use along each axis.
                                             Can be 'central' for a central difference scheme, or a
                                             combination of 'f' and 'b' for forward and backward differences,
                                             respectively.  For example, 'fb' would use a forward difference
                                             in the first direction and a backward difference in the second.
        
                Raises:
                    ValueError: If an unsupported scheme type is provided.
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
        """
        Shifts points based on the specified finite difference mode to refine the approximation of derivatives. This is crucial for constructing accurate finite difference schemes used in solving differential equations.
        
                Args:
                    diff (list): Values of finite differences.
                    axis (int): Axis along which the shift is applied.
                    mode (str): The finite difference mode ('f' for forward, 'b' for backward).
        
                Returns:
                    list: A list containing three shifted point configurations, corresponding to the second-order finite difference scheme.
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
        """
        Builds a numerical scheme for the Crank-Nicolson variant, crucial for constructing finite difference approximations of differential operators. This scheme mirrors the structure of the first-order method but incorporates a shift determined by 'second_order_shift' to enhance accuracy.
        
                Args:
                    self: The instance of the `Second_order_scheme` class.
        
                Returns:
                    list: A list representing the numerical scheme, where each element corresponds to a finite difference approximation.
        
                Why:
                    This method constructs the core numerical scheme used to discretize the differential equation. The Crank-Nicolson variant, with its second-order shift, aims to improve the accuracy and stability of the numerical solution, which is essential for reliable equation discovery and modeling.
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
        """
        Generates the coefficients for the second-order finite difference scheme.
        
        This method computes the numerical coefficients based on the specified
        directions (forward or backward) and discretization parameter `h`. These
        coefficients are essential for approximating derivatives in the numerical
        solution of differential equations. The signs and magnitudes of these
        coefficients determine the accuracy and stability of the approximation.
        
        Args:
            h (float, optional): Discretization parameter (grid resolution). Defaults to 1/2.
        
        Returns:
            list: A list of coefficients for the finite difference scheme.
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
        Initializes a finite difference scheme configuration.
        
                This configuration defines the differentiation order, the number of independent variables,
                and the finite difference scheme type (central, forward, or backward) to be used for a specific term
                in the discovered differential equation. This setup is crucial for constructing the operators
                that approximate derivatives within the evolutionary search process.
        
                Args:
                    term (list): Differentiation order for each independent variable.
                                 For example, [0, 0] represents d²u/dx² if x is the first variable in the grid.
                    nvars (int): Number of independent variables in the problem.
                                 For example, if the grid is defined by grid(x, t), then nvars = 2.
                    axes_scheme_type (str): Type of finite difference scheme to use along each axis.
                                             Can be 'central' or a combination of 'f' (forward) and 'b' (backward).
        
                Returns:
                    None
        """

        self.term = term
        self.nvars = nvars
        self.axes_scheme_type = axes_scheme_type

    def scheme_choose(self, scheme_label: str, h:float = 1 / 2) -> list:
        """
        Selects a numerical scheme based on the specified order and discretization parameter.
        
        This method determines which finite difference scheme (first or second order) to use for approximating derivatives,
        based on the user's choice. The selected scheme and its associated sign conventions are then returned.
        This choice influences the accuracy and stability of the numerical solution.
        
        Args:
            scheme_label (str): '2' for the second-order scheme (used only at boundary points),
                '1' for the first-order scheme.
            h (float, optional): Discretization parameter (grid resolution). Defaults to 1/2.
        
        Returns:
            list: A list containing the numerical scheme (as a symbolic expression) at index 0
                  and a list of sign conventions for each term in the scheme at index 1.
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
