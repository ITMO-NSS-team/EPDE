from copy import copy

flatten_list = lambda t: [item for sublist in t for item in sublist]


class Finite_diffs():
    # the idea is simple - central difference changes [0]->([1]-[-1])/(2h) (in terms of grid nodes position)
    @staticmethod
    def finite_diff_shift(diff, axis, mode):
        """
        we do the [0]->([1]-[-1])/(2h) transitions to the axes we need
        as an example d2u/dxdt
        u=[0,0]
        u-> du/dx:

        [0,0]->([1,0]-[-1,0])/(2h)

        du/dx->d2u/dxdt:

        [1,0]->([1,1]-[1,-1])/(2h*2tau)

        [-1,0]->([-1,1]-[-1,-1])/(2h*2tau)

        But we do not want to take signs into account (too complex), so

        u-> du/dx:

        [0,0]->[[1,0],[-1,0]]

        du/dx->d2u/dxdt:

        [[1,0],[-1,0]]->[[1,1],[1,-1],[-1,1],[-1,-1]]

        Since order is preserved we can compute signs afterwards

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

    @staticmethod
    def scheme_build(axes, varn, axes_mode):
        order = len(axes)
        finite_diff = []
        direction_list = []
        # we generate [0,0,0,...] for number of variables (varn)
        for i in range(varn):
            finite_diff += [0]
        # just to make this [[0,0,...]]
        finite_diff = [finite_diff]
        # when we increase differential order
        for i in range(order):
            diff_list = []
            for diff in finite_diff:
                # we use [0,0]->[[1,0],[-1,0]] rule for the axis
                if axes_mode == 'central':
                    f_diff = Finite_diffs.finite_diff_shift(diff, axes[i], 'central')
                else:
                    f_diff = Finite_diffs.finite_diff_shift(diff, axes[i], axes_mode[axes[i]])
                if len(diff_list) == 0:
                    # and put it to the pool of differentials if it is empty
                    diff_list = f_diff
                else:
                    # or add to the existing pool
                    for diffs in f_diff:
                        diff_list.append(diffs)
            # the we go to the next differential if needed
            finite_diff = diff_list
            direction_list.append(axes_mode[axes[i]])
        return finite_diff, direction_list

    @staticmethod
    def sign_order(order, mode, h=1 / 2):
        """
        From transormations above, we always start from +1 (1)

        Every +1 changes to ->[+1,-1] when order of differential rises

        [0,0] (+1) ->([1,0]-[-1,0]) ([+1,-1])

        Every -1 changes to [-1,+1]

        [[1,0],[-1,0]] ([+1,-1])->[[1,1],[1,-1],[-1,1],[-1,-1]] ([+1,-1,-1,+1])

        """
        sign_list = [1]
        for i in range(order):
            start_list = []
            for sign in sign_list:
                if mode == 'central':
                    start_list.append([sign * (1 / (2 * h)), -sign * (1 / (2 * h))])
                else:
                    start_list.append([sign / h, -sign / h])
            sign_list = flatten_list(start_list)
        return sign_list
    """
    The following functions are forward and backward schemes combined
    """

    @staticmethod
    def second_order_shift(diff, axis, mode):
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

    @staticmethod
    def second_order_scheme_build(axes, varn, axes_mode):
        order = len(axes)
        finite_diff = []
        direction_list = []
        # we generate [0,0,0,...] for number of variables (varn)
        for i in range(varn):
            finite_diff += [0]
        # just to make this [[0,0,...]]
        finite_diff = [finite_diff]
        # when we increase differential order
        for i in range(order):
            diff_list = []
            for diff in finite_diff:
                # we use [0,0]->[[1,0],[-1,0]] rule for the axis
                if axes_mode == 'central':
                    f_diff = Finite_diffs.second_order_shift(diff, axes[i], 'central')
                else:
                    f_diff = Finite_diffs.second_order_shift(diff, axes[i], axes_mode[axes[i]])
                if len(diff_list) == 0:
                    # and put it to the pool of differentials if it is empty
                    diff_list = f_diff
                else:
                    # or add to the existing pool
                    for diffs in f_diff:
                        diff_list.append(diffs)
            # the we go to the next differential if needed
            finite_diff = diff_list
            direction_list.append(axes_mode[axes[i]])
        return finite_diff, direction_list

    @staticmethod
    def second_order_sign_order(order, mode, h=1/2):
        sign_list = [1]
        for i in range(order):
            start_list = []
            for sign in sign_list:
                if mode[i] == 'f':
                    start_list.append([3 * (1 / (2 * h)) * sign, -4 * (1 / (2 * h)) * sign, (1 / (2 * h)) * sign])
                elif mode[i] == 'b':
                    start_list.append([-3 * (1 / (2 * h)) * sign, 4 * (1 / (2 * h)) * sign, -(1 / (2 * h)) * sign])
            sign_list = flatten_list(start_list)
        return sign_list

