import os
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple, Optional

import numpy as np

from epde.structure.main_structures import Equation
import epde.globals as global_var

# debug log. Just for my PC
# LOGPATH = "/tmp/deepxde_adapter_debug.txt"
# 
# try:
#     with open(LOGPATH, "a", encoding="utf-8") as _f:
#         _f.write(f"\n=== DeepXDEAdapter log started at PID {os.getpid()} ===\n")
# except Exception:
#     LOGPATH = None
# 
# 
# def #_log(msg: Any) -> None:
#     """
#     Lightweight debug logger used in this module.
# 
#     NOT FOR PRODUCTION. JUST SIMPLE CASE FOR DEBUG ON MY PC
#     """
#     print(msg)
#     sys.stdout.flush()
#     if LOGPATH:
#         try:
#             with open(LOGPATH, "a", encoding="utf-8") as f:
#                 f.write(str(msg) + "\n")
#         except Exception:
#             pass


class DeepXDEAdapter:
    """
    Adapter that converts EPDE-style Equation objects into DeepXDE
    PDE functions and runs DeepXDE models.

    Parameters
    ----------
    pretrained_net:
        Optional pre-built network object (left opaque here; passed through if needed).
    config:
        Optional keyword configuration for network, training, sampling, etc.
    """

    def __init__(self, pretrained_net: Optional[Any] = None, **config: Any) -> None:
        self.pretrained_net: Optional[Any] = pretrained_net
        self.config: Dict[str, Any] = dict(config or {})
        self.net: List[int] = list(self.config.get("net", [50, 50, 50, 50]))
        self.activation: str = str(self.config.get("activation", "tanh"))
        self.optimizer: str = str(self.config.get("optimizer", "adam"))
        self.lr: float = float(self.config.get("lr", 1e-3))
        self.kernel_initializer: str = str(self.config.get("kernel_initializer", "Glorot normal"))
        self.num_domain: int = int(self.config.get("num_domain", 2000))
        self.num_boundary: int = int(self.config.get("num_boundary", 500))
        self.num_initial: int = int(self.config.get("num_initial", 500))
        self.epochs: int = int(self.config.get("epochs", 10000))
        self.bc_type: str = str(self.config.get("bc_type", "Dirichlet"))
        self.fallback_bc_value: float = float(self.config.get("fallback_bc_value", 0.0))

        self.coordinate_mapping: Optional[Dict[str, int]] = self.config.get("coordinate_mapping", None)
        self.coord_names: Optional[Sequence[str]] = None
        self.coord_map: Optional[Dict[str, int]] = None

    def _set_coordinate_info(self, coord_names: Sequence[str]) -> None:
        """
        Set internal coordinate names and build mapping between EPDE coord names
        and DeepXDE coordinate order.

        By default the implementation assumes the first coord is time and the rest
        are spatial. The mapping created maps canonical coordinate label -> index
        used by DeepXDE tensors.
        """
        self.coord_names = list(coord_names)
        if self.coordinate_mapping is not None:
            self.coord_map = dict(self.coordinate_mapping)
        else:
            # Default mapping: first coordinate is time, DeepXDE expects spatial coords first, then time
            self.coord_map = {}
            spatial_dim = len(coord_names) - 1
            for i, name in enumerate(coord_names):
                if i == 0:  # time
                    self.coord_map[name] = spatial_dim
                else:  # spatial
                    self.coord_map[name] = i - 1
        #_log(f"[DeepXDEAdapter] coord_map = {self.coord_map}")

    def _equation_to_pde_func(self, dde: Any, equation: Equation) -> Callable[[Any, Any], Any]:
        """
        Convert EPDE `Equation` into a PDE function compatible with DeepXDE.

        Returns a callable `pde(x, y)` capturing `coord_map` and using adapter helper
        methods to evaluate terms and derivatives.
        """
        chosen: Dict[str, Any] = {"map": self.coord_map}

        def pde(x: Any, y: Any) -> Any:
            coord_map = chosen["map"]
            use_weights = getattr(equation, "weights_final_evald", False) and hasattr(equation, "weights_final")
            residual = y * 0.0

            all_terms = equation.structure
            for term_idx, term in enumerate(all_terms):
                if term_idx == equation.target_idx:
                    continue
                coeff = float(equation.weights_final[term_idx]) if use_weights else 1.0
                term_val = 1.0
                for factor in term.structure:
                    fv = self._factor_value_with_map(dde, factor, x, y, coord_map)
                    term_val *= fv
                residual += coeff * term_val

            # constant (bias) term if present in weights
            if use_weights and len(equation.weights_final) > len(all_terms):
                residual += float(equation.weights_final[-1]) * (y * 0.0 + 1.0)

            target_idx = getattr(equation, "target_idx", 0)
            target = equation.structure[target_idx]
            target_val = 1.0
            for factor in target.structure:
                fv = self._factor_value_with_map(dde, factor, x, y, coord_map)
                target_val *= fv
            residual -= target_val
            return residual

        return pde

    def _factor_value_with_map(
        self, dde: Any, factor: Any, x: Any, y: Any, coord_map: Optional[Dict[str, int]]
    ) -> Any:
        """
        Evaluate one factor (variable, derivative, coordinate token or constant)
        given DeepXDE tensors `x`, `y` and the coordinate mapping.

        Returns a tensor-like object compatible with DeepXDE operations.
        """
        # derivative
        if getattr(factor, "is_deriv", False) and getattr(factor, "deriv_code", None):
            val = y if getattr(factor, "variable", None) == "u" else (y * 0.0)
            for ax in factor.deriv_code:
                if ax is None:
                    continue
                try:
                    ax_int = int(ax)
                except (ValueError, TypeError):
                    continue
                # ax_int — index in original coord_names list
                coord_name = self.coord_names[ax_int]
                dde_ax = coord_map.get(coord_name, None) if coord_map is not None else None
                if dde_ax is not None:
                    val = dde.grad.jacobian(val, x, i=0, j=dde_ax)
                else:
                    return y * 0.0
            return val

        # main variable u^p
        if getattr(factor, "variable", None) == "u":
            params = getattr(factor, "params", [1.0])
            p = float(params[-1])
            return y ** p

        # constant
        if len(getattr(factor, "structure", [])) == 0 or "const" in str(getattr(factor, "name", "")).lower():
            return y * 0.0 + 1.0

        # coordinate token (t, x, y...)
        label = getattr(factor, "cache_label", None)
        if label:
            if isinstance(label, tuple):
                label = str(label[0]).lower()
            else:
                label = str(label).lower()
            idx = coord_map.get(label, None) if coord_map is not None else None
            if idx is not None:
                return x[:, int(idx) : int(idx) + 1]
        return y * 0.0 + 1.0  # fallback

    # ---------- solvers for different dimensionalities ----------
    def _solve_1d(self, equation: Equation, grids: Sequence[np.ndarray], data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve 1D-in-time problem (only time axis) via DeepXDE.

        Returns a tuple (y_pred, final_loss).
        """
        import deepxde as dde
        from scipy.spatial import cKDTree

        t = grids[0]
        geom = dde.geometry.TimeDomain(t.min(), t.max())

        mask = global_var.grid_cache.g_func_mask
        coords_masked = t[mask].reshape(-1, 1)
        data_masked = data.ravel()

        eps_t = (t.max() - t.min()) * 1e-5
        initial_idx = np.where(np.abs(t[mask] - t.min()) < eps_t)[0]

        if len(initial_idx) > 0:
            tree_init = cKDTree(coords_masked[initial_idx])

            def ic_func(x: Any) -> np.ndarray:
                if hasattr(x, "detach"):
                    x_np = x.detach().cpu().numpy()
                else:
                    x_np = np.asarray(x)
                _, idx = tree_init.query(x_np)
                return data_masked[initial_idx][idx].reshape(-1, 1)

        else:

            def ic_func(x: Any) -> np.ndarray:
                return np.full((x.shape[0], 1), self.fallback_bc_value)

        bcs = [dde.icbc.IC(geom, ic_func, lambda _, on_initial: on_initial)]

        pde_func = self._equation_to_pde_func(dde, equation)
        data_obj = dde.data.PDE(
            geom,
            pde_func,
            bcs,
            num_domain=self.num_domain,
            num_boundary=self.num_boundary,
            num_test=500,
        )

        layer_size = [1] + self.net + [1]
        net = dde.nn.FNN(layer_size, self.activation, self.kernel_initializer)
        model = dde.Model(data_obj, net)
        model.compile(self.optimizer, lr=self.lr)
        try:
            losshistory, train_state = model.train(epochs=self.epochs)
            final_loss = float(losshistory.loss_train[-1][0]) if losshistory.loss_train else float("nan")
        except Exception:
            y_pred = np.full(data.shape, np.nan)
            final_loss = float("nan")
            return y_pred, final_loss

        t_flat = t.reshape(-1, 1)
        y_pred = model.predict(t_flat).reshape(-1)
        return y_pred, final_loss

    def _solve_2d(self, equation: Equation, grids: Sequence[np.ndarray], data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve 1-space + time (2D) problems using DeepXDE GeometryXTime.

        Returns (y_pred, final_loss).
        """
        import deepxde as dde
        from scipy.spatial import cKDTree

        #_log("[DeepXDEAdapter] Solving 2D problem (1 space + time)")
        t, x = grids[0], grids[1]
        geom = dde.geometry.Interval(x.min(), x.max())
        timedomain = dde.geometry.TimeDomain(t.min(), t.max())
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        mask = global_var.grid_cache.g_func_mask
        masked_coords = np.stack([g[mask] for g in grids], axis=1)
        masked_coords_swapped = masked_coords[:, [1, 0]]
        data_masked = data.ravel()

        eps_x = (x.max() - x.min()) * 1e-5
        eps_t = (t.max() - t.min()) * 1e-5

        left_idx = np.where(np.abs(masked_coords_swapped[:, 0] - x.min()) < eps_x)[0]
        right_idx = np.where(np.abs(masked_coords_swapped[:, 0] - x.max()) < eps_x)[0]
        initial_idx = np.where(np.abs(masked_coords_swapped[:, 1] - t.min()) < eps_t)[0]

        def make_bc_func(indices: Sequence[int]) -> Callable[[Any], np.ndarray]:
            if len(indices) == 0:
                return lambda x_in: np.full((x_in.shape[0], 1), self.fallback_bc_value)
            tree = cKDTree(masked_coords_swapped[indices])

            def func(x_in: Any) -> np.ndarray:
                if hasattr(x_in, "detach"):
                    x_np = x_in.detach().cpu().numpy()
                else:
                    x_np = np.asarray(x_in)
                _, idx = tree.query(x_np)
                return data_masked[indices][idx].reshape(-1, 1)

            return func

        bcs = []
        if len(left_idx) > 0:
            bcs.append(
                dde.icbc.DirichletBC(
                    geomtime,
                    make_bc_func(left_idx),
                    lambda _, on_boundary: on_boundary and np.isclose(_.x[0], x.min(), rtol=1e-5, atol=eps_x),
                )
            )
        if len(right_idx) > 0:
            bcs.append(
                dde.icbc.DirichletBC(
                    geomtime,
                    make_bc_func(right_idx),
                    lambda _, on_boundary: on_boundary and np.isclose(_.x[0], x.max(), rtol=1e-5, atol=eps_x),
                )
            )
        if len(initial_idx) > 0:
            bcs.append(dde.icbc.IC(geomtime, make_bc_func(initial_idx), lambda _, on_initial: on_initial))

        pde_func = self._equation_to_pde_func(dde, equation)
        data_obj = dde.data.TimePDE(
            geomtime,
            pde_func,
            bcs,
            num_domain=self.num_domain,
            num_boundary=self.num_boundary,
            num_initial=self.num_initial,
            num_test=500,
        )

        layer_size = [geomtime.dim] + self.net + [1]
        net = dde.nn.FNN(layer_size, self.activation, self.kernel_initializer)
        model = dde.Model(data_obj, net)
        model.compile(self.optimizer, lr=self.lr)
        #_log("[DeepXDEAdapter] Starting training (2D)")
        try:
            losshistory, train_state = model.train(epochs=self.epochs)
            final_loss = float(losshistory.loss_train[-1][0]) if losshistory.loss_train else float("nan")
        except Exception as e:
            #_log(f"[DeepXDEAdapter] Training failed: {e}")
            y_pred = np.full(data.shape, np.nan)
            final_loss = float("nan")
            return y_pred, final_loss

        coords_pred = np.stack([x.flatten(), t.flatten()], axis=1)
        y_pred = model.predict(coords_pred).reshape(-1)

        #_log(f"[DeepXDEAdapter] Training complete, final_loss {final_loss}")
        return y_pred, final_loss

    def _solve_3d(self, equation: Equation, grids: Sequence[np.ndarray], data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve 2-space + time (3D) problems using DeepXDE GeometryXTime.

        Returns (y_pred, final_loss).
        """
        import deepxde as dde
        from scipy.spatial import cKDTree

        #_log("[DeepXDEAdapter] Solving 3D problem (2 space + time)")
        t, x, y = grids[0], grids[1], grids[2]
        geom = dde.geometry.Rectangle([x.min(), y.min()], [x.max(), y.max()])
        timedomain = dde.geometry.TimeDomain(t.min(), t.max())
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        mask = global_var.grid_cache.g_func_mask
        masked_coords = np.stack([g[mask] for g in grids], axis=1)
        # DeepXDE expects (x, y, t)
        masked_coords_swapped = masked_coords[:, [1, 2, 0]]
        data_masked = data.ravel()

        # Set coordinate info from grid_cache (keys order must match grids order)
        eps_x = (x.max() - x.min()) * 1e-5
        eps_y = (y.max() - y.min()) * 1e-5
        eps_t = (t.max() - t.min()) * 1e-5

        x_min_idx = np.where(np.abs(masked_coords_swapped[:, 0] - x.min()) < eps_x)[0]
        x_max_idx = np.where(np.abs(masked_coords_swapped[:, 0] - x.max()) < eps_x)[0]
        y_min_idx = np.where(np.abs(masked_coords_swapped[:, 1] - y.min()) < eps_y)[0]
        y_max_idx = np.where(np.abs(masked_coords_swapped[:, 1] - y.max()) < eps_y)[0]
        initial_idx = np.where(np.abs(masked_coords_swapped[:, 2] - t.min()) < eps_t)[0]

        def make_bc_func(indices: Sequence[int]) -> Callable[[Any], np.ndarray]:
            if len(indices) == 0:
                return lambda x_in: np.full((x_in.shape[0], 1), self.fallback_bc_value)
            tree = cKDTree(masked_coords_swapped[indices])

            def func(x_in: Any) -> np.ndarray:
                if hasattr(x_in, "detach"):
                    x_np = x_in.detach().cpu().numpy()
                else:
                    x_np = np.asarray(x_in)
                _, idx = tree.query(x_np)
                return data_masked[indices][idx].reshape(-1, 1)

            return func

        bcs: List[Any] = []
        if len(x_min_idx) > 0:
            bcs.append(
                dde.icbc.DirichletBC(
                    geomtime,
                    make_bc_func(x_min_idx),
                    lambda _, on_boundary: on_boundary and np.isclose(_.x[0], x.min(), rtol=1e-5, atol=eps_x),
                )
            )
        if len(x_max_idx) > 0:
            bcs.append(
                dde.icbc.DirichletBC(
                    geomtime,
                    make_bc_func(x_max_idx),
                    lambda _, on_boundary: on_boundary and np.isclose(_.x[0], x.max(), rtol=1e-5, atol=eps_x),
                )
            )
        if len(y_min_idx) > 0:
            bcs.append(
                dde.icbc.DirichletBC(
                    geomtime,
                    make_bc_func(y_min_idx),
                    lambda _, on_boundary: on_boundary and np.isclose(_.x[1], y.min(), rtol=1e-5, atol=eps_y),
                )
            )
        if len(y_max_idx) > 0:
            bcs.append(
                dde.icbc.DirichletBC(
                    geomtime,
                    make_bc_func(y_max_idx),
                    lambda _, on_boundary: on_boundary and np.isclose(_.x[1], y.max(), rtol=1e-5, atol=eps_y),
                )
            )
        if len(initial_idx) > 0:
            bcs.append(dde.icbc.IC(geomtime, make_bc_func(initial_idx), lambda _, on_initial: on_initial))

        pde_func = self._equation_to_pde_func(dde, equation)
        data_obj = dde.data.TimePDE(
            geomtime,
            pde_func,
            bcs,
            num_domain=self.num_domain,
            num_boundary=self.num_boundary,
            num_initial=self.num_initial,
            num_test=500,
        )

        layer_size = [geomtime.dim] + self.net + [1]
        net = dde.nn.FNN(layer_size, self.activation, self.kernel_initializer)
        model = dde.Model(data_obj, net)
        model.compile(self.optimizer, lr=self.lr)
        #_log("[DeepXDEAdapter] Starting training (3D)")
        try:
            losshistory, train_state = model.train(epochs=self.epochs)
            final_loss = float(losshistory.loss_train[-1][0]) if losshistory.loss_train else float("nan")
        except Exception as e:
            #_log(f"[DeepXDEAdapter] Training failed: {e}")
            y_pred = np.full(data.shape, np.nan)
            final_loss = float("nan")
            return y_pred, final_loss

        coords_pred = np.stack([x.flatten(), y.flatten(), t.flatten()], axis=1)
        y_pred = model.predict(coords_pred).reshape(-1)

        #_log(f"[DeepXDEAdapter] Training complete, final_loss {final_loss}")
        return y_pred, final_loss

    def solve(self, equation: Equation, grids: Sequence[np.ndarray], data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Public entrypoint: detect problem dimension and call the appropriate solver.

        Parameters
        ----------
        equation:
            EPDE-style Equation object describing PDE structure and fitted weights.
        grids:
            Sequence of numpy arrays describing coordinate axes. The expected ordering
            inside the EPDE stack is (t, x, y, ...). The adapter maps to DeepXDE ordering.
        data:
            Numpy array of observed values on the grid (flattenable to ravel()).

        Returns
        -------
        (y_pred, final_loss)
            y_pred is a flat numpy array of predictions (shape matches data.ravel()).
            final_loss is the final training loss (float) or#_log NaN on failure.
        """
        dim = len(grids)
        #_log(f"[DeepXDEAdapter] Detected dimension: {dim}")

        # Set coordinate info from grid_cache (keys order must match grids order)
        keys, _ = global_var.grid_cache.get_all(mode="numpy")
        self._set_coordinate_info(keys)

        if dim == 1:
            return self._solve_1d(equation, grids, data)
        elif dim == 2:
            return self._solve_2d(equation, grids, data)
        elif dim == 3:
            return self._solve_3d(equation, grids, data)
        else:
            raise NotImplementedError(f"Dimension {dim} not supported (only 1,2,3)")