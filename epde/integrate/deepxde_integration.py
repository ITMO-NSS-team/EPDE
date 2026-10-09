"""DeepXDE backend for ``SolverBasedFitness`` -- fixed-coefficient PINN with
an observation term, scored on held-out time.

One solve per trajectory. The loss has no tuned weights::

    loss = sum_e  mean(residual_e^2)          / Var(target_e, train window)
         + sum_v  mean((u_v - u_obs_v)^2)     / Var(u_obs_v,   train window)

* the residual uses the candidate's OWN fitted coefficients
  (``weights_final``, from EPDE's finite-difference design matrix) -- they are
  fixed, the net only has to produce a field that satisfies them;
* the observation term (``PointSetBC``) supervises the observed field on the
  first ``train_frac`` of the time levels only, while the residual is
  enforced on collocation points over the WHOLE domain;
* each term is divided by the variance of its own channel, so both are a
  fraction of their signal left unexplained and weight 1 is commensurate.

The caller scores the solution on the held-out time levels
(``time_split``), where only the candidate equation drives it. That is the
design the PINN experiments on Allen-Cahn validated: the fit on the training
window barely separated the true equation from a wrong one, the held-out
tail separated them ~20x on every seed.

Replaces the former IC/BC strategies, which (a) were fed the equation's
TARGET-term values as if they were the field ``u`` and (b) searched for
IC/BC points at the full-grid edges among inner-domain coordinates, so 2-D/3-D
solves got no conditions at all and 1-D solves pinned ``u(t0) = 0``.
"""

import contextlib
import functools
import math
import os
import warnings
from typing import List, NamedTuple, Tuple

import numpy as np
import torch as _torch

from epde.structure.main_structures import Equation, SoEq
import epde.globals as global_var
from epde.globals import EPDEUsageWarning

#: ``import deepxde`` calls ``torch.set_default_device('cuda')`` at import time
#: whenever a GPU is visible (``deepxde/backend/pytorch/tensor.py``) and never
#: puts it back. That is PROCESS-WIDE and permanent: every bare
#: ``torch.tensor`` / ``torch.zeros`` / ``torch.as_tensor`` built afterwards --
#: here, in the caller's script, in the next test module -- lands on the GPU,
#: and the next ``.numpy()`` on one raises "can't convert cuda:0 device type
#: tensor to numpy". It also silently overrode ``solver.device='cpu'``, which
#: ``search_config`` documents as the way to force the CPU for a bit-identical
#: A/B -- so on this backend that flag used to defeat itself.
#:
#: Snapshot the default, let the import have its way, then give it straight
#: back. ``_solve_device`` re-enters DeepXDE's choice for the length of one
#: solve, which is the only place it is wanted.
_DEFAULT_DEVICE_BEFORE_DDE = _torch.get_default_device()
import deepxde as dde                                            # noqa: E402
#: The device DeepXDE picked for itself, or ``None`` when it changed nothing
#: (a CPU-only machine, where all of this is inert).
DDE_SOLVE_DEVICE = _torch.get_default_device()
if DDE_SOLVE_DEVICE == _DEFAULT_DEVICE_BEFORE_DDE:
    DDE_SOLVE_DEVICE = None
else:
    _torch.set_default_device(_DEFAULT_DEVICE_BEFORE_DDE)


@contextlib.contextmanager
def _solve_device():
    """DeepXDE's own default device, scoped to a single solve.

    Nothing in DeepXDE takes a device argument: its net, its collocation
    tensors and its optimiser state all come from bare torch factories, so the
    default device IS the configuration. Restoring it at import and re-entering
    it only here keeps the GPU where it pays -- the ~120 s PINN solve -- and
    leaves the rest of the process, and the caller, on the device they had.
    """
    if DDE_SOLVE_DEVICE is None:
        yield
        return
    previous = _torch.get_default_device()
    _torch.set_default_device(DDE_SOLVE_DEVICE)
    try:
        yield
    finally:
        _torch.set_default_device(previous)


def _on_solve_device(fn):
    """Run ``fn`` under DeepXDE's default device (see ``_solve_device``)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _solve_device():
            return fn(*args, **kwargs)
    return wrapper


os.makedirs(os.path.expanduser('~/.deepxde'), exist_ok=True)


# The split and its yardsticks live in ``heldout`` (numpy only), so a backend
# that never imports deepxde can share them. Re-exported: every existing
# ``from epde.integrate.deepxde_integration import time_split`` keeps working.
from epde.integrate.heldout import (DeepXDEConfigError, TimeSplit,   # noqa: E402,F401
                                    time_split, variance_weight)


def _grad_norms(losses, params) -> List[float]:
    """``||grad_params L_i||`` per loss term (torch), one backward each.

    A term detached from the parameters gives an all-``None`` gradient tuple;
    ``sum()`` over it returns the int 0 and ``torch.sqrt(0)`` raises TypeError,
    so that case is answered with 0.0 and left for the caller's floor.
    """
    import torch

    norms = []
    for idx in range(len(losses)):
        grads = torch.autograd.grad(losses[idx], params, retain_graph=True,
                                    allow_unused=True)
        live = [g for g in grads if g is not None]
        if not live:
            norms.append(0.0)
            continue
        norms.append(float(torch.sqrt(sum((g ** 2).sum() for g in live))))
    return norms


class _ValCheckpoint:
    """Keeps the parameters that scored best on the validation block.

    A PURE OBSERVER. It reads ``net(x)`` under ``no_grad`` and clones a
    state_dict; it never touches ``loss_weights``, optimiser state, the RNG, or
    ``dde.grad``'s cache, so a run with checkpointing on follows the identical
    trajectory to one with it off.

    Why not select on the training loss instead: within the L-BFGS phase the
    last iterate already IS its argmin (strong-Wolfe accepts only descent), so
    such a rule recovers nothing; and across candidates the training loss is not
    even comparable, since each candidate's weights are built from its own
    target term's variance or its own initial gradient norms. The validation
    error is a pure function of the parameters, which is what makes one
    incumbent meaningful across both optimiser phases.

    ``model.predict`` is deliberately NOT used: it fires predict callbacks and
    calls ``dde.grad.clear()``, and that cache is the one
    ``_PerStepGradNormStep`` walks across its per-term backward passes.
    """

    def __init__(self, net, x_val, y_val, scales, err_func, period_adam,
                 period_lbfgs):
        import torch

        self._torch = torch
        self.net = net
        self.x_val = x_val                      #: (n_val_points, n_dims) tensor
        self.y_val = y_val                      #: (n_val_points, n_vars) tensor
        self.scales = scales                    #: per-variable train-block std
        self.err_func = err_func
        self.period = {'adam': max(1, int(period_adam)),
                       'lbfgs': max(1, int(period_lbfgs))}
        self.phase = 'adam'
        self.calls = 0
        self.best = float('inf')
        self.best_state = None
        self.best_step = None
        self.best_phase = None
        self.last = float('inf')
        self.n_checkpoints = 0
        self.n_snapshots = 0
        self.n_skipped = 0
        self.curve = []

    def score(self) -> float:
        torch = self._torch
        with torch.no_grad():
            pred = self.net(self.x_val)
        total = 0.0
        for var in range(self.y_val.shape[1]):
            resid = (pred[:, var] - self.y_val[:, var]) / self.scales[var]
            total += self.err_func(resid)
        return float(total / self.y_val.shape[1])

    def record(self, force: bool = False):
        """Evaluate and, if improved, snapshot. ``force`` ignores the period --
        used for the terminal iterate, which reaches no hook because the L-BFGS
        loop breaks before its callbacks fire."""
        self.calls += 1
        if not force and (self.calls - 1) % self.period[self.phase]:
            return
        value = self.score()
        self.n_checkpoints += 1
        if not np.isfinite(value):
            self.n_skipped += 1
            return
        self.last = value
        self.curve.append((self.calls, self.phase, value))
        # Strict '<': the step()-entry closure re-evaluates the point it just
        # moved to, so duplicates are guaranteed and '<=' would silently prefer
        # the later copy. The earliest tied iterate wins.
        if value < self.best:
            self.best = value
            self.best_state = {k: v.detach().clone()
                               for k, v in self.net.state_dict().items()}
            self.best_step, self.best_phase = self.calls, self.phase
            self.n_snapshots += 1

    def restore(self) -> str:
        if self.best_state is None:
            return 'last'
        self.net.load_state_dict(self.best_state)
        return 'best'

    def summary(self, selected: str) -> dict:
        return {'best_val': self.best, 'last_val': self.last,
                'improvement_ratio': (self.last / self.best
                                      if np.isfinite(self.best) and self.best > 0
                                      else float('nan')),
                'best_step': self.best_step, 'best_phase': self.best_phase,
                #: ``best_step == last_step`` means the final iterate won, i.e.
                #: selection changed nothing on this solve.
                'last_step': self.calls,
                'best_is_last': self.best_step == self.calls,
                'n_checkpoints': self.n_checkpoints,
                'n_snapshots': self.n_snapshots,
                'n_skipped_nonfinite': self.n_skipped,
                'n_val_points': int(self.y_val.shape[0]),
                'selected': selected, 'curve': self.curve}


def _install_val_callback(model, ckpt):
    """Adam phase: ``_train_sgd`` fires callbacks every iteration, around
    ``_train_step``, so this observes the same points under the stock step and
    under ``_PerStepGradNormStep``."""
    class _Cb(dde.callbacks.Callback):
        def on_epoch_end(self):
            ckpt.record()

    model.callbacks = dde.callbacks.CallbackList(callbacks=[_Cb()])
    model.callbacks.set_model(model)
    return model


def _install_val_closure_hook(model, ckpt):
    """L-BFGS phase: wrap ``opt.step`` so the checkpointer sees the parameter
    vectors the optimiser EVALUATES.

    Chunking the phase via ``iter_per_step`` would give accepted iterates only,
    but it is NOT equivalent to one long call: torch's ``max_ls`` budget is
    computed per ``step()`` (``max_eval - current_evals``), so a short chunk
    truncates individual line searches and changes which step is accepted, and
    its ``current_evals >= max_eval`` break can end the whole phase while
    looking like convergence. This seam adds no ``step()`` boundary, no extra
    closure evaluation and no extra ``_test``, leaving the trajectory bitwise
    unchanged; the cost is that rejected strong-Wolfe trial points are also
    scored. Their validation error is still a legitimate measurement of that
    parameter vector, and ``_directional_evaluate`` restores the parameters
    afterwards, so nothing leaks back into the optimisation.
    """
    opt = model.opt
    stock_step = opt.step

    def step(closure=None, *args, **kwargs):
        def watched():
            out = closure()
            ckpt.record()
            return out
        return stock_step(watched if closure is not None else closure,
                          *args, **kwargs)

    opt.step = step
    return lambda: setattr(opt, 'step', stock_step)


def _install_lbfgs_stall_guard(model):
    """DeepXDE's L-BFGS loop cannot terminate when torch converges AT ENTRY.

    ``torch.optim.LBFGS.step`` tests ``max|grad| <= tolerance_grad`` before its
    first iteration and returns immediately (``lbfgs.py``: ``if opt_cond:
    return orig_loss``) WITHOUT incrementing ``n_iter``. DeepXDE's
    ``_train_pytorch_lbfgs`` then reads the unchanged counter and evaluates its
    convergence test ``prev_n_iter == n_iter - 1`` as ``prev == prev - 1``,
    which is False; it adds ``n_iter - prev_n_iter == 0`` to the step counter,
    leaves ``prev_n_iter`` where it was, and spins on
    ``while prev_n_iter < maxiter`` forever, re-printing one row.

    The condition has never fired here -- gtol has never held at entry -- but
    ``lbfgs_gtol_rel`` is a knob on exactly that test, so enabling it would hang
    a solve rather than stop it. This reinstalls DeepXDE's loop ON THE INSTANCE
    with the same logic plus a stall break, and records why the phase ended:
    candidates are otherwise scored at materially different degrees of
    convergence (measured 5829-7982 iterations against ``maxiter=15000``) with
    nothing saying which stopped why.

    ``model.lbfgs_exit['reason']``:

    * ``'gtol'`` -- the gradient test ended the phase (DeepXDE's absolute
      ``gtol``, or ``lbfgs_gtol_rel``'s). A ``step`` that adds no iteration is
      ALWAYS this: the entry return above is the only way ``step`` can finish
      without iterating. A gradient stop mid-``step`` ends that call early and
      the next one returns at entry, so it lands here too.
    * ``'maxiter'`` -- the iteration budget ran out. The last ``step`` is
      bounded to what the budget still allows, so ``iterations`` is exactly
      ``maxiter`` -- not, as it once was, up to ``iter_per_step - 1`` past it.
    * ``'maxeval'`` -- a ``step``'s only iteration used up that call's
      evaluation budget (``fun_per_step``); a matter of tiny budgets.
    * ``'converged'`` -- DeepXDE's own test (a ``step`` made exactly one
      iteration) for any other reason. Measured: the float32 line search
      returns a zero step, torch breaks, and the next call does the same.
    * ``'nonfinite'`` -- the loss or its gradient is NaN/inf: the phase
      diverged. DeepXDE's ``_test`` stops on a NaN loss, so this also
      replaces the ``stop_training`` that stop would otherwise read as.
    * ``'stop_training'`` -- a callback asked to stop.
    * ``'no_step'`` -- the optimiser holds no state after a step; nothing ran.

    A one-iteration ``step`` is where these overlap, so it is resolved in the
    order nonfinite, maxiter (what the loop itself stops on, whatever the
    gradient), gtol, maxeval, converged.

    ``model.lbfgs_exit['iterations']`` is torch's own ``n_iter``: the
    iterations actually taken. DeepXDE's test breaks before it counts the
    iteration it detects, so ``prev_n_iter`` alone is one short there.
    """
    import types

    def loop(self, verbose=1):
        prev_n_iter = 0
        reason = 'maxiter'
        maxiter = dde.optimizers.config.LBFGS_options['maxiter']
        # torch re-reads ``max_iter`` from the param group at the top of every
        # ``step`` (``lbfgs.py``), so hand the LAST call only what the budget
        # still allows. Without this the cap is tested BETWEEN calls only, and a
        # call that ends early -- normally on torch's own PER-CALL evaluation
        # budget ``max_eval``, not on any maxiter arithmetic -- is followed by a
        # fresh full ``iter_per_step``: measured, ``lbfgs_maxiter=10`` ran 18
        # iterations, and the shipped 2000 could in principle run 2999. On the
        # two gate systems it never did -- 2000 is an exact multiple of the
        # 1000-iteration chunk, so AC stopped at exactly 2000 and Duffing
        # converged below the cap, and the shipped gate numbers are unchanged
        # by this fix. What it does change is every cap that is NOT a multiple
        # of the chunk, which silently bought more iterations than it asked
        # for: at lr=1e-4 a cap of 1500 scored the 2000-iteration result.
        # Only the tail is cut. Under a strong-Wolfe line search ``max_iter``
        # bounds nothing but the iteration ``while``, so every iteration up to
        # the cap is bitwise what it was, and a phase whose calls never end
        # early is untouched. ``max_eval`` is deliberately NOT scaled with it:
        # shortening a line search's evaluation budget would change which step
        # is accepted. ``prev_n_iter < maxiter`` keeps the bound >= 1.
        per_step = self.opt.param_groups[0]['max_iter']
        while prev_n_iter < maxiter:
            self.opt.param_groups[0]['max_iter'] = min(per_step,
                                                       maxiter - prev_n_iter)
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()
            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size))
            evals_before = self.opt.state_dict()['state'].get(0, {}).get(
                'func_evals', 0)
            self._train_step(self.train_state.X_train, self.train_state.y_train,
                             self.train_state.train_aux_vars)
            state = self.opt.state_dict()['state'].get(0)
            if state is None or 'n_iter' not in state:
                # Stop rather than raise inside the solve, and say so.
                reason = 'no_step'
                break
            n_iter = state['n_iter']
            if n_iter == prev_n_iter:
                reason = 'gtol'                         # the guard
                break
            if prev_n_iter == n_iter - 1:               # DeepXDE's own test
                prev_n_iter = n_iter
                gradient = _lbfgs_gradient_test(self)
                if gradient == 'nonfinite':
                    reason = 'nonfinite'
                elif n_iter >= maxiter:
                    reason = 'maxiter'
                elif gradient == 'gtol':
                    reason = 'gtol'
                elif (state['func_evals'] - evals_before
                      >= self.opt.param_groups[0]['max_eval']):
                    reason = 'maxeval'
                else:
                    reason = 'converged'
                break
            self.train_state.iteration += n_iter - prev_n_iter
            self.train_state.step += n_iter - prev_n_iter
            prev_n_iter = n_iter
            self._test(verbose=verbose)
            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()
            if self.stop_training:
                # ``_test``'s own stop rule; anything else was a callback.
                diverged = (np.isnan(self.train_state.loss_train).any()
                            or np.isnan(self.train_state.loss_test).any())
                reason = 'nonfinite' if diverged else 'stop_training'
                break
        self.lbfgs_exit = {'reason': reason, 'iterations': int(prev_n_iter)}

    model._train_pytorch_lbfgs = types.MethodType(loop, model)
    return model


def _lbfgs_gradient_test(model):
    """torch's stopping test ``max|grad| <= tolerance_grad``, re-run at the
    current parameters: ``'gtol'`` if it holds, ``'nonfinite'`` if the loss or
    gradient is NaN/inf (where torch's test is simply False), else ``None``.

    Rebuilds the loss exactly as DeepXDE's L-BFGS closure does (the same
    training batch, ``outputs_losses_train`` with the model's weights applied,
    ``torch.sum``), and compares in torch like ``lbfgs.py`` does, so the answer
    is the one torch itself reached at that point. ``autograd.grad`` leaves
    ``.grad`` alone, and nothing after the phase reads it anyway.
    """
    import torch

    state = model.train_state
    losses = model.outputs_losses_train(state.X_train, state.y_train,
                                        state.train_aux_vars)[1]
    total = torch.sum(losses)
    if not bool(torch.isfinite(total)):
        return 'nonfinite'
    group = model.opt.param_groups[0]
    grads = torch.autograd.grad(total, group['params'], allow_unused=True)
    flat = [g.reshape(-1) for g in grads if g is not None]
    if not flat:
        return 'gtol'                    # torch reads missing grads as zeros
    flat = torch.cat(flat)
    if not bool(torch.isfinite(flat).all()):
        return 'nonfinite'
    return 'gtol' if bool(flat.abs().max() <= group['tolerance_grad']) else None


def normalised_grad_weights(norms, floor_rel: float = 1e-3, cap: float = 1e4,
                            stats: dict = None):
    """``1/||grad L_i||``, floored RELATIVE to the largest norm and rescaled to
    ``mean(w) == 1``; ``None`` when the vector is unusable.

    An ABSOLUTE floor is safe once, at initialisation, where no term has a
    vanishing gradient. Per step it is the expected end state of a satisfied
    term, and ``1/tiny`` overflows to ``inf`` in the float32 loss tensor --
    which ``np.isnan`` does not catch, so the run would continue on a corrupted
    objective. Rescaling to mean 1 keeps the total gradient scale fixed across
    refreshes, so the EMA below compares like with like.
    """
    g = np.asarray(norms, dtype=np.float64)
    if g.size == 0 or not np.isfinite(g).all():
        return None
    g_max = float(g.max())
    if g_max <= 0.0:
        return None
    floored = bool(np.any(g < floor_rel * g_max))
    w = 1.0 / np.maximum(g, floor_rel * g_max)
    w = w / float(np.mean(w))
    capped = bool(np.any(w > cap))
    w = np.minimum(w, cap)
    if stats is not None:
        # Whether the two guards actually BIND: a guard that never binds cannot
        # be responsible for a result, and one that binds often is a knob.
        stats['floored'] = stats.get('floored', 0) + int(floored)
        stats['capped'] = stats.get('capped', 0) + int(capped)
        stats['spread'] = max(stats.get('spread', 0.0),
                              float(g_max / max(float(g.min()), 1e-300)))
    return [float(x) for x in w] if np.isfinite(w).all() else None


class _PerStepGradNormStep:
    """Drop-in for ``Model.train_step`` that refreshes ``1/||grad L_i||`` from
    the SAME forward pass the step consumes.

    DeepXDE's stock step is one forward plus one backward of ``sum(losses)``
    (``model.py`` ``_compile_pytorch``). Taking the per-term gradients instead
    and assembling ``p.grad = sum_i w_i g_i`` gives the identical update, so the
    only extra cost is N-1 backward passes -- no second forward. ``period > 1``
    refreshes every k steps and takes the ordinary single-backward path in
    between.

    Installed AFTER the first-order ``compile`` (which reassigns
    ``train_step``), and wiped by the next ``compile`` -- which is how the
    L-BFGS phase gets its frozen weights.
    """

    def __init__(self, model, weights, ema: float = 0.9, floor_rel: float = 1e-3,
                 cap: float = 1e4, period: int = 1):
        self.model = model
        self.w = [float(x) for x in weights]
        self.ema = float(ema)
        self.floor_rel = float(floor_rel)
        self.cap = float(cap)
        self.period = max(1, int(period))
        self.calls = 0
        self.refreshes = 0
        self.rejected = 0
        #: How often each guard BOUND, and the widest gradient-norm spread seen
        #: -- so a result can be attributed to the weighting rather than to the
        #: floor/cap, or not.
        self.stats = {'floored': 0, 'capped': 0, 'spread': 0.0}

    def summary(self) -> dict:
        out = dict(self.stats)
        out.update(refreshes=self.refreshes, rejected=self.rejected,
                   weights=[float(w) for w in self.w],
                   floor_rel=self.floor_rel, cap=self.cap,
                   ema=self.ema, period=self.period)
        return out

    def install(self):
        # We weight inside the step, so DeepXDE must not weight again; leaving
        # it None also means losshistory records RAW per-term losses.
        self.model.loss_weights = None
        self.model.train_step = self
        return self

    def __call__(self, inputs, targets, auxiliary_vars):
        import torch

        model = self.model
        params = [p for p in model.net.parameters() if p.requires_grad]
        _, losses = model.outputs_losses_train(inputs, targets, auxiliary_vars)
        if len(losses) != len(self.w):
            raise RuntimeError(
                f'DeepXDE returned {len(losses)} loss terms but the weight vector '
                f'has {len(self.w)}')

        refresh = (self.calls % self.period == 0)
        self.calls += 1
        model.opt.zero_grad(set_to_none=True)

        if not refresh:
            total = sum(w * loss for w, loss in zip(self.w, losses))
            total.backward()
        else:
            grads, norms = [], []
            for idx in range(len(losses)):
                g = torch.autograd.grad(losses[idx], params,
                                        retain_graph=(idx < len(losses) - 1),
                                        allow_unused=True)
                grads.append(g)
                live = [z for z in g if z is not None]
                norms.append(float(torch.sqrt(sum((z ** 2).sum() for z in live)))
                             if live else 0.0)
            fresh = normalised_grad_weights(norms, self.floor_rel, self.cap,
                                            stats=self.stats)
            if fresh is not None:
                # EMA, so a term that momentarily flattens cannot take over in
                # one step; a rejected (non-finite) vector keeps the old weights.
                self.w = [self.ema * old + (1.0 - self.ema) * new
                          for old, new in zip(self.w, fresh)]
                self.refreshes += 1
            else:
                self.rejected += 1
            for param_idx, param in enumerate(params):
                acc = None
                for w, g in zip(self.w, grads):
                    if g[param_idx] is None:
                        continue
                    contrib = w * g[param_idx]
                    acc = contrib if acc is None else acc + contrib
                param.grad = acc
        model.opt.step()
        if getattr(model, 'lr_scheduler', None) is not None:
            model.lr_scheduler.step()


def _input_columns(grids) -> np.ndarray:
    """Coordinates in DeepXDE's input order ``[space..., t]``.

    EPDE's axis 0 is time; ``GeometryXTime`` puts time LAST. This is the same
    ordering ``DeepXDEAdapter._set_coordinate_info`` encodes in ``coord_map``
    for the residual's derivatives.
    """
    columns = [np.asarray(g).reshape(-1) for g in grids[1:]]
    columns.append(np.asarray(grids[0]).reshape(-1))
    return np.stack(columns, axis=1)


def _geometry(grids):
    """The solve domain over the FULL grid extent (the residual holds
    everywhere, not just on the inner domain)."""
    t = np.asarray(grids[0])
    time = dde.geometry.TimeDomain(float(t.min()), float(t.max()))
    space = [np.asarray(g) for g in grids[1:]]
    if not space:
        return time
    if len(space) == 1:
        geom = dde.geometry.Interval(float(space[0].min()), float(space[0].max()))
    else:
        geom = dde.geometry.Rectangle([float(s.min()) for s in space],
                                      [float(s.max()) for s in space])
    return dde.geometry.GeometryXTime(geom, time)


class DeepXDEAdapter:
    def __init__(self, pretrained_net=None, **config):
        self.pretrained_net = pretrained_net
        self.config = config or {}
        self.net = self.config.get('net', [50, 50, 50, 50])
        self.activation = self.config.get('activation', 'tanh')
        self.optimizer = self.config.get('optimizer', 'adam')
        self.lr = self.config.get('lr', 1e-3)
        self.kernel_initializer = self.config.get('kernel_initializer', 'Glorot normal')
        self.num_domain = int(self.config.get('num_domain', 2000))
        self.num_test = int(self.config.get('num_test', 500))
        self.epochs = int(self.config.get('epochs', 10000))
        #: Learning-rate schedule for the first-order phase, passed to
        #: ``Model.compile(decay=...)`` -- e.g. ``('cosine', epochs, 1e-5)``
        #: or ``('step', 2000, 0.5)``. None = constant ``lr``.
        decay = self.config.get('decay', None)
        self.decay = tuple(decay) if decay is not None else None
        #: Adam iterations are followed by this many L-BFGS iterations (0 = off).
        self.lbfgs_maxiter = int(self.config.get('lbfgs_maxiter', 0))
        #: Fraction of the time levels the observation term sees; the rest is
        #: where the fitness host scores the solution.
        #: How the terms are made commensurate. 'variance' (default): each is
        #: divided by its own channel's variance, a yardstick computed once
        #: from the data. 'gradient': each is divided by the norm of its own
        #: gradient at INITIALISATION, so every term starts with the same
        #: gradient magnitude and no data yardstick is needed at all -- the
        #: weights are frozen after that, because weights recomputed per step
        #: are not the gradient of any fixed objective (L-BFGS's curvature
        #: history and line search both assume one) and a term approaching
        #: satisfaction would take over as its gradient vanishes.
        #: 'gradient_per_step' refreshes those weights every ``grad_weight_period``
        #: first-order steps (EMA-smoothed) and FREEZES the last vector for the
        #: L-BFGS phase -- see ``_PerStepGradNormStep`` and the note in
        #: ``_solve_trajectory`` on why the second-order phase must not float.
        self.loss_weight_mode = self.config.get('loss_weight_mode', 'variance')
        if self.loss_weight_mode not in ('variance', 'gradient', 'gradient_per_step'):
            raise ValueError("deepxde_config loss_weight_mode must be 'variance', "
                             f"'gradient' or 'gradient_per_step', got "
                             f"{self.loss_weight_mode!r}")
        self.grad_weight_period = max(1, int(self.config.get('grad_weight_period', 1)))
        self.grad_weight_ema = float(self.config.get('grad_weight_ema', 0.9))
        if not 0.0 <= self.grad_weight_ema < 1.0:
            raise ValueError('deepxde_config grad_weight_ema must lie in [0, 1), '
                             f'got {self.grad_weight_ema!r}')
        self.grad_weight_floor_rel = float(self.config.get('grad_weight_floor_rel', 1e-3))
        self.grad_weight_cap = float(self.config.get('grad_weight_cap', 1e4))
        #: Rescale the FROZEN gradient weights to mean 1. Off by default so the
        #: measured 'gradient' arms reproduce, but see ``lbfgs_gtol_rel``: raw
        #: 1/||grad|| has an arbitrary overall SCALE (mean ~6.5 on Allen-Cahn),
        #: and the L-BFGS stopping test is absolute, so the scale silently buys
        #: or costs convergence.
        self.grad_weight_normalise = bool(self.config.get('grad_weight_normalise', False))
        #: Multiplies the PHYSICS terms' weights only -- the knob for asking
        #: whether the physics:data balance matters at all.
        self.phys_weight_scale = float(self.config.get('phys_weight_scale', 1.0))
        #: Stop L-BFGS at this fraction of the gradient it STARTS with, instead
        #: of DeepXDE's absolute gtol=1e-8 on max|grad| (config.py:15 ->
        #: torch tolerance_grad, lbfgs.py:370). Without it, two arms whose
        #: losses differ by an overall factor stop at different convergence,
        #: which confounds any comparison between them.
        rel = self.config.get('lbfgs_gtol_rel', None)
        self.lbfgs_gtol_rel = float(rel) if rel is not None else None
        # ---- learning-rate-free options. All off by default; each is ONE
        # mechanism, measured against its control on the gate harness
        # (projects/pic/data/solver_gate).
        #: 'affine' maps every input column onto [-1, 1] from the grid's own
        #: bounds (coordinates, not data). Duffing feeds t in [0, 20] to tanh.
        self.input_transform = self.config.get('input_transform', None)
        #: 'train_moments' makes the output mu + sd * net, with mu, sd of the
        #: observed field on the train window (the observation term's yardstick).
        self.output_transform = self.config.get('output_transform', None)
        #: where training starts. None: the Glorot init. 'data_prefit': an
        #: observation-only L-BFGS fit from it (``prefit_maxiter`` iterations).
        #: 'lstsq_last_layer': the output layer solved by least squares on the
        #: train observations at the initial hidden features. Applied AFTER the
        #: gradient-weight probe, so the weights are those of the Glorot init:
        #: at a data fit the observation gradient is ~0 and 1/||grad|| explodes.
        self.init = self.config.get('init', None)
        self.prefit_maxiter = int(self.config.get('prefit_maxiter', 1000))
        #: 'shared_data_fit': ONE observation-only fit per dataset, run to its
        #: natural L-BFGS stop (this is a safety cap, not a budget), cached and
        #: loaded as the start of every candidate. ``shared_fit_seed`` None: the
        #: fit starts from this solve's own initial weights (paired arms on a
        #: seeded harness); an int: from a fixed-seed Glorot draw inside
        #: ``torch.random.fork_rng`` (the global RNG is untouched), so a live
        #: search gets ONE start for all candidates. ``shared_fit_dir`` persists
        #: the cache across processes.
        self.shared_fit_maxiter = int(self.config.get('shared_fit_maxiter', 20000))
        seed = self.config.get('shared_fit_seed', None)
        self.shared_fit_seed = int(seed) if seed is not None else None
        self.shared_fit_dir = self.config.get('shared_fit_dir', None)
        #: 'float64' runs the shared data fit on a double-precision copy and
        #: loads the result back at the net's own precision. In float32 the
        #: fit's strong-Wolfe line search stalls at a zero step once the
        #: decrease it could confirm drops below the loss's rounding noise --
        #: measured on Duffing (see the Sep 29 2026 note in the harness).
        self.shared_fit_precision = self.config.get('shared_fit_precision', 'float32')
        #: the observation (data) loss. 'mse': u - u_obs, as always. The
        #: observation equation u = u_obs has two terms, so its scale-invariant
        #: form divides by their mass: 'sinv_live' by |u| + |u_obs| (the net's
        #: own value, differentiated; the ratio lies in [0, 1]), 'sinv_data' by
        #: 2 |u_obs| (frozen; unbounded where the data cross zero). No floor: an
        #: exact 0/0 counts as satisfied (``residual_terms.cancellation_ratio``).
        #: The gradient loss weights are taken on this loss, as for ``pde_loss``.
        self.obs_loss = self.config.get('obs_loss', 'mse')
        #: 'float64': the solve in double precision. The net is still BUILT in
        #: float32 and upcast, so its initial weights equal the float32 arms'.
        self.precision = self.config.get('precision', 'float32')
        #: the physics residual. 'mse': as is. 'sinv_live' / 'sinv_data': divided
        #: pointwise by the term mass |target| + sum_k |c_k term_k| + |b| (the
        #: denominator of ``Discrepancy._compute_scale_invariant``) of the net's
        #: own field (differentiated) / of the data on the train window (frozen,
        #: nearest inner-grid point; the tail copies the last train level).
        #: Gradient weighting is unchanged: the probe then weights the ratio.
        self.pde_loss = self.config.get('pde_loss', 'mse')
        #: 0: no floor (the ratio needs none, see ``residual_terms.cancellation_ratio``)
        self.sinv_floor_rel = float(self.config.get('sinv_floor_rel', 0.0))
        #: sha256 of the initial weights (one device-to-host copy before
        #: training), for pairing arms that must share them
        self.pairing_digest = bool(self.config.get('pairing_digest', False))
        #: the first-order phase: 'adam' (``optimizer``/``lr``, as shipped) or
        #: an optimiser with no learning rate -- 'dog' (``dog_reps_rel``) or
        #: 'prodigy' (``prodigy_d0``); see ``epde.integrate.lr_free``
        self.first_order = self.config.get('first_order', 'adam')
        self.dog_reps_rel = float(self.config.get('dog_reps_rel', 1e-6))
        self.prodigy_d0 = float(self.config.get('prodigy_d0', 1e-6))
        #: an Adam lr taken FROM THE DATA instead of the config.
        #: 'prefit_distance': median |theta_prefit - theta0| / epochs, the
        #: prefit being an observation-only L-BFGS fit from theta0
        #: (``prefit_maxiter``) on a copy; Adam moves a weight by at most ~lr
        #: per step, so this is the smallest lr that lets the median weight
        #: reach its data-fit value within the budget. 'data_selected': the lr
        #: in ``lr_grid`` whose Adam run (``epochs`` steps, on a copy) leaves
        #: the lowest observation-only loss on the train window.
        self.lr_rule = self.config.get('lr_rule', None)
        self.lr_grid = tuple(float(v) for v in self.config.get('lr_grid',
                                                                (1e-5, 1e-4, 1e-3, 1e-2)))
        #: the second-order phase: 'lbfgs' (as shipped) or 'lm' (Levenberg-
        #: Marquardt on the stacked residual vector, ``lm_maxiter`` iterations,
        #: ``lm_tau`` = initial damping relative to max diag(J^T J))
        self.second_order = self.config.get('second_order', 'lbfgs')
        self.lm_maxiter = int(self.config.get('lm_maxiter', 500))
        self.lm_tau = float(self.config.get('lm_tau', 1e-3))
        self.lm_chunk = int(self.config.get('lm_chunk', 256))
        #: 'auto': every spatial axis the DATA show to be periodic
        #: (``data_profile.build_profile``, train window) enters the net as
        #: (cos, sin)(2 pi (x - lo) / L) instead of x, which makes the network
        #: exactly periodic in it with the data's own period L; no other
        #: constant. Time and non-periodic axes are unchanged (or affine-mapped
        #: under ``input_transform='affine'``).
        self.periodic_embedding = self.config.get('periodic_embedding', None)
        #: where the physics is enforced. None: DeepXDE's Hammersley set.
        #: 'data_gradient': the SAME points moved by the sequential (Rosenblatt)
        #: inverse CDF of the train-window data-gradient density -- time first,
        #: then each spatial axis conditional on the earlier ones -- so fronts
        #: get the points; the held-out levels get the last train level's
        #: pattern at the train-window mean density. 'flat': the same warp with
        #: a flat density -- the placement control (the identity up to rounding).
        self.collocation_density = self.config.get('collocation_density', None)
        choices = {'input_transform': (None, 'affine'),
                   'output_transform': (None, 'train_moments'),
                   'init': (None, 'data_prefit', 'lstsq_last_layer', 'shared_data_fit'),
                   'precision': ('float32', 'float64'),
                   'pde_loss': ('mse', 'sinv_live', 'sinv_data'),
                   'first_order': ('adam', 'dog', 'prodigy'),
                   'lr_rule': (None, 'prefit_distance', 'data_selected'),
                   'second_order': ('lbfgs', 'lm'),
                   'periodic_embedding': (None, 'auto'),
                   'collocation_density': (None, 'data_gradient', 'flat'),
                   'shared_fit_precision': ('float32', 'float64'),
                   'obs_loss': ('mse', 'sinv_live', 'sinv_data')}
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise DeepXDEConfigError(f'deepxde_config {key} must be one of {allowed}, '
                                         f'got {getattr(self, key)!r}')
        # ``lr=None`` is how an L-BFGS-only run says it has no learning rate.
        # Anything else must be a positive number: a string used to reach
        # ``compile`` inside the solve's blanket except and come back as an
        # all-NaN "solve failed".
        if self.lr_rule is not None and (self.epochs <= 0 or self.first_order != 'adam'):
            raise DeepXDEConfigError('deepxde_config lr_rule sets the Adam lr; it needs '
                                     "epochs > 0 and first_order='adam'")
        if self.first_order != 'adam' and self.loss_weight_mode == 'gradient_per_step':
            raise DeepXDEConfigError("deepxde_config first_order='dog'/'prodigy' does not "
                                     "combine with loss_weight_mode='gradient_per_step'")
        if self.second_order == 'lm' and self.obs_loss != 'mse':
            raise DeepXDEConfigError("deepxde_config second_order='lm' builds the observation "
                                     "rows itself (u - u_obs); obs_loss must be 'mse'")
        if self.second_order == 'lm' and self.lbfgs_maxiter > 0:
            raise DeepXDEConfigError("deepxde_config second_order='lm' replaces L-BFGS; "
                                     'set lbfgs_maxiter=0')
        if self.second_order == 'lm' and float(self.config.get('val_frac', 0.0)) > 0.0:
            raise DeepXDEConfigError("deepxde_config second_order='lm' does not feed the "
                                     'validation checkpoint; use val_frac=0')
        if self.first_order != 'adam' and self.decay is not None:
            raise DeepXDEConfigError("deepxde_config decay schedules Adam's lr; an lr-free "
                                     "first_order has none to schedule")
        # checked here, not inside the solve: there, a bad value would reach the
        # blanket except and come back as a silent all-NaN solve
        for key, value, low in (('lm_maxiter', self.lm_maxiter, 1),
                                ('lm_chunk', self.lm_chunk, 1),
                                ('prefit_maxiter', self.prefit_maxiter, 0),
                                ('shared_fit_maxiter', self.shared_fit_maxiter, 1)):
            if value < low:
                raise DeepXDEConfigError(f'deepxde_config {key} must be >= {low}, got {value!r}')
        for key, value in (('lm_tau', self.lm_tau), ('dog_reps_rel', self.dog_reps_rel),
                           ('prodigy_d0', self.prodigy_d0)):
            if not value > 0.0:
                raise DeepXDEConfigError(f'deepxde_config {key} must be > 0, got {value!r}')
        if not self.lr_grid or not all(v > 0.0 for v in self.lr_grid):
            raise DeepXDEConfigError(f'deepxde_config lr_grid must be non-empty and positive, '
                                     f'got {self.lr_grid!r}')
        if self.sinv_floor_rel < 0.0:
            raise DeepXDEConfigError('deepxde_config sinv_floor_rel must be >= 0')
        if self.lr is None:
            lr_free = (self.epochs == 0 or self.first_order != 'adam'
                       or self.lr_rule is not None)
            if not lr_free:
                raise DeepXDEConfigError('deepxde_config lr=None needs epochs=0, an lr-free '
                                         'first_order or an lr_rule (a first-order phase '
                                         'needs a step size)')
        else:
            try:
                self.lr = float(self.lr)
            except (TypeError, ValueError):
                raise DeepXDEConfigError(f'deepxde_config lr must be a number or None, '
                                         f'got {self.lr!r}') from None
            if not self.lr > 0.0:
                raise DeepXDEConfigError(f'deepxde_config lr must be > 0, got {self.lr!r}')
        #: filled per solve by the options above
        self.last_init_stats = None
        self.last_pairing = None
        #: Filled by the last dynamic solve: how often the floor/cap bound, how
        #: wide the gradient-norm spread got, and the final weights.
        self.last_weight_stats = None
        #: Why the last L-BFGS phase ended and after how many iterations --
        #: 'gtol' | 'maxiter' | 'maxeval' | 'converged' | 'nonfinite' |
        #: 'stop_training' | 'no_step' (see ``_install_lbfgs_stall_guard``).
        #: Candidates are otherwise compared at unrecorded, and unequal,
        #: degrees of convergence. Cleared at the start of every solve, so a
        #: solve that fails leaves None rather than its predecessor's record.
        self.last_lbfgs_exit = None
        self.train_frac = float(self.config.get('train_frac', 0.8))
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError(f'deepxde_config train_frac must lie strictly between '
                             f'0 and 1, got {self.train_frac!r}')
        #: Validation block, carved out of the FIT window (never out of the
        #: scored tail, so the test block is invariant to this). 0 = no val set,
        #: no checkpointing, last iterate -- bit-identical to the two-way split.
        #: It is a real trade: every level it takes is a level the observation
        #: term loses.
        self.val_frac = float(self.config.get('val_frac', 0.0))
        #: 'best_val' returns the checkpoint with the lowest validation error;
        #: 'last' returns the final iterate. Inert when ``val_frac == 0``.
        self.select_by = self.config.get('select_by', 'best_val')
        if self.select_by not in ('best_val', 'last'):
            raise DeepXDEConfigError("deepxde_config select_by must be 'best_val' "
                                     f"or 'last', got {self.select_by!r}")
        #: Target number of validation evaluations PER PHASE. A resolution
        #: budget rather than a period in iterations, so it auto-scales with
        #: ``epochs``/``lbfgs_maxiter`` instead of needing a tuned integer per
        #: system, and bounds the overhead a priori (~0.3% of a solve).
        self.val_checkpoints = max(1, int(self.config.get('val_checkpoints', 100)))
        #: Off puts checkpoints in the Adam phase only -- strictly less
        #: information, zero risk. See ``_install_val_closure_hook``.
        self.checkpoint_lbfgs = bool(self.config.get('checkpoint_lbfgs', True))
        #: Filled when a validation block exists: the selection trace and
        #: whether 'best' differed from 'last' at all.
        self.last_val_stats = None

        self.coordinate_mapping = self.config.get('coordinate_mapping', None)
        self.coord_names = None
        self.coord_map = None
        #: Trajectory whose domain the current solve runs on; set by ``solve``.
        self.domain_key = None

    @property
    def domain_mask(self) -> np.ndarray:
        """The solved trajectory's inner-domain mask, GRID-SHAPED.

        Keyed by the domain being solved (the mask belongs to a trajectory).
        Grid-shaped rather than flat because it indexes the grids
        (``g[mask]``), and those keep their grid shape.
        """
        key = self.domain_key
        if key is None:
            key = global_var.samples_manager.trajecatoryIDs[0]
        return np.asarray(global_var.samples_manager.gFunc('m')[key])

    def _set_coordinate_info(self, coord_names):
        self.coord_names = coord_names
        if self.coordinate_mapping is not None:
            self.coord_map = self.coordinate_mapping
        else:
            spatial_dim = len(coord_names) - 1
            self.coord_map = {}
            for i, name in enumerate(coord_names):
                if i == 0:
                    self.coord_map[name] = spatial_dim
                else:
                    self.coord_map[name] = i - 1

    def _equation_system_to_pde_func(self, dde, eq_list, var_names):
        var_idx_map = {name: i for i, name in enumerate(var_names)}
        from epde.integrate.residual_terms import cancellation_ratio
        sinv = getattr(self, 'pde_loss', 'mse')
        floors = getattr(self, '_sinv_floors', None)
        data_weight = getattr(self, '_sinv_data_weight', None)

        def magnitude(v):
            return v.abs() if hasattr(v, 'abs') else abs(v)

        def pde(x, y):
            residuals = []
            for eq_idx, eq in enumerate(eq_list):
                mass = 0.0
                use_weights = getattr(eq, "weights_final_evald", False) and hasattr(eq, "weights_final")
                residual = y[:, eq_idx:eq_idx + 1] * 0.0
                all_terms = eq.structure
                tgt = eq.target_idx
                for term_idx, term in enumerate(all_terms):
                    if term_idx == tgt:
                        continue
                    # ``weight_index``: both weight vectors SKIP the target, so
                    # indexing them by raw structure position read the wrong
                    # coefficient for every term past it.
                    coeff = (float(eq.weights_final[eq.weight_index(term_idx, tgt)])
                             if use_weights else 1.0)
                    term_val = 1.0
                    for factor in term.structure:
                        fv = self._factor_value_with_map(dde, factor, x, y, self.coord_map, var_idx_map)
                        term_val *= fv
                    residual += coeff * term_val
                    if sinv == 'sinv_live':
                        mass = mass + magnitude(coeff * term_val)
                # The PDE carries a free coefficient (bias) only when the
                # equation HAS one: its support slot ``weights_internal[-1]`` is
                # non-zero (the intercept rule of the coefficient step, which
                # leaves ``weights_final[-1]`` at 0.0 otherwise -- but a bias the
                # equation does not have must not reach the residual whatever
                # the trailing value says). The former presence sniff
                # ``len(weights_final) > len(all_terms)`` was never true.
                has_free = use_weights and float(eq.weights_internal[-1]) != 0.0
                if has_free:
                    residual += float(eq.weights_final[-1]) * (y[:, 0:1] * 0.0 + 1.0)
                target = eq.target
                target_val = 1.0
                for factor in target.structure:
                    fv = self._factor_value_with_map(dde, factor, x, y, self.coord_map, var_idx_map)
                    target_val *= fv
                residual -= target_val
                if sinv == 'sinv_live':
                    mass = (mass + magnitude(target_val)
                            + (abs(float(eq.weights_final[-1])) if has_free else 0.0))
                    residual = cancellation_ratio(residual, mass + floors[eq_idx], _torch)
                elif sinv == 'sinv_data':
                    residual = residual * data_weight(eq_idx, x)
                residuals.append(residual)
            return residuals

        return pde

    @staticmethod
    def _factor_params(factor) -> dict:
        """The factor's parameters BY NAME, from ``params_description``
        (``{'power': ..., 'dim': ..., 'freq': ...}``)."""
        description = getattr(factor, "params_description", None) or {}
        params = getattr(factor, "params", None)
        if params is None:
            return {}
        return {info["name"]: float(params[idx]) for idx, info in description.items()
                if isinstance(info, dict) and "name" in info}

    @classmethod
    def _factor_power(cls, factor) -> float:
        """The factor's ``power`` parameter (1.0 when it has none)."""
        return cls._factor_params(factor).get("power", 1.0)

    @staticmethod
    def _power(value, power: float):
        return value if power == 1.0 else value ** power

    def _axis_column(self, axis, coord_map) -> int:
        """DeepXDE input column of EPDE grid axis ``axis`` (0 = time)."""
        name = self.coord_names[int(round(float(axis)))]
        column = coord_map.get(name, None)
        if column is None:
            raise ValueError(f"grid axis {axis} ({name!r}) has no DeepXDE input column "
                             f"in coord_map {coord_map}")
        return int(column)

    def _factor_value_with_map(self, dde, factor, x, y, coord_map, var_idx_map=None):
        """One factor of the residual as a DeepXDE tensor, dispatched on the
        factor's FAMILY (``ftype``).

        The former chain tested ``variable is not None`` for "a system
        variable", but grid and trig factors carry their family name there
        (``'grids'``, ``'trigonometric'``) and were silently evaluated as
        ``u**power`` through a default index; the grid branch after it was
        unreachable, and an unrecognised factor fell back to 1.0. Families
        the residual cannot express now fail loudly instead.
        """
        params = self._factor_params(factor)
        power = params.get("power", 1.0)
        family = str(getattr(factor, "ftype", ""))
        label = str(getattr(factor, "label", ""))
        variable = getattr(factor, "variable", None)
        var_idx_map = var_idx_map or {}

        # Grid token: grids[dim] ** power (epde.evaluators.grid_eval_fun_np)
        if family == "grids":
            column = self._axis_column(params["dim"], coord_map)
            return self._power(x[:, column:column + 1], power)

        # Trig token: sin/cos(freq * grids[dim]) ** power (trig_eval_fun_np)
        if family == "trigonometric":
            funcs = {"sin": dde.backend.sin, "cos": dde.backend.cos}
            if label not in funcs:
                raise NotImplementedError(f"DeepXDE residual: unknown trigonometric token {label!r}")
            column = self._axis_column(params["dim"], coord_map)
            return self._power(funcs[label](params["freq"] * x[:, column:column + 1]), power)

        # Constant token: value ** power (const_eval_fun_np)
        if family == "constants":
            return self._power(y[:, 0:1] * 0.0 + params["value"], power)

        if variable not in var_idx_map:
            raise NotImplementedError(
                f"DeepXDE residual cannot express factor {label!r} of family {family!r} "
                f"(variable {variable!r}; system variables {sorted(var_idx_map)})")
        idx = var_idx_map[variable]
        # A plain variable token also carries ``is_deriv=True`` with
        # ``deriv_code=[None]``; only a code with a real axis is a derivative.
        # The former truthiness test sent ``u{power: 3}`` down the derivative
        # branch, which skipped the None axis and returned ``u`` -- the power
        # was dropped, so Allen-Cahn's ``5u - 5u^3`` cancelled to zero and the
        # PINN was asked to satisfy the heat equation.
        axes = [ax for ax in (getattr(factor, "deriv_code", None) or []) if ax is not None]
        value = y[:, idx:idx + 1]
        if getattr(factor, "is_deriv", False) and axes:
            for ax in axes:
                value = dde.grad.jacobian(value, x, i=0, j=self._axis_column(ax, coord_map))
        return self._power(value, power)

    def loss_weights(self, eq_list: List[Equation], data_list, train_mask) -> List[float]:
        """``[1/Var(target_e)] * n_eqs + [1/Var(u_obs_v)] * n_vars``, both on
        the training window.

        The order is DeepXDE's: one PDE-residual loss per equation, then the
        conditions in the order they were added (``data/pde.py``
        ``losses``) -- here, one observation term per variable.
        """
        weights = []
        for eq in eq_list:
            if getattr(self, 'pde_loss', 'mse') != 'mse':
                weights.append(1.0)          # the ratio is dimensionless already
                continue
            target = np.asarray(eq.evaluate(active_only=True)[0][self.domain_key]).reshape(-1)
            weights.append(variance_weight(target[train_mask]))
        for observed in data_list:
            weights.append(variance_weight(np.asarray(observed).reshape(-1)[train_mask]))
        return weights

    @staticmethod
    def _unweighted_losses(model):
        """Per-term losses on the training batch, with DeepXDE's own weighting
        switched off around the call."""
        inputs, targets = model.data.train_next_batch(None)[:2]
        weights_before, model.loss_weights = model.loss_weights, None
        try:
            _, losses = model.outputs_losses_train(inputs, targets, None)
        finally:
            model.loss_weights = weights_before
        return losses

    @classmethod
    def _gradient_weights(cls, model) -> List[float]:
        """``1 / ||grad L_i||`` per loss term at the CURRENT state, so every
        term contributes the same gradient magnitude.

        Costs one backward pass per term, once. The caller freezes the result:
        see ``loss_weight_mode``.
        """
        losses = cls._unweighted_losses(model)
        params = [p for p in model.net.parameters() if p.requires_grad]
        tiny = float(np.finfo(np.float64).tiny)
        return [1.0 / max(norm, tiny) for norm in _grad_norms(losses, params)]

    @classmethod
    def _reported_loss(cls, model, weights) -> float:
        """``sum_i w_i L_i`` on the training batch under the REFERENCE weights.

        Read from a fresh unweighted forward rather than from
        ``losshistory.loss_train[-1]``: that row carries whatever weighting was
        live when it was recorded, which under 'gradient_per_step' is a moving
        yardstick (and is raw during the dynamic phase). This keeps the reported
        loss on one scale across modes.
        """
        losses = cls._unweighted_losses(model)
        return float(sum(float(w) * float(loss.detach())
                         for w, loss in zip(weights, losses)))

    @classmethod
    def _max_abs_grad(cls, model, weights) -> float:
        """``max|grad|`` of the weighted loss at the current parameters -- the
        quantity DeepXDE's ``gtol`` is compared against."""
        import torch

        losses = cls._unweighted_losses(model)
        params = [p for p in model.net.parameters() if p.requires_grad]
        total = sum(float(w) * loss for w, loss in zip(weights, losses))
        grads = torch.autograd.grad(total, params, allow_unused=True)
        return max((float(g.abs().max()) for g in grads if g is not None), default=0.0)

    @staticmethod
    def _set_lbfgs_budget(maxiter: int, gtol: float = None):
        """``set_LBFGS_options(maxiter=...)`` alone does NOT bound the run:
        ``iter_per_step`` is derived once at import from the default 15000, so
        torch's LBFGS is built with ``max_iter=1000`` and a smaller cap
        overshoots (measured: 37 iterations for ``maxiter=20``). Setting the
        per-step budget too makes the configured cap real.

        This call is TOTAL, which is what keeps one solve from inheriting the
        previous one's budget. ``set_LBFGS_options`` takes every option as a
        keyword with a default and writes all six unconditionally, so naming
        only ``maxiter`` here does not "leave the rest alone" -- it restores
        ``maxcor``/``ftol``/``gtol``/``maxls`` to the library defaults, which
        are the values this adapter wants. The two derived keys that
        ``set_LBFGS_options`` does NOT touch (DeepXDE computes them once at
        import, ``optimizers/config.py``) are rewritten below. This function
        being the only caller in the tree, the resulting global state is a pure
        function of ``(maxiter, gtol)`` -- pinned by
        ``TestTheLBFGSBudget::test_the_budget_does_not_depend_on_call_order``.
        Adding a knob here means passing it on EVERY call, not just when it is
        set, or that property is lost."""
        if gtol is None:
            dde.optimizers.set_LBFGS_options(maxiter=int(maxiter))
        else:
            dde.optimizers.set_LBFGS_options(maxiter=int(maxiter), gtol=float(gtol))
        try:
            from deepxde.optimizers import config as _dde_opt_config
            per_step = min(int(maxiter), 1000)
            _dde_opt_config.LBFGS_options['iter_per_step'] = per_step
            _dde_opt_config.LBFGS_options['fun_per_step'] = int(1.25 * per_step)
        except Exception:                                    # pragma: no cover
            pass

    def _make_checkpoint(self, net, inner_coords, data_list, split):
        """A ``_ValCheckpoint`` for this solve, or ``None`` when no validation
        block was asked for (in which case nothing is installed and the run is
        bit-identical to the two-way split).

        Residuals are divided by each variable's TRAIN-block standard deviation
        -- the observation terms' own yardstick, reused rather than reinvented.
        For a single-variable system that is a positive constant, so the
        selection is exactly argmin RMSE; it only bites for a system of
        equations, where one net must yield one checkpoint and the largest-scale
        variable would otherwise own the choice.
        """
        import torch

        if self.val_frac <= 0.0 or not split.val.any():
            return None
        device = next(net.parameters()).device
        dtype = next(net.parameters()).dtype
        y_val = np.stack([np.asarray(v, dtype=np.float64).reshape(-1)[split.val]
                          for v in data_list], axis=1)
        scales = [max(float(np.std(np.asarray(v, dtype=np.float64)
                                   .reshape(-1)[split.train])),
                      float(np.finfo(np.float64).tiny)) for v in data_list]
        metric = str(getattr(self, 'error_metric', 'rmse')).lower()
        if metric in ('l2', 'rmse'):
            err = lambda r: float(torch.sqrt(torch.mean(r ** 2)))
        elif metric == 'mae':
            err = lambda r: float(torch.mean(torch.abs(r)))
        else:
            raise DeepXDEConfigError(f'no validation metric for {metric!r}')
        adam_period = math.ceil(max(1, self.epochs) / self.val_checkpoints)
        lbfgs_period = math.ceil(max(1, int(1.25 * self.lbfgs_maxiter))
                                 / self.val_checkpoints)
        return _ValCheckpoint(
            net,
            torch.as_tensor(inner_coords[split.val], dtype=dtype, device=device),
            torch.as_tensor(y_val, dtype=dtype, device=device),
            scales, err, adam_period, lbfgs_period)

    def _solve_trajectory(self, eq_list, var_names, grids, data_list):
        # Per-solve diagnostics: a solve that raises must not leave the
        # previous candidate's records behind to be read as its own.
        self.last_lbfgs_exit = None
        self.last_val_stats = None
        self.last_weight_stats = None
        self.last_init_stats = None
        self.last_pairing = None
        self._sinv_floors, self._sinv_data_weight = None, None
        self.last_sinv_stats = None
        self.last_lr_stats = None
        self.last_lm_stats = None
        grids = [np.asarray(g) for g in grids]
        mask = self.domain_mask
        inner_coords = _input_columns([g[mask] for g in grids])
        split = time_split(grids[0][mask], self.train_frac, self.val_frac)
        train = split.train

        bcs = []
        self.last_obs_stats = None
        for var_idx, observed in enumerate(data_list):
            observed = np.asarray(observed, dtype=np.float64).reshape(-1)
            if observed.size != inner_coords.shape[0]:
                raise ValueError(
                    f'observed field for {var_names[var_idx]!r} has {observed.size} '
                    f'values, the inner domain has {inner_coords.shape[0]} points')
            bcs.append(self._observation_bc(dde, inner_coords[train],
                                            observed[train].reshape(-1, 1), var_idx))
        loss_weights = self.loss_weights(eq_list, data_list, train)

        geom = _geometry(grids)
        self._embedding = None
        # A data profile that cannot be built fails every candidate alike: a
        # setup error, not a NaN score that reads as "the search found nothing".
        try:
            if self.periodic_embedding is not None or self.collocation_density is not None:
                self._raw = global_var.samples_manager.raw_fields(self.domain_key)
            if self.periodic_embedding is not None:
                from epde.integrate.data_profile import build_profile
                profile = build_profile(grids, {v: self._raw[v] for v in var_names},
                                        split.t_train)
                self._embedding = [(a.axis, a.lo, a.period) for a in profile.axes if a.periodic]
        except (KeyError, ValueError) as exc:
            raise DeepXDEConfigError(f'periodic_embedding / collocation_density: the data '
                                     f'profile of this trajectory failed: {exc}') from exc
        if self.pde_loss != 'mse':
            self._setup_sinv(eq_list, grids, mask, split)
        pde_func = self._equation_system_to_pde_func(dde, eq_list, var_names)
        if len(grids) == 1:
            data_obj = dde.data.PDE(geom, pde_func, bcs, num_domain=self.num_domain,
                                    num_boundary=0, num_test=self.num_test)
        else:
            data_obj = dde.data.TimePDE(geom, pde_func, bcs, num_domain=self.num_domain,
                                        num_boundary=0, num_initial=0,
                                        num_test=self.num_test)

        if self.collocation_density is not None:
            try:
                warped = self._warped_collocation(np.asarray(data_obj.train_x_all), grids,
                                                  var_names, split)
            except (KeyError, ValueError) as exc:
                raise DeepXDEConfigError(f'collocation_density: the warp of this trajectory '
                                         f'failed: {exc}') from exc
            data_obj.replace_with_anchors(warped)
        n_embed = len(self._embedding) if self._embedding else 0
        layer_size = [len(grids) + n_embed] + list(self.net) + [len(var_names)]
        net = self._build_net(layer_size)
        self._apply_transforms(net, grids, data_list, split)
        if self.pairing_digest:
            self.last_pairing = {'theta0': self._param_digest(net)}
        model = dde.Model(data_obj, net)
        dynamic = self.loss_weight_mode == 'gradient_per_step'
        ckpt = self._make_checkpoint(net, inner_coords, data_list, split)
        #: ``_max_abs_grad``/``_gradient_weights`` read ``outputs_losses_train``,
        #: which only exists once something has been compiled.
        compiled = False
        try:
            if self.loss_weight_mode in ('gradient', 'gradient_per_step'):
                # Compiled once unweighted only to read the per-term gradients.
                # The optimiser built here is discarded; an L-BFGS-only run
                # (lr=None) compiles 'L-BFGS', which needs no step size.
                if self.lr is None:
                    model.compile('L-BFGS')
                else:
                    model.compile(self.optimizer, lr=self.lr)
                compiled = True
                loss_weights = self._gradient_weights(model)
                if dynamic or self.grad_weight_normalise:
                    loss_weights = (normalised_grad_weights(
                        [1.0 / w for w in loss_weights], self.grad_weight_floor_rel,
                        self.grad_weight_cap) or loss_weights)
            if self.phys_weight_scale != 1.0:
                # Physics terms come first (DeepXDE orders pde losses before bcs).
                loss_weights = [w * self.phys_weight_scale if idx < len(eq_list) else w
                                for idx, w in enumerate(loss_weights)]
            reference_weights = list(loss_weights)
            if self.last_pairing is not None:
                self.last_pairing['loss_weights'] = [float(w) for w in reference_weights]
            if self.init is not None:
                self.last_init_stats = self._apply_init(net, inner_coords, data_list, split)
                if getattr(self, '_stop_after_init', False):
                    return None, None
            adam_lr = self.lr
            if self.lr_rule is not None:
                self.last_lr_stats = self._data_driven_lr(net, inner_coords, data_list, split)
                adam_lr = self.last_lr_stats['lr']
            per_step = None
            losshistory = None
            # ``epochs=0`` skips the first-order phase outright (L-BFGS from
            # the initialisation); DeepXDE's own train() would still compile
            # and log a zero-iteration run.
            if self.epochs > 0:
                if self.first_order == 'adam':
                    model.compile(self.optimizer, lr=adam_lr, decay=self.decay,
                                  loss_weights=(None if dynamic else loss_weights))
                else:
                    from epde.integrate import lr_free
                    params = [p for p in net.parameters() if p.requires_grad]
                    opt = (lr_free.DoG(params, reps_rel=self.dog_reps_rel)
                           if self.first_order == 'dog'
                           else lr_free.Prodigy(params, d0=self.prodigy_d0))
                    model.compile(opt, loss_weights=loss_weights)
                compiled = True
                if dynamic:
                    per_step = _PerStepGradNormStep(
                        model, loss_weights, ema=self.grad_weight_ema,
                        floor_rel=self.grad_weight_floor_rel,
                        cap=self.grad_weight_cap,
                        period=self.grad_weight_period).install()
                if ckpt is not None:
                    ckpt.phase = 'adam'
                    _install_val_callback(model, ckpt)
                losshistory, _ = model.train(iterations=self.epochs)
                if ckpt is not None:
                    ckpt.record(force=True)     # the terminal Adam iterate
                if per_step is not None:
                    self.last_weight_stats = per_step.summary()
            elif self.lbfgs_maxiter <= 0 and self.second_order != 'lm':
                raise ValueError('deepxde_config: epochs=0 needs lbfgs_maxiter > 0, '
                                 'otherwise nothing trains')
            if self.second_order == 'lm':
                from epde.integrate import lr_free
                frozen = per_step.w if per_step is not None else loss_weights
                # compiled only so the reported loss can be read afterwards;
                # LM moves the parameters itself
                model.compile('L-BFGS', loss_weights=frozen)
                compiled = True
                X_col = _torch.as_tensor(data_obj.train_x_all,
                                         dtype=next(net.parameters()).dtype,
                                         device=next(net.parameters()).device)
                stats = lr_free.levenberg_marquardt(
                    net, data_obj.pde, X_col, data_obj.bcs, list(frozen), len(eq_list),
                    maxiter=self.lm_maxiter, tau=self.lm_tau, gtol=1e-8,
                    chunk=self.lm_chunk, clear_cache=dde.grad.clear)
                self.last_lm_stats = {k: v for k, v in stats.items() if k != 'history'}
                self.last_lm_stats['history'] = [list(h) for h in stats['history'][-50:]]
                self.last_lbfgs_exit = {'reason': 'lm_' + stats['reason'],
                                        'iterations': stats['iterations']}
            if self.lbfgs_maxiter > 0:
                # The second-order phase always runs on FROZEN weights (the last
                # EMA vector under 'gradient_per_step'): L-BFGS's curvature
                # history and its strong-Wolfe line search both compare loss
                # values across evaluations, which is meaningless if the
                # objective moves between them. Recompiling also restores
                # DeepXDE's own train_step, undoing the dynamic install.
                frozen = per_step.w if per_step is not None else loss_weights
                gtol = None
                if self.lbfgs_gtol_rel is not None:
                    if not compiled:
                        # 'variance' weighting with epochs=0 reaches here having
                        # compiled NOTHING, so the probe below used to raise a
                        # TypeError that the blanket handler turned into a silent
                        # all-NaN "solve failed". Compile first; the optimiser
                        # built here is thrown away by the real compile two lines
                        # down, which is what actually applies the budget.
                        model.compile('L-BFGS', loss_weights=frozen)
                        compiled = True
                    gtol = self.lbfgs_gtol_rel * self._max_abs_grad(model, frozen)
                self._set_lbfgs_budget(self.lbfgs_maxiter, gtol)
                model.compile('L-BFGS', loss_weights=frozen)
                _install_lbfgs_stall_guard(model)
                uninstall = None
                if ckpt is not None:
                    ckpt.phase = 'lbfgs'
                    if self.checkpoint_lbfgs:
                        uninstall = _install_val_closure_hook(model, ckpt)
                losshistory, _ = model.train()
                if uninstall is not None:
                    uninstall()
                if ckpt is not None:
                    # The guard breaks BEFORE the loop's callbacks, so the final
                    # L-BFGS parameters reach no hook. Forcing the record keeps
                    # the last iterate in the candidate set, which is what makes
                    # the selected model never worse on val than today's.
                    ckpt.record(force=True)
                self.last_lbfgs_exit = getattr(model, 'lbfgs_exit', None)
            if ckpt is not None:
                # Restore BEFORE the reported loss and the prediction, so the
                # returned field, loss and diagnostics all describe ONE
                # parameter vector.
                selected = ckpt.restore() if self.select_by == 'best_val' else 'last'
                self.last_val_stats = ckpt.summary(selected)
            final_loss = self._reported_loss(model, reference_weights)
        except (NotImplementedError, DeepXDEConfigError):
            # A factor the residual cannot express, or a config that cannot
            # describe a run: surface the message instead of masking it as a NaN
            # loss. A misconfiguration fails every candidate identically, so it
            # would otherwise read as "the search found nothing" rather than as
            # the setup error it is.
            raise
        except Exception:
            return [np.full(grids[0].size, np.nan) for _ in var_names], np.nan

        pred = model.predict(_input_columns(grids))
        solutions = [pred[:, i].reshape(-1) for i in range(len(var_names))]
        return solutions, final_loss

    @_on_solve_device
    def solve(self, equation_or_system, grids: list, data, domain_key: int = None):
        """Solve one trajectory.

        ``grids`` is that trajectory's per-dimension FULL coordinate list
        (``samples_manager.grids()[key]``); ``data`` is the OBSERVED field per
        variable on the inner domain (``samples_manager.get((var, (1.0,)))``),
        in ``vars_to_describe`` order for a system. Returns the full-grid
        prediction per variable (flattened, C order) and the final loss.
        """
        self.domain_key = (domain_key if domain_key is not None
                           else global_var.samples_manager.trajecatoryIDs[0])
        dim = len(grids)
        if dim not in (1, 2, 3):
            raise NotImplementedError(
                f'DeepXDE integration covers 1-D to 3-D domains; got {dim} '
                'coordinate axes.')

        keys = global_var.samples_manager.grid_keys
        self._set_coordinate_info(keys)

        if isinstance(equation_or_system, Equation):
            eq_list = [equation_or_system]
            var_names = [equation_or_system.main_var_to_explain]
            data_list = [data] if isinstance(data, np.ndarray) else data
        elif isinstance(equation_or_system, SoEq):
            var_names = equation_or_system.vars_to_describe
            eq_list = [equation_or_system.vals[var] for var in equation_or_system.vars_to_describe]
            if isinstance(data, np.ndarray):
                raise ValueError("For SoEq, data must be a list of arrays (one per variable).")
            data_list = data
        else:
            raise TypeError("Unsupported equation type")

        with self._precision_scope():
            return self._solve_trajectory(eq_list, var_names, grids, data_list)

    @contextlib.contextmanager
    def _precision_scope(self):
        """DeepXDE's float type for ONE solve, then back. It is process-wide
        (DeepXDE's ``config.real`` and torch's default dtype), so a float64
        candidate must not leak into the next one. float32 is a no-op."""
        if self.precision == 'float32':
            yield
            return
        real = dde.config.real
        previous_bits, previous_dtype = real.precision, _torch.get_default_dtype()
        real.set_float64()
        _torch.set_default_dtype(_torch.float64)
        try:
            yield
        finally:
            {16: real.set_float16, 32: real.set_float32, 64: real.set_float64}[previous_bits]()
            _torch.set_default_dtype(previous_dtype)

    def _build_net(self, layer_size):
        """The FNN. Under float64 it is BUILT in float32 and upcast, so its
        initial weights are exactly those of the float32 arms (Glorot draws in
        float64 would be different numbers) and the arms stay paired."""
        if self.precision != 'float64':
            return dde.nn.FNN(layer_size, self.activation, self.kernel_initializer)
        real = dde.config.real
        real.set_float32()
        _torch.set_default_dtype(_torch.float32)
        try:
            net = dde.nn.FNN(layer_size, self.activation, self.kernel_initializer)
        finally:
            real.set_float64()
            _torch.set_default_dtype(_torch.float64)
        return net.double()

    def _apply_transforms(self, net, grids, data_list, split):
        """Input / output transforms (they define the network, so they go in
        before the gradient-weight probe). DeepXDE differentiates through them."""
        p0 = next(net.parameters())
        dev, dt = p0.device, p0.dtype
        embedding = getattr(self, '_embedding', None)
        if self.input_transform == 'affine' or embedding:
            # DeepXDE's column order [space..., t] (``_input_columns``)
            cols = [np.asarray(g) for g in grids[1:]] + [np.asarray(grids[0])]
            lo = np.array([float(c.min()) for c in cols])
            hi = np.array([float(c.max()) for c in cols])
            if self.input_transform == 'affine':
                scale, shift = 2.0 / (hi - lo), -1.0 - 2.0 * lo / (hi - lo)
            else:
                scale, shift = np.ones_like(lo), np.zeros_like(lo)
            scale = _torch.as_tensor(scale, dtype=dt, device=dev)
            shift = _torch.as_tensor(shift, dtype=dt, device=dev)
            # EPDE axis a -> DeepXDE column (time is last)
            n_axes = len(grids)
            periodic = {}
            for axis, a_lo, period in (embedding or []):
                col = axis - 1 if axis > 0 else n_axes - 1
                periodic[col] = (2.0 * math.pi / period, a_lo)
            keep = [c for c in range(n_axes) if c not in periodic]

            def transform(x):
                parts = [x[:, keep] * scale[keep] + shift[keep]] if keep else []
                for col, (w, a_lo) in sorted(periodic.items()):
                    phase = w * (x[:, col:col + 1] - a_lo)
                    parts += [_torch.cos(phase), _torch.sin(phase)]
                return _torch.cat(parts, dim=1)

            net.apply_feature_transform(transform)
        if self.output_transform == 'train_moments':
            obs = [np.asarray(v, dtype=np.float64).reshape(-1)[split.train] for v in data_list]
            tiny = float(np.finfo(np.float64).tiny)
            mu = _torch.as_tensor([float(o.mean()) for o in obs], dtype=dt, device=dev)
            sd = _torch.as_tensor([max(float(o.std()), tiny) for o in obs], dtype=dt, device=dev)
            net.apply_output_transform(lambda x, y: mu + sd * y)

    def _warped_collocation(self, points, grids, var_names, split):
        """DeepXDE's own collocation points, moved by the sequential inverse CDF
        (Rosenblatt map) of the train-window data-gradient density
        ('data_gradient') or of a flat one ('flat'), on any number of axes.

        Time goes first, through the density's time marginal (its mean over
        every spatial axis). Each spatial EPDE axis a = 1, 2, ... follows in
        order, through the density on the grid cell the earlier axes landed in
        (the NEAREST node on each earlier axis), averaged over the later axes.
        Every 1-D inverse CDF is the trapezoid rule on the axis's own coordinates.

        The density is the mean over variables of ``gradient_density``, which is
        scale-free per variable (1 + |grad u| / mean |grad u|), so no variable's
        units weigh more. It reads the train levels only. The held-out levels get
        the last train level's pattern scaled to the train-window mean density:
        copying that level as it stands would let one level (in 1-D, one number)
        set how many physics points the scored tail receives.

        Deterministic and identical for every candidate: the density depends on
        the data only. It is > 0 everywhere (>= 1 on the train levels; the tail
        rescale can take it below 1), so every CDF is strictly increasing and a
        point on a face of the box stays on that face. A flat density makes every
        CDF the normalised coordinate itself: 'flat' is the identity up to
        rounding on any grid -- the placement control."""
        from epde.integrate.data_profile import (axis_coordinates, gradient_density,
                                                 train_level_mask)
        grids = [np.asarray(g, dtype=np.float64) for g in grids]
        n = len(grids)
        coords = [axis_coordinates(grids[a], a) for a in range(n)]
        if self.collocation_density == 'flat':
            dens = np.ones(grids[0].shape)
        else:
            dens = sum(gradient_density(grids, self._raw[v], split.t_train, kind='all')
                       for v in var_names) / len(var_names)
            train = train_level_mask(coords[0], split.t_train)
            if (~train).any():
                last = dens[train][-1]
                dens[~train] = last * (float(dens[train].mean()) / float(np.mean(last)))
        out = np.array(points, dtype=np.float64, copy=True)
        column = [n - 1] + list(range(n - 1))      # EPDE axis -> DeepXDE column: time last

        def inverse_cdf(weights, axis_coords, u):
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (weights[1:] + weights[:-1])
                                                     * np.diff(axis_coords))])
            cdf /= cdf[-1]
            return np.interp(u, cdf, axis_coords)

        def unit(a):                               # DeepXDE's own coordinate, on [0, 1]
            lo, hi = coords[a][0], coords[a][-1]
            return (points[:, column[a]] - lo) / (hi - lo)

        def nearest(axis_coords, values):
            i = np.clip(np.searchsorted(axis_coords, values), 1, len(axis_coords) - 1)
            left = np.abs(values - axis_coords[i - 1]) <= np.abs(axis_coords[i] - values)
            return np.where(left, i - 1, i)

        marginal_t = dens.reshape(len(coords[0]), -1).mean(axis=1)
        out[:, column[0]] = inverse_cdf(marginal_t, coords[0], unit(0))
        cell = None                                # C-order flat index of the cell on axes < a
        for a in range(1, n):
            node = nearest(coords[a - 1], out[:, column[a - 1]])
            cell = node if cell is None else cell * len(coords[a - 1]) + node
            u = unit(a)
            earlier = tuple(len(c) for c in coords[:a])
            for key in np.unique(cell):
                sel = cell == key
                block = dens[np.unravel_index(key, earlier)]
                # the last axis takes its row as is; an intermediate one averages
                # over the axes after it
                weights = block if block.ndim == 1 else \
                    block.reshape(len(coords[a]), -1).mean(axis=1)
                out[sel, column[a]] = inverse_cdf(weights, coords[a], u[sel])
        return out.astype(np.asarray(points).dtype)

    def _observation_bc(self, dde, points, values, component):
        """The observation term of one variable: DeepXDE's PointSetBC, or, under a
        scale-invariant ``obs_loss``, the same points with the observation
        equation's cancellation ratio as the error."""
        if self.obs_loss == 'mse':
            return dde.icbc.PointSetBC(points, values, component=component)
        from epde.integrate.residual_terms import cancellation_ratio, mass_diagnostics
        mode = self.obs_loss

        class _ScaleInvariantPointSetBC(dde.icbc.PointSetBC):
            def error(self, X, inputs, outputs, beg, end, aux_var=None):
                r = super().error(X, inputs, outputs, beg, end, aux_var)
                y = outputs[beg:end, self.component:self.component + 1]
                v = self.values
                mass = (y.abs() + v.abs()) if mode == 'sinv_live' else 2.0 * v.abs()
                return cancellation_ratio(r, mass, _torch)

        stats = getattr(self, 'last_obs_stats', None) or {'obs_loss': mode, 'data_mass': []}
        stats['data_mass'].append(mass_diagnostics(2.0 * np.abs(np.asarray(values,
                                                                           dtype=np.float64))))
        self.last_obs_stats = stats
        return _ScaleInvariantPointSetBC(points, values, component=component)

    @staticmethod
    def _param_digest(net) -> str:
        import hashlib
        h = hashlib.sha256()
        for p in net.parameters():
            # float32 bytes: a float64 run upcasts the SAME float32 draw
            # (``_build_net``), so it pairs with its float32 control; a float32
            # net's digest is unchanged by the cast
            h.update(p.detach().to(_torch.float32).cpu().numpy().tobytes())
        return h.hexdigest()[:24]

    def _apply_init(self, net, inner_coords, data_list, split):
        """The start-point options, in place, AFTER the weight probe."""
        import time as _time
        p0 = next(net.parameters())
        dev, dt = p0.device, p0.dtype
        X = _torch.as_tensor(inner_coords[split.train], dtype=dt, device=dev)
        Y = np.stack([np.asarray(v, dtype=np.float64).reshape(-1)[split.train]
                      for v in data_list], axis=1)
        t0 = _time.perf_counter()
        if self.init == 'lstsq_last_layer':
            if net._output_transform is not None:
                raise DeepXDEConfigError("init='lstsq_last_layer' with an output transform "
                                         'is two mechanisms; combine them explicitly')
            with _torch.no_grad():
                h = X if net._input_transform is None else net._input_transform(X)
                act = net.activation
                for j, lin in enumerate(net.linears[:-1]):
                    h = (act[j] if isinstance(act, list) else act)(lin(h))
            H = h.detach().cpu().double().numpy()
            A = np.hstack([H, np.ones((H.shape[0], 1))])      # bias column
            sol, _, rank, sv = np.linalg.lstsq(A, Y, rcond=None)
            last = net.linears[-1]
            with _torch.no_grad():
                last.weight.copy_(_torch.as_tensor(sol[:-1].T, dtype=dt, device=dev))
                last.bias.copy_(_torch.as_tensor(sol[-1], dtype=dt, device=dev))
            return {'init': self.init, 'rank': int(rank),
                    'cond': float(sv[0] / sv[-1]) if sv[-1] > 0 else float('inf'),
                    'w_out_norm': float(np.linalg.norm(sol[:-1])),
                    'seconds': _time.perf_counter() - t0}
        if self.init == 'shared_data_fit':
            stats = self._shared_data_fit(net, X, Y)
            stats['seconds_total'] = _time.perf_counter() - t0
            return stats
        # 'data_prefit': observation-only L-BFGS from the Glorot init.
        stats = self._data_fit(net, X, Y, self.prefit_maxiter)
        stats['init'] = self.init
        return stats

    def _data_driven_lr(self, net, inner_coords, data_list, split):
        """The Adam lr from the data (``lr_rule``), on COPIES of the network,
        inside ``fork_rng`` so neither the weights nor any RNG stream the solve
        uses is touched."""
        import copy
        import time as _time
        p0 = next(net.parameters())
        dev, dt = p0.device, p0.dtype
        X = _torch.as_tensor(inner_coords[split.train], dtype=dt, device=dev)
        Y = np.stack([np.asarray(v, dtype=np.float64).reshape(-1)[split.train]
                      for v in data_list], axis=1)
        t0 = _time.perf_counter()
        devices = [dev] if dev.type == 'cuda' else []
        with _torch.random.fork_rng(devices=devices):
            if self.lr_rule == 'prefit_distance':
                theta0 = _torch.cat([p.detach().reshape(-1) for p in net.parameters()])
                fitted = copy.deepcopy(net)
                fit = self._data_fit(fitted, X, Y, self.prefit_maxiter)
                theta = _torch.cat([p.detach().reshape(-1) for p in fitted.parameters()])
                median = float((theta - theta0).abs().median())
                lr = median / max(self.epochs, 1)
                return {'rule': self.lr_rule, 'lr': lr, 'median_displacement': median,
                        'prefit': fit, 'seconds': _time.perf_counter() - t0}
            # 'data_selected': Adam on the observation-only loss, one copy per lr
            w = [variance_weight(Y[:, v]) for v in range(Y.shape[1])]
            Yt = _torch.as_tensor(Y, dtype=dt, device=dev)
            losses = {}
            for lr in self.lr_grid:
                trial = copy.deepcopy(net)
                opt = _torch.optim.Adam(trial.parameters(), lr=lr)
                for _ in range(self.epochs):
                    opt.zero_grad()
                    pred = trial(X)
                    loss = sum(w[v] * _torch.mean((pred[:, v] - Yt[:, v]) ** 2)
                               for v in range(Y.shape[1]))
                    loss.backward()
                    opt.step()
                with _torch.no_grad():
                    pred = trial(X)
                    final = float(sum(w[v] * _torch.mean((pred[:, v] - Yt[:, v]) ** 2)
                                      for v in range(Y.shape[1])))
                losses[lr] = final if math.isfinite(final) else float('inf')
            best = min(losses, key=lambda k: losses[k])
            return {'rule': self.lr_rule, 'lr': best,
                    'losses': {str(k): v for k, v in losses.items()},
                    'at_grid_edge': best in (min(self.lr_grid), max(self.lr_grid)),
                    'seconds': _time.perf_counter() - t0}

    #: shared data fits of this process, by cache key (see ``_shared_data_fit``)
    _SHARED_FITS = {}

    @staticmethod
    def _data_fit(net, X, Y, maxiter):
        """Observation-only L-BFGS on ``net``, in place: DeepXDE's L-BFGS
        settings (lr=1, strong Wolfe, history 100, gtol 1e-8, ftol 0), raw
        torch, no DeepXDE model and no global options touched. The variables
        are weighted by 1/Var, the observation term's own yardstick."""
        import time as _time
        t0 = _time.perf_counter()
        p0 = next(net.parameters())
        dev, dt = p0.device, p0.dtype
        w = [variance_weight(Y[:, v]) for v in range(Y.shape[1])]
        Yt = _torch.as_tensor(Y, dtype=dt, device=dev)
        params = [p for p in net.parameters() if p.requires_grad]
        opt = _torch.optim.LBFGS(params, lr=1, max_iter=maxiter,
                                 max_eval=int(1.25 * maxiter),
                                 tolerance_grad=1e-8, tolerance_change=0.0,
                                 history_size=100, line_search_fn='strong_wolfe')

        def data_loss():
            pred = net(X)
            return sum(w[v] * _torch.mean((pred[:, v] - Yt[:, v]) ** 2)
                       for v in range(Y.shape[1]))

        def closure():
            opt.zero_grad()
            loss = data_loss()
            loss.backward()
            return loss

        with _torch.no_grad():
            before = float(data_loss())
        opt.step(closure)
        with _torch.no_grad():
            after = float(data_loss())
        state = opt.state[opt._params[0]] if opt.state else {}
        n_iter = int(state.get('n_iter', 0))
        # the gradient AT the final parameters: torch's prev_flat_grad is the
        # one from the start of the last iteration, which mislabels a gtol stop
        grads = _torch.autograd.grad(data_loss(), params, allow_unused=True)
        gmax = max((float(g.abs().max()) for g in grads if g is not None),
                   default=float('nan'))
        if n_iter >= maxiter:
            reason = 'maxiter'
        elif int(state.get('func_evals', 0)) >= int(1.25 * maxiter):
            reason = 'maxeval'
        elif gmax <= 1e-8:
            reason = 'gtol'
        elif float(state.get('t', 1.0)) == 0.0:
            reason = 'zero_step'           # the line search found no decrease it could confirm
        else:
            reason = 'no_change'
        for p in params:
            p.grad = None
        return {'prefit_iterations': n_iter, 'data_loss_before': before,
                'data_loss_after': after, 'stop': reason, 'gmax': gmax,
                'dtype': str(dt), 'seconds': _time.perf_counter() - t0}

    def _shared_data_fit(self, net, X, Y):
        """Load (or fit, then cache) the shared observation-only start.

        The cache key is the start's weights, the train observations, the
        architecture and the cap: two candidates on one dataset with the same
        start share one fit, and nothing about the candidate equation enters
        it. The fit sees the TRAIN window only (``X``, ``Y``), so the scored
        tail cannot leak into the start -- unlike the retired
        ``solution_guess_nn`` path, whose net was trained on every point."""
        import copy
        import hashlib
        p0 = next(net.parameters())
        dev, dt = p0.device, p0.dtype
        start = net
        if self.shared_fit_seed is not None:
            from deepxde.nn import initializers
            start = copy.deepcopy(net)
            init = initializers.get(self.kernel_initializer)
            zeros = initializers.get('zeros')
            devices = [dev] if dev.type == 'cuda' else []
            with _torch.random.fork_rng(devices=devices):
                _torch.manual_seed(self.shared_fit_seed)
                with _torch.no_grad():
                    for lin in start.linears:
                        init(lin.weight)
                        zeros(lin.bias)
        h = hashlib.sha256()
        for p in start.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        h.update(np.ascontiguousarray(X.detach().cpu().numpy()).tobytes())
        h.update(np.ascontiguousarray(Y).tobytes())
        key_parts = (list(self.net), self.activation, self.input_transform,
                     self.output_transform, str(dt), self.shared_fit_maxiter)
        if self.shared_fit_precision != 'float32':
            # only a non-default precision enters the key, so every fit cached
            # before this option existed keeps its key
            key_parts = key_parts + (self.shared_fit_precision,)
        if getattr(self, 'periodic_embedding', None) is not None:
            # the embedding is data-determined, but it changes the network the
            # fit belongs to; name it rather than rely on the weight bytes
            key_parts = key_parts + (('periodic', self.periodic_embedding),)
        h.update(repr(key_parts).encode())
        key = h.hexdigest()[:32]
        path = (os.path.join(self.shared_fit_dir, key + '.pt')
                if self.shared_fit_dir else None)
        stats = {'init': 'shared_data_fit', 'key': key, 'seed': self.shared_fit_seed}
        state, source = self._SHARED_FITS.get(key), 'memory'
        if state is None and path is not None and os.path.exists(path):
            blob = _torch.load(path, map_location='cpu')
            state, source = blob['state'], 'disk'
            stats['fit'] = blob['stats']
            self._SHARED_FITS[key] = state
        if state is None:
            if self.shared_fit_precision == 'float64' and dt != _torch.float64:
                fitted = copy.deepcopy(start).double()
                fit = self._data_fit(fitted, X.double(), Y, self.shared_fit_maxiter)
                state = {k: v.detach().to(dt).cpu().clone()
                         for k, v in fitted.state_dict().items()}
            else:
                fit = self._data_fit(start, X, Y, self.shared_fit_maxiter)
                state = {k: v.detach().cpu().clone() for k, v in start.state_dict().items()}
            source = 'fitted'
            stats['fit'] = fit
            self._SHARED_FITS[key] = state
            if path is not None:
                os.makedirs(self.shared_fit_dir, exist_ok=True)
                _torch.save({'state': state, 'stats': fit}, path)
        net.load_state_dict({k: v.to(device=dev) for k, v in state.items()})
        stats['source'] = source
        return stats

    def prepare_shared_fit(self, equation_or_system, grids: list, data, domain_key: int = None):
        """Run a solve only as far as the shared data fit, so it lands in the
        cache (and in ``shared_fit_dir``). The gate harness does this in its own
        process first, so every candidate process LOADS the fit and all of them
        share one process history."""
        if self.init != 'shared_data_fit':
            raise DeepXDEConfigError("prepare_shared_fit needs init='shared_data_fit'")
        self._stop_after_init = True
        try:
            self.solve(equation_or_system, grids, data, domain_key)
        finally:
            self._stop_after_init = False
        return self.last_init_stats

    def _setup_sinv(self, eq_list, grids, mask, split):
        """Floors, and for 'sinv_data' the frozen term mass of the DATA.

        The mass is EPDE's own term values on the inner domain (the denominator
        of ``Discrepancy._compute_scale_invariant``, in-place term set), kept on
        the train window. A collocation point reads its nearest inner-grid
        point; past the last train level it reads that level (the tail is
        never read)."""
        from epde.operators.common.objectives import cancellation_parts
        from epde.integrate.residual_terms import inverse_mass, mass_diagnostics
        key = self.domain_key
        m = np.asarray(mask).astype(bool)
        axes = [np.unique(np.asarray(grids[a])[m]) for a in range(len(grids))]
        shape = tuple(len(c) for c in axes)
        if int(m.sum()) != int(np.prod(shape)):
            raise NotImplementedError("pde_loss 'sinv_*' needs a box-shaped inner domain")
        n_train = int(np.sum(axes[0] <= split.t_train))
        windows, floors, diagnostics = [], [], []
        for eq in eq_list:
            parts = cancellation_parts(eq, active_only=False)
            if parts is None:                      # a lone target: its own magnitude
                values = np.abs(np.asarray(eq.evaluate(active_only=False)[0][key]))
            else:
                values = parts[1][key]
            mass = np.asarray(values, dtype=np.float64).reshape(shape)[:n_train]
            floors.append(self.sinv_floor_rel * float(mass.mean()))
            windows.append(inverse_mass(mass + floors[-1]))
            diagnostics.append(mass_diagnostics(mass + floors[-1]))
        self._sinv_floors = floors
        self.last_sinv_stats = {'floors': floors, 'data_mass': diagnostics}
        if self.pde_loss != 'sinv_data':
            self._sinv_data_weight = None
            return
        axes[0] = axes[0][:n_train]
        columns = [self._axis_column(a, self.coord_map) for a in range(len(grids))]
        cache = {}

        def nearest(values, grid):
            if grid.numel() == 1:
                return _torch.zeros_like(values, dtype=_torch.long)
            i = _torch.bucketize(values.contiguous(), grid).clamp(1, grid.numel() - 1)
            left = (values - grid[i - 1]).abs() < (grid[i] - values).abs()
            return _torch.where(left, i - 1, i)

        def data_weight(eq_idx, x):
            k = (x.device, x.dtype)
            if k not in cache:
                cache[k] = ([_torch.as_tensor(c, dtype=x.dtype, device=x.device) for c in axes],
                            [_torch.as_tensor(wd, dtype=x.dtype, device=x.device)
                             for wd in windows])
            grids_t, windows_t = cache[k]
            idx = tuple(nearest(x[:, col].detach(), g) for col, g in zip(columns, grids_t))
            return windows_t[eq_idx][idx].reshape(-1, 1)

        self._sinv_data_weight = data_weight
