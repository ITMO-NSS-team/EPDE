#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The PINN pieces every system's ``pinn_test_autoscale.py`` shares.

dp, duffing and wave were three near-copies: the same MLP up to its
input/output width, the same Adam-then-LBFGS driver, the same truth-free
checkpoint rule, the same per-term statistic scorer. Three copies of one
rule is how the systems silently stop being comparable -- the same
argument that put ``observation_loss`` and the chi/het statistics in
``dp/cv_metric.py`` behind ``stat_common``. This module is the other
half of that: the MODEL and the TRAINING LOOP, so a change to either
lands on all three at once.

What stays in each system's script, because it genuinely differs:
    * loading the dataset and its truth coefficients,
    * the derivative operator (one t, two t, or (x, t) second order),
    * the library / feature construction and the equation set,
    * which yardsticks the IC and BC terms divide by,
    * what goes into the .npz.

What lives here:
    MLP            n_in -> hidden(tanh) -> n_out, xavier_normal + zero
                   bias, identical to what all three had inline
    stat_scores    the truth-free constancy statistic as a loss term
    Checkpoint     truth-free ('loss') or oracle ('err') selection
    assert_train_only  hard guard that the loss never reads past t_split
    paired_t/sign_p/t_crit  seed-paired verdicts, from the NN project
    train          Adam then LBFGS/strong_wolfe, with the per-term
                   gradient diagnostic wired in

The statistics themselves and ``observation_loss``/``grad_norms`` come
from ``stat_common`` (i.e. from ``dp/cv_metric.py``); nothing is
reimplemented here.
"""

import math
import time

import torch
import torch.nn as nn

from stat_common import (bounded, chi2_per_term, global_ols, grad_norms,
                         het_per_window)


# ============================================================ model
class MLP(nn.Module):
    """The one network. dp had [1]->[2], duffing [1]->[1], wave [2]->[1]
    and were otherwise character-identical, initialiser included."""

    def __init__(self, n_in, n_out, hidden):
        super().__init__()
        layers = [n_in] + list(hidden) + [n_out]
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ============================================================ statistic
def stat_scores(y, A, mask, stat, *, bound=True, float64=True, floor=False,
                ridge=1e-12):
    """The truth-free constancy statistic, already dimensionless and
    (after ``bounded``) bounded -- so it enters the loss at weight 1.

    Returns ``(score_or_None, theta_hat)`` with theta_hat the SAME global
    OLS the physics term reads when COEF_SOURCE='ols': one loss must not
    contain two different theta_hats. Note the het branch takes theta from
    ``global_ols`` and NOT from ``het_per_window``'s own return -- that is
    what dp and duffing already did, and it is the theta the physics
    residual uses.

    ``stat='none'`` returns ``(None, theta)``: the caller adds nothing to
    the loss but still needs the recovery.

    wave does NOT use this. Its statistic runs one score path PER GRID
    AXIS (``_chi_per_axis``), which is the whole escape from the
    residual-normalisation trap -- a genuinely different rule, kept
    separate rather than folded in here.
    """
    if stat == "chi":
        yy, AA = (y.double(), A.double()) if float64 else (y, A)
        out = chi2_per_term(yy, AA, ridge=ridge, use_floor=floor)
        s = bounded(out["score"]) if bound else out["score"]
        return s, out["theta"]
    if stat == "het":
        out = het_per_window(y, A, mask, ridge=ridge)
        c, _ = global_ols(y, A, ridge=ridge)
        return out["score"], c
    c, _ = global_ols(y, A, ridge=ridge)
    return None, c


# ============================================================ verdicts
# Taken from ~/PycharmProjects/NN (online_snr_pruning.ipynb), which keeps
# per-seed results UNAGGREGATED so two configurations can be compared
# PAIRED. That is the missing piece here: duffing's coefficient error
# moves 10x on the seed alone, so a single-seed point comparison cannot
# support a verdict, and several earlier ones in this project were stated
# more confidently than the spread allowed.
def paired_t(d):
    """Paired t over an array of per-seed differences (nan when it cannot
    be formed)."""
    import numpy as np
    d = np.asarray(d, dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))


def sign_p(n_pos, n):
    """Exact two-sided binomial p-value at p=0.5 -- the assumption-free
    companion to the t."""
    k = min(n_pos, n - n_pos)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2.0 ** n)


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 23: 2.069, 30: 2.042}


def t_crit(n):
    """Two-sided 5% critical value of t with df = n-1, so a verdict can
    state its own bar."""
    df = max(1, n - 1)
    return _T95.get(df, _T95[min(_T95, key=lambda k: abs(k - df))]
                    if df < 30 else 1.960)


def mde(sd, n):
    """Smallest paired difference this design could have called
    significant at 5%. Printed beside every unresolved contrast: "no
    difference" and "this design cannot see a difference this small" are
    different statements, and only the second one is usually true here."""
    return t_crit(n) * sd / math.sqrt(n) if n > 1 else float("nan")


def seeds_needed(d):
    """Smallest n at which the observed effect and spread would clear the
    ACTUAL bar on the row it is printed next to, i.e. t_crit(n).

    DELIBERATE DEVIATION from NN, which uses the 2-sigma approximation
    ``(2*sd/|mean|)^2``. NN runs 10-24 seeds, where t_crit is ~2.1 and the
    approximation is harmless. This project runs 3-6, where t_crit is 2.57
    to 4.30 -- so NN's formula would print "needs ~5 seeds" on a row whose
    own stated bar 5 seeds do not reach. Two numbers on one line that
    disagree is worse than either alone.

    Clamped from below by the n already run, as NN clamps it: this is only
    ever printed when the contrast did NOT resolve, so a recommendation
    below the n already run would read as "that was already enough".

    Returns inf when the effect is zero, or when the difference contains a
    nan -- a diverged integration is an expected outcome here, and it must
    not take the report down with it (``int(math.ceil(nan))`` raises).
    """
    import numpy as np
    d = np.asarray(d, dtype=float)
    if len(d) < 2 or not np.isfinite(d).all():
        return float("inf")
    m, sd = abs(d.mean()), d.std(ddof=1)
    if m == 0 or not math.isfinite(sd):
        return float("inf")
    if sd == 0:
        return len(d)                 # perfect separation: already enough
    for n in range(len(d), 501):
        if m / (sd / math.sqrt(n)) >= t_crit(n):
            return n
    return float("inf")


def contrast(name, a, b, lower_is_better=True):
    """One PRE-SPECIFIED paired comparison of two arms over shared seeds.

    Pairing is what makes this worth doing: arm A and arm B at seed s
    share the initialisation and the batch order, so the per-seed
    DIFFERENCE has far less spread than either column. duffing's
    coefficient error moves ~10x on the seed alone while the arm gap is a
    few percent -- unpaired, that gap is invisible.

    ``lower_is_better`` because every metric here is an error: 'wins'
    counts seeds where A is BELOW B.
    """
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"{name}: unpaired arms, {len(a)} vs {len(b)} seeds")
    d = a - b
    n = len(d)
    finite = np.isfinite(d)
    n_win = int((d < 0).sum() if lower_is_better else (d > 0).sum())
    t = paired_t(d)
    tc = t_crit(n)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    # Perfect separation -- every seed differs by the same non-zero amount
    # -- gives sd = 0, and paired_t returns nan for it. Left alone, the
    # STRONGEST evidence the design can produce would print as "not
    # resolved", which is exactly backwards.
    if finite.all() and n > 1 and sd == 0 and d.mean() != 0:
        t = float("inf") * (1.0 if d.mean() > 0 else -1.0)
    resolved = bool(finite.all() and t == t and abs(t) >= tc)
    return dict(name=name, n=n, d=d, mean=float(d.mean()), sd=sd,
                t=t, t_crit=tc, sign_p=sign_p(n_win, n), wins=n_win,
                n_bad=int((~finite).sum()),
                resolved=resolved,
                better=bool(resolved and (d.mean() < 0 if lower_is_better
                                          else d.mean() > 0)),
                mde=mde(sd, n), needed=seeds_needed(d))


def arm_table(results, order=None, noise_arms=None, label="metric",
              fmt="%.4e", show=6):
    """NN's arm table: per-seed values stay UNAGGREGATED and are printed.

    ``range`` is max-min WITHIN an arm, and the mean of those ranges over
    ``noise_arms`` is the noise yardstick every contrast is read against.
    Recipe/probe arms are excluded from that mean by passing an explicit
    ``noise_arms``: an arm that differs from the others by construction
    would otherwise widen the bar that the real comparisons are judged by.

    Returns the yardstick.
    """
    import numpy as np
    order = list(order or results)
    lens = {len(results[k]) for k in order}
    if len(lens) > 1:
        raise ValueError(
            "arm_table: arms have different seed counts %s. The yardstick is "
            "a mean of within-arm ranges, and a short arm contributes an "
            "artificially small one (zero, for a single seed), which shrinks "
            "the bar every contrast is judged against." % sorted(lens))
    mean = {k: float(np.mean(results[k])) for k in order}
    spread = {k: float(np.max(results[k]) - np.min(results[k])) for k in order}
    noise = float(np.mean([spread[k] for k in (noise_arms or order)]))
    w = max(len(k) for k in order)
    print("%-*s %12s %12s   %s" % (w, "arm", "mean " + label, "range",
                                   "per-seed"))
    print("-" * (w + 30 + 14 * min(show, max(len(results[k]) for k in order))))
    for k in order:
        v = results[k]
        tail = "  ..." if len(v) > show else ""
        shown = "  ".join(fmt % x for x in v[:show]) + tail
        print("%-*s %12s %12s   %s"
              % (w, k, fmt % mean[k], fmt % spread[k], shown))
    print(
        "\ntypical WITHIN-arm seed range = %s   <- anything smaller"
        " than this is NOT resolved by a point comparison, however"
        " it is printed." % (fmt % noise))
    return noise


def contrast_table(contrasts, label="metric", fmt="%.4e"):
    """The paired-contrast block. Every row states its own bar (t_crit)
    and, when it fails to clear it, what it would have taken -- an
    unresolved contrast is a statement about the design, not about the
    world, and reporting it as 'no difference' is the error this whole
    layer exists to stop."""
    w = max(len(c["name"]) for c in contrasts)
    print("%-*s %6s %12s %8s %8s %10s   %s"
          % (w, "paired contrast (A - B)", "seeds", "mean diff", "t",
             "t_crit", "sign p", "verdict"))
    print("-" * (w + 70))
    for c in contrasts:
        if c.get("n_bad"):
            verdict = ("UNREADABLE: %d/%d seeds have no finite value "
                       "(diverged)" % (c["n_bad"], c["n"]))
        elif c["resolved"]:
            verdict = "A WINS" if c["better"] else "A LOSES"
        else:
            verdict = ("not resolved (needs ~%s seeds; this design could only "
                       "see %s)" % (c["needed"], fmt % c["mde"]))
        print("%-*s %6d %12s %8.2f %8.2f %10.4f   %s"
              % (w, c["name"], c["n"], fmt % c["mean"], c["t"], c["t_crit"],
                 c["sign_p"], verdict))
        print("%-*s %6s %12s   wins %d/%d"
              % (w, "", "", "", c["wins"], c["n"]))


# ============================================================ hold-out
def assert_train_only(name, coords, t_split, *, tol=1e-6):
    """Fail loudly if anything the LOSS reads reaches past t_split.

    Reviewing the slices by eye is how the first version of this split
    shipped leaky: physics collocation, BC sampling, the FD design and
    the yardstick variances each have to be bounded separately, and
    getting three of four right looks exactly like getting four of four
    right in the printed output. This is called on every loss-feeding
    tensor at construction, so a future edit that widens one of them
    stops the run instead of quietly improving the numbers.

    ``coords`` is the time coordinate of whatever is being checked (a 1-D
    tensor, or the t column of an (x, t) batch). ``tol`` absorbs float32
    representation of t_split itself, nothing more.
    """
    import torch
    c = coords.detach() if torch.is_tensor(coords) else torch.as_tensor(coords)
    hi = float(c.max())
    if hi > t_split + tol:
        raise AssertionError(
            f"TRAIN/TEST LEAK in {name!r}: max t = {hi:.6g} > t_split = "
            f"{t_split:.6g}. The loss must not see the held-out tail.")
    return coords


# ============================================================ selection
class Checkpoint:
    """Best-state tracker. ``key="loss"`` is truth-free and is what the
    experiments select on; ``key='err'`` is the oracle probe, kept because
    the gap between the two IS a result (a loss that cannot find its own
    best state is not a usable loss)."""

    def __init__(self, net, key="loss", err_fn=None, select_fn=None,
                 extra_fn=None):
        # Selection and diagnostic are SEPARATE: dp selects on one
        # coefficient key while always recording the global one, and the
        # gap between truth-free and oracle selection is itself a result.
        self.net, self.key, self.err_fn = net, key, err_fn
        self.select_fn = select_fn if select_fn is not None else err_fn
        if key != "loss" and self.select_fn is None:
            raise ValueError(f"BEST_KEY={key!r} needs select_fn or err_fn")
        # Anything else worth freezing at the best step (wave keeps the
        # c_hat that went with it, so the reported coefficient and the
        # reported state are the same moment).
        self.extra_fn = extra_fn
        self.extra = {}
        self.best = float("inf")
        self.loss = float("inf")
        self.err = float("inf")
        self.iter = -1
        self.state = None

    def update(self, loss_val, it, aux=None):
        err = float(self.err_fn(aux)) if self.err_fn is not None else float("nan")
        k = loss_val if self.key == "loss" else float(self.select_fn(aux))
        if k < self.best:
            self.best, self.loss, self.err, self.iter = k, loss_val, err, it
            self.state = {kk: v.detach().clone()
                          for kk, v in self.net.state_dict().items()}
            if self.extra_fn is not None:
                self.extra = self.extra_fn(aux)

    def restore(self):
        if self.state is not None:
            self.net.load_state_dict(self.state)


# ============================================================ driver
def train(net, loss_fn, *, adam_iters, adam_lr, lbfgs_max, checkpoint,
          sampler=None, gen=None, log_every=500, log_fn=None,
          grad_terms=None, on_iter=None):
    """Adam, then LBFGS with strong_wolfe -- the driver all three had.

    ``loss_fn(batch) -> (loss, terms, aux)``:
        loss   scalar to differentiate
        terms  {name: tensor} for logging AND for the gradient diagnostic
        aux    passed through to the checkpoint's err_fn / log_fn
    ``sampler(gen) -> batch`` is called per Adam iteration and ONCE more
    for a fixed LBFGS batch; pass None for a fixed-grid system (dp).

    ``grad_terms``: which term names to measure ||grad|| for at logging
    iterations. Taken BEFORE backward with retain_graph=True. A term can
    be tiny in value and still own the gradient, so the loss print alone
    cannot tell you which term is driving training.
    """
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=adam_lr)
    t0 = time.time()

    for it in range(adam_iters):
        if on_iter is not None:
            on_iter(it)                 # dp captures its learning GIF here
        opt.zero_grad()
        batch = sampler(gen) if sampler is not None else None
        loss, terms, aux = loss_fn(batch)
        gsh = None
        if grad_terms and it % log_every == 0:
            gsh = grad_norms({k: terms[k] for k in grad_terms if k in terms},
                             params)
        loss.backward()
        cur = loss.item()
        checkpoint.update(cur, it, aux)
        opt.step()
        if it % log_every == 0 and log_fn is not None:
            log_fn(it, cur, terms, aux, gsh)

    lbfgs = torch.optim.LBFGS(
        params, max_iter=lbfgs_max, max_eval=lbfgs_max, tolerance_grad=1e-8,
        tolerance_change=1e-10, history_size=50, line_search_fn="strong_wolfe")
    fixed = sampler(gen) if sampler is not None else None

    def closure():
        lbfgs.zero_grad()
        loss, _terms, aux = loss_fn(fixed)
        loss.backward()
        checkpoint.update(loss.item(), adam_iters, aux)
        return loss

    final = lbfgs.step(closure)
    return final, time.time() - t0


def as_float(v):
    """Loggable float from a tensor or a plain number."""
    return float(v.detach()) if torch.is_tensor(v) else float(v)


def rel_l1(c, ref):
    """Summed relative error, the coefficient metric the experiments rank
    on -- trajectory error swings 3x on seed alone and cannot rank."""
    import numpy as np
    c = np.asarray(c, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float((np.abs(c - ref) / np.abs(ref)).sum())


def grad_share(gsh, a="phys", b="dat"):
    """Fraction of the measured gradient norm owned by term ``a``.
    Reported because 'the terms are the same order at convergence' is not
    evidence that both are doing work."""
    if not gsh:
        return float("nan")
    return gsh.get(a, 0.0) / max(gsh.get(a, 0.0) + gsh.get(b, 0.0), 1e-30)
