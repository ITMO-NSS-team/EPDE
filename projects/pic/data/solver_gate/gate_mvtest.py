"""Discrimination gate for EPDE's solver-based fitness, one fresh process per solve.

The gate is the question the fitness exists to answer: does the TRUE equation
score better than a WRONG one on the held-out tail? Per seed,

    gate = ln(RMSE_wrong / RMSE_true)          (> 0: the true form wins)

and an arm's headline is the MINIMUM over seeds. Each solve goes through the
real fitness host (``SolverBasedFitness`` + ``Discrepancy('deepxde')``), with the
candidate's coefficients refit by ``LinRegBasedCoeffsEquation`` -- exactly the
call sequence of the scratchpad harnesses (``gtol_experiment.py``,
``methods_isolated.py``) whose numbers are the stored references.

WHY ONE PROCESS PER SOLVE: a DeepXDE solve's result depends on what the process
solved before it (measured: -20% on AC, -53% on Duffing after one intervening
solve), and reseeding does not remove it. Every solve here starts from an
identical process history. The duplicate-control arm ('B-ctl', entered last)
must come back bitwise equal to 'B'; that is the evidence isolation holds.

MODES
  solve   one solve, minimal instrumentation (phase timing, exit record)
  trace   as solve, plus the held-out RMSE at EVERY L-BFGS iterate
          (``torch.optim.lbfgs._strong_wolfe`` wrapped, read-only). One trace at
          cap C predicts the gate at every cap <= C: under a strong-Wolfe line
          search ``max_iter`` bounds only the iteration loop, so the path up to
          iterate k is the same whatever the cap (validated bit-exact against
          60 real solves, Sep 2026). ``analyze.py`` does the replay.
  drive   the parent loop: arms x forms x seeds, one subprocess each, rows
          appended to results/<tag>.jsonl (resumable; existing rows skipped)
  probe   several solves IN ONE process (the run-order artifact diagnostic)

usage
  python gate.py drive  --system ac --arms B,B-ctl --seeds 0-2 [--mode trace]
                        [--tag baseline] [--cpu] [--force]
  python gate.py solve  --system ac --arm B --form true --seed 0 [--mode trace]
  python gate.py probe  --system duffing --arm B --forms true,wrong,true [--cpu]
  python gate.py list
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(REPO, "projects", "pic", "data")
RESULTS = os.path.join(HERE, "results")

FORMS = ("true", "wrong")

# Modules whose content decides a solve. Their sha256 goes into every row, so a
# row can be matched to the code that produced it long after the tree moved.
PROVENANCE_FILES = (
    "epde/integrate/deepxde_integration.py",
    "epde/operators/common/fitness.py",
    "epde/operators/common/objectives.py",
    "epde/interface/search_config.py",
    "epde/integrate/heldout.py",
    "epde/integrate/data_profile.py",
    "epde/integrate/residual_terms.py",
    "epde/integrate/basis_integration.py",
    "epde/integrate/lr_free.py",
    "projects/pic/data/solver_gate/gate.py",
)


# ============================================================ arms
_S0 = {"epochs": 0, "lbfgs_maxiter": 4000, "pairing_digest": True}
#: where ``gate.py prefit`` leaves the shared data fits (S9/B9 only load them)
SHARED_FITS = os.path.join(RESULTS, "shared_fits")

# PRE-SPECIFIED in source. An arm is a backend plus overrides on that backend's
# shipped default config. 'control_of' marks a byte-for-byte duplicate, which
# drive() always runs LAST in its batch.
ARMS = {
    # shipped recipe: Adam lr 1e-4 x 2000, then L-BFGS <= 2000
    "B":      {"backend": "deepxde", "config": {}},
    "B-ctl":  {"backend": "deepxde", "config": {}, "control_of": "B"},
    # the shipped recipe with the L-BFGS cap raised, for gate(cap) up to 4000
    # in trace mode; its replay at 2000 must equal 'B' bitwise
    "B-4k":   {"backend": "deepxde", "config": {"lbfgs_maxiter": 4000}},
    # ---- Stage 1a: learning-rate-free network, L-BFGS only (no first-order
    # phase). S0 is the control of every S arm: Glorot init, L-BFGS from it.
    # Each S arm adds ONE mechanism to S0. pairing_digest records a digest of
    # the initial weights so the pairing can be checked, not assumed.
    "S0":     {"backend": "deepxde", "config": _S0},
    "S0-ctl": {"backend": "deepxde", "config": _S0, "control_of": "S0"},
    "S1":     {"backend": "deepxde", "config": dict(_S0, input_transform="affine")},
    "S2":     {"backend": "deepxde", "config": dict(_S0, output_transform="train_moments")},
    "S3":     {"backend": "deepxde", "config": dict(_S0, init="data_prefit")},
    "S4":     {"backend": "deepxde", "config": dict(_S0, init="lstsq_last_layer")},
    "S6":     {"backend": "deepxde", "config": dict(_S0, precision="float64")},
    # the scale-invariant physics loss (gradient weighting unchanged)
    "S7":     {"backend": "deepxde", "config": dict(_S0, pde_loss="sinv_live")},
    "S8":     {"backend": "deepxde", "config": dict(_S0, pde_loss="sinv_data")},
    # ONE observation-only network per dataset (train window, L-BFGS to its
    # natural stop), reused as the start of every candidate (user, Sep 28);
    # B9 is the same start under the shipped Adam + L-BFGS recipe
    "S9":     {"backend": "deepxde", "config": dict(_S0, init="shared_data_fit",
                                                    shared_fit_dir=SHARED_FITS)},
    "B9":     {"backend": "deepxde", "config": {"init": "shared_data_fit",
                                                "shared_fit_dir": SHARED_FITS,
                                                "pairing_digest": True}},
    # ---- Stage 1c (Sep 29, user-selected follow-ups; each ONE mechanism)
    # S1 with its L-BFGS capped so a REAL solve fits the shipped time: 2000 is
    # the shipped L-BFGS cap and sits under the largest cap measured to fit T_B
    # on every system (2040, from the Stage-1 S1 traces; the gate was not read)
    "S1-cap": {"backend": "deepxde", "config": dict(_S0, input_transform="affine",
                                                    lbfgs_maxiter=2000)},
    # the shared data fit, now on the input-scaled network (its fair test)
    "S10":    {"backend": "deepxde", "config": dict(_S0, input_transform="affine",
                                                    init="shared_data_fit",
                                                    shared_fit_dir=SHARED_FITS)},
    # the scale-invariant losses with a floor of 1e-3 x the train-window mean
    # data term mass
    "S7f":    {"backend": "deepxde", "config": dict(_S0, pde_loss="sinv_live",
                                                    sinv_floor_rel=1e-3)},
    "S8f":    {"backend": "deepxde", "config": dict(_S0, pde_loss="sinv_data",
                                                    sinv_floor_rel=1e-3)},
    # ---- Stage 1b: the other lr-free routes, each vs the shipped recipe B
    "P1":     {"backend": "deepxde", "config": {"first_order": "dog", "lr": None,
                                                "pairing_digest": True}},
    "P2":     {"backend": "deepxde", "config": {"first_order": "prodigy", "lr": None,
                                                "pairing_digest": True}},
    "D1":     {"backend": "deepxde", "config": {"lr_rule": "prefit_distance", "lr": None,
                                                "pairing_digest": True}},
    "D2":     {"backend": "deepxde", "config": {"lr_rule": "data_selected", "lr": None,
                                                "pairing_digest": True}},
    # Levenberg-Marquardt on a [32]x4 net (full size would take ~26 h), with
    # its matched L-BFGS control on the same net
    "S0-32":  {"backend": "deepxde", "config": dict(_S0, net=[32, 32, 32, 32])},
    "S5-32":  {"backend": "deepxde", "config": {"epochs": 0, "lbfgs_maxiter": 0,
                                                "second_order": "lm", "lm_maxiter": 500,
                                                "net": [32, 32, 32, 32], "lr": None,
                                                "pairing_digest": True}},
    # null checks: must equal S0 bitwise (seed 0 is enough)
    "S0-nolr":  {"backend": "deepxde", "config": dict(_S0, lr=None), "null_of": "S0"},
    "S3-null":  {"backend": "deepxde", "config": dict(_S0, init="data_prefit",
                                                      prefit_maxiter=0), "null_of": "S0"},
    # ---- Stage 1b: the fixed basis, L-BFGS (deterministic: one seed). The
    # default is the backend's shipped config; every other cell changes ONE
    # entry of it (a sweep, not a tuning: nothing is adjusted until all cells
    # are in).
    "BL":          {"backend": "basis", "config": {}},
    "BL-n1":       {"backend": "basis", "config": {"refine": "off", "nyquist_factor": 1.0}},
    "BL-n2":       {"backend": "basis", "config": {"refine": "off", "nyquist_factor": 2.0}},
    "BL-n3":       {"backend": "basis", "config": {"refine": "off", "nyquist_factor": 3.0}},
    "BL-n4":       {"backend": "basis", "config": {"refine": "off", "nyquist_factor": 4.0}},
    "BL-r5":       {"backend": "basis", "config": {"refine_tol": 1e-5}},
    "BL-data":     {"backend": "basis", "config": {"start": "data_lstsq"}},
    "BL-noprec":   {"backend": "basis", "config": {"precondition": "none"}},
    "BL-t4":       {"backend": "basis", "config": {"time_factor": 4.0}},
    "BL-c3":       {"backend": "basis", "config": {"collocation_factor": 3}},
    "BL-sinv-live": {"backend": "basis", "config": {"pde_loss": "sinv_live"}},
    "BL-sinv-data": {"backend": "basis", "config": {"pde_loss": "sinv_data"}},
    # Stage 1c: the resolution chosen by convergence of the prediction; the
    # floored scale-invariant losses
    "BL-pred":     {"backend": "basis", "config": {"refine": "prediction"}},
    "BL-sinv-live-f": {"backend": "basis", "config": {"pde_loss": "sinv_live",
                                                      "sinv_floor_rel": 1e-3}},
    "BL-sinv-data-f": {"backend": "basis", "config": {"pde_loss": "sinv_data",
                                                      "sinv_floor_rel": 1e-3}},
    # harness mechanics only -- never a result
    "_smoke": {"backend": "deepxde", "config": {"epochs": 20, "lbfgs_maxiter": 30,
                                                "num_domain": 200, "num_test": 50}},
}

SEEDLESS_BACKENDS = ("basis",)  # deterministic backends run one seed only


def arm_backend(arm):
    return ARMS[arm]["backend"]


def arm_config(arm):
    """The full config the arm solves with: shipped default + overrides."""
    spec = ARMS[arm]
    sys.path.insert(0, REPO)
    if spec["backend"] == "deepxde":
        from epde.interface.search_config import _default_deepxde_config
        base = _default_deepxde_config()
    elif spec["backend"] == "basis":
        from epde.interface.search_config import _default_basis_config
        base = _default_basis_config()
    else:
        raise ValueError(f"unknown backend {spec['backend']!r}")
    return dict(base, **spec["config"])


# ============================================================ systems
#: variables per system, in pool order = vars_to_describe = the net's output
#: columns = the order the translate dict must follow
SYSTEM_VARS = {"ac": ["u"], "duffing": ["u"], "burgers": ["u"],
               "lv": ["u", "v"], "ns": ["u", "v", "p"], "lorenz": ["u", "v", "w"]}
#: the variables the gate RMSE averages (the host's force_out_of_place rule is
#: the mean over ALL equations). NS leaves p out: only grad p enters the
#: physics, so p's tail carries a free gauge drift c(t) -- noise, not verdict.
GATE_VARS = {"ns": ["u", "v"]}
SYSTEMS = tuple(SYSTEM_VARS)
# Lorenz-63 window: t in [20.0, 25.2] of the stored run (dt 1e-3), every 5th
# sample, re-zeroed (the system is autonomous, so the shift is exact).
# Inner [0.05, 5.15]; train_frac 0.8 -> tail 1.02 time units = 0.92 T_lambda
# (lambda_1 = 0.900 measured). The repo's t[:1000] is an off-attractor
# transient with a 0.14 T_lambda tail.
LORENZ_WINDOW = {"i0": 20000, "n": 1041, "step": 5, "boundary_width": 10}


def gate_vars(system):
    return GATE_VARS.get(system, SYSTEM_VARS[system])


def _lorenz_window():
    t = np.load(os.path.join(DATA, "lorenz", "t.npy"))
    X = np.load(os.path.join(DATA, "lorenz", "lorenz.npy"))
    w = LORENZ_WINDOW
    idx = w["i0"] + w["step"] * np.arange(w["n"])
    return t[idx] - t[idx[0]], X[idx]


def _duffing_params():
    d = np.load(os.path.join(DATA, "duffing", "duffing.npz"))
    return {k: float(d[k]) for k in ("delta", "alpha", "beta", "gamma", "omega")}


def system_forms(system):
    """The (true, wrong) equation strings. Coefficients are REFIT from the data
    by ``LinRegBasedCoeffsEquation``; only the structure matters here."""
    if system == "ac":
        return {
            "true": ("0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + "
                     "5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}"),
            "wrong": ("-5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} + 0.0 = "
                      "du/dx0{power: 1.0}"),
        }
    if system == "duffing":
        p = _duffing_params()
        common = (f"{-p['delta']} * du/dx0{{power: 1.0}} + {-p['alpha']} * u{{power: 1.0}} + ")
        forcing = (f"{p['gamma']} * cos{{power: 1.0, freq: {p['omega']}, dim: 0.0}} + 0.0 = "
                   f"d^2u/dx0^2{{power: 1.0}}")
        return {
            "true": common + f"{-p['beta']} * u{{power: 3.0}} + " + forcing,
            "wrong": common + forcing,
        }
    if system == "burgers":
        # PDE-FIND viscous Burgers: u_t = -u u_x + 0.1 u_xx, periodic in x.
        return {
            "true": ("0.1 * d^2u/dx1^2{power: 1.0} + "
                     "-1.0 * u{power: 1.0} * du/dx1{power: 1.0} + 0.0 = "
                     "du/dx0{power: 1.0}"),
            "wrong": ("-1.0 * u{power: 1.0} * du/dx1{power: 1.0} + 0.0 = "
                      "du/dx0{power: 1.0}"),
        }
    if system == "lv":
        # Synthetic Lotka-Volterra, alpha=beta=gamma=delta=20, IC (4, 2), h=1/301
        # (projects/hunter-prey/data_preparation.py; its hand-written RK4 has two
        # stage-3 bugs, so the record is LV only to ~0.4% -- the refit absorbs it).
        # The wrong form drops the prey growth: the refit intercept becomes
        # constant recruitment, a bounded damped spiral (the LOWEST of the four
        # single-term drops; dropping predator mortality blows up in the tail).
        v_eq = ("-20.0 * v{power: 1.0} + 20.0 * u{power: 1.0} * v{power: 1.0} + 0.0 = "
                "dv/dx0{power: 1.0}")
        return {
            "true": {"u": ("20.0 * u{power: 1.0} + -20.0 * u{power: 1.0} * v{power: 1.0} + "
                           "0.0 = du/dx0{power: 1.0}"),
                     "v": v_eq},
            "wrong": {"u": "-20.0 * u{power: 1.0} * v{power: 1.0} + 0.0 = du/dx0{power: 1.0}",
                      "v": v_eq},
        }
    if system == "ns":
        # x-momentum (u: target u_t), continuity (v: target v_y), y-momentum
        # solved for p_y (p) -- the mapping of main:ns/cv_metric.py. dx0=t,
        # dx1=y, dx2=x. The wrong form drops the pressure gradient from
        # x-momentum (u becomes 2-D viscous Burgers); static FD residual 10x true.
        adv_u = ("-1.0 * u{power: 1.0} * du/dx2{power: 1.0} + "
                 "-1.0 * v{power: 1.0} * du/dx1{power: 1.0} + ")
        visc_u = ("0.01 * d^2u/dx2^2{power: 1.0} + 0.01 * d^2u/dx1^2{power: 1.0} + "
                  "0.0 = du/dx0{power: 1.0}")
        cont = "-1.0 * du/dx2{power: 1.0} + 0.0 = dv/dx1{power: 1.0}"
        mom_v = ("-1.0 * dv/dx0{power: 1.0} + -1.0 * u{power: 1.0} * dv/dx2{power: 1.0} + "
                 "-1.0 * v{power: 1.0} * dv/dx1{power: 1.0} + "
                 "0.01 * d^2v/dx2^2{power: 1.0} + 0.01 * d^2v/dx1^2{power: 1.0} + "
                 "0.0 = dp/dx1{power: 1.0}")
        return {
            "true": {"u": adv_u + "-1.0 * dp/dx2{power: 1.0} + " + visc_u,
                     "v": cont, "p": mom_v},
            "wrong": {"u": adv_u + visc_u, "v": cont, "p": mom_v},
        }
    if system == "lorenz":
        # Lorenz-63, sigma=10, rho=28, beta=8/3 (x, y, z -> u, v, w). The wrong
        # form drops -x*z, the y-equation's only nonlinearity: (u, v) becomes a
        # damped linear oscillator of about the right period that never switches
        # lobes. Oracle gate +3.8..+5.1 over 8 window starts.
        eq_u = "10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}"
        eq_w = ("1.0 * u{power: 1.0} * v{power: 1.0} + -2.6666666666666665 * w{power: 1.0} "
                "+ 0.0 = dw/dx0{power: 1.0}")
        return {
            "true": {"u": eq_u,
                     "v": ("28.0 * u{power: 1.0} + -1.0 * u{power: 1.0} * w{power: 1.0} + "
                           "-1.0 * v{power: 1.0} + 0.0 = dv/dx0{power: 1.0}"),
                     "w": eq_w},
            "wrong": {"u": eq_u,
                      "v": "28.0 * u{power: 1.0} + -1.0 * v{power: 1.0} + 0.0 = dv/dx0{power: 1.0}",
                      "w": eq_w},
        }
    raise ValueError(f"unknown system {system!r}")


def build_search(system):
    """The EPDE pool the candidate is translated on -- the same calls as the
    historical harnesses for AC and Duffing, and ``burgers_discovery``'s
    settings for Burgers."""
    import epde
    search = epde.EpdeSearch(use_solver=False,
                             verbose_params={"show_iter_idx": False}, device="cpu")
    if system == "ac":
        data = np.load(os.path.join(DATA, "ac", "ac_data.npy"))
        grids = np.meshgrid(np.linspace(0.0, 1.0, 51),
                            np.linspace(-1.0, 0.984375, 128), indexing="ij")
        _, domain = search.createDomain((grids[0], grids[1]), boundary_width=(5, 12), ID=0)
        search.set_preprocessor(default_preprocessor_type="FD", preprocessor_kwargs={})
        _, trajectory = search.createTrajectory({"u": data}, domain, cache_id=0)
        search.create_pool(data=[trajectory], max_deriv_order=(2, 3), additional_tokens=[])
    elif system == "duffing":
        from epde import TrigonometricTokens
        d = np.load(os.path.join(DATA, "duffing", "duffing.npz"))
        t, x = d["t"].astype(np.float64), d["x"].astype(np.float64)
        omega = float(d["omega"])
        _, domain = search.createDomain(t, boundary_width=10, ID=0)
        search.set_preprocessor(default_preprocessor_type="FD", preprocessor_kwargs={})
        _, trajectory = search.createTrajectory({"u": x}, domain, cache_id=0)
        search.create_pool(data=[trajectory], max_deriv_order=(2,), data_fun_pow=3,
                           additional_tokens=[TrigonometricTokens(
                               dimensionality=0, freq=(omega - 1e-3, omega + 1e-3))])
    elif system == "burgers":
        from scipy.io import loadmat
        m = loadmat(os.path.join(DATA, "burgers", "burgers.mat"))
        t, x = np.ravel(m["t"]), np.ravel(m["x"])
        data = np.transpose(np.real(m["usol"]))              # (t, x)
        grids = np.meshgrid(t, x, indexing="ij")
        _, domain = search.createDomain((grids[0], grids[1]), boundary_width=20, ID=0)
        search.set_preprocessor(default_preprocessor_type="FD", preprocessor_kwargs={})
        _, trajectory = search.createTrajectory({"u": data}, domain, cache_id=0)
        search.create_pool(data=[trajectory], max_deriv_order=(2, 3), data_fun_pow=3,
                           additional_tokens=[])
    elif system == "lv":
        # all 301 levels (lv.py's t[:150] would leave a ~0.07 tail); t in [0, 1)
        t = np.load(os.path.join(DATA, "lv", "t_20.npy")).astype(np.float64)
        data = np.load(os.path.join(DATA, "lv", "data_20.npy")).astype(np.float64)
        _, domain = search.createDomain(t, boundary_width=10, ID=0)
        search.set_preprocessor(default_preprocessor_type="FD", preprocessor_kwargs={})
        _, trajectory = search.createTrajectory({"u": data[:, 0], "v": data[:, 1]},
                                                domain, cache_id=0)
        search.create_pool(data=[trajectory], max_deriv_order=(1,), data_fun_pow=1,
                           additional_tokens=[])
    elif system == "ns":
        # Raissi et al. cylinder wake (Nektar DNS, Re=100; D=1, U_inf=1,
        # nu=0.01), float64 .mat (the npz's float32 grids trip EPDE's
        # non-uniform-axis warning). X_star is x-fastest, so a snapshot reshapes
        # to (y, x); EPDE axes x0=t, x1=y, x2=x. GATE SUBSET (the full window
        # puts ~359k points in every DeepXDE step): the first 50 levels (dt 0.1;
        # the p_y-target refit degrades at dt 0.2), every 2nd y on [-1.59, 1.51],
        # every 2nd x on [1, 5.95]. Boundary (2, 2, 7): the x-width keeps the
        # scored region ~1 time unit of advection downstream of the open inflow.
        from scipy.io import loadmat
        m = loadmat(os.path.join(DATA, "ns", "cylinder_nektar_wake.mat"))
        t_all = np.ravel(m["t"])
        x_all, y_all = np.unique(m["X_star"][:, 0]), np.unique(m["X_star"][:, 1])

        def field(a):                                   # (N, T) -> (t, y, x)
            return a.T.reshape(len(t_all), len(y_all), len(x_all))

        ts, ys, xs = slice(0, 50), slice(5, 45, 2), slice(0, 72, 2)
        full = {"u": field(m["U_star"][:, 0, :]), "v": field(m["U_star"][:, 1, :]),
                "p": field(m["p_star"])}
        data = {k: np.ascontiguousarray(f[ts][:, ys][:, :, xs]) for k, f in full.items()}
        grids = np.meshgrid(t_all[ts], y_all[ys], x_all[xs], indexing="ij")
        _, domain = search.createDomain((grids[0], grids[1], grids[2]),
                                        boundary_width=(2, 2, 7), ID=0)
        search.set_preprocessor(default_preprocessor_type="FD", preprocessor_kwargs={})
        _, trajectory = search.createTrajectory(data, domain, cache_id=0)
        search.create_pool(data=[trajectory], max_deriv_order=(1, 2, 2), data_fun_pow=1,
                           additional_tokens=[])
    elif system == "lorenz":
        t, X = _lorenz_window()
        _, domain = search.createDomain(t, boundary_width=LORENZ_WINDOW["boundary_width"],
                                        ID=0)
        search.set_preprocessor(default_preprocessor_type="FD", preprocessor_kwargs={})
        _, trajectory = search.createTrajectory({"u": X[:, 0], "v": X[:, 1], "w": X[:, 2]},
                                                domain, cache_id=0)
        search.create_pool(data=[trajectory], max_deriv_order=(1,), data_fun_pow=1,
                           additional_tokens=[])
    else:
        raise ValueError(f"unknown system {system!r}")
    return search


def fitted_system(search, text, system="ac"):
    """Translate + refit EVERY equation. Returns (SoEq, [equations in
    vars_to_describe order]). A single-variable text stays a str (the str
    overload, as the historical harnesses); a coupled one is a dict."""
    from epde.interface.equation_translator import translate_equation
    from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
    all_vars = SYSTEM_VARS[system]
    eq_system = translate_equation(text, search.pool, all_vars=list(all_vars))
    if list(eq_system.vars_to_describe) != list(all_vars):
        raise RuntimeError(f"vars_to_describe {eq_system.vars_to_describe} != {all_vars}")
    eqs = [eq_system.vals[var] for var in all_vars]
    for var, eq in zip(all_vars, eqs):
        eq.main_var_to_explain = var
    eq_system.use_default_singleobjective_function()
    # translate raises both *_evald flags on every equation, so an unrefit one
    # would pass the host's guard and be solved with the TEXT coefficients
    for eq in eqs:
        eq.weights_internal = np.ones(len(eq.structure))
        eq.weights_internal_evald = True
        LinRegBasedCoeffsEquation().apply(eq, {})
    return eq_system, eqs


def _score_fields(system, eqs):
    """The row's RMSE fields. 'rmse' is the gate RMSE: the mean over the gate
    variables (one variable: that variable's RMSE, exactly)."""
    per = {var: float(eq.fitness_value) for var, eq in zip(SYSTEM_VARS[system], eqs)}
    out = {"rmse": float(np.mean([per[v] for v in gate_vars(system)]))}
    if len(eqs) > 1:
        out["rmse_per_var"] = per
        out["coeffs"] = {var: [float(w) for w in np.asarray(eq.weights_final).reshape(-1)]
                         for var, eq in zip(SYSTEM_VARS[system], eqs)}
        out["equation"] = {var: getattr(eq, "text_form", None)
                           for var, eq in zip(SYSTEM_VARS[system], eqs)}
    else:
        out["coeffs"] = [float(w) for w in np.asarray(eqs[0].weights_final).reshape(-1)]
        out["equation"] = getattr(eqs[0], "text_form", None)
    return out


# ============================================================ provenance
def _sha(path):
    try:
        with open(os.path.join(REPO, path), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return None


def provenance():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                                  check=True).stdout
        except Exception:                                      # noqa: BLE001
            return b""
    return {
        "head": git("rev-parse", "HEAD").decode().strip(),
        "diff_epde": hashlib.sha256(git("diff", "HEAD", "--", "epde")).hexdigest()[:16],
        "files": {p: _sha(p) for p in PROVENANCE_FILES},
    }


def environment():
    import torch
    env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                          "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}
    env["torch_threads"] = torch.get_num_threads()
    env["torch"] = torch.__version__
    # With CUDA_VISIBLE_DEVICES="" on Windows, is_available() can still say
    # True while device 0 does not exist; a report must never cost a solve.
    try:
        env["cuda"] = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        env["gpu"] = torch.cuda.get_device_name(0) if env["cuda"] else None
    except Exception as exc:                                   # noqa: BLE001
        env["cuda"], env["gpu"] = False, f"unavailable ({exc})"
    mod = sys.modules.get("deepxde")
    env["deepxde"] = getattr(mod, "__version__", None) if mod is not None else None
    return env


# ============================================================ one solve
def _sync():
    """Finish queued GPU work, so a timed probe measures itself only."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def trace_path(tag, system, arm, form, seed):
    return os.path.join(RESULTS, "traces", tag, f"{system}__{arm}__{form}__{seed}.json")


def _held_out_replica(dxi, train_frac, val_frac, system="ac"):
    """Held-out RMSE of a net, computed exactly as the host scores it:
    prediction on the FULL grid, inner mask, test block of ``time_split``;
    per variable, then the mean over the gate variables (as ``_score_fields``)."""
    import torch
    samples = dxi.global_var.samples_manager
    key = samples.trajecatoryIDs[0]
    g_full = [np.asarray(g) for g in samples.grids()[key]]
    flat_mask = np.asarray(samples.gFunc("m")[key]).reshape(-1)
    t_inner = g_full[0].reshape(-1)[flat_mask]
    held = dxi.time_split(t_inner, train_frac, val_frac).test
    cols = [SYSTEM_VARS[system].index(v) for v in gate_vars(system)]
    obs = [np.asarray(samples.get((v, (1.0,)))[key]).reshape(-1)[held]
           for v in gate_vars(system)]
    X_np = np.asarray(dxi._input_columns(g_full))
    cache = {}

    def rmse(net):
        p0 = next(net.parameters())
        k = (p0.device, p0.dtype)
        if k not in cache:
            cache[k] = torch.as_tensor(X_np, dtype=p0.dtype, device=p0.device)
        with torch.no_grad():
            p = net(cache[k]).detach().cpu().numpy()
        errs = []
        for c, o in zip(cols, obs):
            s = p[:, c].reshape(-1)[flat_mask][held]
            errs.append(float(np.sqrt(np.mean((s - o) ** 2))))
        return float(np.mean(errs))

    return rmse


def run_basis_solve(system, arm, form, seed, mode, tag):
    """One fixed-basis solve through the real host. No deepxde import."""
    if mode != "solve":
        raise SystemExit("the basis backend has no trace mode (it refines between levels); "
                         "use --mode solve")
    t_start = time.perf_counter()
    sys.path.insert(0, REPO)
    import epde                                                    # noqa: F401
    from epde.operators.common.fitness import SolverBasedFitness
    from epde.operators.common.objectives import Discrepancy
    cfg = arm_config(arm)
    search = build_search(system)
    eq_system, eqs = fitted_system(search, system_forms(system)[form], system)
    prelude = time.perf_counter() - t_start
    host = SolverBasedFitness(["penalty_coeff", "error_metric", "basis_config"],
                              primary=Discrepancy("deepxde"), backend="basis")
    host.params = {"penalty_coeff": 0.2, "error_metric": "rmse", "basis_config": cfg}
    for eq in eqs:
        eq.fitness_calculated = False
    t0 = time.perf_counter()
    host.apply(eq_system, {})
    seconds = time.perf_counter() - t0
    adapter = host.adapter
    stats = adapter.last_solve_stats or {}
    return {
        "tag": tag, "system": system, "arm": arm, "form": form, "seed": seed,
        "mode": mode, "backend": "basis", "config": cfg,
        **_score_fields(system, eqs), "seconds": seconds, "prelude_seconds": prelude,
        "lbfgs_exit": adapter.last_lbfgs_exit,
        "basis_stats": stats,
        "deepxde_imported": "deepxde" in sys.modules,
        "env": environment(), "provenance": provenance(), "pid": os.getpid(),
    }


def run_solve(system, arm, form, seed, mode="solve", tag="adhoc", write_trace=True):
    """One solve in THIS process. Returns the row (and writes the trace)."""
    backend = arm_backend(arm)
    if backend == "basis":
        return run_basis_solve(system, arm, form, seed, mode, tag)
    t_start = time.perf_counter()
    sys.path.insert(0, REPO)
    if backend != "deepxde":
        raise NotImplementedError(backend)
    if mode == "trace" and ARMS[arm]["config"].get("second_order") == "lm":
        raise SystemExit(f"{arm} has no L-BFGS phase to trace; use --mode solve")
    # Import order of the historical harnesses: deepxde first. The stored
    # reference RMSEs were produced this way, and bitwise reproduction is the
    # first check every batch has to pass.
    import deepxde as dde
    import torch
    import torch.optim.lbfgs as tl
    import epde                                                    # noqa: F401
    import epde.integrate.deepxde_integration as dxi
    from epde.operators.common.fitness import SolverBasedFitness
    from epde.operators.common.objectives import Discrepancy

    cfg = arm_config(arm)
    search = build_search(system)
    eq_system, eqs = fitted_system(search, system_forms(system)[form], system)
    prelude = time.perf_counter() - t_start

    rec = {"phases": [], "rows": [], "chunks": [], "loss_weights_lbfgs": None,
           "rmse_lbfgs_entry": None, "t_lbfgs_start": None, "G0_probe": None,
           "instr_seconds": 0.0}
    state = {"model": None}
    T0 = time.perf_counter()

    # ---- phase timing (both modes; pure Python bookkeeping)
    orig_train = dde.Model.train

    def timed_train(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_train(self, *a, **k)
        rec["phases"].append({"optimizer": str(self.opt_name),
                              "iterations": k.get("iterations"),
                              "seconds": time.perf_counter() - t0})
        return out

    dde.Model.train = timed_train

    rmse_now = None
    if mode == "trace":
        rmse_now = _held_out_replica(dxi, float(cfg.get("train_frac", 0.8)),
                                     float(cfg.get("val_frac", 0.0)), system)

    orig_install = dxi._install_lbfgs_stall_guard

    def install(model):
        state["model"] = model
        rec["t_lbfgs_start"] = time.perf_counter() - T0
        rec["loss_weights_lbfgs"] = [float(w) for w in model.loss_weights]
        if mode == "trace":
            _sync()
            ti = time.perf_counter()
            try:
                rec["G0_probe"] = float(dxi.DeepXDEAdapter._max_abs_grad(model, model.loss_weights))
                rec["rmse_lbfgs_entry"] = rmse_now(model.net)
            except Exception as exc:                           # noqa: BLE001
                rec["instr_error"] = repr(exc)
                raise
            rec["instr_seconds"] += time.perf_counter() - ti
            opt = model.opt
            stock = opt.step

            def step(closure=None, *a, **k):
                st = opt.state.get(opt._params[0], {})
                n0 = int(st.get("n_iter", 0))
                res = stock(closure, *a, **k)
                n1 = int(opt.state[opt._params[0]]["n_iter"])
                rec["chunks"].append([n0, n1, time.perf_counter() - T0])
                return res

            opt.step = step
        return orig_install(model)

    dxi._install_lbfgs_stall_guard = install

    if mode == "trace":
        orig_sw = tl._strong_wolfe

        def sw(obj_func, x, t, dvec, f, g, gtd, *a, **k):
            m = state["model"]
            if m is None:                  # not the physics L-BFGS phase
                return orig_sw(obj_func, x, t, dvec, f, g, gtd, *a, **k)
            try:
                n = int(m.opt.state[m.opt._params[0]]["n_iter"])
                _sync()          # queued solve work must not be billed to the probe
                ti = time.perf_counter()
                r_before = rmse_now(m.net)
            except Exception as exc:                           # noqa: BLE001
                # the adapter's blanket except would turn this into a NaN
                # "solve failed"; record it so the row is refused instead
                rec["instr_error"] = repr(exc)
                raise
            rec["instr_seconds"] += time.perf_counter() - ti
            g_before = float(g.abs().max())
            out = orig_sw(obj_func, x, t, dvec, f, g, gtd, *a, **k)
            f_new, g_new, t_new, evals = out
            rec["rows"].append([n, r_before, g_before, float(f_new),
                                float(g_new.abs().max()), float(t_new), int(evals),
                                time.perf_counter() - T0, rec["instr_seconds"]])
            return out

        tl._strong_wolfe = sw

    # ---- the solve, exactly as the historical harnesses
    dde.config.set_random_seed(seed)
    host = SolverBasedFitness(["penalty_coeff", "error_metric", "deepxde_config"],
                              primary=Discrepancy("deepxde"), backend="deepxde")
    host.params = {"penalty_coeff": 0.2, "error_metric": "rmse", "deepxde_config": cfg}
    for eq in eqs:
        eq.fitness_calculated = False
    t0 = time.perf_counter()
    host.apply(eq_system, {})
    seconds = time.perf_counter() - t0
    if rec.get("instr_error"):
        raise RuntimeError(f"trace instrumentation failed inside the solve: {rec['instr_error']}")

    m = state["model"]
    final_n = None
    if m is not None and m.opt._params and m.opt.state.get(m.opt._params[0]):
        final_n = int(m.opt.state[m.opt._params[0]]["n_iter"])
    adapter = host.adapter
    row = {
        "tag": tag, "system": system, "arm": arm, "form": form, "seed": seed,
        "mode": mode, "backend": "deepxde", "config": cfg,
        **_score_fields(system, eqs), "seconds": seconds, "prelude_seconds": prelude,
        "phases": rec["phases"],
        "lbfgs_exit": getattr(adapter, "last_lbfgs_exit", None),
        "init_stats": getattr(adapter, "last_init_stats", None),
        "sinv_stats": getattr(adapter, "last_sinv_stats", None),
        "lr_stats": getattr(adapter, "last_lr_stats", None),
        "lm_stats": getattr(adapter, "last_lm_stats", None),
        "pairing": getattr(adapter, "last_pairing", None),
        "final_n_iter": final_n,
        "loss_weights_lbfgs": rec["loss_weights_lbfgs"],
        "solve_device": str(dxi.DDE_SOLVE_DEVICE),
        "net_device": (str(next(m.net.parameters()).device) if m is not None else None),
        "net_dtype": (str(next(m.net.parameters()).dtype) if m is not None else None),
        "env": environment(), "provenance": provenance(), "pid": os.getpid(),
    }
    if mode == "trace":
        rec["rmse_final_replica"] = rmse_now(m.net) if m is not None else None
        row["rmse_final_replica"] = rec["rmse_final_replica"]
        row["instr_seconds"] = rec["instr_seconds"]
        row["n_trace_rows"] = len(rec["rows"])
        rec.update({k: row[k] for k in ("system", "arm", "form", "seed", "rmse",
                                        "final_n_iter", "lbfgs_exit", "config",
                                        "seconds")})
        rec["T_solve_start"] = t0 - T0
        if write_trace:
            path = trace_path(tag, system, arm, form, seed)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(rec, fh)
            row["trace"] = os.path.relpath(path, HERE)
    return row


# ============================================================ shared fits
def run_prefit(system, arm, seed):
    """Build the shared data fit of (system, arm's architecture, seed) and
    leave it in SHARED_FITS. Same imports, same seeding and the same call path
    up to the net as ``run_solve``, so the S9/B9 solves find it by key."""
    sys.path.insert(0, REPO)
    import deepxde as dde
    import epde                                                    # noqa: F401
    import epde.integrate.deepxde_integration as dxi
    cfg = arm_config(arm)
    if cfg.get("init") != "shared_data_fit":
        raise SystemExit(f"arm {arm} has no shared data fit")
    search = build_search(system)
    eq_system, _ = fitted_system(search, system_forms(system)["true"], system)
    samples = dxi.global_var.samples_manager
    key = samples.trajecatoryIDs[0]
    # one observed array per variable, in the host's order: the cache key
    # hashes them, so anything else would never match the solve's Y
    observed = [np.asarray(samples.get((v, (1.0,)))[key]).reshape(-1)
                for v in SYSTEM_VARS[system]]
    dde.config.set_random_seed(seed)
    adapter = dxi.DeepXDEAdapter(**cfg)
    t0 = time.perf_counter()
    stats = adapter.prepare_shared_fit(eq_system, grids=samples.grids()[key],
                                       data=observed, domain_key=key)
    return {"system": system, "arm": arm, "seed": seed, "seconds": time.perf_counter() - t0,
            "stats": stats, "pairing": adapter.last_pairing, "pid": os.getpid()}


def drive_prefit(system, arm, seeds, cpu=False):
    os.makedirs(SHARED_FITS, exist_ok=True)
    log = os.path.join(RESULTS, "shared_fits.jsonl")
    for seed in seeds:
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "prefit-one",
               "--system", system, "--arm", arm, "--seed", str(seed)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=_child_env(cpu))
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("PREFIT ")), None)
        if line is None:
            print(f"  prefit {system} seed {seed}: FAILED rc={proc.returncode}\n"
                  f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}", flush=True)
            continue
        row = json.loads(line[7:])
        with open(log, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        fit = (row["stats"] or {}).get("fit") or {}
        print(f"  prefit {system} seed {seed}: {row['stats'].get('source')} "
              f"key {row['stats'].get('key')}  iters {fit.get('prefit_iterations')}  "
              f"data loss {fit.get('data_loss_before'):.3e} -> {fit.get('data_loss_after'):.3e}  "
              f"{row['seconds']:.0f}s", flush=True)


# ============================================================ probe
def run_probe(system, arm, forms, seed):
    """Several solves in ONE process, same arm and seed, reseeded before each.
    If a repeat of form X differs from its first solve, the process carries
    state from one solve to the next (the run-order artifact)."""
    out = []
    for position, form in enumerate(forms, start=1):
        row = run_solve(system, arm, form, seed, mode="solve", tag="probe",
                        write_trace=False)
        row["position"] = position
        out.append(row)
        print(f"PROBE {position} {form:<6} rmse={row['rmse']!r} "
              f"{row['seconds']:.0f}s exit={row['lbfgs_exit']}", flush=True)
    seen = {}
    for r in out:
        if r["form"] in seen:
            first = seen[r["form"]]
            print(f"PROBE repeat of {r['form']}: first={first!r} now={r['rmse']!r} "
                  f"identical={first == r['rmse']} "
                  f"rel={abs(r['rmse'] - first) / first:.3e}", flush=True)
        else:
            seen[r["form"]] = r["rmse"]
    return out


# ============================================================ parent loop
def results_path(tag):
    return os.path.join(RESULTS, f"{tag}.jsonl")


def load_rows(tag):
    path = results_path(tag)
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _child_env(cpu):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def drive(system, arms, seeds, forms, mode, tag, cpu=False, force=False):
    # controls last: a duplicate entered after its original is the only
    # ordering under which "bitwise equal" rules out drift across the batch
    arms = sorted(arms, key=lambda a: "control_of" in ARMS[a])
    done = {(r["system"], r["arm"], r["form"], r["seed"], r["mode"])
            for r in load_rows(tag)}
    os.makedirs(RESULTS, exist_ok=True)
    started = time.time()
    for arm in arms:
        print(f"######## {system} {arm} ({mode})  {time.strftime('%H:%M:%S')}", flush=True)
        arm_seeds = (seeds[:1] if (arm_backend(arm) in SEEDLESS_BACKENDS
                                   or "null_of" in ARMS[arm]) else seeds)
        for form in forms:
            for seed in arm_seeds:
                if (system, arm, form, seed, mode) in done and not force:
                    print(f"  {arm:<8} {form:<5} seed {seed}: already in {tag}", flush=True)
                    continue
                cmd = [sys.executable, "-u", os.path.abspath(__file__), "solve",
                       "--system", system, "--arm", arm, "--form", form,
                       "--seed", str(seed), "--mode", mode, "--tag", tag]
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      env=_child_env(cpu))
                line = next((ln for ln in proc.stdout.splitlines()
                             if ln.startswith("ROW ")), None)
                if line is None:
                    print(f"  {arm:<8} {form:<5} seed {seed}: FAILED rc={proc.returncode}\n"
                          f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}", flush=True)
                    continue
                row = json.loads(line[4:])
                row["cpu_only"] = bool(cpu)
                with open(results_path(tag), "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                ex = row.get("lbfgs_exit") or {}
                extra = ""
                if mode == "trace":
                    extra = (f"  replica==host:{row['rmse'] == row['rmse_final_replica']}"
                             f"  instr {row['instr_seconds']:.1f}s")
                print(f"  {arm:<8} {form:<5} seed {seed}: tail RMSE {row['rmse']:.12e}  "
                      f"{row['seconds']:.0f}s (+{row['prelude_seconds']:.1f}s)  "
                      f"exit={ex.get('reason')}@{ex.get('iterations')}{extra}", flush=True)
    print(f"total {(time.time() - started) / 60:.1f} min -> {results_path(tag)}", flush=True)


def _parse_seeds(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["solve", "drive", "probe", "list", "prefit",
                                        "prefit-one"])
    ap.add_argument("--system", choices=SYSTEMS)
    ap.add_argument("--arm")
    ap.add_argument("--arms")
    ap.add_argument("--form", choices=FORMS)
    ap.add_argument("--forms", default=",".join(FORMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--mode", choices=["solve", "trace"], default="solve")
    ap.add_argument("--tag", default="adhoc")
    ap.add_argument("--cpu", action="store_true",
                    help="hide the GPU from the solve processes (CUDA_VISIBLE_DEVICES='')")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if a.command == "list":
        for name, spec in ARMS.items():
            print(f"{name:<8} {spec['backend']:<8} {json.dumps(spec['config'])}"
                  + (f"  (control of {spec['control_of']})" if "control_of" in spec else "")
                  + (f"  (null check of {spec['null_of']})" if "null_of" in spec else ""))
        return
    if a.command == "solve":
        row = run_solve(a.system, a.arm, a.form, a.seed, a.mode, a.tag)
        print("ROW " + json.dumps(row), flush=True)
        return
    if a.command == "prefit-one":
        print("PREFIT " + json.dumps(run_prefit(a.system, a.arm, a.seed)), flush=True)
        return
    if a.command == "prefit":
        drive_prefit(a.system, a.arm, _parse_seeds(a.seeds), cpu=a.cpu)
        return
    if a.command == "probe":
        if a.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        rows = run_probe(a.system, a.arm, a.forms.split(","), a.seed)
        os.makedirs(RESULTS, exist_ok=True)
        name = f"probe_{a.system}_{a.arm}_{'cpu' if a.cpu else 'gpu'}.json"
        with open(os.path.join(RESULTS, name), "w") as fh:
            json.dump(rows, fh, indent=1)
        return
    arms = a.arms.split(",")
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; see `gate.py list`")
    drive(a.system, arms, _parse_seeds(a.seeds), a.forms.split(","), a.mode, a.tag,
          cpu=a.cpu, force=a.force)


if __name__ == "__main__":
    main()
