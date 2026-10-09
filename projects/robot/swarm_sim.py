r"""Do the discovered laws reproduce the SWARM? Simulate every robot of a recording together.

Each robot steps at the fitted lag (10 frames = 0.4 s), with its tokens recomputed from the
simulated state and history every step (same definitions as swarm_data.contact_features):
    r_i <- r_i + dt * (K_i^x . tok_i,  K_i^y . tok_i) + dt * sd_i * xi
    tok = [c, s, push_x, push_y, wall_x, wall_y, vb_x, vb_y, x, y, 1, m1_x, m1_y, .., m4_x, m4_y]
          x, y centred on the arena; vb = velocity over the last step; m_j = velocity over earlier
          windows (slow memory): 0.4-0.8, 0.8-1.6, 1.6-3.6, 3.6-9.6 s ago
    phi_i <- phi_i + w_i dt + sd_phi_i * xi                             (own spin, own phase noise)
then hard-disk projection: overlapping pairs pushed apart to the recording's core distance, robots
outside the recording's largest radius put back on it (a physical constraint, not a fitted law;
swarm_contact_law.py replaces the pair part by the learned contact law).
sd_i = the model's residual sd on the recording (velocity units). The simulation starts after
9.6 s of recorded history (the longest memory window) and runs to the end of the recording.

Laws (coefficients K per robot, all fitted on the same recording):
  pooled        one law for all robots (shared propulsion v0 e^{i theta}, push, wall, persistence)
  own_prop      own propulsion per robot, the rest shared (joint least squares; epde_style.py)
  epde          every robot its OWN EPDE equations (results/swarm_robot_epde.jsonl, lowest-
                discrepancy member of each front)
  pipeline      every robot its own best-subset pipeline law with slow-memory tokens available
                (results/robot_pipeline.jsonl, lowest-WAPE member of each front)
  <law>+mem     the same plus slow memory m1..m4 (shared, isotropic; for epde fitted on the
                residuals of each robot's EPDE equations)
Statistics of the simulated swarm vs the recording (sampled at the same 0.4 s step):
  MSD at 0.4 .. 100 s, contact-number fractions, wall fraction, radial layering (total variation
  distance of the r/R histogram), speed by contact number, circulation (mean tangential velocity),
  and per-robot correlations sim vs data of MSD(10 s), mean radius and tangential velocity.

python swarm_sim.py [--seeds 4] [--laws pooled,own_prop,...]      -> results/swarm_sim.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from swarm_data import (swarm_files, swarm_meta, load_swarm, arena_envelope, spin_rates,  # noqa: E402
                        contact_features, ROBOT_DIAMETER_PX as SIGMA)
import swarm_robot_epde as SRE                                                             # noqa: E402

LAG = 10
MEM = [(1, 2), (2, 4), (4, 9), (9, 24)]             # memory windows, in steps back from now
HIST = MEM[-1][1]
TOK = ["c", "s", "push_x", "push_y", "wall_x", "wall_y", "vb_x", "vb_y", "x", "y", "1"] + \
      [f"m{j}_{v}" for j in range(1, len(MEM) + 1) for v in ("x", "y")]
TI = {k: i for i, k in enumerate(TOK)}
MEMX = [TI[f"m{j}_x"] for j in range(1, len(MEM) + 1)]
MEMY = [TI[f"m{j}_y"] for j in range(1, len(MEM) + 1)]
MSD_LAGS = [1, 2, 5, 10, 25, 50, 100, 250]          # steps of 0.4 s
RBINS = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.05]
LAWS = ["pooled", "pooled+mem", "own_prop", "own_prop+mem", "epde", "epde+mem", "pipeline"]


# ------------------------------------------------------------------ recording
def recording(path):
    t, x, y, phi, ids = load_swarm(path)
    dt = t[1] - t[0]
    xc, yc, r_env = arena_envelope(x, y)
    w = spin_rates(t, phi)
    F = contact_features(x, y, w, xc, yc, r_env)
    rows = np.arange(HIST * LAG, len(t) - LAG)
    tok = np.zeros((len(rows), len(ids), len(TOK)))
    tok[..., TI["c"]], tok[..., TI["s"]] = np.cos(phi[rows]), np.sin(phi[rows])
    for k in ("push_x", "push_y", "wall_x", "wall_y"):
        tok[..., TI[k]] = F[k][rows]
    for v, pos in (("x", x), ("y", y)):
        tok[..., TI[f"vb_{v}"]] = (pos[rows] - pos[rows - LAG]) / (LAG * dt)
        for j, (a, b) in enumerate(MEM, 1):
            tok[..., TI[f"m{j}_{v}"]] = (pos[rows - a * LAG] - pos[rows - b * LAG]) / ((b - a) * LAG * dt)
    tok[..., TI["x"]], tok[..., TI["y"]] = x[rows] - xc, y[rows] - yc
    tok[..., TI["1"]] = 1.0
    fx, fy = (x[rows + LAG] - x[rows]) / (LAG * dt), (y[rows + LAG] - y[rows]) / (LAG * dt)
    # data on the simulation's clock: every LAG frames
    samp = np.arange(0, len(t), LAG)
    pos = np.stack([x[samp], y[samp]], -1)
    dphi = phi[samp[1:]] - phi[samp[:-1]]
    nn = []
    for k in range(0, len(samp), 5):
        d = np.linalg.norm(pos[k][:, None] - pos[k][None], axis=-1)
        d[np.diag_indices_from(d)] = np.inf
        nn.append(d.min(1))
    r_all = np.hypot(pos[..., 0] - xc, pos[..., 1] - yc)
    return dict(tag=swarm_meta(path)["tag"], ids=[int(i) for i in ids], tok=tok, fx=fx, fy=fy, w=w,
                dt=LAG * dt, xc=xc, yc=yc, r_env=r_env, pos=pos, phi=phi[samp],
                sd_phi=dphi.std(0), core=float(np.percentile(np.concatenate(nn), 1)),
                r_max=float(np.percentile(r_all, 99.9)))


# ------------------------------------------------------------------ laws -> per-robot K (N, 2, len(TOK))
def fit_joint(R, own_prop, memory, rows=slice(None)):
    """Joint least squares of the isotropic law: propulsion (own or shared), shared push, wall,
    persistence, constants (+ shared slow memory)."""
    N = len(R["ids"])
    nm = len(MEM) if memory else 0
    ns = 5 + nm
    p = ns + (2 * N if own_prop else 2)
    G, g = np.zeros((p, p)), np.zeros(p)
    for i in range(N):
        T = R["tok"][rows, i]
        one, zero = np.ones(len(T)), np.zeros(len(T))
        Ax = np.column_stack([T[:, TI["push_x"]], T[:, TI["wall_x"]], T[:, TI["vb_x"]], one, zero]
                             + [T[:, j] for j in MEMX[:nm]] + [T[:, TI["c"]], -T[:, TI["s"]]])
        Ay = np.column_stack([T[:, TI["push_y"]], T[:, TI["wall_y"]], T[:, TI["vb_y"]], zero, one]
                             + [T[:, j] for j in MEMY[:nm]] + [T[:, TI["s"]], T[:, TI["c"]]])
        A = np.vstack([Ax, Ay])
        y = np.concatenate([R["fx"][rows, i], R["fy"][rows, i]])
        idx = list(range(ns)) + ([ns + 2 * i, ns + 2 * i + 1] if own_prop else [ns, ns + 1])
        G[np.ix_(idx, idx)] += A.T @ A
        g[idx] += A.T @ y
    c = np.linalg.solve(G + 1e-9 * np.trace(G) / p * np.eye(p), g)
    muP, muW, a, cx, cy = c[:5]
    mem = c[5:ns]
    K = np.zeros((N, 2, len(TOK)))
    for i in range(N):
        ai, bi = (c[ns + 2 * i], c[ns + 2 * i + 1]) if own_prop else (c[ns], c[ns + 1])
        K[i, 0, [TI["c"], TI["s"], TI["push_x"], TI["wall_x"], TI["vb_x"], TI["1"]]] = [ai, -bi, muP, muW, a, cx]
        K[i, 1, [TI["c"], TI["s"], TI["push_y"], TI["wall_y"], TI["vb_y"], TI["1"]]] = [bi, ai, muP, muW, a, cy]
        K[i, 0, MEMX[:nm]], K[i, 1, MEMY[:nm]] = mem, mem
    return K, mem


def add_memory(R, K, rows=slice(None)):
    """Shared isotropic slow memory fitted on the residuals of fixed per-robot laws K."""
    res, M = [], []
    for comp, (f, mi) in enumerate(((R["fx"], MEMX), (R["fy"], MEMY))):
        res.append((f[rows] - np.einsum("tik,ik->ti", R["tok"][rows], K[:, comp])).ravel())
        M.append(R["tok"][rows][..., mi].reshape(-1, len(MEM)))
    mem = np.linalg.lstsq(np.vstack(M), np.concatenate(res), rcond=None)[0]
    K = K.copy()
    K[:, 0, MEMX] += mem
    K[:, 1, MEMY] += mem
    return K, mem


def epde_K(R, runs):
    """Every robot's own EPDE equations (most complete front member, solved for the derivative)."""
    N = len(R["ids"])
    K = np.zeros((N, 2, len(TOK)))
    for i, rid in enumerate(R["ids"]):
        for comp, var in enumerate(("x", "y")):
            row = runs.get((R["tag"], rid, var))
            eq = SRE.most_complete(row)[0] if row else None
            if eq is None:
                continue
            terms, icpt, _ = eq
            for name, coef in terms.items():
                K[i, comp, TI[name]] += coef
            K[i, comp, TI["1"]] += icpt
    return K


def pipeline_K(R, pipe):
    """Every robot's own pipeline law (lowest-WAPE member of its (WAPE, win14) front; no intercept)."""
    N = len(R["ids"])
    K = np.zeros((N, 2, len(TOK)))
    for i, rid in enumerate(R["ids"]):
        for comp, var in enumerate(("x", "y")):
            r = pipe.get((R["tag"], rid, var))
            if r is None:
                continue
            best = min(r["front"], key=lambda j: r["cands"][j]["wape"])
            for name, coef in r["cands"][best]["terms"].items():
                K[i, comp, TI[name]] += coef
    return K


def pipeline_K_np(R, pipe):
    """pipeline_K for laws that may hold the neighbour-drive tokens: (K, Knp)."""
    N = len(R["ids"])
    K, Knp = np.zeros((N, 2, len(TOK))), np.zeros((N, 2, len(NP_TOK)))
    for i, rid in enumerate(R["ids"]):
        for comp, var in enumerate(("x", "y")):
            r = pipe.get((R["tag"], rid, var))
            if r is None:
                continue
            best = min(r["front"], key=lambda j: r["cands"][j]["wape"])
            for name, coef in r["cands"][best]["terms"].items():
                if name in NP_TOK:
                    Knp[i, comp, NP_TOK.index(name)] += coef
                else:
                    K[i, comp, TI[name]] += coef
    return K, Knp


# ------------------------------------------------------------------ neighbour drive
#: Tokens built from the NEIGHBOURS' own propulsion p_j = v0_j e(phi_j + theta_j), read off robot j's
#: own law (its c and s coefficients):  np = sum over touching j of p_j;  np_in = sum over touching j
#: of the part of p_j pointing into robot i along the bond, max(0, p_j . u_ji) u_ji.
NP_TOK = ["np_x", "np_y", "np_in_x", "np_in_y"]


def propulsion_vec(Ksrc, c, s):
    """p_j from a law's c and s coefficients; c, s (..., N) -> (..., N, 2)."""
    return np.stack([Ksrc[:, 0, TI["c"]] * c + Ksrc[:, 0, TI["s"]] * s,
                     Ksrc[:, 1, TI["c"]] * c + Ksrc[:, 1, TI["s"]] * s], -1)


def drive_tokens(r, p):
    """r, p (..., N, 2) -> (..., N, 4) = [np_x, np_y, np_in_x, np_in_y]."""
    d = r[..., None, :, :] - r[..., :, None, :]              # [i, j] = r_j - r_i
    D = np.linalg.norm(d, axis=-1)
    n = D.shape[-1]
    D[..., np.arange(n), np.arange(n)] = np.inf
    C = (D < 1.1 * SIGMA).astype(float)
    u = -d / D[..., None]                                   # unit vector from j to i
    npv = np.einsum("...ij,...jk->...ik", C, p)
    proj = np.clip(np.einsum("...ijk,...jk->...ij", u, p), 0, None) * C
    npin = np.einsum("...ij,...ijk->...ik", proj, u)
    return np.concatenate([npv, npin], -1)


def neighbour_drive(R, Ksrc):
    """The recording's neighbour-drive tokens (T_rows, N, 4), propulsion read from law Ksrc."""
    pos = np.stack([R["tok"][..., TI["x"]], R["tok"][..., TI["y"]]], -1)
    p = propulsion_vec(Ksrc, R["tok"][..., TI["c"]], R["tok"][..., TI["s"]])
    out = np.zeros(pos.shape[:2] + (len(NP_TOK),))
    for t0 in range(0, len(pos), 500):
        sl = slice(t0, min(len(pos), t0 + 500))
        out[sl] = drive_tokens(pos[sl], p[sl])
    return out


def fit_drive(R, K, NP, which=(0, 1), rows=slice(None)):
    """Shared isotropic coefficients of the neighbour-drive tokens on the residuals of law K.
    which: 0 = np, 1 = np_in. Returns Knp (N, 2, 4)."""
    px, py = predict(R, K, rows)
    ex, ey = R["fx"][rows] - px, R["fy"][rows] - py
    cols_x, cols_y = [[0, 2][w] for w in which], [[1, 3][w] for w in which]
    A = np.vstack([NP[rows][..., cols_x].reshape(-1, len(which)), NP[rows][..., cols_y].reshape(-1, len(which))])
    coef = np.linalg.lstsq(A, np.concatenate([ex.ravel(), ey.ravel()]), rcond=None)[0]
    Knp = np.zeros((len(R["ids"]), 2, len(NP_TOK)))
    Knp[:, 0, cols_x], Knp[:, 1, cols_y] = coef, coef
    return Knp


def predict_np(R, K, Knp, NP, rows=slice(None)):
    px, py = predict(R, K, rows)
    return (px + np.einsum("tik,ik->ti", NP[rows], Knp[:, 0]), py + np.einsum("tik,ik->ti", NP[rows], Knp[:, 1]))


def residual_sd_np(R, K, Knp, NP):
    px, py = predict_np(R, K, Knp, NP)
    return np.stack([(R["fx"] - px).std(0), (R["fy"] - py).std(0)], -1)


# ------------------------------------------------------------------ distance-resolved push (contact law)
#: Push resolved by the pair distance (robot diameters): push_k = sum over neighbours j with D_ij in
#: band k of the unit vector from j to i. The deepest band also covers any deeper overlap.
PUSH_EDGES = np.array([0.0, 0.93, 0.97, 1.0, 1.05, 1.1])


def band_push(r):
    """r (..., N, 2) -> (..., N, n_bands, 2)."""
    d = r[..., None, :, :] - r[..., :, None, :]              # [i, j] = r_j - r_i
    D = np.linalg.norm(d, axis=-1)
    n = D.shape[-1]
    D[..., np.arange(n), np.arange(n)] = np.inf
    u = -d / D[..., None]                                   # unit vector from j to i
    k = np.digitize(D / SIGMA, PUSH_EDGES) - 1
    return np.stack([np.einsum("...ij,...ijk->...ik", (k == b).astype(float), u)
                     for b in range(len(PUSH_EDGES) - 1)], -2)


def recording_band_push(R):
    """The recording's banded push tokens (T_rows, N, n_bands, 2)."""
    pos = np.stack([R["tok"][..., TI["x"]], R["tok"][..., TI["y"]]], -1)
    out = np.zeros(pos.shape[:2] + (len(PUSH_EDGES) - 1, 2))
    for t0 in range(0, len(pos), 500):
        sl = slice(t0, min(len(pos), t0 + 500))
        out[sl] = band_push(pos[sl])
    return out


def fit_contact(R, K, NP, BP, Knp=None, rows=slice(None)):
    """Shared isotropic contact law on the residuals of law K: a correction of the push per distance
    band, plus np and np_in unless the law already holds its own neighbour drive Knp.
    Returns (Kb (n_bands,), Knp (N, 2, 4))."""
    px, py = predict(R, K, rows) if Knp is None else predict_np(R, K, Knp, NP, rows)
    ex, ey = R["fx"][rows] - px, R["fy"][rows] - py
    nb = BP.shape[-2]
    Ax = [BP[rows][..., b, 0] for b in range(nb)] + ([NP[rows][..., 0], NP[rows][..., 2]] if Knp is None else [])
    Ay = [BP[rows][..., b, 1] for b in range(nb)] + ([NP[rows][..., 1], NP[rows][..., 3]] if Knp is None else [])
    A = np.vstack([np.stack([a.ravel() for a in Ax], -1), np.stack([a.ravel() for a in Ay], -1)])
    coef = np.linalg.lstsq(A, np.concatenate([ex.ravel(), ey.ravel()]), rcond=None)[0]
    if Knp is None:
        Knp = np.zeros((len(R["ids"]), 2, len(NP_TOK)))
        Knp[:, 0, [0, 2]], Knp[:, 1, [1, 3]] = coef[nb:], coef[nb:]
    return coef[:nb], Knp


def predict_contact(R, K, Knp, NP, Kb, BP, rows=slice(None)):
    px, py = predict_np(R, K, Knp, NP, rows)
    return px + BP[rows][..., 0] @ Kb, py + BP[rows][..., 1] @ Kb


def residual_sd_contact(R, K, Knp, NP, Kb, BP):
    px, py = predict_contact(R, K, Knp, NP, Kb, BP)
    return np.stack([(R["fx"] - px).std(0), (R["fy"] - py).std(0)], -1)


def predict(R, K, rows=slice(None)):
    return (np.einsum("tik,ik->ti", R["tok"][rows], K[:, 0]), np.einsum("tik,ik->ti", R["tok"][rows], K[:, 1]))


def residual_sd(R, K):
    """Per robot and axis, the law's residual sd on the recording (velocity units)."""
    px, py = predict(R, K)
    return np.stack([(R["fx"] - px).std(0), (R["fy"] - py).std(0)], -1)


def heldout_r2(R, own_prop, memory):
    """First half -> second half (a lag-wide gap), x and y pooled."""
    T = R["tok"].shape[0]
    tr, te = slice(0, T // 2 - LAG), slice(T // 2 + LAG, T)
    K, _ = fit_joint(R, own_prop, memory, tr)
    px, py = predict(R, K, te)
    y = np.concatenate([R["fx"][te].ravel(), R["fy"][te].ravel()])
    e = y - np.concatenate([px.ravel(), py.ravel()])
    return 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)


# ------------------------------------------------------------------ simulation
def features(r, R):
    d = r[None, :, :] - r[:, None, :]                       # [i, j] = r_j - r_i
    D = np.linalg.norm(d, axis=-1)
    D[np.diag_indices_from(D)] = np.inf
    C = D < 1.1 * SIGMA
    P = np.where(C[..., None], -d / D[..., None], 0.0).sum(1)
    rel = r - [R["xc"], R["yc"]]
    rr = np.linalg.norm(rel, axis=1)
    W = np.where((rr > R["r_env"] - 0.25 * SIGMA)[:, None], -rel / rr[:, None], 0.0)
    return P, W, rel


def project(r, R, iters=30, core=None):
    """Hard disks: separate overlapping pairs to the core distance (the recording's by default; 0 =
    no pair projection) and keep centres inside r_max, both in every iteration (the wall must not
    push robots back into each other)."""
    core = R["core"] if core is None else core
    c0 = np.array([R["xc"], R["yc"]])
    for _ in range(iters):
        d = r[None, :, :] - r[:, None, :]
        D = np.linalg.norm(d, axis=-1)
        D[np.diag_indices_from(D)] = np.inf
        ov = np.clip(core - D, 0, None)
        r = r - 0.5 * (ov[..., None] * d / D[..., None]).sum(1)
        rel = r - c0
        rr = np.linalg.norm(rel, axis=1, keepdims=True)
        out = rr[:, 0] > R["r_max"]
        r = np.where(out[:, None], c0 + rel / rr * R["r_max"], r)
        if ov.max() < 1e-3 and not out.any():
            break
    return r


def shared_noise(r, rng, lam, ell):
    """Unit-variance noise per robot and axis, a fraction lam shared through a smooth random field:
    corr(i, j) = lam exp(-d_ij^2 / 2 ell^2) for i != j, 1 on the diagonal (x and y independent)."""
    D2 = np.sum((r[:, None] - r[None]) ** 2, -1)
    C = lam * np.exp(-D2 / (2 * ell ** 2))
    C[np.diag_indices_from(C)] = 1.0
    return np.linalg.cholesky(C + 1e-9 * np.eye(len(r))) @ rng.standard_normal(r.shape)


def simulate(R, K, sd, rng, shared=None, track=False, phi_replay=False, Knp=None, Ksrc=None, Kb=None,
             core=None):
    """From the recorded state after HIST steps of history, to the end of the recording.
    shared = (lam, ell): neighbour-correlated noise (see shared_noise). track: also return each
    step's residual of the law (realised step velocity minus the law's prediction). phi_replay:
    headings taken from the recording instead of the constant-spin model (a diagnostic).
    Knp: coefficients of the neighbour-drive tokens, the neighbours' propulsion read from law Ksrc.
    Kb: push correction per distance band (band_push). core: pair-projection distance (0 = none)."""
    dt = R["dt"]
    hist = list(R["pos"][:HIST + 1].copy())               # positions of the last HIST steps and now
    phi = R["phi"][HIST].copy()
    resid = []
    for k in range(len(R["pos"]) - HIST - 1):
        if phi_replay:
            phi = R["phi"][HIST + k]
        r = hist[-1]
        P, W, rel = features(r, R)
        vb = (r - hist[-2]) / dt
        mem = [(hist[-1 - a] - hist[-1 - b]) / ((b - a) * dt) for a, b in MEM]
        tok = np.column_stack([np.cos(phi), np.sin(phi), P, W, vb, rel, np.ones(len(r))] + mem)
        v = np.stack([(K[:, 0] * tok).sum(1), (K[:, 1] * tok).sum(1)], -1)
        if Knp is not None:
            dtok = drive_tokens(r, propulsion_vec(Ksrc, np.cos(phi), np.sin(phi)))
            v = v + np.stack([(Knp[:, 0] * dtok).sum(1), (Knp[:, 1] * dtok).sum(1)], -1)
        if Kb is not None:
            v = v + np.einsum("ibk,b->ik", band_push(r), Kb)
        xi = rng.standard_normal(r.shape) if shared is None else shared_noise(r, rng, *shared)
        hist.append(project(r + dt * v + dt * sd * xi, R, core=core))
        if track:
            resid.append((hist[-1] - r) / dt - v)
        phi = phi + R["w"] * dt + R["sd_phi"] * rng.standard_normal(len(phi))
    pos = np.array(hist[HIST:])
    return (pos, np.array(resid)) if track else pos


# ------------------------------------------------------------------ swarm statistics
def stats(pos, R):
    dt = R["dt"]
    rel = pos - [R["xc"], R["yc"]]
    rr = np.linalg.norm(rel, axis=-1) / R["r_env"]
    msd = {L: float(np.mean(np.sum((pos[L:] - pos[:-L]) ** 2, -1))) for L in MSD_LAGS if L < len(pos)}
    robot_msd10 = np.mean(np.sum((pos[25:] - pos[:-25]) ** 2, -1), 0)
    n = np.array([(np.linalg.norm(p[None] - p[:, None], axis=-1) < 1.1 * SIGMA).sum(1) - 1 for p in pos])
    v = (pos[1:] - pos[:-1]) / dt
    speed = np.linalg.norm(v, axis=-1)
    tang = (rel[:-1, :, 0] * v[..., 1] - rel[:-1, :, 1] * v[..., 0]) / np.linalg.norm(rel[:-1], axis=-1)
    cls = np.minimum(n[:-1], 4)
    hist = np.histogram(np.clip(rr, 0, RBINS[-1] - 1e-9), bins=RBINS)[0] / rr.size
    return dict(msd=msd, contacts=[float(np.mean(np.minimum(n, 4) == k)) for k in range(5)],
                wall=float(np.mean(rr > 1 - 0.25 * SIGMA / R["r_env"])), rhist=hist.tolist(),
                speed_by_n=[float(speed[cls == k].mean()) if np.any(cls == k) else np.nan for k in range(5)],
                circulation=float(tang.mean()),
                robot=dict(msd10=robot_msd10.tolist(), radius=rr.mean(0).tolist(), tang=tang.mean(0).tolist()))


def compare(sim, data):
    """Summary errors of one simulation set (list over seeds) against the recording."""
    msd = {L: np.mean([s["msd"][L] for s in sim]) for L in data["msd"]}
    rob = {k: np.mean([s["robot"][k] for s in sim], 0) for k in data["robot"]}
    mean = lambda key: np.mean([s[key] for s in sim], 0)                          # noqa: E731
    return dict(
        msd_ratio={L: float(msd[L] / data["msd"][L]) for L in msd},
        contacts=mean("contacts").tolist(), wall=float(mean("wall")),
        rhist_tv=float(0.5 * np.abs(mean("rhist") - np.array(data["rhist"])).sum()),
        speed_by_n=mean("speed_by_n").tolist(), circulation=float(mean("circulation")),
        corr={k: float(np.corrcoef(rob[k], data["robot"][k])[0, 1]) for k in rob})


def laws_for(R, runs, pipe, names):
    out, mem = {}, {}
    for name in names:
        base, plus = name.split("+")[0], name.endswith("+mem")
        if base in ("pooled", "own_prop"):
            out[name], m = fit_joint(R, base == "own_prop", plus)
            if plus:
                mem[name] = m
        elif base == "epde":
            K = epde_K(R, runs)
            if plus:
                K, mem[name] = add_memory(R, K)
            out[name] = K
        elif base == "pipeline":
            out[name] = pipeline_K(R, pipe)
    return out, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--laws", default=",".join(LAWS))
    a = ap.parse_args()
    names = a.laws.split(",")
    runs = {(r["tag"], r["robot_id"], r["var"]): r for r in map(json.loads, open(SRE.RESULTS))
            if r["status"] == "ok"}
    pipe_path = os.path.join(HERE, "results", "robot_pipeline.jsonl")
    pipe = ({(r["tag"], r["robot_id"], r["var"]): r for r in map(json.loads, open(pipe_path))}
            if os.path.exists(pipe_path) else {})
    res = {}
    for f in swarm_files():
        R = recording(f)
        data = stats(R["pos"][HIST:], R)
        laws, mem = laws_for(R, runs, pipe, names)
        res[R["tag"]] = dict(data={k: v for k, v in data.items() if k != "robot"}, core=R["core"],
                             r_max=R["r_max"], r_env=R["r_env"],
                             memory={k: v.tolist() for k, v in mem.items()},
                             heldout_r2={f"{k}{'+mem' if m else ''}": heldout_r2(R, k == "own_prop", m)
                                         for k in ("pooled", "own_prop") for m in (False, True)})
        h = res[R["tag"]]["heldout_r2"]
        print(f"\n### {R['tag']}  ({len(R['ids'])} robots, core {R['core']:.0f} px, "
              f"{len(R['pos']) - HIST} steps of {R['dt']:.1f} s, {a.seeds} seeds)")
        print("  held-out R2 (1st half -> 2nd): " + ", ".join(f"{k} {v:.4f}" for k, v in h.items()))
        for k, m in mem.items():
            print(f"  {k} memory coefficients (0.4-0.8 / 0.8-1.6 / 1.6-3.6 / 3.6-9.6 s ago): "
                  + " / ".join(f"{v:+.3f}" for v in m))
        print(f"  {'':12s} {'MSD ratio sim/data at 0.4 / 2 / 10 / 40 / 100 s':>46s} | contacts 0/1/2/3/4+ "
              f"| wall | layer TV | speed by contacts 0..4+ | circ | corr per robot: MSD10 radius tang")
        d = data
        print(f"  {'data':12s} {'':>46s} | " + "/".join(f"{v:.2f}" for v in d["contacts"])
              + f" | {d['wall']:.2f} |    -     | " + "/".join(f"{v:.0f}" for v in d["speed_by_n"])
              + f" | {d['circulation']:+.1f}")
        for name, K in laws.items():
            sd = residual_sd(R, K)
            sims = [stats(simulate(R, K, sd, np.random.default_rng(s)), R) for s in range(a.seeds)]
            c = compare(sims, data)
            res[R["tag"]][name] = c
            mr = c["msd_ratio"]
            print(f"  {name:12s} {' / '.join(f'{mr[L]:.2f}' for L in (1, 5, 25, 100, 250) if L in mr):>46s} | "
                  + "/".join(f"{v:.2f}" for v in c["contacts"]) + f" | {c['wall']:.2f} |  {c['rhist_tv']:.3f}   | "
                  + "/".join(f"{v:.0f}" for v in c["speed_by_n"]) + f" | {c['circulation']:+.1f} | "
                  + "  ".join(f"{c['corr'][k]:+.2f}" for k in ("msd10", "radius", "tang")), flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "swarm_sim.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
