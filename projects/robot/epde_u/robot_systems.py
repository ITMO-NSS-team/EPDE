"""Robot systems for the arm harness (run_auto.py / sys_trace.py in this directory).

TRACE_SYSTEM names:
    spin_<TAG>_<ID>      heading vector (c, s) = (cos phi, sin phi) of one robot, central FD (2 variables)
    driftx_<TAG>_<ID>    one coordinate of one robot (x or y; never both in one system: the shared future
    drifty_<TAG>_<ID>    step leaks), forward increments over LAG = 10 frames (0.4 s), time order 1
    field_<TAG>          mesoscale fields rho, Px, Py, Wx, Wy on the inscribed 14 x 14 square, sampled every
                         0.4 s (swarm_fields.coarse_time, tau 5), forward increments over one sample
TAG = a swarm recording (N30_PWM10_D41, N40_PWM10_D40, ...) or N1_PWM10 (the lone robot, ID 1).

USER Nov 2 2026: "Increase search space if possible, mb we need some tokens to account error". On top of run_auto's
common space (variables to power 3, derivative powers 1..2, grid and data-frequency trig tokens):
  * terms of up to 3 factors for the 1-D systems (spin, drift), probabilities 0.5 / 0.35 / 0.15; fields keep 1-2
    factors (5 variables: a 3-factor library would not fit the PVFP Gram);
  * physical tokens that can carry what the bare law leaves as error:
      spin   contacts {n_contact, n_same, n_opp} (swarm only): spin slows in contact, more for same-spin pairs
      drift  heading {c, s}; interaction {push, wall of this coordinate and the other}; contacts {n_contact};
             memory {vb, m1..m4 of this coordinate} (velocity over the last 0.4 s and earlier windows up to 9.6 s:
             the cage-restoring kernel); neighbour {np, np_in of this coordinate} (sum of touching robots'
             propulsion, from the recording's own-propulsion law, swarm_sim.neighbour_drive); position {the
             other coordinate}
      field  static {rho_bar} (each point's time-mean density: relaxation of the packing)
    Each family is one CacheStoredTokens (unique_token_type: one token of a family per term), power 1.

Reference forms (NOT ground truth; earlier best-subset / pipeline laws of this data) classify equations for the
harness logs: 'law' = exactly a reference form, 'law+' = a reference form plus terms. The constant is ignored.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = os.path.dirname(HERE)                                    # projects/robot (data + swarm_* modules)
LONE = 'N1_PWM10'
LAG = 10
FACTORS_1D = {'factors_num': [1, 2, 3], 'probas': [0.5, 0.35, 0.15]}
CACHE = os.path.join(HERE, 'robot_cache')


def _robot_path():
    if ROBOT not in sys.path:
        sys.path.append(ROBOT)


def parse(name):
    """(kind, tag, robot id or None) or None for a non-robot system."""
    # spinb (Nov 11): spin without the contacts family; spinvc / spinvs (Nov 12): one heading component as the only
    # variable, the other a value-only data token (no derivative of it, power 1)
    for kind in ('spin', 'spinb', 'spinvc', 'spinvs', 'driftx', 'drifty'):
        if name.startswith(kind + '_'):
            tag, rid = name[len(kind) + 1:].rsplit('_', 1)
            return kind, tag, int(rid)
    if name.startswith('field_'):
        return 'field', name[len('field_'):], None
    for k in ('fieldv5', 'fieldv6'):        # fieldv5: fieldv4 + div_V2 (contact-resolved robot model); fieldv6: the
        if name.startswith(k + '_'):        # same with every field smoothed 1.5 cap diameters wide
            var, tag = name[len(k) + 1:].split('_', 1)
            return k, tag, var
    if name.startswith('fieldv4_'):         # fieldv4_<VAR>_<TAG>: fieldv3 plus one family of divergence tokens
        var, tag = name[len('fieldv4_'):].split('_', 1)
        return 'fieldv4', tag, var
    if name.startswith('fieldv3_'):         # fieldv3_<VAR>_<TAG>: fieldv2 plus the track-built fields Q and Push
        var, tag = name[len('fieldv3_'):].split('_', 1)
        return 'fieldv3', tag, var
    if name.startswith('fieldv2_'):         # fieldv2_<VAR>_<TAG>: the same with space orders 2 (boundary 2: 10 x 10)
        var, tag = name[len('fieldv2_'):].split('_', 1)
        return 'fieldv2', tag, var
    if name.startswith('fieldv_'):          # fieldv_<VAR>_<TAG>: one field alone (robot id slot = the field name)
        var, tag = name[len('fieldv_'):].split('_', 1)
        return 'fieldv', tag, var
    return None


def is_robot(name):
    return parse(name) is not None


# ------------------------------------------------------------------ data
def _swarm_path(tag):
    _robot_path()
    from swarm_data import swarm_files, swarm_meta
    return next(p for p in swarm_files() if swarm_meta(p)['tag'] == tag)


def _neighbour(tag):
    """swarm_sim.neighbour_drive of the recording (rows HIST*LAG .. T-LAG), cached as npz; (ids, NP, names)."""
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, 'np_%s.npz' % tag)
    if os.path.exists(out):
        z = np.load(out)
        return [int(i) for i in z['ids']], z['NP'], [str(n) for n in z['names']], int(z['head'])
    _robot_path()
    import swarm_sim as SS
    R = SS.recording(_swarm_path(tag))
    NP = SS.neighbour_drive(R, SS.fit_joint(R, True, False)[0])
    np.savez(out, ids=np.array(R['ids']), NP=NP, names=np.array(SS.NP_TOK), head=SS.HIST * SS.LAG)
    return R['ids'], NP, list(SS.NP_TOK), SS.HIST * SS.LAG


def _track(tag, rid):
    """t, x, y (pixels, centred on the arena), phi (rad), contact features {label: (T,)} of one robot."""
    _robot_path()
    if tag == LONE:
        from robot_discovery import load_track
        t, x, y, phi = load_track()
        return t, x - x.mean(), y - y.mean(), phi, {}
    from swarm_data import load_swarm, arena_envelope, spin_rates, contact_features
    t, X, Y, PHI, ids = load_swarm(_swarm_path(tag))
    xc, yc, r_env = arena_envelope(X, Y)
    F = contact_features(X, Y, spin_rates(t, PHI), xc, yc, r_env)
    k = list(ids).index(rid)
    return t, X[:, k] - xc, Y[:, k] - yc, PHI[:, k], {lab: F[lab][:, k] for lab in F}


def _nonconstant(d):
    return {k: v for k, v in d.items() if np.ptp(v) > 0}


def _memory(pos, dt):
    _robot_path()
    from robot_pipeline_per_robot import memory_tokens
    return memory_tokens(pos, dt)


def load(name):
    """axes, fields, spec (extra token families as {family: {label: array}}, prep, orders, factors)."""
    kind, tag, rid = parse(name)
    if kind == 'field':
        return _load_field(tag)
    if kind in ('fieldv', 'fieldv2'):
        return _load_fieldv(tag, rid, (1, 2, 2) if kind == 'fieldv2' else None)
    if kind == 'fieldv3':
        return _load_fieldv3(tag, rid)
    if kind == 'fieldv4':
        return _load_fieldv4(tag, rid)
    if kind in ('fieldv5', 'fieldv6'):
        return _load_fieldv5(tag, rid, 1.0 if kind == 'fieldv5' else 1.5)
    t, x, y, phi, F = _track(tag, rid)
    if kind in ('spin', 'spinb'):
        # spinb: no data-token family, so EPDE's weak rows accept it (weak_rows.check_families refuses
        # CacheStoredTokens families such as contacts)
        fams = {} if kind == 'spinb' else \
            {'contacts': _nonconstant({k: F[k] for k in ('n_contact', 'n_same', 'n_opp') if k in F})}
        fams = {k: v for k, v in fams.items() if v}
        spec = dict(families=fams, prep_kwargs=None, orders=(2,), factors=FACTORS_1D)
        return [t], {'c': np.cos(phi), 's': np.sin(phi)}, spec
    if kind in ('spinvc', 'spinvs'):
        # Nov 12 (USER "Need a way to avoid unit-circle relation"): the other component enters as a value token only,
        # so no derivative of it (c c' + s s' = 0 and its derivatives) and, at power 1, no s^2 (c^2 + s^2 = 1)
        var, other = ('c', 's') if kind == 'spinvc' else ('s', 'c')
        vals = {'c': np.cos(phi), 's': np.sin(phi)}
        fams = {'heading': {other: vals[other]},
                'contacts': _nonconstant({k: F[k] for k in ('n_contact', 'n_same', 'n_opp') if k in F})}
        fams = {k: v for k, v in fams.items() if v}
        spec = dict(families=fams, prep_kwargs=None, orders=(2,), factors=FACTORS_1D)
        return [t], {var: vals[var]}, spec
    var, other = ('x', 'y') if kind == 'driftx' else ('y', 'x')
    pos, opos = (x, y) if var == 'x' else (y, x)
    dt = t[1] - t[0]
    vb = np.zeros_like(pos)
    vb[LAG:] = (pos[LAG:] - pos[:-LAG]) / (LAG * dt)
    memory = {'vb_' + var: vb}
    for j, m in enumerate(_memory(pos, dt), 1):
        memory['m%d_%s' % (j, var)] = m
    fams = {'heading': {'c': np.cos(phi), 's': np.sin(phi)},
            'interaction': _nonconstant({k: F[k] for k in ('push_x', 'push_y', 'wall_x', 'wall_y') if k in F}),
            'contacts': _nonconstant({k: F[k] for k in ('n_contact',) if k in F}),
            'memory': memory,
            'position': {'pos_' + other: opos}}
    if tag != LONE:
        ids, NP, names, head = _neighbour(tag)
        col = ids.index(rid)
        drive = {}
        for k, n in enumerate(names):
            if n.endswith('_' + var):
                full = np.zeros_like(pos)
                full[head:head + NP.shape[0]] = NP[:, col, k]
                drive[n] = full
        fams['neighbour'] = _nonconstant(drive)
    fams = {k: v for k, v in fams.items() if v}
    # boundary: the memory tokens look back HIST = 24 x 0.4 s (240 frames) and the neighbour drive starts there;
    # rows before it would carry zero-filled tokens. Symmetric, so it also covers the forward tail (LAG).
    spec = dict(families=fams, prep_kwargs={'scheme': 'forward', 'lag': LAG, 'accuracy': 2}, orders=(1,),
                factors=FACTORS_1D, boundary=[24 * LAG])
    if os.environ.get('EPDE_FORM') == 'weak':
        # Nov 11: weak rows integrate the FULL grid (no domain boundary), so cut the zero-filled head (memory /
        # neighbour drive start at 24 LAG) and the neighbour drive's empty tail (last LAG frames) off the data
        sl = slice(24 * LAG, len(t) - LAG)
        fams = {f: {k: np.ascontiguousarray(a[sl]) for k, a in d.items()} for f, d in fams.items()}
        spec['families'] = fams
        return [t[sl]], {var: np.ascontiguousarray(pos[sl])}, spec
    return [t], {var: pos}, spec


def _load_field(tag):
    _robot_path()
    import swarm_fields as SF
    F = SF.coarse_time(SF.build_fields(_swarm_path(tag)), 5)
    fields = {v: np.ascontiguousarray(F[v], dtype=float) for v in SF.VARS}
    shape = fields['rho'].shape
    fams = {'static': {'rho_bar': np.ascontiguousarray(np.broadcast_to(F['rho_bar'], shape), dtype=float)}}
    # run_auto's common space has space axes to order 4 (USER Nov 2: widen the space); time stays order 1 -- a
    # forward second difference would share the target's future sample (leak)
    orders = tuple(int(o) for o in os.environ.get('ROBOT_FIELD_ORDERS', '1,4,4').split(','))
    spec = dict(families=fams, prep_kwargs={'scheme': 'forward', 'lag': 1, 'accuracy': 2}, orders=orders,
                factors={'factors_num': [1, 2], 'probas': [0.65, 0.35]})
    return [F['t'], F['g'], F['g'].copy()], fields, spec


def _load_fieldv(tag, var, orders=None):
    """Nov 10: one field as the only EPDE variable (the Nov 2 joint field system leaked: the other fields' forward
    increments were features). The other fields enter as DATA tokens, the pipeline's library (swarm_fields.atoms_for):
    each other field v is one family {v, v_x, v_y, lap_v} (central differences, np.gradient edge_order 2), plus
    rho_bar. The own field's space derivatives (orders 4) come from EPDE; time = forward increment over one sample."""
    _robot_path()
    import swarm_fields as SF
    F = SF.coarse_time(SF.build_fields(_swarm_path(tag)), 5)
    shape = F[var].shape
    c = lambda a: np.ascontiguousarray(a, dtype=float)       # noqa: E731
    fams = {}
    for v in SF.VARS:
        if v == var:
            continue
        gx, gy = SF.spatial(F, F[v], 1), SF.spatial(F, F[v], 2)
        fams['f_' + v] = {v: c(F[v]), v + '_x': c(gx), v + '_y': c(gy),
                          'lap_' + v: c(SF.spatial(F, gx, 1) + SF.spatial(F, gy, 2))}
    fams['static'] = {'rho_bar': c(np.broadcast_to(F['rho_bar'], shape))}
    # orders (1, 4, 4): auto boundary [1, 4, 4] -> 6 x 6 interior (as the Nov 2 joint run); fieldv2 = (1, 2, 2):
    # boundary 2 -> the pipeline's 10 x 10 inner domain
    orders = orders or tuple(int(o) for o in os.environ.get('ROBOT_FIELD_ORDERS', '1,4,4').split(','))
    spec = dict(families=fams, prep_kwargs={'scheme': 'forward', 'lag': 1, 'accuracy': 2}, orders=orders,
                factors={'factors_num': [1, 2], 'probas': [0.65, 0.35]})
    return [F['t'], F['g'], F['g'].copy()], {var: c(F[var])}, spec


def _load_fieldv3(tag, var):
    """fieldv2 plus four token families built from the tracks at the same instant with the density's own kernel:
    Q = sum A v0_i p_i K (each robot's own propulsion speed times its heading; P weights every robot equally) and
    Push = sum A push_i K (excluded-volume push of touching neighbours, swarm_data.contact_features), each
    component one family {f, f_x, f_y, lap_f} as the other fields."""
    axes, fields, spec = _load_fieldv(tag, var, (1, 2, 2))
    import swarm_fields as SF
    from swarm_data import load_swarm, arena_envelope, contact_features
    path = _swarm_path(tag)
    F = SF.build_fields(path)
    t, x, y, phi, ids = load_swarm(path)
    xc, yc, r_env = arena_envelope(x, y)
    s = SF.SS.px_per_cm()
    sl = slice(None, None, SF.STEP * 5)
    xs, ys, ph = x[sl], y[sl], phi[sl]
    g, ell = F['g'], SF.SS.CAP_CM
    norm = np.pi * (ell / 2) ** 2 / (2 * np.pi * ell ** 2)
    kx = np.exp(-(g[None, None, :] - ((xs - xc) / s)[:, :, None]) ** 2 / (2 * ell ** 2))
    ky = np.exp(-(g[None, None, :] - ((ys - yc) / s)[:, :, None]) ** 2 / (2 * ell ** 2))
    field = lambda wt: norm * np.einsum('ti,tix,tiy->txy', wt, kx, ky)     # noqa: E731
    C = contact_features(xs, ys, F['w'], xc, yc, r_env)
    built = {'Qx': field(F['v0'] * np.cos(ph + F['theta'])), 'Qy': field(F['v0'] * np.sin(ph + F['theta'])),
             'Pushx': field(C['push_x']), 'Pushy': field(C['push_y'])}
    c = lambda a: np.ascontiguousarray(a, dtype=float)       # noqa: E731
    shape = fields[var].shape
    for name, a in built.items():
        assert a.shape == shape, (name, a.shape, shape)
        gx, gy = SF.spatial(F, a, 1), SF.spatial(F, a, 2)
        spec['families']['f_' + name] = {name: c(a), name + '_x': c(gx), name + '_y': c(gy),
                                         'lap_' + name: c(SF.spatial(F, gx, 1) + SF.spatial(F, gy, 2))}
    return axes, fields, spec


def _load_fieldv4(tag, var):
    """fieldv3 plus one family of divergence tokens: div_P, div_Q, div_Push and div_V. V = sum A v_i K with v_i each
    robot's PREDICTED velocity over one sample: a regression (one coefficient per feature for both components, the
    whole record) on its own propulsion v0_i p_i, push, wall, gear (swarm_data.contact_features), the touching
    neighbours' propulsion pointing at it (npin) and its own propulsion spent pushing into a neighbour (block). Every
    feature is known at the start of the sample."""
    axes, fields, spec = _load_fieldv3(tag, var)
    import swarm_fields as SF
    from swarm_data import load_swarm, arena_envelope, contact_features, ROBOT_DIAMETER_PX
    path = _swarm_path(tag)
    F = SF.build_fields(path)
    t, x, y, phi, ids = load_swarm(path)
    xc, yc, r_env = arena_envelope(x, y)
    s = SF.SS.px_per_cm()
    sl = slice(None, None, SF.STEP * 5)
    xs, ys, ph = x[sl], y[sl], phi[sl]
    X0, Y0 = (xs - xc) / s, (ys - yc) / s
    T, N = X0.shape
    dt = axes[0][1] - axes[0][0]
    g, ell = F['g'], SF.SS.CAP_CM
    norm = np.pi * (ell / 2) ** 2 / (2 * np.pi * ell ** 2)
    kx = np.exp(-(g[None, None, :] - X0[:, :, None]) ** 2 / (2 * ell ** 2))
    ky = np.exp(-(g[None, None, :] - Y0[:, :, None]) ** 2 / (2 * ell ** 2))
    field = lambda wt: norm * np.einsum('ti,tix,tiy->txy', wt, kx, ky)     # noqa: E731
    prop = F['v0'] * np.exp(1j * (ph + F['theta']))
    C = contact_features(xs, ys, F['w'], xc, yc, r_env)
    npin, block = np.zeros((T, N), dtype=complex), np.zeros((T, N), dtype=complex)
    idx = np.arange(N)
    for t0 in range(0, T, 400):
        cut = slice(t0, min(T, t0 + 400))
        dX = xs[cut][:, None, :] - xs[cut][:, :, None]              # [t, i, j] = x_j - x_i
        dY = ys[cut][:, None, :] - ys[cut][:, :, None]
        R = np.hypot(dX, dY)
        R[:, idx, idx] = np.inf
        touch = R < 1.1 * ROBOT_DIAMETER_PX
        u = np.where(touch, -(dX + 1j * dY) / R, 0.0)               # unit vector j -> i
        toward = np.maximum(0.0, (prop[cut][:, None, :] * np.conj(u)).real)
        into = np.maximum(0.0, -(prop[cut][:, :, None] * np.conj(u)).real)
        npin[cut], block[cut] = (toward * u).sum(2), (into * u).sum(2)
    feats = [prop, C['push_x'] + 1j * C['push_y'], C['wall_x'] + 1j * C['wall_y'], C['gear_x'] + 1j * C['gear_y'],
             npin, block]
    v = ((X0[1:] - X0[:-1]) + 1j * (Y0[1:] - Y0[:-1])) / dt
    A = np.stack([f[:-1] for f in feats], axis=-1).reshape(-1, len(feats))
    coef, *_ = np.linalg.lstsq(np.concatenate([A.real, A.imag]), np.concatenate([v.real.ravel(), v.imag.ravel()]),
                               rcond=None)
    pred = np.stack(feats, axis=-1) @ coef
    print('ROBOT VELOCITY MODEL %s: coefficients (prop, push, wall, gear, npin, block) %s, R2 %.3f' % (
        tag, np.round(coef, 3).tolist(),
        1.0 - float(np.sum(np.abs(v - pred[:-1]) ** 2) / np.sum(np.abs(v - v.mean()) ** 2))))
    Vx, Vy = field(pred.real), field(pred.imag)
    fam = spec['families']
    c = lambda a: np.ascontiguousarray(a, dtype=float)       # noqa: E731
    fam['f_div'] = {'div_P': c(fam['f_Px']['Px_x'] + fam['f_Py']['Py_y']),
                    'div_Q': c(fam['f_Qx']['Qx_x'] + fam['f_Qy']['Qy_y']),
                    'div_Push': c(fam['f_Pushx']['Pushx_x'] + fam['f_Pushy']['Pushy_y']),
                    'div_V': c(SF.spatial(F, Vx, 1) + SF.spatial(F, Vy, 2))}
    return axes, fields, spec


def _contact_resolved(xs, ys, prop, xc, yc, r_env, sigma, sweeps=40):
    """Each frame: the velocities closest to ``prop`` (complex, (T, N)) with no touching pair approaching and no
    robot on the wall moving outwards (projected Gauss-Seidel on the contacts)."""
    T, N = xs.shape
    res = np.zeros((T, N), dtype=complex)
    for t in range(T):
        x, y = xs[t], ys[t]
        dX, dY = x[None, :] - x[:, None], y[None, :] - y[:, None]
        R = np.hypot(dX, dY)
        ii, jj = np.nonzero(np.triu(R < 1.1 * sigma, 1))
        nx, ny = (dX[ii, jj] / R[ii, jj]).tolist(), (dY[ii, jj] / R[ii, jj]).tolist()
        r = np.hypot(x - xc, y - yc)
        wi = np.nonzero(r > r_env - 0.25 * sigma)[0]
        wx, wy = ((x[wi] - xc) / r[wi]).tolist(), ((y[wi] - yc) / r[wi]).tolist()
        ii, jj, wi = ii.tolist(), jj.tolist(), wi.tolist()
        vx, vy = prop[t].real.tolist(), prop[t].imag.tolist()
        lam, lw = [0.0] * len(ii), [0.0] * len(wi)
        for _ in range(sweeps):
            for k in range(len(ii)):
                i, j = ii[k], jj[k]
                new = lam[k] - ((vx[j] - vx[i]) * nx[k] + (vy[j] - vy[i]) * ny[k]) / 2
                if new < 0.0:
                    new = 0.0
                d = new - lam[k]
                if d:
                    lam[k] = new
                    vx[j] += d * nx[k]
                    vy[j] += d * ny[k]
                    vx[i] -= d * nx[k]
                    vy[i] -= d * ny[k]
            for k in range(len(wi)):
                i = wi[k]
                new = lw[k] + vx[i] * wx[k] + vy[i] * wy[k]
                if new < 0.0:
                    new = 0.0
                d = new - lw[k]
                if d:
                    lw[k] = new
                    vx[i] -= d * wx[k]
                    vy[i] -= d * wy[k]
        res[t] = np.array(vx) + 1j * np.array(vy)
    return res


def _load_fieldv5(tag, var, factor):
    """One field as the only variable, every field and token built from the tracks with a Gaussian kernel ``factor``
    cap diameters wide (1.0 reproduces the cached fields). Families: the other fields {v, v_x, v_y, lap_v}, rho_bar,
    Q and Push components as fieldv3, and the divergence tokens div_P, div_Q, div_Push, div_V, div_V2.
      div_V   flux of the robot velocity predicted from own propulsion, push, wall, gear, npin, block (fieldv4)
      div_V2  the same model with the contact-resolved propulsion (_contact_resolved) and the mean velocity over
              the windows 1-2, 2-4, 4-9, 9-24 samples back
    Robot models: one coefficient per feature for both components, the whole record."""
    _robot_path()
    import swarm_fields as SF
    from swarm_data import load_swarm, arena_envelope, contact_features, ROBOT_DIAMETER_PX
    path = _swarm_path(tag)
    F0 = SF.build_fields(path)
    t, x, y, phi, ids = load_swarm(path)
    xc, yc, r_env = arena_envelope(x, y)
    s = SF.SS.px_per_cm()
    g, ell = F0['g'], factor * SF.SS.CAP_CM
    norm = np.pi * (SF.SS.CAP_CM / 2) ** 2 / (2 * np.pi * ell ** 2)

    def kernel(px, py):
        kx = np.exp(-(g[None, None, :] - px[:, :, None]) ** 2 / (2 * ell ** 2))
        ky = np.exp(-(g[None, None, :] - py[:, :, None]) ** 2 / (2 * ell ** 2))
        return lambda wt: norm * np.einsum('ti,tix,tiy->txy', wt, kx, ky)

    fine = slice(None, None, SF.STEP)
    rho_bar = kernel((x[fine] - xc) / s, (y[fine] - yc) / s)(np.ones(x[fine].shape)).mean(0)
    sl = slice(None, None, SF.STEP * 5)
    xs, ys, ph = x[sl], y[sl], phi[sl]
    pos = ((xs - xc) + 1j * (ys - yc)) / s
    T, N = pos.shape
    tt = F0['t'][::5]
    dt = tt[1] - tt[0]
    field = kernel(pos.real, pos.imag)
    head = np.exp(1j * (ph + F0['theta']))
    w, prop = F0['w'], F0['v0'] * head
    base = {'rho': field(np.ones((T, N))), 'Px': field(head.real), 'Py': field(head.imag),
            'Wx': field(w * head.real), 'Wy': field(w * head.imag)}
    C = contact_features(xs, ys, w, xc, yc, r_env)
    npin, block = np.zeros((T, N), dtype=complex), np.zeros((T, N), dtype=complex)
    idx = np.arange(N)
    for t0 in range(0, T, 400):
        cut = slice(t0, min(T, t0 + 400))
        d = (xs[cut][:, None, :] - xs[cut][:, :, None]) + 1j * (ys[cut][:, None, :] - ys[cut][:, :, None])
        R = np.abs(d)
        R[:, idx, idx] = np.inf
        u = np.where(R < 1.1 * ROBOT_DIAMETER_PX, -d / R, 0.0)      # unit vector j -> i
        toward = np.maximum(0.0, (prop[cut][:, None, :] * np.conj(u)).real)
        into = np.maximum(0.0, -(prop[cut][:, :, None] * np.conj(u)).real)
        npin[cut], block[cut] = (toward * u).sum(2), (into * u).sum(2)
    six = [prop, C['push_x'] + 1j * C['push_y'], C['wall_x'] + 1j * C['wall_y'], C['gear_x'] + 1j * C['gear_y'],
           npin, block]
    memory = []
    for a, b in ((1, 2), (2, 4), (4, 9), (9, 24)):
        m = np.zeros((T, N), dtype=complex)
        m[b:] = (pos[b - a:T - a] - pos[:T - b]) / ((b - a) * dt)
        memory.append(m)
    v = (pos[1:] - pos[:-1]) / dt

    def model(feats, start):
        A = np.stack([f[start:-1] for f in feats], axis=-1).reshape(-1, len(feats))
        y_ = v[start:]
        coef, *_ = np.linalg.lstsq(np.concatenate([A.real, A.imag]),
                                   np.concatenate([y_.real.ravel(), y_.imag.ravel()]), rcond=None)
        pred = np.stack(feats, axis=-1) @ coef
        r2 = 1.0 - float(np.sum(np.abs(y_ - pred[start:-1]) ** 2) / np.sum(np.abs(y_ - y_.mean()) ** 2))
        return pred, coef, r2

    p1, c1, r1 = model(six, 0)
    resolved = _contact_resolved(xs, ys, prop, xc, yc, r_env, ROBOT_DIAMETER_PX)
    p2, c2, r2 = model([resolved] + six + memory, 24)
    print('ROBOT VELOCITY MODELS %s (kernel %.1f): V R2 %.3f, coefficients %s; V2 R2 %.3f, coefficients %s' % (
        tag, factor, r1, np.round(c1, 3).tolist(), r2, np.round(c2, 3).tolist()))
    G = {'g': g}
    sp = lambda a, axis: SF.spatial(G, a, axis)              # noqa: E731
    c = lambda a: np.ascontiguousarray(a, dtype=float)       # noqa: E731
    shape = base[var].shape

    def family(name, a):
        gx, gy = sp(a, 1), sp(a, 2)
        return {name: c(a), name + '_x': c(gx), name + '_y': c(gy), 'lap_' + name: c(sp(gx, 1) + sp(gy, 2))}

    div = lambda ax, ay: c(sp(ax, 1) + sp(ay, 2))            # noqa: E731
    fams = {'f_' + f: family(f, base[f]) for f in SF.VARS if f != var}
    fams['static'] = {'rho_bar': c(np.broadcast_to(rho_bar, shape))}
    Qx, Qy = field(prop.real), field(prop.imag)
    Pushx, Pushy = field(C['push_x']), field(C['push_y'])
    for name, a in (('Qx', Qx), ('Qy', Qy), ('Pushx', Pushx), ('Pushy', Pushy)):
        fams['f_' + name] = family(name, a)
    fams['f_div'] = {'div_P': div(base['Px'], base['Py']), 'div_Q': div(Qx, Qy), 'div_Push': div(Pushx, Pushy),
                     'div_V': div(field(p1.real), field(p1.imag)), 'div_V2': div(field(p2.real), field(p2.imag))}
    spec = dict(families=fams, prep_kwargs={'scheme': 'forward', 'lag': 1, 'accuracy': 2}, orders=(1, 2, 2),
                factors={'factors_num': [1, 2], 'probas': [0.65, 0.35]})
    return [tt, g, g.copy()], {var: c(base[var])}, spec


# ------------------------------------------------------------------ reference forms
def truth(name):
    kind, tag, rid = parse(name)
    if kind in ('fieldv4', 'fieldv5', 'fieldv6') and rid == 'rho':
        ref = [('transport', ['Px_x', 'Py_y']), ('transport_relax', ['Px_x', 'Py_y', 'rho', 'rho_bar']),
               ('q_transport', ['Qx_x', 'Qy_y']), ('p_div', ['div_P']), ('q_div', ['div_Q']),
               ('q_div_relax', ['div_Q', 'rho', 'rho_bar']), ('v_div', ['div_V']),
               ('v_div_relax', ['div_V', 'rho', 'rho_bar'])]
        if kind != 'fieldv4':
            ref += [('v2_div', ['div_V2']), ('v2_div_relax', ['div_V2', 'rho', 'rho_bar'])]
        return {rid: [(k, 'd%s/dx0' % rid, keys) for k, keys in ref]}
    if kind == 'fieldv3' and rid == 'rho':
        ref = [('transport', ['Px_x', 'Py_y']), ('transport_relax', ['Px_x', 'Py_y', 'rho', 'rho_bar']),
               ('q_transport', ['Qx_x', 'Qy_y']), ('q_transport_relax', ['Qx_x', 'Qy_y', 'rho', 'rho_bar']),
               ('q_transport_push', ['Qx_x', 'Qy_y', 'Pushx_x', 'Pushy_y']),
               ('q_transport_push_relax', ['Qx_x', 'Qy_y', 'Pushx_x', 'Pushy_y', 'rho', 'rho_bar'])]
        return {rid: [(k, 'd%s/dx0' % rid, keys) for k, keys in ref]}
    if kind in ('fieldv', 'fieldv2'):
        # the field pipeline's balanced laws (results/swarm_fields_tau5.txt) and the expected forms with diffusion
        lap = lambda f: ['d^2%s/dx1^2' % f, 'd^2%s/dx2^2' % f]             # noqa: E731
        ref = {'rho': [('transport', ['Px_x', 'Py_y']), ('transport_relax', ['Px_x', 'Py_y', 'rho', 'rho_bar']),
                       ('expected', ['Px_x', 'Py_y'] + lap('rho'))],
               'Px': [('decay_rot', ['Px', 'Wy']), ('expected', ['Px', 'Wy'] + lap('Px'))],
               'Py': [('decay_rot', ['Py', 'Wx']), ('expected', ['Py', 'Wx'] + lap('Py'))],
               'Wx': [('decay_rot', ['Wx', 'Py']), ('chiral', ['Wx', 'Py', 'Wy']), ('expected', ['Wx', 'Py'] + lap('Wx'))],
               'Wy': [('decay_rot', ['Wy', 'Px']), ('chiral', ['Wy', 'Px', 'Wx']), ('expected', ['Wy', 'Px'] + lap('Wy'))]}
        return {rid: [(k, 'd%s/dx0' % rid, keys) for k, keys in ref[rid]]}
    if kind in ('spin', 'spinb'):
        return {'c': [('rot', 'dc/dx0', ['s']), ('osc', 'd^2c/dx0^2', ['c']), ('unit', 'c*dc/dx0', ['ds/dx0*s'])],
                's': [('rot', 'ds/dx0', ['c']), ('osc', 'd^2s/dx0^2', ['s']), ('unit', 'ds/dx0*s', ['c*dc/dx0'])]}
    if kind == 'spinvc':
        return {'c': [('rot', 'dc/dx0', ['s']), ('osc', 'd^2c/dx0^2', ['c'])]}
    if kind == 'spinvs':
        return {'s': [('rot', 'ds/dx0', ['c']), ('osc', 'd^2s/dx0^2', ['s'])]}
    if kind in ('driftx', 'drifty'):
        v = 'x' if kind == 'driftx' else 'y'
        d = 'd%s/dx0' % v
        return {v: [('prop', d, ['c', 's']), ('prop_push', d, ['c', 's', 'push_' + v]),
                    ('prop_push_wall', d, ['c', 's', 'push_' + v, 'wall_' + v]),
                    ('prop_np', d, ['c', 's', 'np_in_' + v]),
                    ('prop_push_wall_vb', d, ['c', 's', 'push_' + v, 'wall_' + v, 'vb_' + v])]}
    return {'rho': [('transport', 'drho/dx0', ['dPx/dx1', 'dPy/dx2']),
                    ('transport_relax', 'drho/dx0', ['dPx/dx1', 'dPy/dx2', 'rho', 'rho_bar']),
                    ('expected', 'drho/dx0', ['dPx/dx1', 'dPy/dx2', 'd^2rho/dx1^2', 'd^2rho/dx2^2'])],
            'Px': [('decay_rot', 'dPx/dx0', ['Px', 'Wy'])],
            'Py': [('decay_rot', 'dPy/dx0', ['Py', 'Wx'])],
            'Wx': [('decay_rot', 'dWx/dx0', ['Wx', 'Py']), ('chiral', 'dWx/dx0', ['Wx', 'Py', 'Wy'])],
            'Wy': [('decay_rot', 'dWy/dx0', ['Wy', 'Px']), ('chiral', 'dWy/dx0', ['Wy', 'Px', 'Wx'])]}
