"""Equivariant translation library from the collision catalog. Products precede the neighbor sum."""
import ast
from itertools import product
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.signal import lfilter

from physics import CAP, CUTOFF, DT, EDGES, LAG, SIGMA, heading, input_at, robot_parameters, rotate, wall_vectors

V = CAP / DT
TAU = LAG                         # history time in frames
H = 3                             # velocity-jump half-window in frames
WARMUP = 5 * LAG
BENCHMARK = ('C002', 'C020', 'C081', 'C101')
PRIORITY = (27, 64, 51, 72, 126, 119, 123, 137, 156, 161, 103, 169)
NATIVE_VECTORS = (*BENCHMARK, 'C035','C051','C119','C123','C126','C137','C156')
GATES = ('Gwall','Gspin')
BOOK = Path(__file__).with_name('candidate_terms.xlsx')


def dot(x, y):
    return (x * y).sum(axis=-1)


def pos(x):
    return np.maximum(x, 0)


def catalog():
    wb = load_workbook(BOOK, read_only=True)
    try:
        rows = [(id_, family, scope, expr, direction) for id_, family, scope, _, expr, direction, *_ in
                list(wb['Terms'].values)[1:186]]
    finally:
        wb.close()
    assert [r[0] for r in rows] == [f'C{i:03}' for i in range(1, 186)]
    # Distinct physical scalar factors multiplying a single normal or tangent.
    # This operation is performed per pair, including at simultaneous contacts.
    for direction, last in [('n', ('same', 'opp', 'N_i', 'w_i', 'h_i')),
                            ('t', ('Sigma', 'z', 's', 'ell_j'))]:
        for factors in product(('q', 'Icap', 'sqrt(delta)'), ('c', 'b', 'D_j'), last):
            rows.append((f'P{len(rows)-184:03}', 'Three-factor interactions', 'Pair',
                         '*'.join(factors), direction))
    for vector in NATIVE_VECTORS:
        for gates in [('Gwall',),('Gspin',),GATES]:
            label = '*'.join((vector,*gates))
            rows.append((label,'Robot modulation of contact','Modulated vector',label,'As written'))
    return rows


TERMS = catalog()
IDS = [r[0] for r in TERMS]
FUN = dict(abs=np.abs, sqrt=np.sqrt, tanh=np.tanh, exp=np.exp, log=np.log, sign=np.sign,
           min=np.minimum, max=np.maximum, dot=dot, pos=pos, norm=np.linalg.norm)


def expression(weight):
    # Only arithmetic and the explicitly registered NumPy functions are valid workbook syntax.
    tree = ast.parse(weight.replace('^', '**').replace('F_i@n', 'Fin').replace('F_j@n', 'Fjn'), mode='eval')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
    for node in ast.walk(tree):
        if not isinstance(node, allowed) or isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name) or node.func.id not in FUN):
            raise ValueError(weight)
    return compile(tree, str(BOOK), 'eval')


EXPR = [expression(r[3]) for r in TERMS]


def rolling_sum(x, width):
    cs = np.r_[0., np.cumsum(x, dtype=float)]
    return cs[1:] - cs[np.maximum(np.arange(len(x)) + 1 - width, 0)]


def decayed(x):
    """Past jumps become observable at onset+h and are retained through three tau."""
    out = lfilter([1.], [1., -np.exp(-1/TAU)], x, axis=0)
    horizon = 3*TAU + 1
    out[horizon:] -= np.exp(-horizon/TAU) * out[:-horizon].copy()
    return out


def history(track, rows, observed=None):
    p, ph = track.pos, track.phi
    frames, robots = p.shape[:2]
    left, right = np.triu_indices(robots, 1)
    pairs = len(left)
    index = np.zeros((robots, robots), int)
    index[left, right] = index[right, left] = np.arange(pairs)
    v, w = np.zeros_like(p), np.zeros_like(ph)
    v[LAG:] = (p[LAG:] - p[:-LAG]) / CAP
    w[LAG:] = (ph[LAG:] - ph[:-LAG]) / 2
    keys = ['active_age','fresh','c_entry','q_peak','unload','H_cap','H_c','repeat_count',
            'X_slip','H_spin']
    hist = {k: np.zeros((len(rows), pairs), np.float32) for k in keys}
    release = np.zeros((len(rows), robots, 2, 2))
    ready = np.ones((len(rows), robots), bool)
    counts = np.zeros((frames, robots), np.int16)
    events = []
    times = np.arange(frames)
    for k, (i, j) in enumerate(zip(left, right)):
        d = p[:, i] - p[:, j]
        r = np.linalg.norm(d, axis=1)
        measured = r if observed is None else np.linalg.norm(observed.pos[:,i]-observed.pos[:,j],axis=1)
        core = measured < CAP
        cleared = measured >= 1.03 * CAP
        observable_clear = cleared & (times >= LAG)
        first_clear = np.flatnonzero(observable_clear)
        first = first_clear[0] if len(first_clear) else frames
        if first > LAG:
            ready[np.ix_(rows < first + 3*TAU + H, [i,j])] = False
        if not core.any():
            events.append(np.array([], int))
            continue
        n = d / np.maximum(r[:, None], 1e-12)
        c = pos(-dot(v[:, i] - v[:, j], n))
        spin = w[:, i] + w[:, j]
        z = dot(v[:, i] - v[:, j], rotate(n)) - spin
        q = pos((CUTOFF-r)/SIGMA)
        last_clear = np.maximum.accumulate(np.where(observable_clear, times, -1))
        last_core = np.maximum.accumulate(np.where(core, times, -1))
        active = (last_clear >= 0) & (last_core > last_clear)
        starts = np.flatnonzero(np.diff(active.astype(int), prepend=0) == 1)
        ends = np.flatnonzero(np.diff(active.astype(int), append=0) == -1) + 1
        events.append(starts)
        counts[starts, i] += 1
        counts[starts, j] += 1
        hist['H_cap'][:, k] = rolling_sum(core, TAU)[rows]/TAU
        hist['H_c'][:, k] = rolling_sum(core*c, TAU)[rows]/TAU
        hist['H_spin'][:, k] = rolling_sum(core*spin, TAU)[rows]/TAU
        onsets = np.zeros(frames)
        onsets[starts] = 1
        hist['repeat_count'][:, k] = rolling_sum(onsets, TAU)[rows]
        for idx, (a, b) in enumerate(zip(starts, ends)):
            entry = pos(-dot(v[a-1,i] - v[a-1,j], n[a]))
            # Left rectangles integrate only intervals already elapsed.
            slip = np.r_[0., np.cumsum(z[a:b])]/LAG
            sl = slice(*np.searchsorted(rows, [a,b]))
            ages = rows[sl] - a
            hist['active_age'][sl,k] = ages/LAG
            hist['fresh'][sl,k] = np.exp(-ages/TAU)
            hist['c_entry'][sl,k] = entry
            peak = np.maximum.accumulate(q[a:b])[ages]
            hist['q_peak'][sl,k] = peak
            hist['unload'][sl,k] = peak-q[rows[sl]]
            hist['X_slip'][sl,k] = slip[ages]
            if b < frames:
                end = min(b+3*TAU+1, starts[idx+1] if idx+1 < len(starts) else frames)
                sl = slice(*np.searchsorted(rows, [b,end]))
                vec = np.stack([n[b]*entry, rotate(n[b])*slip[-1]], axis=-1)
                carry = np.exp(-(rows[sl]-b)/TAU)[:,None,None] * vec
                release[sl,i] += carry
                release[sl,j] -= carry
    # A simultaneous onset contributes once to each robot. Removing a partner
    # removes the jump only if this was its sole partner at that onset.
    jump = np.zeros((frames,robots,3))
    dv = (p[2*H:] - 2*p[H:-H] + p[:-2*H]) * LAG/(H*CAP)
    jump[2*H:,:,:2] = dv
    jump[2*H:,:,2] = np.linalg.norm(dv, axis=-1)
    jumps = np.zeros_like(jump)
    jumps[H:] = jump[H:] * (counts[:-H] > 0)[...,None]
    total = decayed(jumps)[rows]
    sole = np.zeros((len(rows), pairs, 2, 3), np.float32)
    for k, (i, j, starts) in enumerate(zip(left, right, events)):
        starts = starts[starts + H < frames]
        if len(starts):
            for side, robot in enumerate((i,j)):
                a = starts[counts[starts,robot] == 1] + H
                impulses = np.zeros((frames,3))
                impulses[a] = jump[a,robot]
                sole[:,k,side] = decayed(impulses)[rows]
    return hist, index, total, sole, release, ready


def columns(track, rows, slots, memory, params, observed=None):
    lookup, common, arenas = params
    hist, index, total, sole, release, ready = memory
    p, e, vel, omega = input_at(track, rows)
    robots = len(track.ids)
    arena = arenas[int(track.tag[-2:])]
    calib = robot_parameters(track, lookup)
    prop = (calib[:,0,None]*e + calib[:,1,None]*rotate(e))/V
    base = V*prop + common[0]*vel + common[1]*wall_vectors(p,arena)
    target = (track.pos[rows+LAG]-p)/DT
    v = vel/V
    w = omega*(CAP/2/V)
    offset = arena[:2]-p
    radius = np.linalg.norm(offset, axis=-1)
    m = offset / np.maximum(radius[...,None],1e-12)
    wall = pos(1-pos(arena[2]-radius)/CAP) * (radius > 0)
    d = p[:,:,None,:]-p[:,None,:,:]
    distances = np.linalg.norm(d,axis=-1)
    f,i,j = np.nonzero((distances>0) & (distances<CUTOFF))
    r = distances[f,i,j]
    n = d[f,i,j] / r[:,None]
    t = rotate(n)
    q = (CUTOFF-r)/SIGMA
    rho = r/CAP
    measured = r if observed is None else np.linalg.norm(observed.pos[rows[f],i]-observed.pos[rows[f],j],axis=1)
    delta = pos(1-measured/CAP)
    rel = v[f,i]-v[f,j]
    vn, s = dot(rel,n), dot(rel,t)
    c,b = pos(-vn),pos(vn)
    wi,wj = w[f,i],w[f,j]
    spin = wi+wj
    z = s-spin
    flat = f*robots+i
    size = len(rows)*robots

    def aggregate(weight):
        return np.bincount(flat, weights=weight, minlength=size).reshape(len(rows),robots)

    def aggregate_vec(vec):
        return np.stack((aggregate(vec[:,0]),aggregate(vec[:,1])), axis=-1)

    crowd = aggregate(np.ones_like(q))
    density = aggregate(q)
    approach = q[:,None]*c[:,None]*n
    load = aggregate_vec(approach)
    # The 2D symmetric tensor has three independent components.
    xx,xy,yy = (aggregate(q*n[:,a]*n[:,b_]) for a,b_ in [(0,0),(0,1),(1,1)])
    def fabric(robot):
        full = np.stack((xx[f,robot]*n[:,0]+xy[f,robot]*n[:,1],
                         xy[f,robot]*n[:,0]+yy[f,robot]*n[:,1]),axis=-1)
        return full-q[:,None]*n

    pair = index[i,j]
    si = (i>j).astype(int)
    ki = total[slots[f],i]-sole[slots[f],pair,si]
    kj = total[slots[f],j]-sole[slots[f],pair,1-si]
    env = dict(q=q, rho=rho, rhoc=CUTOFF/CAP, delta=delta, Icap=(measured<CAP), c=c,b=b,s=s,
               wi=wi,wj=wj,Sigma=spin,OmegaDiff=wi-wj,same=(wi*wj>0),opp=(wi*wj<0),z=z,
               n=n,t=t,m=m[f,i], w_i=wall[f,i],w_j=wall[f,j],h_i=-dot(e[f,i],n),h_j=dot(e[f,j],n),
               ell_i=dot(e[f,i],t),ell_j=dot(e[f,j],t),align=dot(e[f,i],e[f,j]),
               cross_e=dot(rotate(e[f,i]),e[f,j]),p_j=prop[f,j],D_i=pos(-dot(prop[f,i],n)),
               D_j=pos(dot(prop[f,j],n)),T_i=dot(prop[f,i],t),T_j=dot(prop[f,j],t),
               K_i=ki[:,:2],K_j=kj[:,:2],L_i=ki[:,2],L_j=kj[:,2],
               N_i=crowd[f,i]-1,N_j=crowd[f,j]-1,rho_i=density[f,i]-q,rho_j=density[f,j]-q,
               P_i=load[f,i]-approach,P_j=load[f,j]+approach,Fin=fabric(i),Fjn=fabric(j),
               q0=.1,c0=.1,s0=.1,z0=.1,x0=1.,eps=.01)
    env.update({k:u[slots[f],pair] for k,u in hist.items()})
    env.update({f'B{k}':(r>=a*SIGMA)&(r<b_*SIGMA) for k,(a,b_) in enumerate(zip(EDGES[:-1],EDGES[1:]))})
    # Use explicit vector broadcasting for the few vector-valued weights.
    overrides = {'C122':env['unload'], 'C135':q[:,None]*env['K_j']}
    rc = dict(w_i=wall, c_w=pos(-dot(v,m)), z_w=dot(v,rotate(m))-w, wi=w,z0=.1)
    robot_controls = [v, v-(track.pos[rows-LAG]-track.pos[rows-2*LAG])/CAP, prop,
                      w[...,None]*rotate(v), abs(w)[...,None]*v, np.linalg.norm(v,axis=-1)[...,None]*v,prop-v]
    x = np.empty((*v.shape,len(TERMS)),np.float32)
    globals_ = {'__builtins__':{}, **FUN}
    gates = dict(Gwall=wall,Gspin=abs(w))
    for k,(id_,_,scope,_,direction) in enumerate(TERMS):
        if scope == 'Modulated vector':
            vector,*modifiers = id_.split('*')
            value = x[...,IDS.index(vector)]
            for modifier in modifiers:
                value = value*gates[modifier][...,None]
            x[...,k] = value
            continue
        if scope == 'Recent pair':
            vec = release[slots,...,k-130]
        elif scope == 'Robot':
            if k >= 178:
                vec = robot_controls[k-178]
            else:
                weight = eval(EXPR[k],globals_,rc)
                vec = weight[...,None] * (m if direction == 'm' else rotate(m))
        else:
            weight = overrides[id_] if id_ in overrides else eval(EXPR[k],globals_,env)
            vec = weight if direction == 'As written' else np.asarray(weight)[...,None]*(n if direction=='n' else t)
            vec = aggregate_vec(vec)
        x[...,k] = vec*V
    return x, base, target, ready[slots], (f,i,j,n,t), gates
