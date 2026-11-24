"""Solver-free fixed-panel screen; see protocol.txt for scope and adaptations."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import re
import time
from functools import lru_cache
from dataclasses import replace

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path[:0] = [str(REPO), str(REPO/'projects/pic'),
               str(ROOT.parent/'slab_validation_2026-11-16')]
import numpy as np
from scipy.fft import dct, idct
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.optimize import minimize_scalar
from epde_bench import datasets
from epde_bench.reconstruction import parse_equation
from epde.preprocessing import weak_form as WF
from epde.operators.common import survival as SV
from epde.interface.search_config import active_config, set_active_config


def kernels(m, h, q, frequencies, max_order):
    """Kepler product-rule quadrature, with scale-aware PDE precision checks."""
    base = WF.window_kernels((m,), (h,), (q,), (max_order,))[0]
    s = np.arange(-m,m+1)/m
    length = m*h
    phase = np.exp(1j*frequencies[:,None]*length*s[None,:])
    out = [phase*base[0]]
    degree = min(2*m,2*q)
    v = np.polynomial.legendre.legvander(s,degree)
    right = np.linalg.solve(v.T@v,v.T)
    error,relative = 0.,0.
    for order in range(1,max_order+1):
        raw = phase*sum(math.comb(order,j)*(1j*frequencies[:,None])**(order-j)*base[j]
                        for j in range(order+1))
        derivatives = np.column_stack([np.polynomial.legendre.legval(s,
            np.polynomial.legendre.legder(np.eye(degree+1)[j],m=order))/length**order
            for j in range(degree+1)])
        wanted = (-1)**order*(out[0]@derivatives)
        corrected = raw+(wanted-raw@v)@right
        # Preserve the production DC quadrature exactly; cancellation at large
        # derivative scales makes an absolute 1e-10 bound inappropriate.
        corrected[frequencies == 0] = base[order]
        err = float(np.max(np.abs(corrected@v-wanted)))
        scale = max(1.,float(np.max(np.abs(wanted))),float(np.max(np.sum(np.abs(corrected),axis=1))))
        error,relative = max(error,err),max(relative,err/scale)
        out.append(corrected)
    if relative > 1e-10: raise ValueError(f'Weak moment precision: {relative}')
    return out,dict(zero_frequency_kernel_max_error=0.,polynomial_IBP_max_error=error,
                    polynomial_IBP_scaled_error=relative)


def dump(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2, default=convert), encoding='utf-8')
    temp.replace(path)


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def axes_of(problem):
    return [np.unique(g) for g in problem.grids]


def atom(name, **params):
    return (name, tuple(sorted(dict(power=1., **params).items())))


def canon(factors):
    return tuple(sorted((name, tuple(sorted(pars))) for name, pars in factors))


def parse(text):
    terms, constant, target, target_coef = parse_equation(text)
    cols = [canon(f) for c, f in terms if c != 0]
    coefficients = [c/target_coef for c, f in terms if c != 0]
    if constant != 0:
        cols.append(())
        coefficients.append(constant/target_coef)
    return canon(target), cols, coefficients


def panel(problem):
    candidates = []
    for slot, truth in enumerate(problem.truth):
        target, cols, exact = parse(truth)
        # Slot ordering follows the catalog, including NS continuity as slot 2.
        variable = next((v for name, pars in target for v in problem.variables
                         if name == v or re.match(r'd(?:\^\d+)?'+re.escape(v)+r'/dx', name)),
                        problem.variables[min(slot, len(problem.variables)-1)])
        base = dict(slot=slot, variable=variable, target=target)
        seen = set()
        def add(label, features, y=target, valid=False, coefficients=None, source=None):
            features = list(dict.fromkeys(f for f in features if f != y))
            key = (y, tuple(sorted(features)))
            if key in seen: return
            seen.add(key)
            candidates.append(dict(base, id=f'e{slot}:{label}', label=label, target=y,
                                   terms=features, registered=valid,
                                   exact=coefficients, source=source))
        add('reference', cols, valid=True, coefficients=exact, source=truth)
        for i, alternative in enumerate(problem.truth_alternatives):
            y, xs, cs = parse(alternative[slot])
            add(f'equivalent_{i}', xs, y, True, cs, alternative[slot])
        for j in range(len(cols)):
            add(f'drop_{j}', cols[:j]+cols[j+1:])
        t, t2 = (atom('x', dim=0.),), (('x', (('dim', 0.), ('power', 2.))),)
        sn, co = (atom('sin', freq=2., dim=0.),), (atom('cos', freq=2., dim=0.),)
        extras = [(), t, t2, sn, co]
        for v in problem.variables:
            extras.extend([((v, (('power', float(p)),)),) for p in (1,2,3)])
        for j, f in enumerate(extras):
            if f not in cols and f != target: add(f'add_{j}', cols+[f])
        add('time_poly', [(), t, t2])
        add('time_harmonic', [(), sn, co])
        add('field_harmonic', [(), (atom(variable),), sn, co])
        first = (atom(f'd{variable}/dx0'),)
        add('first_order_fit', [(), (atom(variable),),
                              ((variable, (('power', 2.),)),),
                              ((variable, (('power', 3.),)),), t, t2, sn, co], first)
    labels = sorted({x for c in candidates for x in [c['target'], *c['terms']]})
    index = {s:i for i,s in enumerate(labels)}
    for c in candidates:
        c['yi'] = index[c['target']]
        c['xi'] = [index[s] for s in c['terms']]
    return candidates, labels


def symbols(problem, labels):
    dim = len(problem.shape)
    zero = (0,)*dim
    terms, opaque = [], {}
    for label in labels:
        data, coords, ops = [], [0]*dim, []
        for name, params in label:
            params = dict(params)
            power = params.get('power', 1.)
            if power != int(power) or power < 0: raise ValueError((name, params))
            power = int(power)
            derivative = re.fullmatch(r'd(?:\^(\d+))?([A-Za-z_][A-Za-z0-9_]*)/dx(\d+)(?:\^\d+)?', name)
            if name in problem.data:
                data.extend([(name, zero)]*power)
            elif derivative:
                order, var, axis = derivative.groups()
                orders = [0]*dim
                orders[int(axis)] = int(order or 1)
                data.extend([(var, tuple(orders))]*power)
            elif name == 'x':
                coords[int(params['dim'])] += power
            else:
                key = (name, tuple(sorted(params.items())))
                if name in ('sin', 'cos'):
                    values = getattr(np, name)(params['freq']*problem.grids[int(params['dim'])])**power
                else:
                    values = problem.named_arrays[name][0]**power
                opaque[key] = values
                ops.append(key)
        terms.append(WF.WeakTerm(data, coords, ops))
    return terms, opaque


def pooled_gcv_filter(power, eigenvalues):
    """Choose shrinkage 1/(1+lambda*d); stable GCV includes endpoints."""
    d = np.asarray(eigenvalues)
    n = len(d)
    positive = d > 0
    ds = d[positive]
    ps = power[positive]
    def objective(loglam):
        lam = np.exp(loglam)
        a = ds/(1+lam*ds)
        return float(n*np.dot(a*a, ps)/np.sum(a)**2)
    grid = np.unique(np.log(1/ds))
    vals = np.array([objective(x) for x in grid])
    best = int(np.argmin(vals))
    lo, hi = grid[max(0,best-1)], grid[min(len(grid)-1,best+1)]
    choices = [(float(vals[best]), float(np.exp(grid[best]))),
               (float(n*np.dot(ds*ds, ps)/np.sum(ds)**2), 0.),
               (float(n*np.sum(ps)/len(ds)**2), math.inf)]
    if hi > lo:
        result = minimize_scalar(objective, bounds=(lo, hi), method='bounded')
        choices.append((float(result.fun), float(np.exp(result.x))))
    score, lam = min(choices, key=lambda x:x[0])
    filt = (~positive).astype(float) if np.isinf(lam) else 1/(1+lam*d)
    return filt, dict(lam=lam, gcv=score, effective_df=float(np.sum(filt)))


@lru_cache(maxsize=32)
def spline_basis(full_tuple, indices_tuple):
    full = np.array(full_tuple)
    indices = np.array(indices_tuple)
    t = (full-full[0])/(full[-1]-full[0])
    knots = t[indices]
    n = len(knots)
    h = np.diff(knots)
    q = np.zeros((n,n-2))
    j = np.arange(n-2)
    q[j,j], q[j+1,j], q[j+2,j] = 1/h[:-1], -1/h[:-1]-1/h[1:], 1/h[1:]
    r = np.diag((h[:-1]+h[1:])/3)
    r += np.diag(h[1:-1]/6, 1)+np.diag(h[1:-1]/6, -1)
    penalty = q@np.linalg.solve(r,q.T)
    d, u = np.linalg.eigh(penalty)
    d[:2] = 0.
    if np.any(d[2:] <= 0): raise ValueError('Spline penalty lost positive rank')
    interpolation = CubicSpline(knots, np.eye(n), bc_type='natural')(t)
    return d, u, interpolation@u


def reconstruct(raw, axes, indices):
    d, u, interpolated_basis = spline_basis(tuple(axes[0]), tuple(indices))
    values, info = {}, {}
    for var, y in raw.items():
        flat = y.reshape(len(indices), -1)
        transformed = u.T@flat
        filt, entry = pooled_gcv_filter(np.mean(transformed**2, axis=1), d)
        signal = (interpolated_basis@(filt[:,None]*transformed)).reshape((len(axes[0]),)+y.shape[1:])
        details = [entry]
        for axis in range(1, y.ndim):
            z = dct(signal, axis=axis, norm='ortho')
            power = np.mean(np.moveaxis(z,axis,0).reshape(y.shape[axis],-1)**2, axis=1)
            spatial_d = (np.arange(y.shape[axis], dtype=float)/y.shape[axis])**4
            f, entry = pooled_gcv_filter(power, spatial_d)
            shape = [1]*y.ndim
            shape[axis] = len(f)
            signal = idct(z*f.reshape(shape), axis=axis, norm='ortho')
            details.append(entry)
        values[var], info[var] = signal, details
    return values, info


def bandwidth(raw, axes):
    """Pool normalized variables/other-axis replicates; exact projection LOO."""
    output = {'gcv': [], 'loo': []}
    diagnostics = []
    for axis, ax in enumerate(axes):
        n = len(ax)
        y = np.concatenate([np.moveaxis(v/np.std(v), axis, 0).reshape(n,-1)
                            for v in raw.values()], axis=1)
        z = dct(y, norm='ortho', axis=0)
        energy = np.mean(z*z, axis=1)
        rss = np.cumsum(energy[::-1])[::-1][1:]
        gcv = n*rss/(n-np.arange(1,n))**2
        residual = y.copy()
        remaining = np.ones(n)
        loo = []
        position = np.arange(n)+.5
        for k in range(n-1):
            v = np.full(n,1/np.sqrt(n)) if k == 0 else np.sqrt(2/n)*np.cos(np.pi*position*k/n)
            residual -= v[:,None]*z[k]
            remaining -= v*v
            loo.append(float(np.mean(np.mean(residual*residual, axis=1)/(remaining**2))))
        spacing = np.pi/(n*(ax[1]-ax[0]))
        row = {'n': n, 'spacing': spacing}
        for method, score in [('gcv',gcv),('loo',loo)]:
            k = int(np.argmin(score))
            # Include BOTH spatial signs: tensor modes with mixed signs matter.
            f = np.r_[np.arange(k+1), -np.arange(1,k+1)]*spacing
            output[method].append(f)
            row[method] = {'max_mode':k, 'minimum':float(score[k])}
        diagnostics.append(row)
    return output, diagnostics


def common_geometry(axes, terms):
    dim = len(axes)
    q = (3,)+(5,)*(dim-1)
    monos = [t.monomials(psi_cap=q) for t in terms]
    reach = np.max([WF.leftover_orders(m,dim) for m in monos], axis=0)*WF.FD_REACH
    cut = reach.astype(int)
    cut[0] += 2  # all three interpolation domains overlap here
    return q, monos, tuple(cut)


def local_plan(axes, terms, frequencies, width):
    q, monos, cut = common_geometry(axes, terms)
    ms = []
    for axis, (a, f, c, smoothness) in enumerate(zip(axes, frequencies, cut, q)):
        n = len(a)-2*c
        fmax = max(np.abs(f))
        # Half a retained wavelength, or the full available interval at DC.
        radius = (np.pi/fmax)/(a[1]-a[0]) if fmax > 0 else (n-1)/2
        m = max(smoothness, min(int(round(radius)), (n-1)//4))
        if 2*m+1 > n: raise ValueError('Not enough points for weak moments')
        ms.append(m)
    ss = ms.copy()
    while True:
        shape = tuple((len(a)-2*c-2*m-1)//s+1 for a,c,m,s in zip(axes,cut,ms,ss))
        if np.prod(shape)>width+1 and SV.row_floor('dyad', shape, width) is None: break
        if max(ss) == 1: raise ValueError(f'Dyad has insufficient rows: {shape}')
        ss = [max(1,s//2) for s in ss]
    return dict(q=q,cut=cut,ms=tuple(ms),ss=tuple(ss),shape=shape)


class LocalBank:
    def __init__(self, axes, terms, opaque, plan):
        self.axes, self.terms, self.opaque, self.plan = axes, terms, opaque, plan
        self.monos = [t.monomials(psi_cap=plan['q']) for t in terms]
    def matrix(self, values):
        f = WF.Fields(values, self.axes, self.opaque)
        p = self.plan
        a, shape = WF.weak_matrix(self.monos, [t.opaque for t in self.terms], f,
                                 p['cut'], p['ms'], p['ss'], p['q'])
        assert shape == p['shape']
        assert np.isfinite(a).all()
        return a


class SpectralBank:
    def __init__(self, axes, terms, opaque, frequencies):
        self.axes, self.terms, self.opaque = axes, terms, opaque
        q, self.monos, cut = common_geometry(axes, terms)
        self.kernels, self.slices, self.validation = [], [], []
        for axis, (a,c,qs,f) in enumerate(zip(axes, cut, q, frequencies)):
            m = (len(a)-2*c-1)//2
            order = max(key[2][axis] for mono in self.monos for key in mono)
            ks, checks = kernels(m,float(a[1]-a[0]),qs,f,order)
            self.kernels.append(ks)
            self.slices.append(slice(c,c+2*m+1))
            self.validation.append(checks)
        self.slices = tuple(self.slices)
        self.shape = tuple(len(f) for f in frequencies)
        self.metadata = dict(shape=self.shape, rows=int(np.prod(self.shape)), validation=self.validation)
    def matrix(self, values):
        fields = WF.Fields(values,self.axes,self.opaque)
        cache, integrals, columns = {}, {}, []
        for term, monos in zip(self.terms,self.monos):
            col = np.zeros(self.shape,dtype=complex)
            for (data,coords,orders),coef in monos.items():
                key = (data,coords,term.opaque)
                if key not in cache:
                    cache[key] = fields.point(data,coords,term.opaque)[self.slices]
                ikey = (key,orders)
                if ikey not in integrals:
                    arr = cache[key]
                    # Contract the most compressive axis first.
                    order_axes = sorted(range(len(self.axes)), key=lambda a:len(self.kernels[a][0])/arr.shape[a])
                    for axis in order_axes:
                        arr = np.moveaxis(np.tensordot(self.kernels[axis][orders[axis]],arr,axes=(1,axis)),0,axis)
                    integrals[ikey] = arr
                col += coef*integrals[ikey]
            columns.append(col.reshape(-1))
        matrix = np.stack(columns,axis=-1)
        assert np.isfinite(matrix).all()
        return matrix


def signed_fit(gram, observed, xi, yi):
    if not xi: return np.array([]),dict(rank=0,negative=0)
    scale = np.linalg.norm(observed[:,xi],axis=0)
    scale = np.where(scale>0,scale,1.)
    g = gram[np.ix_(xi,xi)]/scale[:,None]/scale[None,:]
    rhs = gram[xi,yi]/scale
    eig,v = np.linalg.eigh((g+g.T)/2)
    threshold = SV._DYAD_RCOND*np.max(np.abs(eig))
    active = np.abs(eig)>threshold
    inv = np.divide(1,eig,out=np.zeros_like(eig),where=active)
    coef = (v@(inv*(v.T@rhs)))/scale
    assert np.isfinite(coef).all()
    return coef,dict(rank=int(active.sum()),negative=int(np.sum(eig < -threshold)))


def fit_training(bank, fields, candidates, counts):
    pilot = bank.matrix(fields[0])
    total = np.zeros_like(pilot)
    gram = np.zeros((pilot.shape[1],pilot.shape[1]))
    result = {}
    for b,values in enumerate(fields[1:],1):
        mat = bank.matrix(values)
        total += mat
        gram += mat.T@mat
        if b in counts:
            mean = total/b
            shifted = 2*pilot-mean
            covariance = (gram-b*mean.T@mean)/(b-1)
            corrected = shifted.T@shifted-(1+1/b)*covariance
            result[b] = [signed_fit(corrected,pilot,c['xi'],c['yi']) for c in candidates]
    return result


def spectral_means(bank, fields, counts):
    pilot = bank.matrix(fields[0])
    total = np.zeros_like(pilot)
    means = {}
    for b,values in enumerate(fields[1:],1):
        total += bank.matrix(values)
        if b in counts: means[b] = total.copy()/b
    return pilot, means


def dominance(d, instability, candidates):
    result = []
    for slot in sorted({c['slot'] for c in candidates}):
        ids = [i for i,c in enumerate(candidates) if c['slot']==slot]
        ref = next(i for i in ids if candidates[i]['label']=='reference')
        dominates = [i for i in ids if i != ref and d[i]<=d[ref] and instability[i]<=instability[ref]
                     and (d[i]<d[ref] or instability[i]<instability[ref])]
        result.append(dict(slot=slot, reference_discrepancy=float(d[ref]), reference_instability=float(instability[ref]),
                           unregistered=[candidates[i]['id'] for i in dominates if not candidates[i]['registered']],
                           registered=[candidates[i]['id'] for i in dominates if candidates[i]['registered']]))
    return result


def evaluate(candidates, fits, train, a, b, clean):
    rows = []
    for c,(coef,diagnostic) in zip(candidates,fits):
        contrast = np.zeros(train.shape[1])
        contrast[c['yi']] = 1.
        contrast[c['xi']] -= coef
        ra,rb,rc = a@contrast,b@contrast,clean@contrast
        scale = float(np.vdot(train[:,c['yi']],train[:,c['yi']]).real)
        if scale <= 0: raise ValueError('Zero training target')
        cross = float(np.vdot(ra,rb).real/scale)
        ordinary = float((np.vdot(ra,ra).real+np.vdot(rb,rb).real)/(2*scale))
        oracle = float(np.vdot(rc,rc).real/scale)
        rows.append(dict(id=c['id'],cross=cross,ordinary=ordinary,oracle=oracle,
                         coefficients=coef,fit=diagnostic,scale=scale))
    return rows


def run_case(name, seed, draws, output):
    started = time.monotonic()
    problem = datasets.load(name)
    axes = axes_of(problem)
    candidates,labels = panel(problem)
    terms,opaque = symbols(problem,labels)
    width = max(len(c['terms']) for c in candidates)
    config = active_config()
    set_active_config(replace(config,search_space=replace(config.search_space,
        equation_terms_max_number=width,system_rps_mode='repair')))
    assert all(SV._reference_width(p)==width for p in range(width+1))
    raw = problem.noisy(5.,seed)
    print(f'{name} seed={seed}: shape={problem.shape}, candidates={len(candidates)}, columns={len(labels)}',flush=True)
    all_frequencies,selection = bandwidth(raw,axes)
    base_plan = local_plan(axes,terms,all_frequencies['loo'],width)
    baseline = LocalBank(axes,terms,opaque,base_plan)
    raw_mat = baseline.matrix(raw)
    instability = []
    for c in candidates:
        score = SV.dyad_scores(raw_mat[:,c['xi']],raw_mat[:,c['yi']],np.ones(len(raw_mat)),
                               base_plan['shape'],fit_intercept=False) if c['xi'] else np.array([0.])
        instability.append(float(np.sum(score)))
    groups,frequencies,group_info = [],[],[]
    for group in range(3):
        indices = np.arange(group,len(axes[0]),3)
        observed = {v:y[indices] for v,y in raw.items()}
        group_axes = [axes[0][indices],*axes[1:]]
        freq,sel = bandwidth(observed,group_axes)
        frequencies.append(freq)
        pilot,fitinfo = reconstruct(observed,axes,indices)
        model = {v:WF.noise_model(y) for v,y in observed.items()}
        rng = np.random.default_rng(np.random.SeedSequence([1171,seed,group]))
        fields = [pilot]
        for draw in range(draws):
            perturbed = {v:pilot[v][indices]+WF.ar_noise(rng,y.shape,model[v][0],model[v][1])
                         for v,y in observed.items()}
            fields.append(reconstruct(perturbed,axes,indices)[0])
        groups.append(fields)
        group_info.append(dict(indices=indices,selection=sel,smoothing=fitinfo,noise=model,
                               observed_sha256={v:digest(y) for v,y in observed.items()}))
        print(f'  group {group}: reconstructed {draws} draws; modes={[(s["gcv"]["max_mode"],s["loo"]["max_mode"]) for s in sel]}',flush=True)
    counts = sorted(set([draws//2,draws]))
    folds = []
    for training in range(3):
        plan = local_plan(axes,terms,frequencies[training]['loo'],width)
        local = LocalBank(axes,terms,opaque,plan)
        fits = fit_training(local,groups[training],candidates,counts)
        evaluation = [g for g in range(3) if g != training]
        fold = dict(training=training,plan=plan,arms={})
        for method in ('gcv','loo'):
            bank = SpectralBank(axes,terms,opaque,frequencies[training][method])
            train_mat = bank.matrix(groups[training][0])
            clean_mat = bank.matrix(problem.data)
            left = spectral_means(bank,groups[evaluation[0]],counts)
            right = spectral_means(bank,groups[evaluation[1]],counts)
            arms = {}
            for count in counts:
                aa,bb = 2*left[0]-left[1][count],2*right[0]-right[1][count]
                rows = evaluate(candidates,fits[count],train_mat,aa,bb,clean_mat)
                uncorrected = evaluate(candidates,fits[count],train_mat,left[0],right[0],clean_mat)
                for r,u in zip(rows,uncorrected): r['uncorrected_cross']=u['cross']
                arms[str(count)] = rows
            # Clean coefficient/control fit uses only local clean weak columns.
            cm = local.matrix(problem.data)
            clean_fits = [signed_fit(cm.T@cm,cm,c['xi'],c['yi']) for c in candidates]
            clean_control = evaluate(candidates,clean_fits,clean_mat,clean_mat,clean_mat,clean_mat)
            fold['arms'][method] = dict(bank=bank.metadata,counts=arms,clean_control=clean_control)
            print(f'  fold {training} {method}: {bank.metadata["rows"]} weak tests, elapsed={time.monotonic()-started:.1f}s',flush=True)
        folds.append(fold)
    aggregates = {}
    for method in ('gcv','loo'):
        aggregates[method] = {}
        for count in counts:
            arms = [f['arms'][method]['counts'][str(count)] for f in folds]
            entry = {}
            for score in ('cross','ordinary','oracle','uncorrected_cross'):
                average = np.mean([[r[score] for r in a] for a in arms],axis=0)
                entry[score] = dict(scores=average,dominance=dominance(average,instability,candidates))
            clean_average = np.mean([[r['oracle'] for r in f['arms'][method]['clean_control']] for f in folds],axis=0)
            entry['clean_fit_energy'] = clean_average
            aggregates[method][str(count)] = entry
    result = dict(system=name,seed=seed,noise_percent=5.,draws=draws,shape=problem.shape,
                  candidates=candidates,labels=labels,base_plan=base_plan,baseline_selection=selection,
                  instability=instability,dyad_reference_width=width,groups=group_info,folds=folds,aggregates=aggregates,
                  elapsed_seconds=time.monotonic()-started,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  data_sha256={v:digest(y) for v,y in problem.data.items()})
    dump(output/f'{name}_seed{seed}_b{draws}.json',result)
    print(f'DONE {name} seed={seed} {result["elapsed_seconds"]:.1f}s',flush=True)
    return result


def checks():
    # Independent fixed-lambda smoothing-spline comparison.
    t = np.linspace(0,1,71)
    idx = np.arange(1,71,3)
    y = np.sin(5*t[idx])+.02*np.random.default_rng(4).normal(size=len(idx))
    d,u,iu = spline_basis(tuple(t),tuple(idx))
    lam = .002
    ours = iu@((u.T@y)/(1+lam*d))
    other = make_smoothing_spline(t[idx],y,lam=lam)(t)
    # Natural cubic interpolation outside knots is cubic extrapolation in both.
    np.testing.assert_allclose(ours,other,rtol=1e-8,atol=1e-9)
    # Compare 2D spectral zero mode with production weak quadrature.
    ax = [np.linspace(0,2,65),np.linspace(-1,1,57)]
    tt,xx = np.meshgrid(*ax,indexing='ij')
    vals = {'u':np.sin(tt)*np.cos(xx)}
    tr = [WF.WeakTerm([('u',(1,0))],(0,0)), WF.WeakTerm([('u',(0,4))],(0,0)),
          WF.WeakTerm([('u',(0,0)),('u',(0,1))],(0,0)), WF.WeakTerm([],(0,0))]
    sb = SpectralBank(ax,tr,{},[np.array([0.,1.,-1.]),np.array([0.,2.,-2.])])
    actual = sb.matrix(vals)
    q,monos,cut = common_geometry(ax,tr)
    ms = tuple((len(a)-2*c-1)//2 for a,c in zip(ax,cut))
    expected,_ = WF.weak_matrix(monos,[t.opaque for t in tr],WF.Fields(vals,ax,{}),cut,ms,(1,1),q)
    np.testing.assert_allclose(actual[0],expected[0],atol=1e-9)
    # Compare exact LOO selection to brute projection refits.
    yy = np.sin(np.arange(15)*.5)+.1*np.random.default_rng(8).normal(size=15)
    _,selection = bandwidth({'u':yy},[np.arange(15,dtype=float)])
    errors = []
    for k in range(14):
        design = np.sqrt(2/15)*np.cos(np.pi*(np.arange(15)[:,None]+.5)*np.arange(k+1)[None,:]/15)
        design[:,0] = 1/np.sqrt(15)
        residual = []
        for i in range(15):
            mask = np.arange(15)!=i
            beta = np.linalg.lstsq(design[mask],yy[mask]/np.std(yy),rcond=None)[0]
            residual.append(yy[i]/np.std(yy)-design[i]@beta)
        errors.append(np.mean(np.square(residual)))
    assert int(np.argmin(errors)) == selection[0]['loo']['max_mode']
    assert len(datasets.names('core')) == 15
    for name in datasets.names('core'):
        p = datasets.load(name)
        cs,ls = panel(p)
        symbols(p,ls)
        assert len({c['slot'] for c in cs}) == len(p.truth)
    return dict(spline_max_error=float(np.max(np.abs(ours-other))),
                zero_mode_max_error=float(np.max(np.abs(actual[0]-expected[0]))),
                loo_matches_brute=True,all15_parsed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems',nargs='+',default=datasets.names('core'))
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--draws',type=int,default=32)
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--checks-only',action='store_true')
    args = parser.parse_args()
    if os.name == 'nt':
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    validation = checks()
    dump(args.output/'numerical_checks.json',validation)
    print(json.dumps(validation),flush=True)
    if args.checks_only: return
    for name in args.systems:
        for seed in args.seeds:
            path = args.output/f'{name}_seed{seed}_b{args.draws}.json'
            if path.exists():
                print(f'SKIP completed {path.name}',flush=True)
                continue
            run_case(name,seed,args.draws,args.output)


if __name__ == '__main__':
    main()
