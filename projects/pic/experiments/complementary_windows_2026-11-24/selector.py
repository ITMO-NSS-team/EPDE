"""Choose a complementary weak geometry using training fields and the allowed derivative orders.

Call select(fields, plan, orders) with a WF.Fields on regular axes and the native
WindowPlan or a mapping of q, ms, ss, cut and shape. Each returned Geometry can
measure an arbitrary list of WeakTerms. A None second geometry retains all
native measurements. Both noise perturbations are deterministic per call.
"""
import numpy as np
from epde.preprocessing import weak_form as WF


def probes(names, orders):
    zero = (0,) * len(orders)
    terms = []
    for name in sorted(names):
        terms.append(WF.WeakTerm([(name, zero)], zero))
        for axis, limit in enumerate(orders):
            for order in range(1, limit+1):
                derivative = tuple(order if j == axis else 0 for j in range(len(orders)))
                terms.append(WF.WeakTerm([(name, derivative)], zero))
    for axis in range(len(orders)):
        terms.append(WF.WeakTerm([], tuple(int(j == axis) for j in range(len(orders)))))
    return terms


class Geometry:
    """Map native lattice cells to valid integration centers, preserving the physical slab layout."""
    def __init__(self, plan, field_shape, ms, shift):
        self.ms = tuple(ms)
        self.shift = tuple(shift)
        self.output_shape = tuple(plan['shape'])
        self.q, self.cut = tuple(plan['q']), tuple(plan['cut'])
        pos, source_idx, strides = [], [], []
        for length, count, cut, m, old, stride, off in zip(
                field_shape, self.output_shape, plan['cut'], ms, plan['ms'], plan['ss'], shift):
            base = old + off + np.arange(count) * stride
            keep = np.flatnonzero((base >= m) & (base < length-2*cut-m))
            native_stride = int(np.gcd(stride, abs(old+off-m)))
            pos.append(keep)
            source_idx.append((base[keep]-m)//native_stride)
            strides.append(native_stride)
        self.strides = tuple(strides)
        self.source = np.ix_(*source_idx)
        self.positions = np.ix_(*pos)
        mask = np.zeros(self.output_shape, bool)
        mask[self.positions] = True
        parity = np.indices(self.output_shape).sum(axis=0) % 2
        self.replace = (mask & (parity == 1)).ravel()
        self.valid = all(len(p) for p in pos)
        self.native = self.ms == tuple(plan['ms']) and not any(shift)

    def matrix(self, fields, terms, primary=None):
        monos = [t.monomials(psi_cap=self.q) for t in terms]
        values, shape = WF.weak_matrix(monos, [t.opaque for t in terms], fields,
                                       self.cut, self.ms, self.strides, self.q)
        block = values.reshape(*shape, len(terms))[self.source]
        if self.native:
            return block.reshape(-1, len(terms))
        aligned = np.empty((*self.output_shape, len(terms)))
        aligned[self.positions] = block
        result = primary.copy()
        result[self.replace] = aligned.reshape(result.shape)[self.replace]
        return result


def neighbors(plan, shape):
    for d, (m, s) in enumerate(zip(plan['ms'], plan['ss'])):
        for new, off in ((m//2, 0), (2*m, 0), (m, s//2)):
            if new < 2 or (new == m and not off):
                continue
            ms = list(plan['ms'])
            ms[d] = new
            shift = [0]*len(ms)
            shift[d] = off
            g = Geometry(plan, shape, ms, shift)
            if g.valid and g.replace.any():
                yield g


def separation(a, noise):
    # Generalized signal directions from the native selector. Count only those
    # with measured energy above twice the perturbation energy (signal > noise).
    gram = a.T @ a
    active = np.diag(gram) > 0
    if not np.any(active):
        return 0, 0.
    ga = gram[np.ix_(active, active)]
    na = (noise.T @ noise)[np.ix_(active, active)]
    directions = WF.signal_directions(ga, na)
    rank = directions.shape[1]
    scale = 1/np.sqrt(np.diag(ga))
    energy = np.sum(directions * ((scale[:, None] * na * scale[None, :]) @ directions))
    return rank, -float(energy)


def select(fields, plan, orders):
    if isinstance(plan, WF.WindowPlan):
        plan = {k: getattr(plan, k) for k in ('ms', 'ss', 'shape', 'q', 'cut')}
    if len(orders) != fields.D or any(o < 0 or int(o) != o for o in orders):
        raise ValueError('Supply a nonnegative maximum derivative order for each coordinate axis.')
    orders = tuple(int(o) for o in orders)
    terms = probes(fields.values, orders)
    primary = Geometry(plan, fields.shape, plan['ms'], (0,)*fields.D)
    models = {var: WF.noise_model(v)[:2] for var, v in fields.values.items()}
    def perturb(seed):
        rng = np.random.default_rng(seed)
        return WF.Fields({var: WF.ar_noise(rng, fields.shape, *models[var]) for var in sorted(models)}, fields.axes)
    def noise_columns(geometry, random_field, base=None):
        # Every stochastic probe is linear. Coordinates have exactly zero noise.
        mat = geometry.matrix(random_field, terms, base)
        mat[:, -fields.D:] = 0.
        return mat
    a = primary.matrix(fields, terms)
    first, check = perturb(0), perturb(1)
    n = noise_columns(primary, first)
    base = separation(a, n)
    best, selected, chosen_a = base, None, None
    for geom in neighbors(plan, fields.shape):
        mixed = geom.matrix(fields, terms, a)
        rank = separation(mixed, noise_columns(geom, first, n))
        if rank[0] > base[0] and rank > best:
            best, selected, chosen_a = rank, geom, mixed
    audit = dict(original=base[0], proposed=best[0], probe_count=len(terms))
    if selected is not None:
        second = noise_columns(primary, check)
        old = separation(a, second)[0]
        new = separation(chosen_a, noise_columns(selected, check, second))[0]
        audit.update(check_original=old, check_proposed=new)
        if new <= old:
            selected = None
    return primary, selected, audit
