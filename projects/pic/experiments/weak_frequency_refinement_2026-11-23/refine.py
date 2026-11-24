"""Profile a training peak conditional on a sparse weak differential relation."""
from collections import Counter
from itertools import combinations
import math

import numpy as np
from scipy.optimize import minimize_scalar
from epde.preprocessing import weak_form as wf


def orders(dims, axis, order):
    return tuple(order if j == axis else 0 for j in range(dims))


def weak_columns(fields, plan, terms):
    matrix, shape = wf.weak_matrix([t.monomials(psi_cap=plan.q) for t in terms],
                                  [t.opaque for t in terms],fields,plan.cut,plan.ms,plan.ss,plan.q)
    assert shape == tuple(plan.shape) and np.isfinite(matrix).all()
    return matrix


def residualize(columns, basis):
    return columns - basis @ (basis.T @ columns)


def gain_fractions(columns, basis, residual):
    projected = residualize(columns, basis)
    norm2 = np.einsum('ij,ij->j', projected, projected)
    original = np.einsum('ij,ij->j', columns, columns)
    usable = norm2 > np.maximum(1e-20 * original, np.finfo(float).tiny)
    scores = np.zeros(len(norm2))
    residual_norm2 = residual @ residual
    if residual_norm2 > 0:
        scores[usable] = (projected[:,usable].T @ residual)**2 / (norm2[usable]*residual_norm2)
    return scores, projected


class FrequencyProfile:
    def __init__(self, train, axes, plan, settings):
        self.axes = axes
        self.plan = plan
        self.fields = wf.Fields(train, axes)
        self.zero = (0,)*len(axes)
        self.names = sorted(train)
        self.work = Counter()
        self.backgrounds = []
        self.modulators = [('constant', [])]
        library = [('1', wf.WeakTerm([], self.zero))]
        first = orders(len(axes),0,1)
        for v in self.names:
            for power in range(1,settings['data_fun_pow']+1):
                library.append((f'{v}^{power}',wf.WeakTerm([(v,self.zero)]*power,self.zero)))
            self.modulators.extend([(v,[(v,self.zero)]),(f'D0({v})',[(v,first)])])
        for v,w in combinations(self.names,2):
            library.append((f'{v}*{w}',wf.WeakTerm([(v,self.zero),(w,self.zero)],self.zero)))
        for axis in range(len(axes)):
            for power in (1,2):
                library.append((f'x{axis}^{power}',wf.WeakTerm([],orders(len(axes),axis,power))))
            if axis:
                for v in self.names:
                    for k in range(1,settings['max_deriv_order'][axis]+1):
                        library.append((f'D{axis}^{k}({v})',wf.WeakTerm([(v,orders(len(axes),axis,k))],self.zero)))
        common = len(library)
        library.extend((f'D0({v})',wf.WeakTerm([(v,first)],self.zero)) for v in self.names)
        labels, terms = zip(*library)
        matrix = weak_columns(self.fields,plan,terms)
        self.work['background_columns'] = len(terms)
        for k in range(1,settings['max_deriv_order'][0]+1):
            for i,v in enumerate(self.names):
                target = (matrix[:,common+i] if k==1 else weak_columns(self.fields,plan,
                          [wf.WeakTerm([(v,orders(len(axes),0,k))],self.zero)])[:,0])
                count = common if k==1 else len(library)
                selected, basis, residual = self.background(matrix[:,:count], target)
                if np.linalg.norm(residual)>1e-10*np.linalg.norm(target):
                    self.backgrounds.append((v,k,[labels[j] for j in selected],basis,residual))

    def background(self, matrix, target):
        selected = []
        basis = np.empty((len(target),0))
        residual = target
        for _ in range(3):
            scores, _ = gain_fractions(matrix,basis,residual)
            scores[selected] = 0.
            self.work['background_projections'] += len(scores)-len(selected)
            j = int(np.argmax(scores))
            if scores[j]<=1e-12:
                break
            selected.append(j)
            cols = matrix[:,selected]
            basis = np.linalg.qr(cols/np.linalg.norm(cols,axis=0), mode='reduced')[0]
            residual = residualize(target,basis)
            self.work['background_fits'] += 1
        return selected,basis,residual

    def harmonics(self, axis, omega, candidates):
        coordinate = self.fields.coords[axis]
        for trig in {trig for trig,mod in candidates}:
            self.fields.opaque[trig] = np.broadcast_to(getattr(np,trig)(omega*coordinate),self.fields.shape)
        terms = [wf.WeakTerm(self.modulators[mod][1],self.zero,[trig]) for trig,mod in candidates]
        self.work['harmonic_columns'] += len(terms)
        return weak_columns(self.fields,self.plan,terms)

    def calibrate(self, axis, peaks, pos):
        omega = float(peaks[pos])
        grid = self.axes[axis]
        delta = float(np.median(np.diff(grid)))
        width = 2*np.pi/(len(grid)*delta)
        low = max(omega-width/2, np.finfo(float).eps,
                  np.nextafter((peaks[pos-1]+omega)/2,omega) if pos else 0.)
        high = min(omega+width/2,np.pi/delta,
                   np.nextafter((peaks[pos+1]+omega)/2,omega) if pos+1<len(peaks) else np.inf)
        result = dict(axis=axis, initial=omega, installed=omega, lower=float(low), upper=float(high))
        if not low < omega < high or not self.backgrounds:
            return dict(result,decision='no bounded profile')
        candidates = [(trig,mod) for trig in ('sin','cos') for mod in range(len(self.modulators))]
        columns = self.harmonics(axis,omega,candidates)
        best = None
        best_gain = 0.
        for v,k,labels,basis,residual in self.backgrounds:
            scores,_ = gain_fractions(columns,basis,residual)
            self.work['harmonic_projections'] += len(scores)
            for j,(_,mod) in enumerate(candidates):
                if k==1 and self.modulators[mod][0]==f'D0({v})':
                    scores[j]=0.
            j = int(np.argmax(scores))
            if scores[j] > best_gain:
                best_gain = float(scores[j])
                best = (v,k,labels,basis,residual,j)
        if best is None:
            return dict(result,decision='no residual gain')
        v,k,labels,basis,residual,j = best
        trig,mod = candidates[j]
        result.update(target=f'D0^{k}({v})',background=labels,phase=trig,
                      modulator=self.modulators[mod][0],initial_fraction_explained=best_gain)
        # Each retained row's integration window must lie wholly inside its half.
        centers = self.plan.cut[axis]+self.plan.ms[axis]+self.plan.ss[axis]*np.arange(self.plan.shape[axis])
        cut = len(grid)//2
        halves = []
        for take in (centers+self.plan.ms[axis]<cut, centers-self.plan.ms[axis]>=cut):
            shape = [1]*len(self.axes)
            shape[axis] = len(take)
            halves.append(np.broadcast_to(take.reshape(shape),self.plan.shape).ravel())
        result['block_rows'] = [int(x.sum()) for x in halves]
        if min(result['block_rows']) <= len(labels)+1:
            return dict(result,decision='too few disjoint block rows')

        cache = {}
        def evaluate(freq):
            freq=float(freq)
            if freq in cache:
                self.work['profile_cache_hits'] += 1
                return cache[freq]
            x = columns[:,j] if freq==omega else self.harmonics(axis,freq,[(trig,mod)])[:,0]
            z = residualize(x,basis)
            norm = z@z
            # Profiling out linear parameters is equivalent to WLS on [background, x].
            if norm<=max(1e-20*(x@x),np.finfo(float).tiny):
                r = residual
            else:
                r = residual - z*((z@residual)/norm)
            self.work['profile_fits'] += 1
            cache[freq] = (float(r@r),[float(r[h]@r[h]) for h in halves])
            return cache[freq]
        evaluate(omega)
        evaluate(low)
        evaluate(high)
        optimum = minimize_scalar(lambda f:evaluate(f)[0],method='bounded',bounds=(low,high),
                                  options=dict(xatol=width*1e-4,maxiter=24))
        evaluate(optimum.x)
        chosen = min(cache,key=lambda f:(cache[f][0],abs(f-omega)))
        old, old_blocks = cache[omega]
        new, new_blocks = cache[chosen]
        gain = len(residual)*math.log(max(old,np.finfo(float).tiny)/max(new,np.finfo(float).tiny))
        result.update(proposed=chosen,profile_fits=len(cache),initial_sse=old,proposed_sse=new,
                      initial_blocks=old_blocks,proposed_blocks=new_blocks,bic_gain=gain-math.log(len(residual)))
        if gain<=math.log(len(residual)):
            return dict(result,decision='BIC penalty')
        if not all(b<a for a,b in zip(old_blocks,new_blocks)):
            return dict(result,decision='training blocks disagree')
        return dict(result,decision='installed',installed=chosen)
