"""Build and cache the two native weak-window matrices for terms proposed by the search."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'full_resolution_fit_2026-11-18'))
from axis_test import block_specs, choose_axis
from study import fit, np, wls
from epde.preprocessing import weak_form as WF
from epde.structure.weak_rows import weak_term
from epde.operators.common.objectives import LOSS_NAN_VAL


class BlockedScore:
    def __init__(self, weak, slot):
        self.weak = weak
        self.slot = slot
        full = weak.fields[0]
        native = weak.plans[0,slot]
        plan = {k: getattr(native,k) for k in ('q','cut','ms','ss','shape')}
        axis = choose_axis(full.shape,plan)
        if axis is None:
            raise ValueError(f'No split can preserve the native window: {plan}')
        self.geometry = dict(axis=axis, native=plan, halves=[])
        self.blocks = []
        for cuts,p in block_specs(full.shape,plan,axis):
            axes = [a[s] for a,s in zip(full.axes,cuts)]
            fields = WF.Fields({v:y[cuts] for v,y in full.values.items()},axes)
            self.blocks.append((cuts,p,fields))
            self.geometry['halves'].append(p['shape'])
        self.ids = {}
        self.columns = {0: [np.ones(int(np.prod(p['shape']))) for _,p,_ in self.blocks]}
        self.cached = {}
        self.invalid = 0

    def column(self, term):
        key = term.factors_labels
        if key in self.ids:
            return self.ids[key]
        t, factors = weak_term(term,self.weak.fields[0].D,frozenset(self.weak.tokens.get(0,{})))
        opaque = {k:self.weak._opaque_values(f,0) for k,f in factors.items()}
        columns = []
        for cuts,p,field in self.blocks:
            field.opaque.update({k:y[cuts] for k,y in opaque.items()})
            m, shape = WF.weak_matrix([t.monomials(psi_cap=p['q'])], [t.opaque], field,
                                      p['cut'],p['ms'],p['ss'],p['q'])
            assert shape == p['shape']
            columns.append(m[:,0])
        idx = len(self.columns)
        self.ids[key] = idx
        self.columns[idx] = columns
        return idx

    def layout(self, eq):
        target = eq.target_idx
        weights = iter(eq.weights_final[:-1])
        selected = [(i,t) for i,t in enumerate(eq.structure) if i==target or next(weights)!=0.]
        idx = [self.column(t) for _,t in selected]
        pivot = self.column(eq.structure[target])
        if eq.weights_final[-1] != 0:
            idx.append(0)
        return tuple(sorted(idx)), pivot

    def matrices(self, ids):
        return [np.column_stack([self.columns[i][h] for i in ids]) for h in (0,1)]

    def evaluate(self,eq,arm):
        ids, pivot = self.layout(eq)
        key = (arm, ids, pivot if arm=='wls' else None)
        if key not in self.cached:
            matrices = self.matrices(ids)
            values = []
            if len(set(ids))==len(ids) and len(ids)>1:
                for g in (0,1):
                    m = matrices[g]
                    v, status = wls(m,ids.index(pivot)) if arm=='wls' else fit(m)
                    if status != 'ok':
                        break
                    scale2 = np.dot(np.mean(m*m,axis=0),v*v)
                    value = np.mean((matrices[1-g] @ v)**2)/scale2
                    if not np.isfinite(value):
                        break
                    values.append(value)
            self.invalid += int(len(values)!=2)
            self.cached[key] = float(np.mean(values)) if len(values)==2 else float(LOSS_NAN_VAL)
        return self.cached[key]
