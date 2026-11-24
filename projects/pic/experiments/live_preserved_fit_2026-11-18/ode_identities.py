"""Exact check of all oscillator front supports, including constants and phase identities."""
from functools import lru_cache
import json
import re
import sympy as sp
from search import ROOT, ARMS, SEEDS, np
from recheck import labels_of


z,t = sp.symbols('z t',positive=True)


@lru_cache(None)
def expansion(label, snap):
    sin = (z-z**-1)/(2*sp.I)
    cos = (z+z**-1)/2
    u = sin + sp.Rational(13,10)*cos
    ut = cos - sp.Rational(13,10)*sin
    field = {'u':u,'du/dx0':ut,'d^2u/dx0^2':-u,'x':t}
    val = sp.Integer(1)
    for name, params in label:
        p = dict(params)
        if name in ('sin','cos'):
            freq = 1 if snap and abs(p['freq']-1)<.01 else sp.Rational(str(p['freq']))
            a,b = z**freq,z**-freq
            term = (a-b)/(2*sp.I) if name=='sin' else (a+b)/2
        else:
            term = field[name]
        val *= term**int(p['power'])
    coeffs = {}
    for item in sp.Add.make_args(val.expand()):
        powers = item.as_powers_dict()
        m,n = powers.get(t,0),powers.get(z,0)
        key = m,n
        coeffs[key] = coeffs.get(key,0) + item/(t**m*z**n)
    return coeffs


@lru_cache(None)
def identity(labels,snap):
    cols = [expansion(label,snap) for label in labels]
    powers = sorted(set().union(*cols))
    matrix = sp.Matrix([[part(c.get(k,0)) for c in cols] for k in powers for part in (sp.re,sp.im)])
    null = matrix.nullspace()
    # A real null space has a vector using all terms iff none of its coordinates vanishes identically.
    return bool(null) and all(any(v[i]!=0 for v in null) for i in range(len(labels)))


def actual_residual(equation):
    grid = np.linspace(0,4*np.pi,2001)
    u = np.sin(grid)+1.3*np.cos(grid)
    values = {'u':u,'du/dx0':np.cos(grid)-1.3*np.sin(grid),'d^2u/dx0^2':-u,'x':grid}
    def point(label):
        result = np.ones_like(grid)
        for name, params in label:
            p=dict(params)
            field = getattr(np,name)(p['freq']*grid) if name in ('sin','cos') else values[name]
            result *= field**int(p['power'])
        return result
    lhs,rhs=equation.split('=')
    pivot,=labels_of('0 = '+rhs)
    cols=[-point(pivot)]
    for term in re.split(r'(?<![eE])\+',lhs):
        weight=float(term.split('*')[0])
        if weight:
            key,=[l for l in labels_of(term+' = '+rhs) if l!=pivot]
            cols.append(weight*point(key))
    return float(np.linalg.norm(np.sum(cols,axis=0))/np.linalg.norm(cols))


if __name__=='__main__':
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            record=json.loads((ROOT/'results'/f'ode_{seed}_{arm}.json').read_text())
            for item in record['front']:
                equation=item['equation']
                labels=labels_of(equation)
                literal, unit=[identity(labels,snap) for snap in (False,True)]
                assert not item['hit'] or unit
                rows.append(dict(seed=seed,arm=arm,equation=equation,registered=item['hit'],literal=literal,
                                 unit_frequency=unit,printed_relative_residual=actual_residual(equation)))
    (ROOT/'ode_identities.json').write_text(json.dumps(rows,allow_nan=False))
    for arm in ARMS:
        print(arm,[(sum(any(r['seed']==s and r['arm']==arm and r[k] for r in rows) for s in SEEDS))
                   for k in ('registered','literal','unit_frequency')])
