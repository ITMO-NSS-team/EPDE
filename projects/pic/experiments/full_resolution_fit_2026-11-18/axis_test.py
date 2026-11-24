"""Prospective geometry-only blocked scoring on a new single recording per anchor."""
import csv
from concurrent.futures import ProcessPoolExecutor
from math import prod
from pareto_report import outcome
from study import METHODS, ROOT, SYSTEMS, Panel, R, evaluate, fit, matrix_for, np, wls


def block_specs(shape, plan, axis):
    groups = []
    for start, stop in [(0,shape[axis]//2),(shape[axis]//2,shape[axis])]:
        cuts = [slice(None)]*len(shape)
        cuts[axis] = slice(start,stop)
        lengths = list(shape)
        lengths[axis] = stop-start
        levels = tuple(1+(n-1-2*c-2*m)//s for n,c,m,s in zip(lengths,plan['cut'],plan['ms'],plan['ss']))
        if min(levels)<1:
            return []
        groups.append((tuple(cuts), dict(plan,shape=levels)))
    return groups


def choose_axis(shape, plan):
    viable = []
    for axis in range(len(shape)):
        groups = block_specs(shape,plan,axis)
        if groups:
            levels = groups[0][1]['shape']
            viable.append((min(levels), prod(levels), -axis))
    return -max(viable)[2] if viable else None


def block_banks(base, slicing):
    banks = []
    for cuts, plan in slicing:
        axes = [axis[s] for axis,s in zip(base.axes,cuts)]
        opaque = {k:v[cuts] for k,v in base.opaque.items()}
        banks.append((R.LocalBank(axes,base.terms,opaque,plan),cuts))
    return banks


def panel_rows(cs, dyads, matrices):
    rows = []
    for c,dyad in zip(cs,dyads):
        ids = sorted([c['yi'],*c['xi']])
        pivot = ids.index(c['yi'])
        methods = {}
        for method in METHODS:
            values, coefs = [], []
            for g in (0,1):
                train, test = matrices[g], matrices[1-g]
                mat=train[:,ids]
                v, status = wls(mat,pivot) if method=='wls' else fit(mat)
                if status!='ok':
                    continue
                value = evaluate(train,test,test,ids,v)
                prediction = sum(v[j]*test[:,idx] for j,idx in enumerate(ids))
                scale = sum(v[j]**2*np.mean(train[:,idx]**2) for j,idx in enumerate(ids))
                np.testing.assert_allclose(value,np.mean(prediction**2)/scale,rtol=1e-9,atol=1e-12)
                values.append(value)
                if c['label']=='reference' and v[pivot]!=0:
                    coefs.append(float(-v[ids.index(c['xi'][0])]/v[pivot]))
            methods[method] = dict(scores={'held':float(np.mean(values)) if len(values)==2 else None},
                                   first_coefs=coefs)
        rows.append(dict(id=c['id'],slot=c['slot'],label=c['label'],registered=c['registered'],dyad=dyad,methods=methods))
    return rows


def run(system):
    panel = Panel(system)
    base = panel.dyad_bank
    shape = next(iter(panel.problem.data.values())).shape
    axis = choose_axis(shape,base.plan)
    designs = {'time_capped':[(b,s) for b,s in zip(panel.banks[1:],panel.slices[1:])]}
    if axis is not None:
        designs['preserved'] = block_banks(base,block_specs(shape,base.plan,axis))
        if system in ('ac','ac_zero','wave'):
            designs['same_axis_capped'] = block_banks(base,block_specs(shape,panel.banks[0].plan,axis))
    outputs = []
    for anchor in range(12,24):
        raw = panel.recording(anchor,0)
        complete = matrix_for(base,raw,slice(None),panel.const)
        dyads = [panel.dyad(complete,c) for c in panel.cs]
        for design,banks in designs.items():
            matrices = [matrix_for(b,raw,s,panel.const) for b,s in banks]
            rows = panel_rows(panel.cs,dyads,matrices)
            for slot in sorted({r['slot'] for r in rows}):
                group = [r for r in rows if r['slot']==slot]
                reference, = [r for r in group if r['label']=='reference']
                for method in METHODS:
                    vals = outcome(group,method,'held')
                    margin = None
                    if system.startswith('ac'):
                        reduced, = [r for r in group if r['label']=='drop_0']
                        margin = reduced['methods'][method]['scores']['held'] - reference['methods'][method]['scores']['held']
                    coefs = reference['methods'][method]['first_coefs']
                    outputs.append(dict(system=system,slot=slot,anchor=anchor,design=design,method=method,
                                        axis=0 if design=='time_capped' else axis, **vals,
                                        coefficient_a=coefs[0] if len(coefs)==2 else None,
                                        coefficient_b=coefs[1] if len(coefs)==2 else None,diffusion_margin=margin))
    return outputs


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [r for group in pool.map(run,[*SYSTEMS,'ac_zero']) for r in group]
    with (ROOT/'axis_test.csv').open('w',newline='') as stream:
        w = csv.DictWriter(stream,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for des in ('time_capped','preserved'):
        for method in METHODS:
            sub=[r for r in rows if r['system']!='ac_zero' and r['design']==des and r['method']==method]
            print(des,method,len(sub),{k:sum(r[k] for r in sub) for k in ('registered_lower','registered_upper','exact_lower','additions_lower','unresolved')})
