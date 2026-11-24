"""Exploratory attribution of the radius cap on the systems where it changed the bank."""
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pareto_report import outcome
from study import METHODS, ROOT, Panel, R, evaluate, fit, matrix_for, np, wls


def control(system):
    panel = Panel(system)
    banks = {'capped': panel.banks[0], 'original': panel.dyad_bank}
    assert banks['capped'].plan['ms'] != banks['original'].plan['ms']
    saved = json.loads((ROOT/'results'/f'{system}.json').read_text())
    prior = {(r['anchor'],r['id']):r for r in saved['rows']}
    output = []
    for anchor in range(12):
        raw = [panel.recording(anchor,i) for i in range(3)]
        matrices = {k:[matrix_for(bank,field,slice(None),panel.const) for field in raw] for k,bank in banks.items()}
        for training, (t,_,_) in matrices.items():
            for testing,(norm,a,b) in matrices.items():
                rows = []
                for c in panel.cs:
                    p = prior[anchor,c['id']]
                    ids = p['ids']
                    pivot = ids.index(c['yi'])
                    ds = {}
                    coefs = {}
                    for meth in METHODS:
                        v, status = wls(t[:,ids],pivot) if meth=='wls' else fit(t[:,ids])
                        assert status=='ok'
                        value = evaluate(norm,a,b,ids,v)
                        if training == testing == 'capped':
                            np.testing.assert_allclose(value,p['methods'][meth]['scores']['record_cross'],rtol=1e-11,atol=1e-13)
                        ds[meth] = {'scores':{'record_cross':value}}
                        if c['label']=='reference':
                            coefs[meth] = (-v / v[pivot])[[ids.index(j) for j in c['xi']]].tolist()
                    rows.append(dict(p,methods=ds,coefficients=coefs))
                ref, = [r for r in rows if r['label']=='reference']
                for meth in METHODS:
                    o = outcome(rows,meth,'record_cross')
                    margin = None
                    if system.startswith('ac'):
                        reduced, = [r for r in rows if r['label']=='drop_0']
                        margin = reduced['methods'][meth]['scores']['record_cross']-ref['methods'][meth]['scores']['record_cross']
                    output.append(dict(system=system,anchor=anchor,training=training,testing=testing,fit=meth,
                                       registered=o['registered_lower'],added=o['additions_lower'],
                                       reference_coefficients=json.dumps(ref['coefficients'][meth]),
                                       diffusion_margin=margin))
    return output


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [r for group in pool.map(control,['ac','ac_zero','wave','burgers_inviscid']) for r in group]
    with (ROOT/'window_control.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for s in ['ac','ac_zero','wave','burgers_inviscid']:
        for method in METHODS:
            for bank in ('capped','original'):
                sample = [r for r in rows if r['system']==s and r['fit']==method and r['training']==r['testing']==bank and r['anchor']>=6]
                print(s,method,bank,sum(r['registered'] for r in sample),
                      np.median([json.loads(r['reference_coefficients']) for r in sample],axis=0),
                      sum(r['diffusion_margin']>0 for r in sample if r['diffusion_margin'] is not None))
