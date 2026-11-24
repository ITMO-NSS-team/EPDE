"""Clean-field and reconstruction controls; oracle information never fits noisy models."""
import run_screen as R
from analyze_screen import common_width
import numpy as np
import json


def one(path):
    x=json.loads(path.read_text(encoding='utf-8'))
    p=R.datasets.load(x['system']);axes=R.axes_of(p)
    cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
    raw=p.noisy(x['noise_percent'],x['seed'])
    rows=[]
    for fold in x['folds']:
        g=fold['training'];idx=np.array(x['groups'][g]['indices'])
        noisy_pilot,_=R.reconstruct({v:y[idx] for v,y in raw.items()},axes,idx)
        clean_pilot,_=R.reconstruct({v:y[idx] for v,y in p.data.items()},axes,idx)
        plan={k:tuple(v) for k,v in fold['plan'].items()}
        local=R.LocalBank(axes,terms,opaque,plan)
        matrices={name:local.matrix(values) for name,values in
                  [('noisy_pilot_ols',noisy_pilot),('clean_split_ols',clean_pilot),
                   ('raw_noisy_ols',raw),('clean_full_ols',p.data)]}
        frequencies=[]
        for selection in x['groups'][g]['selection']:
            k=selection['loo']['max_mode']
            frequencies.append(np.r_[np.arange(k+1),-np.arange(1,k+1)]*selection['spacing'])
        bank=R.SpectralBank(axes,terms,opaque,frequencies)
        clean=bank.matrix(p.data)
        smoothed=bank.matrix(noisy_pilot)
        clean_smoothed=bank.matrix(clean_pilot)
        for i,c in enumerate(cs):
            if c['label']!='reference':continue
            exact=np.array(c['exact'])
            scale=float(np.vdot(clean[:,c['yi']],clean[:,c['yi']]).real)
            def error(mat,coef):
                residual=mat[:,c['yi']]-mat[:,c['xi']]@coef
                return float(np.vdot(residual,residual).real/scale)
            entry=dict(system=x['system'],seed=x['seed'],slot=c['slot'],fold=g,
                       exact_coefficients=exact,
                       clean_exact_energy=error(clean,exact),
                       clean_split_exact_energy=error(clean_smoothed,exact),
                       noisy_pilot_exact_energy=error(smoothed,exact),fits={})
            for name,mat in matrices.items():
                coef,diagnostic=R.signed_fit(mat.T@mat,mat,c['xi'],c['yi'])
                entry['fits'][name]=dict(coefficients=coef,clean_energy=error(clean,coef),
                                         relative_coefficient_error=float(np.linalg.norm(coef-exact)/np.linalg.norm(exact)))
            saved=fold['arms']['loo']['counts'][str(x['draws'])][i]
            coef=np.array(saved['coefficients'])
            entry['fits']['clsm']=dict(coefficients=coef,clean_energy=error(clean,coef),
                relative_coefficient_error=float(np.linalg.norm(coef-exact)/np.linalg.norm(exact)))
            rows.append(entry)
    return rows


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    rows=[]
    for name in R.datasets.names('core'):
        path=R.ROOT/'audited'/f'{name}_seed0_b32.json'
        if path.exists():
            rows.extend(one(path));print('diagnosed',name,flush=True)
    R.dump(R.ROOT/'audited'/'fit_diagnostics.json',rows)


if __name__=='__main__':main()
