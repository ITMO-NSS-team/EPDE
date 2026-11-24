"""Post-fit clean-signal diagnostic at the fixed physical intrinsic coefficients."""
import json
import numpy as np
from pair import PREVIOUS, ROOT
from forcing_profile import READOUTS


def main():
    rows=json.loads((ROOT/'forcing_results.json').read_text())['rows']
    out=[]
    for r in rows:
        if r['off_seed'] is not None:
            continue
        path=PREVIOUS/'forcing_profile'/f"{r['profile']}_{r['seed']}.npz"
        v=np.array([6.,1.,0.,1.])
        with np.load(path) as z:
            results={}
            for b in READOUTS:
                s=z[b+'_grams'][:32].mean(axis=0)
                allowance=r['cutoff']*float(v@s@v)
                results[b]={k:float(np.linalg.norm(z[b+'_'+k]@v)**2/allowance) for k in ('clean','observed')}
        strongest={k:max(g[k] for g in results.values()) for k in ('clean','observed')}
        assert (strongest['observed']<=1)==r['controls']['physical_intrinsic_without_forcing']['passed']
        out.append(dict(profile=r['profile'],seed=r['seed'],by_bank=results,maximum=strongest))
    for p in ['iid','positive','negative']:
        print(p,{k:[round(min(a),3),round(max(a),3)] for k in ('clean','observed')
                 for a in [[r['maximum'][k] for r in out if r['profile']==p]]})
    (ROOT/'noise_diagnostic.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
