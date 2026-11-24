from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
import correlated_fit as F
np=F.np
from scipy.linalg import eigvalsh

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rows=[];inputs={}
    for profile in ('iid','positive','negative'):
        pp=ROOT/f'ks_{profile}_attribution_seed36.json';d=json.loads(pp.read_text());inputs[str(pp)]=sha(pp)
        parent=ROOT/'results'/f'ks_{profile}_curvature_seed36.json';base=json.loads(parent.read_text());frozen=ROOT/'frozen'/parent.name;active=json.loads(frozen.read_text())['active']
        paths=(pp.with_suffix('.npz'),parent.with_suffix('.npz'));inputs.update({str(p):sha(p) for p in (*paths,parent,frozen)})
        with np.load(paths[0]) as z:a={k:z[k] for k in z.files}
        with np.load(paths[1]) as z:b={k:z[k] for k in z.files}
        for bank in ('local','broad'):
            gg=a[bank+'_oracle_error_grams'].mean(axis=0)[np.ix_(active,active)];s=b[bank+'_sigma'][np.ix_(active,active)];scale=np.sqrt(np.diag(s))
            w=b[bank+'_white'];white=w.T@gg@w;ev=np.linalg.eigvalsh((white+white.T)/2);gg=gg/scale[:,None]/scale[None,:];s=s/scale[:,None]/scale[None,:];direct=eigvalsh((gg+gg.T)/2,s);np.testing.assert_allclose(direct,ev,rtol=3e-8,atol=3e-10)
            rows.append(dict(profile=profile,bank=bank,cutoff=base['metrics'][bank]['cutoff'],min_actual_to_calibration_second_moment=float(ev[0]),max_actual_to_calibration_second_moment=float(ev[-1]),
                original_score_quantiles=np.quantile(b[bank+'_actual_operator_scores'],[.05,.5,.95]).tolist(),oracle_score_quantiles=np.quantile(a[bank+'_oracle_scores'],[.05,.5,.95]).tolist(),model=base['model']))
    out=dict(status='COMPLETE',independent_eigenvalue_check='PASS',rows=rows,source_sha256={str(Path(__file__)):sha(Path(__file__))},input_sha256=inputs)
    (ROOT/'ks_calibration_distortion.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
