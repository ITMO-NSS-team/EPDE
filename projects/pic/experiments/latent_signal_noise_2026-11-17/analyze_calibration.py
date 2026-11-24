from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
import correlated_fit as F
np=F.np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rows=[];inputs={}
    for name in ('ns','pde_compound'):
        for seed in (36,37,38):
            p=ROOT/f'{name}_attribution_seed{seed}.json';d=json.loads(p.read_text());inputs[str(p)]=sha(p)
            if d['status']=='SKIPPED':continue
            attr=p.with_suffix('.npz');main=ROOT/'results'/f'{name}_iid_curvature_seed{seed}.npz';inputs[str(attr)]=sha(attr);inputs[str(main)]=sha(main)
            frozen=json.loads((ROOT/'frozen'/f'{name}_iid_curvature_seed{seed}.json').read_text());active=frozen['active']
            with np.load(attr) as z:a={k:z[k] for k in z.files}
            with np.load(main) as z:b={k:z[k] for k in z.files}
            for bank in ('local','broad'):
                w=b[bank+'_white'];gg=a[bank+'_oracle_error_grams'].mean(axis=0);ratio=w.T@gg[np.ix_(active,active)]@w
                eig=np.linalg.eigvalsh((ratio+ratio.T)/2);row=next(r for r in d['rows'] if r['bank']==bank)
                rows.append(dict(system=name,seed=seed,bank=bank,min_actual_to_calibration_second_moment=float(eig[0]),max_actual_to_calibration_second_moment=float(eig[-1]),
                    oracle_mean_score_over_cutoff=row['oracle_mean_score']/row['cutoff'],analytic_bias_score_over_cutoff=row['analytic_bias_cutoff_ratio']))
    out=dict(status='COMPLETE',note='Finite-256-field empirical second moments after oracle correction; diagnostic eigenvalue ratios, not confidence bounds.',rows=rows,
        source_sha256={str(Path(__file__)):sha(Path(__file__))},input_sha256=inputs)
    (ROOT/'calibration_distortion.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
