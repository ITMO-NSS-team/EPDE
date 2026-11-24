"""Read-only explanation of the nine completed forced-KdV certificates."""
from bridge import *


def main():
    files=sorted((ROOT/'results').glob('kdv_cossin_*.json'));assert len(files)==9;rows=[]
    for path in files:
        d=read(path);c=next(c for c in d['candidates'] if c['label']=='reference')
        cert=next(c for c in d['joint']['refit_pilot']['certificates'] if c['candidate']=='e0:reference')
        assert cert['geometry']=='unbounded_certified'
        direction=np.array(cert['common_negative_direction']);v=np.zeros(len(d['labels']));v[c['xi']]=direction
        banks={}
        with np.load(path.with_suffix('.npz')) as z:
            for b in ('local','broad'):
                g=S.gram(z[b+'_observed']);clean=S.gram(z[b+'_clean']);cv=z[b+'_anchor_covariance'];q=d['joint']['refit_pilot']['cutoff']
                residual=float(v@g@v);allowance=float(q*(v@cv@v));assert residual<allowance
                denominator=sum(abs(direction[j])*np.sqrt(max(0.,clean[k,k])) for j,k in enumerate(c['xi']))**2
                banks[b]=dict(residual_over_allowance=residual/allowance,
                              clean_cancellation_ratio=float(v@clean@v)/denominator,
                              clean_feature_correlation_eigenvalues=np.linalg.eigvalsh(clean[np.ix_(c['xi'],c['xi'])]/np.sqrt(np.diag(clean)[c['xi']])[:,None]/np.sqrt(np.diag(clean)[c['xi']])[None,:]).tolist())
        panel=read(ROOT/'panel_results'/path.name)
        row=dict(profile=d['profile'],seed=d['seed'],banks=banks,
                 cutoff_ratio_upper_bound_for_removing_this_ray=max(r['residual_over_allowance'] for r in banks.values()),
                 panel_cutoff_ratio=panel['modes']['refit_pilot']['cutoff_ratio'])
        assert row['panel_cutoff_ratio']>row['cutoff_ratio_upper_bound_for_removing_this_ray']
        rows.append(row)
    out=dict(status='COMPLETE',rows=rows,
             interpretation='This fixed direction remains negative in both banks whenever the cutoff '
             'ratio exceeds the recorded maximum residual/allowance ratio. The panel reduction is '
             'insufficient even to remove this single recession direction. Removing it would not '
             'by itself establish boundedness. Clean cancellation is a post hoc trajectory diagnostic, '
             'not an input to calibration or a prediction for new noise levels.',
             source_sha256=sources('forced_geometry_probe.py'),
             input_sha256={str(p):sha(p) for p in files+[ROOT/'panel_results'/p.name for p in files]})
    dump(ROOT/'forced_geometry_probe.json',out);print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
