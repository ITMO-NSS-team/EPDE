"""Check post-calibration diagnostics directly against archived moment arrays."""
from bridge import *


def main():
    precision=read(ROOT/'precision_summary.json');failure=read(ROOT/'failure_geometry.json');forced=read(ROOT/'forced_geometry_probe.json')
    zeros=coordinates=rays=failures=0
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==69
    for path in files:
        d=read(path);key=tuple(d[k] for k in ('system','profile','seed'))
        def matches(row):return tuple(row[k] for k in ('system','profile','seed'))==key
        with np.load(path.with_suffix('.npz')) as z:
            for row in filter(matches,precision['rows']):
                if row['geometry']!='bounded_outer':continue
                coordinates+=1;c=next(c for c in d['candidates'] if c['id']==row['candidate']);j=row['coefficient']
                cert=next(c for c in d['joint']['refit_pilot']['certificates'] if c['candidate']==row['candidate'])
                assert row['lower']==cert['outer_region']['lower'][j] and row['upper']==cert['outer_region']['upper'][j]
                assert row['full_width']==row['upper']-row['lower']
                witness=row['zero_witness']
                if witness is None:assert not row['zero_witness_found'];continue
                assert row['zero_witness_found'] and witness['coefficients'][j]==0.
                v=S.E.vector(c,witness['coefficients'],len(d['labels']));cut=d['joint']['refit_pilot']['cutoff']
                for b in ('local','broad'):
                    residual=z[b+'_observed']@v;left=float(residual@residual);right=float(cut*(v@z[b+'_anchor_covariance']@v));margin=left-right
                    assert margin<=1e-11*max(left,right,np.finfo(float).tiny)
                    np.testing.assert_allclose(margin,witness['margins'][b],rtol=1e-8,atol=1e-10*max(left,right,np.finfo(float).tiny))
                if row['truth']!=0:zeros+=1
            for row in filter(matches,precision['recession_diagnostics']):
                c=next(c for c in d['candidates'] if c['id']==row['candidate']);v=np.zeros(len(d['labels']));v[c['xi']]=[r['weight'] for r in row['normalized_direction']]
                cut=d['joint']['refit_pilot']['cutoff']
                for b,info in row['banks'].items():
                    y=z[b+'_observed']@v;left=float(y@y);right=float(cut*(v@z[b+'_anchor_covariance']@v));assert left<right
                    np.testing.assert_allclose(left/right,info['curvature_ratio'],rtol=1e-8)
                rays+=1
            for row in filter(matches,failure['failures']):
                b=row['bank'];cut=d['cuts'][row['mode']][b]
                np.testing.assert_allclose(row['observed_over_cutoff'],d['actual_scores'][b]/cut,rtol=1e-12)
                assert row['bootstrap_exceedances']==int(np.sum(z[row['mode']+'_'+b+'_scores']>=d['actual_scores'][b]));failures+=1
            if d['system']=='kdv_cossin':
                row=next(r for r in forced['rows'] if (r['profile'],r['seed'])==(d['profile'],d['seed']))
                c=next(c for c in d['candidates'] if c['label']=='reference')
                for b,info in row['banks'].items():
                    matrix=z[b+'_clean'][:,c['xi']];matrix=matrix/np.linalg.norm(matrix,axis=0)
                    values=np.sort(np.linalg.svd(matrix,compute_uv=False)**2)
                    np.testing.assert_allclose(values,info['clean_feature_correlation_eigenvalues'],rtol=1e-8,atol=1e-10)
    assert zeros==precision['zero_witnesses'];assert coordinates==precision['coordinate_records'];assert rays==len(precision['recession_diagnostics']);assert failures==len(failure['failures'])
    out=dict(status='PASS',datasets=69,zero_witnesses=zeros,coordinate_records=coordinates,recession_directions=rays,bank_failures=failures,
             source_sha256=sources('audit_diagnostics.py','precision_summary.py','failure_geometry.py','forced_geometry_probe.py'),
             input_sha256={str(ROOT/n):sha(ROOT/n) for n in ('precision_summary.json','failure_geometry.json','forced_geometry_probe.json')})
    dump(ROOT/'diagnostic_audit.json',out);print('PASS',zeros,coordinates,rays,failures,flush=True)


if __name__=='__main__':main()
