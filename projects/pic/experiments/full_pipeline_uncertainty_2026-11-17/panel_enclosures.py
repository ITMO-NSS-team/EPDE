"""Preserve valid enclosures when a new cutoff changes certificate weighting.

Reevaluate the original positive combination at the panel cutoff, then intersect
its coordinate bounds with those of the new combination. Both contain the same
joint region. No confidence threshold, coverage result, or region is changed.
"""
from bridge import *
import audit_verified as Audit
import csv


def main():
    files=sorted((ROOT/'panel_results').glob('*.json'));assert len(files)==69;rows=[];checks=0
    for path in files:
        panel=read(path);primary=read(ROOT/'results'/path.name);meta={k:primary[k] for k in ('system','profile','seed')}
        with np.load((ROOT/'results'/path.name).with_suffix('.npz')) as z:
            grams=[S.gram(z[b+'_observed']) for b in ('local','broad')];covs=[z[b+'_anchor_covariance'] for b in ('local','broad')]
            for mode,method in panel['modes'].items():
                for c in primary['candidates']:
                    if c['label']!='reference':continue
                    new=next(r for r in method['certificates'] if r['candidate']==c['id']);old=next(r for r in primary['joint'][mode]['certificates'] if r['candidate']==c['id'])
                    if new['geometry']!=old['geometry'] or new['geometry']!='bounded_outer':continue
                    cut=method['cutoff'];regions=[];weights=[]
                    for certificate in (new,old):
                        lam=certificate['lambda_local'];norm=np.array(certificate['target_normalizers']);assert 0<=lam<=1 and np.all(norm>0)
                        g=lam*grams[0]/norm[0]+(1-lam)*grams[1]/norm[1];s=lam*covs[0]/norm[0]+(1-lam)*covs[1]/norm[1]
                        region=S.quadratic_region(g-cut*s,g,s,c,cut);assert region['status']=='ellipsoid'
                        Audit.verified(region,g-cut*s,g,s,c,cut);checks+=1;regions.append(region);weights.append([lam/norm[0],(1-lam)/norm[1]])
                    lo=np.maximum(regions[0]['lower'],regions[1]['lower']);hi=np.minimum(regions[0]['upper'],regions[1]['upper']);assert np.all(lo<=hi)
                    truth=np.array(c['exact']);assert np.all(lo<=truth) and np.all(truth<=hi)
                    ow=np.array(old['outer_region']['upper'])-old['outer_region']['lower'];nw=np.array(new['outer_region']['upper'])-new['outer_region']['lower']
                    rows.append(dict(**meta,mode=mode,candidate=c['id'],old_full_width=ow.tolist(),
                                     original_panel_width_ratio=float(np.median(nw/ow)),
                                     retained_lower=lo.tolist(),retained_upper=hi.tolist(),
                                     retained_width_ratio=float(np.median((hi-lo)/ow)),
                                     positive_combination_weights=weights))
    out=dict(status='COMPLETE',audit_status='PASS',datasets=69,region_checks=checks,rows=rows,
             high_precision_checks=Audit.proofs,
             retained_width_quantiles={m:np.quantile([r['retained_width_ratio'] for r in rows if r['mode']==m],[0,.5,1]).tolist() for m in ('fixed_pilot','refit_pilot')},
             interpretation='Intersection of coordinate bounds from two separately verified positive-combination '
             'ellipsoids. A valid outer enclosure of the unchanged panel joint region, not its exact coordinate extrema.',
             source_sha256=sources('panel_enclosures.py','audit_verified.py'),
             input_sha256={str(p):sha(p) for p in files+[ROOT/'results'/p.name for p in files]})
    dump(ROOT/'panel_enclosures.json',out)
    with (ROOT/'panel_retained_widths.csv').open('w',encoding='utf-8',newline='') as f:
        names=['system','profile','seed','mode','candidate','original_panel_width_ratio','retained_width_ratio'];w=csv.DictWriter(f,names);w.writeheader();w.writerows({k:r[k] for k in names} for r in rows)
    print('PASS',checks,out['retained_width_quantiles'],flush=True)


if __name__=='__main__':main()
