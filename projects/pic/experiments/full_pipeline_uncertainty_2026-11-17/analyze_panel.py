from bridge import *
from collections import Counter
import csv


def main():
    files=sorted((ROOT/'panel_results').glob('*.json'));assert len(files)==69;rows=[];geometries=[];references=[]
    for path in files:
        d=read(path);p=read(ROOT/'results'/path.name);meta={k:d[k] for k in ('system','profile','seed')}
        ids={c['id'] for c in p['candidates'] if c['label']=='reference'}
        for mode,m in d['modes'].items():
            rows.append(dict(**meta,mode=mode,covered=d['coverage'][mode],original_covered=p['coverage'][mode]['both'],cutoff_ratio=m['cutoff_ratio']))
            for cert in m['certificates']:
                if cert['candidate'] not in ids:continue
                old=next(c for c in p['joint'][mode]['certificates'] if c['candidate']==cert['candidate']);width=None
                if old['geometry']==cert['geometry']=='bounded_outer':
                    ow=np.array(old['outer_region']['upper'])-old['outer_region']['lower'];nw=np.array(cert['outer_region']['upper'])-cert['outer_region']['lower'];width=float(np.median(nw/ow))
                geometries.append(dict(**meta,mode=mode,candidate=cert['candidate'],original=old['geometry'],panel=cert['geometry'],width_ratio=width))
        references.extend(dict(**meta,**r) for r in d['references'])
    summary=[]
    for profile in ('iid','positive','negative'):
        for mode in ('fixed_pilot','refit_pilot'):
            rr=[r for r in rows if (r['profile'],r['mode'])==(profile,mode)]
            summary.append(dict(profile=profile,mode=mode,datasets=len(rr),covered=sum(r['covered'] for r in rr),
                                original_covered=sum(r['original_covered'] for r in rr),
                                original_only=sum(r['original_covered'] and not r['covered'] for r in rr),
                                panel_only=sum(r['covered'] and not r['original_covered'] for r in rr)))
    out=dict(status='COMPLETE',scope='Exploratory post hoc comparison; fixed candidate panel only',datasets=len(files),summary=summary,
             failures=[r for r in rows if not r['covered']],reference_failures=[r for r in references if not(r['noise_covered'] and r['zero_accepted'])],
             geometry={m:dict(Counter(r['panel'] for r in geometries if r['mode']==m)) for m in ('fixed_pilot','refit_pilot')},
             geometry_changes=[r for r in geometries if r['original']!=r['panel']],
             cutoff_ratios={m:np.quantile([r['cutoff_ratio'] for r in rows if r['mode']==m],[0,.5,1]).tolist() for m in ('fixed_pilot','refit_pilot')},
             width_ratios={m:np.quantile([r['width_ratio'] for r in geometries if r['mode']==m and r['width_ratio'] is not None],[0,.5,1]).tolist() for m in ('fixed_pilot','refit_pilot')},
             source_sha256=sources('analyze_panel.py'),input_sha256={str(p):sha(p) for p in files})
    dump(ROOT/'panel_summary.json',out)
    for name,rr in [('panel_coverage.csv',rows),('panel_geometry.csv',geometries),('panel_references.csv',references)]:
        with (ROOT/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
    print(json.dumps({k:v for k,v in out.items() if not k.endswith('sha256')},indent=2),flush=True)


if __name__=='__main__':main()
