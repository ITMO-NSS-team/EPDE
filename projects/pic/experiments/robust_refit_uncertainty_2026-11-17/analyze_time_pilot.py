from common import *
from collections import Counter
import csv

def main():
    files=sorted((ROOT/'time_pilot_results').glob('*.json'));assert len(files)==3;rows=[];params=[];warnings=[];geometries=[]
    for path in files:
        d=read(path);s=read(ROOT/'time_pilot_studentized'/path.name);f=read(ROOT/'time_pilot_frozen'/path.name);seed=d['seed'];warnings+=d['warnings']
        for r in d['statistics']:rows.append(dict(seed=seed,mode=r['mode'],bank=r['bank'],count=r['operator_coverage_count'],coverage=r['operator_coverage_count']/256))
        with np.load((ROOT/'time_pilot_studentized'/path.name).with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
        with np.load(path.with_suffix('.npz')) as z:b={k:z[k] for k in z.files}
        for bank in ('local','broad'):rows.append(dict(seed=seed,mode='studentized',bank=bank,count=s['coverage_counts'][bank],coverage=s['coverage_counts'][bank]/256))
        cut=np.quantile(np.maximum(a['local_check_scores'],a['broad_check_scores']),.95);count=int(np.sum(np.maximum(a['local_actual_scores'],a['broad_actual_scores'])<=cut));rows.append(dict(seed=seed,mode='studentized',bank='both',count=count,coverage=count/256))
        for mode in ('fixed','refit'):count=d['joint'][mode]['operator_coverage_count'];rows.append(dict(seed=seed,mode=mode,bank='both',count=count,coverage=count/256))
        for stream,rr in d['refits'].items():
            for v,base in d['model'].items():
                ratios=[r['model'][v][0]/base[0] for r in rr];rhos=[r['model'][v][1] for r in rr];params.append(dict(seed=seed,stream=stream,variable=v,sigma_ratio_min=min(ratios),sigma_ratio_median=float(np.median(ratios)),sigma_ratio_max=max(ratios),max_abs_rho=float(np.max(np.abs(rhos)))))
        for c in f['candidates']:
            if c['label']!='reference':continue
            v=S.E.vector(c,c['exact'],len(f['labels']));gg=[b[f'refit_{bank}_observed_gram'] for bank in ('local','broad')];ss=[a[bank+'_anchor_sigma'] for bank in ('local','broad')];cert=C.J.certificate(gg,ss,cut,c,seed);geometries.append(dict(seed=seed,candidate=c['id'],geometry=cert['geometry'],accepted=bool(all(v@g@v<=cut*float(v@s@v) for g,s in zip(gg,ss)))))
    summary=[]
    for mode in ('fixed','refit','studentized'):
        for bank in ('local','broad','both'):
            rr=[r for r in rows if (r['mode'],r['bank'])==(mode,bank)];summary.append(dict(mode=mode,bank=bank,coverage=float(np.mean([r['coverage'] for r in rr])),counts=[r['count'] for r in rr]))
    out=dict(status='COMPLETE',anchors=3,paired_records=6,summary=summary,rows=rows,parameter_distributions=params,warnings=warnings,studentized_reference_geometry=geometries,source_sha256=hashes('analyze_time_pilot.py'),input_sha256={str(p):sha(p) for p in files+list((ROOT/'time_pilot_studentized').glob('*.json'))});dump(ROOT/'time_pilot_summary.json',out)
    with (ROOT/'time_pilot_coverage.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2));print('WARNINGS',len(warnings));print('GEOMETRY',geometries)
if __name__=='__main__':main()
