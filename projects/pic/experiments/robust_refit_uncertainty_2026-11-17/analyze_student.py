from common import *
from collections import Counter,defaultdict
import csv
def write(n,rr):
    with (ROOT/n).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
def main():
    files=sorted((ROOT/'studentized_all').glob('*.json'));assert len(files)==69;rows=[];refs=[];geometries=[];widths=[];computing=[]
    for path in files:
        d=read(path);pd=read(ROOT/'results'/path.name);f=read(ROOT/'frozen_fast'/path.name);meta={k:d[k] for k in ('system','profile','seed')}
        with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
        with np.load((ROOT/'results'/path.name).with_suffix('.npz')) as z:b={k:z[k] for k in z.files}
        cut=float(np.quantile(np.maximum(a['local_check_scores'],a['broad_check_scores']),.95));jointcount=int(np.sum(np.maximum(a['local_actual_scores'],a['broad_actual_scores'])<=cut))
        for bank,count in list(d['coverage_counts'].items())+[('both',jointcount)]:
            for mode in ('fixed','refit','studentized'):
                c=count if mode=='studentized' else (pd['joint'][mode]['operator_coverage_count'] if bank=='both' else next(r['operator_coverage_count'] for r in pd['statistics'] if (r['mode'],r['bank'])==(mode,bank)))
                rows.append(dict(**meta,mode=mode,bank=bank,coverage=c/256))
        for c in f['candidates']:
            if c['label']!='reference':continue
            v=S.E.vector(c,c['exact'],len(f['labels']));gobs=[b[f'refit_{bank}_observed_gram'] for bank in ('local','broad')];sig=[a[bank+'_anchor_sigma'] for bank in ('local','broad')]
            cert=C.J.certificate(gobs,sig,cut,c,d['seed']);old=next(x for x in pd['joint']['refit']['certificates'] if x['candidate']==c['id']);geometries.append(dict(**meta,candidate=c['id'],refit_geometry=old['geometry'],studentized_geometry=cert['geometry'],anchor_accepted=all(v@g@v<=cut*float(v@s@v) for g,s in zip(gobs,sig))))
            if old['geometry']==cert['geometry']=='bounded_outer':
                ow=np.array(old['outer_region']['upper'])-old['outer_region']['lower'];nw=np.array(cert['outer_region']['upper'])-cert['outer_region']['lower'];widths.append(dict(**meta,candidate=c['id'],median_coordinate_width_ratio=float(np.median(nw/ow))))
            for bank in ('local','broad'):
                threshold=d['metrics'][bank]['cutoff'];bounds=threshold*np.einsum('i,nij,j->n',v,a[bank+'_actual_covariances'],v);errors=np.einsum('i,nij,j->n',v,b[f'refit_{bank}_actual_error_grams'],v);residual=np.einsum('i,nij,j->n',v,b[f'refit_{bank}_actual_grams'],v)
                refs.append(dict(**meta,candidate=c['id'],bank=bank,uniform_noise_coverage=float(np.mean(errors<=bounds)),uniform_zero_acceptance=float(np.mean(residual<=bounds))))
        computing.append(dict(**meta,exact_metrics=d['exact_metric_calls'],response_metrics=d['response_metric_calls'],patches=len(d['validation']['patches']),disabled_patches=sum(r['status']!='response_validated' for r in d['validation']['patches'])))
    summaries=[];systems=[]
    for group in sorted({(r['profile'],r['mode'],r['bank']) for r in rows}):
        rr=[r['coverage'] for r in rows if (r['profile'],r['mode'],r['bank'])==group];summaries.append(dict(profile=group[0],mode=group[1],bank=group[2],anchors=len(rr),coverage=float(np.mean(rr)),min=min(rr),max=max(rr)))
    for group in sorted({(r['system'],r['profile'],r['mode'],r['bank']) for r in rows}):
        rr=[r['coverage'] for r in rows if (r['system'],r['profile'],r['mode'],r['bank'])==group];systems.append(dict(system=group[0],profile=group[1],mode=group[2],bank=group[3],coverage=float(np.mean(rr))))
    write('studentized_coverage.csv',rows);write('studentized_system_coverage.csv',systems);write('studentized_reference_geometry.csv',geometries);write('studentized_width_ratios.csv',widths);write('studentized_reference_directions.csv',refs)
    out=dict(status='COMPLETE',anchors=69,summary=summaries,systems=systems,geometry={mode:dict(Counter(r[mode+'_geometry'] for r in geometries)) for mode in ('refit','studentized')},reference_anchors=len(geometries),reference_accepted=sum(r['anchor_accepted'] for r in geometries),bounded_width_ratio_quantiles=np.quantile([r['median_coordinate_width_ratio'] for r in widths],[0,.1,.5,.9,1]).tolist(),computing=computing,source_sha256=hashes('analyze_student.py'),input_sha256={str(p):sha(p) for p in files})
    dump(ROOT/'studentized_summary.json',out);print(json.dumps(summaries,indent=2));print('GEOMETRY',out['geometry']);plot(systems)
def plot(systems):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={(r['system'],r['profile'],r['mode'],r['bank']):r['coverage'] for r in systems};cols=[('refit','local'),('studentized','local'),('refit','broad'),('studentized','broad')];data=np.array([[lookup[n,'iid',m,b] for m,b in cols] for n in SYSTEMS])
    fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(data,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_yticks(range(15),SYSTEMS);ax.set_xticks(range(4),['Refit/local','Studentized/local','Refit/broad','Studentized/broad'])
    for i in range(15):
        for j in range(4):ax.text(j,i,f'{100*data[i,j]:.1f}',ha='center',va='center',color='black' if data[i,j]>.6 else 'white',fontsize=9)
    ax.set_title('Exploratory studentization follow-up: nominal 95% coverage');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(ROOT/'studentized_iid.png',dpi=180);fig.savefig(ROOT/'studentized_iid.pdf');plt.close(fig)
if __name__=='__main__':main()
