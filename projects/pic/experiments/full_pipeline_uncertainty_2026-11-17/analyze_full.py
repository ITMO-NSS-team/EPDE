from bridge import *
from collections import Counter
import csv

def write(name,rows):
    if not rows:return
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
    process=read(ROOT/'processes.json');files=sorted((ROOT/'results').glob('*.json'));rows=[];refs=[];geometries=[];widths=[];warnings=[];params=[];ratio=[]
    for path in files:
        d=read(path);meta={k:d[k] for k in ('system','profile','seed')};p=R.datasets.load(d['system'])
        for mode,counts in d['coverage'].items():
            for bank,passed in counts.items():rows.append(dict(**meta,mode=mode,bank=bank,covered=int(passed)))
        refs.extend(dict(**meta,**r) for r in d['references']);warnings.extend(dict(**meta,**r) for r in d['warnings'])
        for v,m in d['model'].items():params.append(dict(**meta,variable=v,sigma_ratio=m[0]/(.05*np.std(p.data[v])),max_rho_error=float(np.max(np.abs(np.array(m[1])-S.PROFILES[d['profile']][1]))),status=d['fit_info'][v]['status']))
        ids={c['id'] for c in d['candidates'] if c['label']=='reference'}
        for bank in ('local','broad'):ratio.append(dict(**meta,bank=bank,cutoff_ratio=d['cuts']['refit_pilot'][bank]/d['cuts']['fixed_pilot'][bank]))
        for mode,j in d['joint'].items():
            for c in j['certificates']:
                if c['candidate'] in ids:geometries.append(dict(**meta,mode=mode,candidate=c['candidate'],geometry=c['geometry']))
        for ident in ids:
            old=next(c for c in d['joint']['fixed_pilot']['certificates'] if c['candidate']==ident);new=next(c for c in d['joint']['refit_pilot']['certificates'] if c['candidate']==ident)
            if old['geometry']==new['geometry']=='bounded_outer':
                ow=np.array(old['outer_region']['upper'])-old['outer_region']['lower'];nw=np.array(new['outer_region']['upper'])-new['outer_region']['lower'];widths.append(dict(**meta,candidate=ident,width_ratio=float(np.median(nw/ow))))
    summary=[];systems=[]
    for profile in ('iid','positive','negative'):
        for mode in ('fixed_pilot','refit_pilot'):
            for bank in ('local','broad','both'):
                rr=[r for r in rows if (r['profile'],r['mode'],r['bank'])==(profile,mode,bank)];summary.append(dict(profile=profile,mode=mode,bank=bank,datasets=len(rr),covered=sum(r['covered'] for r in rr)))
    for name,profile in JOBS:
        for mode in ('fixed_pilot','refit_pilot'):
            for bank in ('local','broad','both'):
                rr=[r for r in rows if (r['system'],r['profile'],r['mode'],r['bank'])==(name,profile,mode,bank)];systems.append(dict(system=name,profile=profile,mode=mode,bank=bank,datasets=len(rr),covered=sum(r['covered'] for r in rr)))
    discordance=[]
    for bank in ('local','broad','both'):
        pairs=[]
        for path in files:
            d=read(path);pairs.append((bool(d['coverage']['fixed_pilot'][bank]),bool(d['coverage']['refit_pilot'][bank])))
        discordance.append(dict(bank=bank,fixed_only=sum(a and not b for a,b in pairs),refit_only=sum(b and not a for a,b in pairs),both=sum(a and b for a,b in pairs),neither=sum(not a and not b for a,b in pairs)))
    out=dict(status='COMPLETE' if len(files)==69 and all(r['exit_code']==0 for r in process['rows']) else 'INCOMPLETE',datasets=len(files),expected=69,summary=summary,systems=systems,discordance=discordance,result_statuses=dict(Counter(read(p)['status'] for p in files)),warnings=warnings,geometry={mode:dict(Counter(r['geometry'] for r in geometries if r['mode']==mode)) for mode in ('fixed_pilot','refit_pilot')},width_ratio_quantiles=np.quantile([r['width_ratio'] for r in widths],[0,.1,.5,.9,1]).tolist() if widths else [],cutoff_ratio_quantiles={bank:np.quantile([r['cutoff_ratio'] for r in ratio if r['bank']==bank],[0,.1,.5,.9,1]).tolist() for bank in ('local','broad')},source_sha256=sources('analyze_full.py'),input_sha256={str(p):sha(p) for p in files+[ROOT/'processes.json']})
    for name,rr in [('coverage.csv',rows),('system_coverage.csv',systems),('reference_directions.csv',refs),('reference_geometry.csv',geometries),('width_ratios.csv',widths),('cutoff_ratios.csv',ratio),('noise_parameters.csv',params),('warnings.csv',warnings)]:write(name,rr)
    dump(ROOT/'summary.json',out);print(json.dumps(dict(summary=summary,discordance=discordance,warnings=len(warnings),geometry=out['geometry']),indent=2));plot(systems)

def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols=[(m,b) for m in ('fixed_pilot','refit_pilot') for b in ('local','broad','both')];data=np.full((len(JOBS),6),np.nan);labels=np.full((len(JOBS),6),'missing',dtype=object)
    for i,(n,p) in enumerate(JOBS):
        for j,(m,b) in enumerate(cols):
            r=next(r for r in rows if (r['system'],r['profile'],r['mode'],r['bank'])==(n,p,m,b));data[i,j]=r['covered']/r['datasets'] if r['datasets'] else np.nan;labels[i,j]=f"{r['covered']}/{r['datasets']}"
    fig,ax=plt.subplots(figsize=(10,10));ax.imshow(data,cmap='RdYlGn',vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(len(JOBS)),[n+' / '+p for n,p in JOBS]);ax.set_xticks(range(6),['Fixed/local','Fixed/broad','Fixed/joint','Refit/local','Refit/broad','Refit/joint']);ax.axvline(2.5,color='black',lw=1)
    for i in range(len(JOBS)):
        for j in range(6):ax.text(j,i,labels[i,j],ha='center',va='center',fontsize=9)
    ax.set_title('Fresh observed datasets covered by the operator bound\nThree datasets per case: a screen, not precise95% certification');fig.tight_layout();fig.savefig(ROOT/'full_pipeline_coverage.png',dpi=180);fig.savefig(ROOT/'full_pipeline_coverage.pdf');plt.close(fig)
if __name__=='__main__':main()
