import study as S
R,np,ROOT=S.R,S.np,S.ROOT
import csv,json,argparse
from collections import Counter

def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def main(cohort):
    files=[p for p in (ROOT/'results').glob('*.json') if ('_iid_iid_' in p.name)==(cohort=='iid')]
    assert len(files)==(45 if cohort=='iid' else 72)
    coverage=[];regions=[];cases=[];directions=[];projections=[]
    for p in sorted(files):
        d=json.loads(p.read_text());f=json.loads((ROOT/'frozen'/p.name).read_text());a=np.load(p.with_suffix('.npz'));cs=f['candidates']
        meta={k:d[k] for k in ('system','seed','profile','model_kind')}
        for bank,metric in d['metrics'].items():
            if metric['status']!='ok':coverage.append(dict(**meta,bank=bank,status=metric['status']));continue
            coverage.append(dict(**meta,bank=bank,status='ok',cutoff=metric['cutoff'],operator_coverage=float(np.mean(a[bank+'_actual_operator_scores']<=metric['cutoff'])),
                covariance_rank=metric['rank'],covariance_dimension=metric['dimension'],
                mean_reference_pointwise_coverage=float(np.mean([r['pointwise_noise_coverage_count']/256 for r in d['statistics'] if r['bank']==bank and r['label']=='reference'])),
                mean_reference_uniform_coverage=float(np.mean([r['uniform_noise_coverage_count']/256 for r in d['statistics'] if r['bank']==bank and r['label']=='reference']))))
            rr=d['regions'][bank]
            for row in rr:
                regions.append(dict(**meta,bank=bank,**{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()}))
                if row['label']=='reference' and row['status']=='ellipsoid':
                    c=next(c for c in cs if c['id']==row['candidate'])
                    for j,(lo,hi,center,truth) in enumerate(zip(row['lower'],row['upper'],row['center'],c['exact'])):
                        projections.append(dict(**meta,bank=bank,candidate=c['id'],coefficient=j,feature=str(c['terms'][j]),center=center,lower=lo,upper=hi,exact=truth))
            for slot in sorted({c['slot'] for c in cs}):
                cc=[r for r in rr if r['slot']==slot];ref=next(r for r in cc if r['label']=='reference');stats=next(r for r in d['statistics'] if r['bank']==bank and r['candidate']==ref['candidate'])
                good=[r for r in cc if r['feasible'] is True];minp=min((r['features'] for r in good),default=None)
                cases.append(dict(**meta,bank=bank,slot=slot,total=len(cc),compatible=len(good),excluded=sum(r['feasible'] is False for r in cc),unresolved=sum(r['feasible'] is None for r in cc),
                    reference_status=ref['status'],reference_compatible=ref['feasible'],reference_exact_accepted=stats['anchor_exact_accepted'],
                    minimum_compatible_features=minp,registered_at_minimum=any(r['registered'] and r['features']==minp for r in good),
                    compatible_drops=sum(r['label'].startswith('drop_') and r['feasible'] is True for r in cc),
                    excluded_drops=sum(r['label'].startswith('drop_') and r['feasible'] is False for r in cc)))
                if ref['status']!='unbounded':continue
                c=next(c for c in cs if c['id']==ref['candidate']);xi=c['xi'];n=len(xi)
                rng=np.random.default_rng(np.random.SeedSequence([940104,d['seed'],slot]));v=rng.normal(size=(1024,n));v/=np.linalg.norm(v,axis=1,keepdims=True)
                v=np.vstack([np.asarray(ref['negative_direction'])[None],np.eye(n),v]);v/=np.linalg.norm(v,axis=1,keepdims=True)
                g=a[bank+'_observed_gram'][np.ix_(xi,xi)];err=a[bank+'_check_grams'][:,xi,:][:,:,xi]
                obs=np.einsum('ki,ij,kj->k',v,g,v);cut=np.quantile(np.einsum('ki,bij,kj->bk',v,err,v),.95,axis=0)
                ratio=np.divide(obs,cut,out=np.full_like(obs,np.inf),where=cut>0);best=int(np.argmin(ratio))
                directions.append(dict(**meta,bank=bank,slot=slot,negative_eigenvector_pointwise_ratio=float(ratio[0]),
                    sampled_minimum_pointwise_ratio=float(ratio[best]),sampled_direction=v[best].tolist(),tested_directions=len(v),
                    pointwise_unbounded_witness=bool(ratio[best]<1)))
    write(ROOT/f'{cohort}_coverage.csv',coverage);write(ROOT/f'{cohort}_regions.csv',regions);write(ROOT/f'{cohort}_cases.csv',cases);write(ROOT/f'{cohort}_projections.csv',projections)
    R.dump(ROOT/f'{cohort}_directional_diagnostics.json',directions)
    summary=[]
    for profile,kind in sorted({(r['profile'],r['model_kind']) for r in coverage}):
        for bank in ('local','broad'):
            vv=[r for r in coverage if r['profile']==profile and r['model_kind']==kind and r['bank']==bank]
            cc=[r for r in cases if r['profile']==profile and r['model_kind']==kind and r['bank']==bank]
            out=dict(profile=profile,model_kind=kind,bank=bank,records=len(vv),incomplete=sum(r['status']!='ok' for r in vv),equation_cases=len(cc))
            if out['incomplete']==0:
                out.update(operator_coverage=float(np.mean([r['operator_coverage'] for r in vv])),
                    reference_pointwise_coverage=float(np.mean([r['mean_reference_pointwise_coverage'] for r in vv])),
                    reference_uniform_coverage=float(np.mean([r['mean_reference_uniform_coverage'] for r in vv])),
                    reference_regions=dict(Counter(r['reference_status'] for r in cc)),
                    exact_reference_accepted=sum(r['reference_exact_accepted'] for r in cc),
                    mean_compatible_candidates=float(np.mean([r['compatible'] for r in cc])),
                    registered_at_minimum=sum(r['registered_at_minimum'] for r in cc),
                    compatible_drops=sum(r['compatible_drops'] for r in cc),excluded_drops=sum(r['excluded_drops'] for r in cc),
                    unresolved_candidates=sum(r['unresolved'] for r in cc))
            summary.append(out)
    result=dict(status='COMPLETE',cohort=cohort,records=len(files),summary=summary,
        source_sha256={__file__:S.E.sha(__file__),str(ROOT/'directional_diagnostic_protocol.txt'):S.E.sha(ROOT/'directional_diagnostic_protocol.txt')})
    R.dump(ROOT/f'{cohort}_summary.json',result);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['iid','correlated'],default='iid');a=ap.parse_args();main(a.cohort)
