"""Existing core-15 constants; does not redefine truth or run new fitting."""
import raw_operator as O
J,R,F,A,np=O.J,O.R,O.F,O.A,O.np
import csv,json,hashlib


def main():
    cases=[];pairs=[];details=[];inputs={};maximum=0.;counts={}
    for name in R.datasets.names('core'):
        p=R.datasets.load(name);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels);ci=labels.index(())
        A.common_width(max(len(c['terms']) for c in cs))
        for seed in range(3):
            _,old=F.baseline(name,seed);path=J.L.ROOT/'results'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            plan={k:tuple(v) for k,v in old['base_plan'].items()}
            raw=p.noisy(5,seed);mat=R.LocalBank(R.axes_of(p),terms,opaque,plan).matrix(raw)
            full=[];without=[]
            for i,c in enumerate(cs):
                per=R.SV.dyad_scores(mat[:,c['xi']],mat[:,c['yi']],np.ones(len(mat)),old['base_plan']['shape'],fit_intercept=False) if c['xi'] else np.array([])
                score=float(np.sum(per));np.testing.assert_allclose(score,old['instability'][i],rtol=5e-12,atol=1e-12)
                maximum=max(maximum,abs(score-old['instability'][i]));full.append(score)
                ix=c['xi'].index(ci) if ci in c['xi'] else None
                charge=float(per[ix]) if ix is not None else 0.;without.append(float(np.sum(np.delete(per,ix))) if ix is not None else score)
                coeff=[f['arms']['local_continuous'][i]['coefficients'] for f in d['folds']]
                intercepts=[co[ix] for co in coeff] if ix is not None else [0.]*3
                details.append(dict(system=name,seed=seed,candidate=c['id'],slot=c['slot'],has_constant=ix is not None,registered=c['registered'],
                    dyad=score,constant_dyad=charge,other_dyad=without[-1],intercepts=intercepts,
                    local_cross=d['aggregates']['local_continuous']['cross']['scores'][i],
                    frozen_clean_energy=d['aggregates']['legacy_continuous']['clean_energy']['scores'][i]))
            scores=d['aggregates']['local_continuous']['cross']['scores']
            for slot in sorted({c['slot'] for c in cs}):
                ri=next(i for i,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
                ai=next(i for i,c in enumerate(cs) if c['slot']==slot and c['label']=='add_0')
                ref,add=details[-len(cs)+ri],details[-len(cs)+ai]
                dom=d['aggregates']['local_continuous']['cross']['dominance'][slot]['unregistered']
                constdom=[cid for cid in dom if ci in next(c['xi'] for c in cs if c['id']==cid)]
                cases.append(dict(system=name,seed=seed,slot=slot,baseline_retained=bool(A.frontier(scores,full,cs,slot)),
                    constant_unscored_retained=bool(A.frontier(scores,without,cs,slot)),dominators=dom,constant_dominators=constdom))
                pairs.append(dict(system=name,seed=seed,slot=slot,reference_D=scores[ri],plus_constant_D=scores[ai],
                    reference_I=full[ri],plus_constant_I=full[ai],constant_I=add['constant_dyad'],
                    reference_clean=ref['frozen_clean_energy'],plus_constant_clean=add['frozen_clean_energy'],
                    intercepts=add['intercepts'],constant_candidate_dominates_reference=cs[ai]['id'] in dom))
    stats=dict(system_runs=45,equation_seed_cases=60,baseline_retained=sum(r['baseline_retained'] for r in cases),
        constant_unscored_retained=sum(r['constant_unscored_retained'] for r in cases),
        failed_cases_with_constant_dominator=sum(not r['baseline_retained'] and bool(r['constant_dominators']) for r in cases),
        failed_cases_with_nonconstant_dominator=sum(not r['baseline_retained'] and len(r['dominators'])>len(r['constant_dominators']) for r in cases),
        reference_plus_constant_dominates=sum(r['constant_candidate_dominates_reference'] for r in pairs),
        adding_constant_improves_clean=sum(r['plus_constant_clean']<r['reference_clean'] for r in pairs),
        max_dyad_reproduction_error=maximum,input_sha256=inputs,
        source_sha256=hashlib.sha256(O.Path(__file__).read_bytes()).hexdigest())
    for name,rows in [('constant_candidates.csv',details),('constant_pairs.csv',pairs),('constant_cases.csv',cases)]:
        with (O.ROOT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    R.dump(O.ROOT/'constant_audit.json',stats)
    print(json.dumps({k:v for k,v in stats.items() if 'sha256' not in k},indent=2))
    for r in cases:
        if not r['baseline_retained']:print(r)


if __name__=='__main__':main()
