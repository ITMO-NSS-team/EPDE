from research_context import *

def main():
    rows=[]
    for path in sorted((ROOT/'repair_results').glob('*.json')):
        d=read(path);e,_,_=primary('ode','iid',51);e.seed=d['seed'];e.raw=e.true_field(0,True);e.model=d['model']
        with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
        e.pilot={v:a['anchor_pilot_'+v] for v in e.raw};refs=[]
        for c in e.cs:
            if c['label']!='reference':continue
            v=S.E.vector(c,c['exact'],len(e.labels))
            for b in e.banks:
                bound=d['cuts'][b]*float(v@a[b+'_covariance']@v);refs.append(dict(bank=b,candidate=c['id'],zero_accepted=float(v@S.gram(a[b+'_observed'])@v)<=bound,noise_covered=float(v@a[b+'_error_gram']@v)<=bound))
        row=dict(seed=e.seed,score_cutoff_ratio={b:d['actual_scores'][b]/d['cuts'][b] for b in e.banks},bootstrap_exceedances={b:int(np.sum(a[b+'_scores']>=d['actual_scores'][b])) for b in e.banks},
                 sigma_truth_ratio={v:e.model[v][0]/(.05*np.std(y)) for v,y in e.p.data.items()},estimated_rhos={v:m[1] for v,m in e.model.items()},references=refs,
                 observed_sigma_change_from_original={v:e.model[v][0]/d['original_anchor'][v]['model'][0]-1 for v in e.raw})
        if not d['coverage']['local']:
            rng=np.random.default_rng(np.random.SeedSequence([984102,e.system_id,e.pid,e.seed]));qq={b:[] for b in e.banks}
            for j,fit in enumerate(d['refits']):
                fields,_=e.draw(rng);assert {v:R.digest(y) for v,y in fields.items()}==d['bootstrap_field_sha256'][j];old={v:r['model'] for v,r in fit['original'].items()};mat=e.matrices(fields,old);cv=e.metric(e.pilot,old)
                for b in e.banks:qq[b].append(Legacy.score(e,S.gram(mat[b]-a[b+'_pilot']),cv[b])[0])
            cuts={b:Legacy.cutoff(v) for b,v in qq.items()};row['old_fit_same_fields']=dict(cutoffs=cuts,coverage={b:d['actual_scores'][b]<=cuts[b] for b in e.banks},note='Paired refit diagnostic conditional on repaired anchor and pilot; not a separately regenerated old procedure.')
            true_model={v:[.05*np.std(y),[0.]*y.ndim,e.model[v][2]] for v,y in e.p.data.items()};m=e.matrices(e.raw,true_model);cv=e.metric(e.pilot,true_model)
            row['oracle_noise_posthoc_only']={b:dict(score=Legacy.score(e,S.gram(m[b]-a[b+'_clean']),cv[b])[0],unchanged_cutoff=d['cuts'][b]) for b in e.banks}
        rows.append(row);print(row,flush=True)
    dump(ROOT/'fresh_failure.json',dict(status='COMPLETE',rows=rows,source_sha256=sources('fresh_failure.py'),input_sha256={str(p):sha(p) for p in (ROOT/'repair_results').glob('*.json')}))

if __name__=='__main__':main()
