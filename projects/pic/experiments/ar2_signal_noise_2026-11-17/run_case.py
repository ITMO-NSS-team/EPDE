import coverage as C
S,R,np,ROOT=C.S,C.R,C.np,C.ROOT
import argparse,time

def run(name,seed,profile,kind):
    start=time.monotonic();e=C.Study(name,seed,profile,kind);key=f'{name}_{profile}_{kind}_seed{seed}';path=ROOT/'results'/f'{key}.json'
    meta=dict(system=name,seed=seed,profile=profile,model_kind=kind,model=e.model,fit_info=e.fit_info,ar_estimates=e.ar,orders=e.orders,scales=e.scales,relative_amplitude=e.delta,
        raw_sha256={v:R.digest(y) for v,y in e.raw.items()},source_sha256=C.sources(),input_sha256=e.inputs)
    if not e.valid:
        R.dump(path,dict(**meta,status='incomplete_noise_model'));print(key,'INCOMPLETE noise model',flush=True);return
    observed=e.matrices(e.raw);pilot=e.matrices(e.pilot,False);arrays={};innovations={}
    for bank in e.banks:arrays[bank+'_observed']=observed[bank];arrays[bank+'_pilot']=pilot[bank]
    for tag,domain in [('calibration',970102),('check',970103)]:
        rng=np.random.default_rng(np.random.SeedSequence([domain,e.profile_id,seed]));stacks={b:[] for b in e.banks};hh=[]
        for b in range(256):
            fields,noise=e.draw(rng)
            if b in (0,255):hh.append({v:R.digest(y) for v,y in noise.items()})
            for bank,m in e.matrices(fields).items():
                error=m-pilot[bank];assert np.all(error[:,e.deterministic]==0);stacks[bank].append(S.gram(error))
        for bank,g in stacks.items():arrays[bank+'_'+tag+'_grams']=np.stack(g)
        innovations[tag]=hh
    metrics={};regions={}
    for bank in e.banks:
        sigma=arrays[bank+'_calibration_grams'].mean(axis=0);g=S.gram(observed[bank]);w,info=S.metric(sigma,e.active)
        arrays[bank+'_sigma']=sigma;arrays[bank+'_observed_gram']=g
        if w is None:metrics[bank]=info;regions[bank]=[];continue
        q=S.operator_scores(arrays[bank+'_check_grams'],w,e.active);cut=float(np.quantile(q,.95));a=g-cut*sigma
        arrays[bank+'_white']=w;arrays[bank+'_check_operator_scores']=q;arrays[bank+'_a']=a;metrics[bank]=dict(info,cutoff=cut)
        regions[bank]=[dict(candidate=c['id'],label=c['label'],slot=c['slot'],registered=c['registered'],**S.quadratic_region(a,g,sigma,c,cut)) for c in e.cs]
    joint=None
    if all(m['status']=='ok' for m in metrics.values()):
        cut=float(np.quantile(np.maximum(arrays['local_check_operator_scores'],arrays['broad_check_operator_scores']),.95))
        joint=dict(cutoff=cut,certificates=[C.J.certificate([arrays[b+'_observed_gram'] for b in e.banks],[arrays[b+'_sigma'] for b in e.banks],cut,c,seed) for c in e.cs])
    frozen=dict(**meta,status='frozen',candidates=e.cs,labels=e.labels,specs=e.specs,active=e.active,deterministic=e.deterministic,
        pilot_sha256={v:R.digest(y) for v,y in e.pilot.items()},innovation_hashes=innovations,array_hashes={k:R.digest(v) for k,v in arrays.items()},metrics=metrics,regions=regions,joint=joint)
    fp=ROOT/'frozen'/path.name;R.dump(fp,frozen);fhash=S.E.sha(fp)
    clean=e.matrices(e.p.data,False);actual={b:[] for b in e.banks};errors={b:[] for b in e.banks};rawhash=[]
    for bank in e.banks:arrays[bank+'_clean']=clean[bank]
    for b in range(256):
        fields=e.true_field(b)
        if b in (0,255):rawhash.append({v:R.digest(y) for v,y in fields.items()})
        for bank,m in e.matrices(fields).items():actual[bank].append(S.gram(m));errors[bank].append(S.gram(m-clean[bank]))
    stats=[]
    for bank in e.banks:
        arrays[bank+'_actual_grams']=np.stack(actual[bank]);arrays[bank+'_actual_error_grams']=np.stack(errors[bank]);info=metrics[bank]
        if info['status']!='ok':continue
        cut=info['cutoff'];sigma=arrays[bank+'_sigma'];q=S.operator_scores(arrays[bank+'_actual_error_grams'],arrays[bank+'_white'],e.active);arrays[bank+'_actual_operator_scores']=q
        for c in e.cs:
            if not c['registered']:continue
            v=S.E.vector(c,c['exact'],len(e.labels));check=np.einsum('i,bij,j->b',v,arrays[bank+'_check_grams'],v);err=np.einsum('i,bij,j->b',v,arrays[bank+'_actual_error_grams'],v)
            residual=np.einsum('i,bij,j->b',v,arrays[bank+'_actual_grams'],v);threshold=float(np.quantile(check,.95));bound=cut*float(v@sigma@v);obs=float(v@arrays[bank+'_observed_gram']@v)
            stats.append(dict(bank=bank,candidate=c['id'],label=c['label'],slot=c['slot'],operator_coverage_count=int(np.sum(q<=cut)),pointwise_noise_coverage_count=int(np.sum(err<=threshold)),
                uniform_noise_coverage_count=int(np.sum(err<=bound)),pointwise_zero_acceptance_count=int(np.sum(residual<=threshold)),uniform_zero_acceptance_count=int(np.sum(residual<=bound)),
                anchor_exact_accepted=bool(obs<=bound),directional_radius=threshold,uniform_bound=bound,anchor_residual=obs,clean_residual=float(np.sum((clean[bank]@v)**2))))
    if joint is not None:
        joint['operator_coverage_count']=int(np.sum(np.maximum(arrays['local_actual_operator_scores'],arrays['broad_actual_operator_scores'])<=joint['cutoff']))
        joint['reference_statistics']=[]
        for c in e.cs:
            if c['label']!='reference':continue
            v=S.E.vector(c,c['exact'],len(e.labels));accept=[];anchor=[]
            for bank in e.banks:
                bound=joint['cutoff']*float(v@arrays[bank+'_sigma']@v);accept.append(np.einsum('i,bij,j->b',v,arrays[bank+'_actual_grams'],v)<=bound);anchor.append(float(v@arrays[bank+'_observed_gram']@v)<=bound)
            joint['reference_statistics'].append(dict(candidate=c['id'],anchor_accepted=all(anchor),actual_acceptance_count=int(np.sum(accept[0]&accept[1]))))
    assert all(np.isfinite(v).all() for v in arrays.values());ap=path.with_suffix('.npz');ap.parent.mkdir(exist_ok=True);np.savez_compressed(ap,**arrays);assert S.E.sha(fp)==fhash
    R.dump(path,dict(**meta,status='COMPLETE',metrics=metrics,regions=regions,joint=joint,statistics=stats,actual_field_hashes=rawhash,frozen_sha256=fhash,array_sha256=S.E.sha(ap),seconds=time.monotonic()-start))
    print(key,'DONE',round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('system');ap.add_argument('seed',type=int);ap.add_argument('profile');ap.add_argument('model');a=ap.parse_args();run(a.system,a.seed,a.profile,a.model)
