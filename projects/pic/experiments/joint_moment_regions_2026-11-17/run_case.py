import study as S
R,np,ROOT=S.R,S.np,S.ROOT
import argparse,time,json


def run(name,seed,profile,kind):
    start=time.monotonic();e=S.Study(name,seed,profile,kind);observed=e.matrices(e.raw);pilot=e.matrices(e.pilot,False)
    key=f'{name}_{profile}_{kind}_seed{seed}';arrays={};stages={};innovations={}
    for bank in e.banks:arrays[bank+'_observed']=observed[bank];arrays[bank+'_pilot']=pilot[bank]
    for tag,domain in [('calibration',940101),('check',940102)]:
        rng=np.random.default_rng(np.random.SeedSequence([domain,e.profile_id,seed]));stacks={bank:[] for bank in e.banks};hh=[]
        for b in range(256):
            field,noise=e.draw(rng)
            if b in (0,255):hh.append({v:R.digest(y) for v,y in noise.items()})
            for bank,mat in e.matrices(field).items():
                error=mat-pilot[bank];assert np.all(error[:,e.deterministic]==0)
                stacks[bank].append(S.gram(error))
        for bank,ss in stacks.items():arrays[bank+'_'+tag+'_grams']=np.stack(ss)
        innovations[tag]=hh
    regions={};metrics={}
    for bank in e.banks:
        sigma=arrays[bank+'_calibration_grams'].mean(axis=0);w,info=S.metric(sigma,e.active);g=S.gram(observed[bank])
        arrays[bank+'_sigma']=sigma;arrays[bank+'_observed_gram']=g
        if w is None:metrics[bank]=info;regions[bank]=[];continue
        q=S.operator_scores(arrays[bank+'_check_grams'],w,e.active);cut=float(np.quantile(q,.95));a=g-cut*sigma
        arrays[bank+'_white']=w;arrays[bank+'_check_operator_scores']=q;arrays[bank+'_a']=a
        metrics[bank]=dict(info,cutoff=cut);regions[bank]=[dict(candidate=c['id'],label=c['label'],slot=c['slot'],registered=c['registered'],
            features=len(c['xi']),**S.quadratic_region(a,g,sigma,c,cut)) for c in e.cs]
    fp=ROOT/'frozen'/f'{key}.json'
    frozen=dict(system=name,seed=seed,profile=profile,model_kind=kind,candidates=e.cs,labels=e.labels,specs=e.specs,model=e.model,
        ar_estimates=e.ar,orders=e.orders,scales=e.scales,relative_amplitude=e.delta,active=e.active,deterministic=e.deterministic,
        raw_sha256={v:R.digest(y) for v,y in e.raw.items()},pilot_sha256={v:R.digest(y) for v,y in e.pilot.items()},
        metrics=metrics,regions=regions,innovation_hashes=innovations,array_hashes={k:R.digest(v) for k,v in arrays.items()},
        source_sha256=S.source_hashes(),input_sha256=e.inputs)
    R.dump(fp,frozen);fh=S.E.sha(fp)
    clean=e.matrices(e.p.data,False)
    for bank in e.banks:arrays[bank+'_clean']=clean[bank]
    actual={bank:[] for bank in e.banks};errors={bank:[] for bank in e.banks};rawhash=[]
    for b in range(256):
        fields=e.true_field(b)
        if b in (0,255):rawhash.append({v:R.digest(y) for v,y in fields.items()})
        for bank,mat in e.matrices(fields).items():actual[bank].append(S.gram(mat));errors[bank].append(S.gram(mat-clean[bank]))
    stats=[]
    for bank in e.banks:
        arrays[bank+'_actual_grams']=np.stack(actual[bank]);arrays[bank+'_actual_error_grams']=np.stack(errors[bank])
        info=metrics[bank]
        if info['status']!='ok':continue
        cut=info['cutoff'];sigma=arrays[bank+'_sigma'];q=S.operator_scores(arrays[bank+'_actual_error_grams'],arrays[bank+'_white'],e.active)
        arrays[bank+'_actual_operator_scores']=q
        for c in e.cs:
            if not c['registered']:continue
            v=S.E.vector(c,c['exact'],len(e.labels));den=float(v@sigma@v)
            check=np.einsum('i,bij,j->b',v,arrays[bank+'_check_grams'],v);err=np.einsum('i,bij,j->b',v,arrays[bank+'_actual_error_grams'],v)
            residual=np.einsum('i,bij,j->b',v,arrays[bank+'_actual_grams'],v);obs=float(v@arrays[bank+'_observed_gram']@v);clean_res=float(np.sum((clean[bank]@v)**2))
            threshold=float(np.quantile(check,.95));bound=cut*den
            stats.append(dict(bank=bank,candidate=c['id'],label=c['label'],slot=c['slot'],operator_coverage_count=int(np.sum(q<=cut)),
                pointwise_noise_coverage_count=int(np.sum(err<=threshold)),uniform_noise_coverage_count=int(np.sum(err<=bound)),
                pointwise_zero_acceptance_count=int(np.sum(residual<=threshold)),uniform_zero_acceptance_count=int(np.sum(residual<=bound)),
                anchor_exact_accepted=bool(obs<=bound),directional_radius=threshold,uniform_bound=bound,anchor_residual=obs,clean_residual=clean_res))
    assert all(np.isfinite(v).all() for v in arrays.values());ap=ROOT/'results'/f'{key}.npz';ap.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(ap,**arrays)
    assert S.E.sha(fp)==fh
    R.dump(ap.with_suffix('.json'),dict(system=name,seed=seed,profile=profile,model_kind=kind,metrics=metrics,regions=regions,statistics=stats,
        actual_field_hashes=rawhash,source_sha256=S.source_hashes(),input_sha256=e.inputs,frozen_sha256=fh,array_sha256=S.E.sha(ap),seconds=time.monotonic()-start))
    print(key,'DONE',round(time.monotonic()-start,1),'s',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('system');ap.add_argument('seed',type=int);ap.add_argument('profile');ap.add_argument('model');a=ap.parse_args();run(a.system,a.seed,a.profile,a.model)
