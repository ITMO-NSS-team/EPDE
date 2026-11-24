import study as S
R,np,ROOT=S.R,S.np,S.ROOT
from pathlib import Path
from scipy.optimize import minimize_scalar
import json,argparse,csv


def certificate(grams,sigmas,cut,c,seed):
    xi,yi=c['xi'],c['yi'];p=len(xi)
    norms=[float(g[yi,yi]+cut*s[yi,yi]) for g,s in zip(grams,sigmas)];assert min(norms)>0
    gg=[g/n for g,n in zip(grams,norms)];ss=[s/n for s,n in zip(sigmas,norms)];aa=[g-cut*s for g,s in zip(gg,ss)]
    if p:
        scale=np.sqrt(np.maximum(np.diag(gg[0]+gg[1]+cut*(ss[0]+ss[1]))[xi],0));scale=np.where(scale>0,scale,1.)
        hh=[a[np.ix_(xi,xi)]/scale[:,None]/scale[None,:] for a in aa]
        def eigmin(lam):return float(np.linalg.eigvalsh(lam*hh[0]+(1-lam)*hh[1])[0])
        opt=minimize_scalar(lambda x:-eigmin(x),bounds=(0.,1.),method='bounded',options={'xatol':1e-12})
        grid=np.linspace(0,1,101);values=np.array([eigmin(x) for x in grid]);choices=[0.,1.,float(opt.x),float(grid[np.argmax(values)])]
        lam=max(choices,key=eigmin);top=eigmin(lam);assert top>=values.max()-1e-12
    else:
        lam=float(aa[0][yi,yi]>=aa[1][yi,yi]);top=None;scale=np.array([]);hh=[]
    g=lam*gg[0]+(1-lam)*gg[1];s=lam*ss[0]+(1-lam)*ss[1]
    region=S.quadratic_region(g-cut*s,g,s,c,cut)
    out=dict(candidate=c['id'],label=c['label'],slot=c['slot'],registered=c['registered'],features=p,
        lambda_local=lam,best_minimum_eigenvalue=top,target_normalizers=norms,outer_region=region,
        geometry='bounded_outer' if region['status']=='ellipsoid' else ('empty_certified' if region['status']=='empty' else ('point' if p==0 else 'unresolved')))
    if p==0:return out
    # Any accepted witness is checked in both original constraints.
    if region.get('witness') is not None:
        v=S.E.vector(c,region['witness'],len(g));margin=[float(v@a@v) for a in aa]
        if max(margin)<=0:out['feasible_witness']=region['witness'];out['witness_margins']=margin
    if out['geometry'] in ('bounded_outer','empty_certified'):return out
    rng=np.random.default_rng(np.random.SeedSequence([940105,seed,c['slot'],c['yi'],len(xi)]));dirs=rng.normal(size=(1024,p));dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    eigendirections=[np.linalg.eigh(h)[1][:,0] for h in hh];dirs=np.vstack([*eigendirections,np.eye(p),dirs])
    curv=np.stack([np.einsum('bi,ij,bj->b',dirs,h,dirs) for h in hh]);tol=1e-12*max(np.linalg.norm(h,2) for h in hh)
    valid=np.flatnonzero(np.all(curv<-tol,axis=0))
    if len(valid):
        j=int(valid[np.argmin(np.max(curv[:,valid],axis=0))]);direction=dirs[j]/scale;t=1.
        for a,curvature in zip(aa,curv[:,j]):
            b=float(a[xi,yi]@direction);y=float(a[yi,yi]);t=max(t,4*abs(b)/(-curvature),np.sqrt(4*max(y,0)/(-curvature)))
        witness=2*t*direction;v=S.E.vector(c,witness,len(g));margin=[float(v@a@v) for a in aa];assert max(margin)<0
        out.update(geometry='unbounded_certified',common_negative_direction=direction,curvatures=curv[:,j],feasible_witness=witness,witness_margins=margin)
    return out


def main(cohort):
    files=[p for p in (ROOT/'results').glob('*.json') if ('_iid_iid_' in p.name)==(cohort=='iid')];assert len(files)==(45 if cohort=='iid' else 72)
    dest=ROOT/'intersection';dest.mkdir(exist_ok=True);rows=[];projections=[]
    for path in sorted(files):
        d=json.loads(path.read_text());f=json.loads((ROOT/'frozen'/path.name).read_text());a=np.load(path.with_suffix('.npz'))
        if any(m['status']!='ok' for m in d['metrics'].values()):
            R.dump(dest/path.name,dict(status='incomplete_covariance'));continue
        q=np.maximum(a['local_check_operator_scores'],a['broad_check_operator_scores']);cut=float(np.quantile(q,.95))
        observed=[a[b+'_observed_gram'] for b in ('local','broad')];sigma=[a[b+'_sigma'] for b in ('local','broad')]
        cc=[certificate(observed,sigma,cut,c,d['seed']) for c in f['candidates']]
        # All geometric certificates now fixed; query exact laws afterwards.
        stats=[]
        qa=np.maximum(a['local_actual_operator_scores'],a['broad_actual_operator_scores'])
        for c in f['candidates']:
            if not c['registered']:continue
            v=S.E.vector(c,c['exact'],len(f['labels']));accepted=[];anchor=[]
            for bank,s,g in zip(('local','broad'),sigma,observed):
                bound=cut*float(v@s@v);accepted.append(np.einsum('i,bij,j->b',v,a[bank+'_actual_grams'],v)<=bound);anchor.append(float(v@g@v)<=bound)
            reg=next(r for r in cc if r['candidate']==c['id']);outer=reg['outer_region'];contains=None
            if outer['status']=='ellipsoid':
                lam=reg['lambda_local'];norm=reg['target_normalizers'];aa=lam*(observed[0]-cut*sigma[0])/norm[0]+(1-lam)*(observed[1]-cut*sigma[1])/norm[1]
                contains=bool(v@aa@v<=0)
                for j,(lo,hi,tr) in enumerate(zip(outer['lower'],outer['upper'],c['exact'])):
                    if c['label']=='reference':projections.append(dict(system=d['system'],seed=d['seed'],profile=d['profile'],model_kind=d['model_kind'],candidate=c['id'],coefficient=j,lower=lo,upper=hi,exact=tr))
            stats.append(dict(candidate=c['id'],label=c['label'],joint_exact_anchor_accepted=all(anchor),actual_exact_acceptance_count=int(np.sum(np.logical_and(*accepted))),outer_exact_accepted=contains))
        result=dict(status='COMPLETE',system=d['system'],seed=d['seed'],profile=d['profile'],model_kind=d['model_kind'],cutoff=cut,
            operator_coverage_count=int(np.sum(qa<=cut)),certificates=cc,statistics=stats,
            source_sha256={str(Path(__file__)):S.E.sha(__file__),str(ROOT/'bank_intersection_protocol.txt'):S.E.sha(ROOT/'bank_intersection_protocol.txt')},
            input_sha256={str(p):S.E.sha(p) for p in [path,path.with_suffix('.npz'),ROOT/'frozen'/path.name]})
        R.dump(dest/path.name,result)
        for reg in cc:
            rows.append(dict(system=d['system'],seed=d['seed'],profile=d['profile'],model_kind=d['model_kind'],candidate=reg['candidate'],label=reg['label'],registered=reg['registered'],geometry=reg['geometry'],lambda_local=reg['lambda_local'],
                feasible_witness='feasible_witness' in reg,operator_coverage_count=result['operator_coverage_count']))
    for name,rr in [(f'{cohort}_intersection.csv',rows),(f'{cohort}_intersection_projections.csv',projections)]:
        with (ROOT/name).open('w',newline='',encoding='utf-8') as file:
            w=csv.DictWriter(file,list(rr[0]) if rr else []);w.writeheader();w.writerows(rr)
    from collections import Counter
    print('REFERENCE GEOMETRY',Counter(r['geometry'] for r in rows if r['label']=='reference'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['iid','correlated'],default='iid');args=ap.parse_args();main(args.cohort)
