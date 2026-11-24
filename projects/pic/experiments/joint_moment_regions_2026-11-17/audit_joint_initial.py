import study as S
R,np,ROOT=S.R,S.np,S.ROOT
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.linalg import eigvalsh
import json,importlib.util

# The historical helpers insert other experiment directories into sys.path.
spec=importlib.util.spec_from_file_location('joint_intersection',ROOT/'intersect_banks.py')
J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)

@lru_cache(None)
def sha(path):return S.E.sha(path)

def hashes(d):
    count=0
    for key in ('source_sha256','input_sha256'):
        for p,h in d.get(key,{}).items():assert sha(p)==h,(key,p);count+=1
    return count

def close(x,y):np.testing.assert_allclose(x,y,rtol=3e-8,atol=3e-10)

def region_check(reg,a,g,s,c,cut):
    replay=S.quadratic_region(a,g,s,c,cut);assert reg['status']==replay['status']
    for key in ('center','lower','upper','negative_direction','witness','minimum'):
        if reg.get(key) is not None:close(reg[key],replay[key])
    xi,yi=c['xi'],c['yi'];p=len(xi)
    if reg.get('witness') is not None:
        v=S.E.vector(c,reg['witness'],len(a));assert v@a@v<=1e-8*(abs(v@g@v)+abs(cut*v@s@v)+1e-300)
    if reg['status']=='unbounded':
        direction=np.asarray(reg['negative_direction']);assert direction@a[np.ix_(xi,xi)]@direction<0
    if reg['status']=='ellipsoid':
        # Independently verify exact coordinate extrema on the quadric.
        scale=np.asarray(reg['scale']);h=a[np.ix_(xi,xi)]/scale[:,None]/scale[None,:]
        inverse=np.linalg.inv(h);center=np.asarray(reg['center']);minimum=float(reg['minimum'])
        assert np.linalg.eigvalsh(h)[0]>0 and minimum<0
        for j in range(p):
            shift=np.sqrt(-minimum/inverse[j,j])*inverse[:,j]/scale
            for sign,key in ((-1,'lower'),(1,'upper')):
                beta=center+sign*shift;close(beta[j],reg[key][j]);v=S.E.vector(c,beta,len(a))
                assert abs(v@a@v)<=2e-7*(abs(v@g@v)+abs(cut*v@s@v)+1e-300)
    if reg['status']=='empty' and p:
        h=a[np.ix_(xi,xi)];scale=np.asarray(reg['scale']);assert np.linalg.eigvalsh(h/scale[:,None]/scale[None,:])[0]>0
        assert reg['minimum']>0

def one(path):
    path=Path(path);d=json.loads(path.read_text());fp=ROOT/'frozen'/path.name;f=json.loads(fp.read_text())
    counts=dict(hash_links=hashes(d)+hashes(f),operator_scores=0,query_scores=0,regions=0,nested_pairs=0,matrix_replays=0,intersection_certificates=0)
    assert sha(fp)==d['frozen_sha256'] and sha(path.with_suffix('.npz'))==d['array_sha256'];counts['hash_links']+=2
    with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    assert all(np.isfinite(v).all() for v in a.values())
    for k,h in f['array_hashes'].items():assert R.digest(a[k])==h;counts['hash_links']+=1
    cs=f['candidates'];active=f['active'];det=f['deterministic']
    for bank in ('local','broad'):
        m=d['metrics'][bank];assert m['status']=='ok';cut=m['cutoff'];g=a[bank+'_observed_gram'];s=a[bank+'_sigma'];aa=g-cut*s
        close(g,S.gram(a[bank+'_observed']));close(s,a[bank+'_calibration_grams'].mean(axis=0));close(aa,a[bank+'_a'])
        sub=s[np.ix_(active,active)];scale=np.sqrt(np.diag(sub));sub=sub/scale[:,None]/scale[None,:]
        for tag in ('check','actual_error'):
            gg=a[bank+'_'+tag+'_grams'];ss=gg[:,active,:][:,:,active]/scale[None,:,None]/scale[None,None,:]
            direct=np.array([eigvalsh((h+h.T)/2,sub)[-1] for h in ss])
            stored=a[bank+('_check_operator_scores' if tag=='check' else '_actual_operator_scores')]
            close(direct,stored);counts['operator_scores']+=len(direct)
            assert np.all(gg[:,det,:]==0) and np.all(gg[:,:,det]==0)
        close(cut,np.quantile(a[bank+'_check_operator_scores'],.95))
        for c,reg in zip(cs,d['regions'][bank]):
            assert c['id']==reg['candidate'];region_check(reg,aa,g,s,c,cut);counts['regions']+=1
            if reg.get('witness') is None:continue
            v=S.E.vector(c,reg['witness'],len(g))
            for other in cs:
                if other['yi']!=c['yi'] or not set(c['xi'])<set(other['xi']):continue
                beta=[reg['witness'][c['xi'].index(j)] if j in c['xi'] else 0. for j in other['xi']]
                np.testing.assert_array_equal(S.E.vector(other,beta,len(g)),v);counts['nested_pairs']+=1
                larger=next(r for r in d['regions'][bank] if r['candidate']==other['id']);assert larger['feasible'] is not False
        for row in [r for r in d['statistics'] if r['bank']==bank]:
            c=next(c for c in cs if c['id']==row['candidate']);v=S.E.vector(c,c['exact'],len(g))
            check=np.einsum('i,bij,j->b',v,a[bank+'_check_grams'],v);actual=np.einsum('i,bij,j->b',v,a[bank+'_actual_error_grams'],v)
            zero=np.einsum('i,bij,j->b',v,a[bank+'_actual_grams'],v);threshold=np.quantile(check,.95);bound=cut*(v@s@v)
            close(threshold,row['directional_radius']);close(bound,row['uniform_bound']);close(v@g@v,row['anchor_residual'])
            close(np.sum((a[bank+'_clean']@v)**2),row['clean_residual'])
            assert row['pointwise_noise_coverage_count']==np.sum(actual<=threshold)
            assert row['uniform_noise_coverage_count']==np.sum(actual<=bound)
            assert row['pointwise_zero_acceptance_count']==np.sum(zero<=threshold)
            assert row['uniform_zero_acceptance_count']==np.sum(zero<=bound)
            assert row['operator_coverage_count']==np.sum(a[bank+'_actual_operator_scores']<=cut)
            assert row['anchor_exact_accepted']==bool(v@g@v<=bound);counts['query_scores']+=770
    inter=json.loads((ROOT/'intersection'/path.name).read_text());assert inter['status']=='COMPLETE';counts['hash_links']+=hashes(inter)
    cut=np.quantile(np.maximum(a['local_check_operator_scores'],a['broad_check_operator_scores']),.95);close(cut,inter['cutoff'])
    assert inter['operator_coverage_count']==np.sum(np.maximum(a['local_actual_operator_scores'],a['broad_actual_operator_scores'])<=cut)
    grams=[a[b+'_observed_gram'] for b in ('local','broad')];sigmas=[a[b+'_sigma'] for b in ('local','broad')]
    for c,cert in zip(cs,inter['certificates']):
        assert c['id']==cert['candidate'];replay=J.certificate(grams,sigmas,cut,c,d['seed']);assert cert['geometry']==replay['geometry'];close(cert['lambda_local'],replay['lambda_local'])
        lam=cert['lambda_local'];assert 0<=lam<=1;norm=np.asarray(cert['target_normalizers']);close(norm,[g[c['yi'],c['yi']]+cut*s[c['yi'],c['yi']] for g,s in zip(grams,sigmas)])
        gg=lam*grams[0]/norm[0]+(1-lam)*grams[1]/norm[1];ss=lam*sigmas[0]/norm[0]+(1-lam)*sigmas[1]/norm[1]
        region_check(cert['outer_region'],gg-cut*ss,gg,ss,c,cut)
        aa=[(g-cut*s)/n for g,s,n in zip(grams,sigmas,norm)]
        if cert.get('feasible_witness') is not None:
            v=S.E.vector(c,cert['feasible_witness'],len(gg));margins=[float(v@q@v) for q in aa];assert max(margins)<=0;close(margins,cert['witness_margins'])
        if cert['geometry']=='unbounded_certified':
            direction=np.asarray(cert['common_negative_direction']);curv=[float(direction@q[np.ix_(c['xi'],c['xi'])]@direction) for q in aa]
            assert max(curv)<0;close(curv,cert['curvatures'])
        counts['intersection_certificates']+=1
    for row in inter['statistics']:
        c=next(c for c in cs if c['id']==row['candidate']);v=S.E.vector(c,c['exact'],len(grams[0]));accept=[];anchor=[]
        for bank,g,s in zip(('local','broad'),grams,sigmas):
            bound=cut*(v@s@v);accept.append(np.einsum('i,bij,j->b',v,a[bank+'_actual_grams'],v)<=bound);anchor.append(v@g@v<=bound)
        assert row['actual_exact_acceptance_count']==np.sum(accept[0]&accept[1]);assert row['joint_exact_anchor_accepted']==bool(all(anchor))
    e=S.Study(d['system'],d['seed'],d['profile'],d['model_kind'])
    assert {v:R.digest(y) for v,y in e.raw.items()}==f['raw_sha256'];assert {v:R.digest(y) for v,y in e.pilot.items()}==f['pilot_sha256']
    close(e.delta,f['relative_amplitude']) if e.delta is not None else None
    for tag,field,correct in [('observed',e.raw,True),('pilot',e.pilot,False),('clean',e.p.data,False)]:
        for bank,mat in e.matrices(field,correct).items():np.testing.assert_array_equal(mat,a[bank+'_'+tag]);counts['matrix_replays']+=1
    for tag,domain in [('calibration',940101),('check',940102)]:
        rng=np.random.default_rng(np.random.SeedSequence([domain,e.profile_id,e.seed]));hh=[]
        for b in range(256):
            noise={v:R.WF.ar_noise(rng,y.shape,*e.model[v][:2]) for v,y in e.raw.items()}
            if b in (0,255):
                hh.append({v:R.digest(y) for v,y in noise.items()})
                for bank,mat in e.matrices({v:e.pilot[v]+y for v,y in noise.items()}).items():
                    np.testing.assert_array_equal(S.gram(mat-a[bank+'_pilot']),a[bank+'_'+tag+'_grams'][b]);counts['matrix_replays']+=1
        assert hh==f['innovation_hashes'][tag]
    for j,b in enumerate((0,255)):
        fields=e.true_field(b);assert {v:R.digest(y) for v,y in fields.items()}==d['actual_field_hashes'][j]
        for bank,mat in e.matrices(fields).items():
            np.testing.assert_array_equal(S.gram(mat),a[bank+'_actual_grams'][b]);np.testing.assert_array_equal(S.gram(mat-a[bank+'_clean']),a[bank+'_actual_error_grams'][b]);counts['matrix_replays']+=1
    return dict(record=path.stem,status='PASS',**counts)

def main():
    checks=json.loads((ROOT/'checks.json').read_text());assert checks['status']=='PASS';hashes(checks)
    expected=set()
    for cohort,n in [('iid',45),('correlated',72)]:
        rr=json.loads((ROOT/(cohort+'_processes.json')).read_text());assert len(rr)==n and all(r['exit_code']==0 for r in rr)
        for r in rr:expected.add(f"{r['system']}_{r['profile']}_{r['model']}_seed{r['seed']}")
    files=sorted((ROOT/'results').glob('*.json'));assert {p.stem for p in files}==expected and len(files)==117
    # Check paired controls really use the same observed anchor/pilot/actual fields.
    for p in files:
        if '_known_rho_' not in p.name:continue
        base=json.loads(p.read_text());fb=json.loads((ROOT/'frozen'/p.name).read_text())
        for kind in ('iid','ar'):
            name=p.name.replace('_known_rho_','_'+kind+'_');d=json.loads((ROOT/'results'/name).read_text());f=json.loads((ROOT/'frozen'/name).read_text())
            assert base['actual_field_hashes']==d['actual_field_hashes'];assert fb['raw_sha256']==f['raw_sha256'] and fb['pilot_sha256']==f['pilot_sha256']
    rows=[]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for future in as_completed([pool.submit(one,str(p)) for p in files]):
            row=future.result();rows.append(row);print(row['record'],'PASS',flush=True)
    totals={k:sum(r[k] for r in rows) for k in rows[0] if k not in ('record','status')}
    out=dict(status='PASS',records=len(rows),totals=totals,rows=rows,paired_fields_identical=True,
        source_sha256={str(Path(__file__)):S.E.sha(__file__),str(ROOT/'analyze.py'):S.E.sha(ROOT/'analyze.py')})
    R.dump(ROOT/'audit_joint.json',out);print(json.dumps(totals,indent=2),flush=True)

if __name__=='__main__':main()
