from common import *
import paired_study as E
from scipy.linalg import eigvalsh
def main():
    files=sorted((ROOT/'attribution').glob('*.json'));assert len(files)==18;scores=matrices=links=0
    for path in files:
        d=read(path)
        for key in ('source_sha256','input_sha256'):
            for p,h in d[key].items():assert sha(p)==h;links+=1
        assert sha(path.with_suffix('.npz'))==d['array_sha256'];r=d['rows'][0];primary=ROOT/'results'/path.name;pd=read(primary);e=E.Study(r['system'],r['seed'],r['profile'])
        true={v:[.05*np.std(y),[e.rho]*y.ndim,[3]*y.ndim] for v,y in e.p.data.items()}
        with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
        with np.load(primary.with_suffix('.npz')) as z:b={k:z[k] for k in z.files}
        for r in d['rows']:
            mode,bank=r['mode'],r['bank'];pre=f'{mode}_{bank}';cut=pd['metrics'][mode][bank]['cutoff'];ss=b[pre+'_sigma'][np.ix_(e.active,e.active)];sc=np.sqrt(np.diag(ss));ss=ss/sc[:,None]/sc[None,:];gg=a[bank+'_oracle_error_grams'][:,e.active,:][:,:,e.active]/sc[None,:,None]/sc[None,None,:]
            direct=np.array([eigvalsh((g+g.T)/2,ss)[-1] for g in gg]);np.testing.assert_allclose(direct,a[pre+'_oracle_scores'],rtol=5e-7,atol=3e-9);assert r['oracle_coverage_count']==np.sum(direct<=cut);scores+=len(direct)
            eig=eigvalsh((gg.mean(axis=0)+gg.mean(axis=0).T)/2,ss);np.testing.assert_allclose([eig[0],eig[-1]],[r['second_moment_min'],r['second_moment_max']],rtol=3e-8,atol=3e-9)
            for tag,field in [('empirical',pre+'_mean_error'),('oracle',bank+'_oracle_mean_error')]:
                score=S.operator_scores(S.gram(a[field])[None],b[pre+'_white'],e.active)[0]/cut;np.testing.assert_allclose(score,r[tag+'_mean_score_cutoff_ratio'],rtol=2e-12,atol=2e-12)
        for j in (0,255):
            for bank,m in e.matrices(e.true_field(j),true).items():
                gram=S.gram(m-b[bank+'_clean']);target=a[bank+'_oracle_error_grams'][j];np.testing.assert_allclose(gram,target,rtol=5e-9,atol=5e-11*max(np.max(np.abs(target)),np.finfo(float).tiny));matrices+=1
    out=dict(status='PASS',records=18,operator_scores=scores,matrix_replays=matrices,hash_links=links,source_sha256=hashes('audit_attribution.py'))
    dump(ROOT/'audit_attribution.json',out);print(out,flush=True)
if __name__=='__main__':main()
