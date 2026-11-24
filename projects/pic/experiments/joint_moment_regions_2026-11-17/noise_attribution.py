import study as S
R,np,ROOT=S.R,S.np,S.ROOT
from pathlib import Path
import json

def slope(order,rho,h):
    def f(r):return R.WF._g(order,r,1)/R.WF._g(order,r,0)
    return (f(rho+h)-f(rho-h))/(2*h)

def main():
    rows=[];inputs={}
    for name in ('ac','kdv_cossin','ks','wave'):
        for seed in (30,31,32):
            for profile in ('positive','negative'):
                e=S.Study(name,seed,profile,'ar');path=ROOT/'frozen'/f'{name}_{profile}_ar_seed{seed}.json';f=json.loads(path.read_text());inputs[str(path)]=S.E.sha(path)
                assert {v:R.digest(y) for v,y in e.raw.items()}==f['raw_sha256']
                for v,y in e.raw.items():
                    pure=y-e.p.data[v];oracle=R.WF.noise_model(pure);truth=.05*float(np.std(e.p.data[v]));raw=e.model[v]
                    pooled=S.amplitude({v:y},{v:e.scales[v]},{v:e.orders[v]},raw[1])*e.scales[v]
                    ss=[]
                    for m in raw[2]:
                        assert m>0;s=slope(m,e.rho,1e-4);ss.append(s);np.testing.assert_allclose(s,slope(m,e.rho,5e-5),rtol=2e-5,atol=1e-9)
                    rows.append(dict(system=name,seed=seed,profile=profile,variable=v,raw_sigma_ratio=raw[0]/truth,
                        pooled_sigma_ratio=pooled/truth,noise_only_sigma_ratio=oracle[0]/truth,raw_rhos=raw[1],raw_orders=raw[2],
                        noise_only_rhos=oracle[1],noise_only_orders=oracle[2],ratio_slopes=ss,order1_slope=slope(1,e.rho,1e-4)))
    summaries=[]
    for profile in ('positive','negative'):
        rr=[r for r in rows if r['profile']==profile];summary=dict(profile=profile)
        for key in ('raw_sigma_ratio','pooled_sigma_ratio','noise_only_sigma_ratio','raw_rhos','noise_only_rhos','raw_orders','noise_only_orders','ratio_slopes'):
            x=np.asarray([r[key] for r in rr]).flatten();summary[key+'_min_median_max']=np.quantile(x,[0,.5,1]).tolist()
        summaries.append(summary)
    R.dump(ROOT/'noise_attribution.json',dict(status='COMPLETE',rows=rows,summary=summaries,source_sha256={**S.source_hashes(),str(Path(__file__)):S.E.sha(__file__),str(ROOT/'noise_attribution_protocol.txt'):S.E.sha(ROOT/'noise_attribution_protocol.txt')},input_sha256=inputs))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
