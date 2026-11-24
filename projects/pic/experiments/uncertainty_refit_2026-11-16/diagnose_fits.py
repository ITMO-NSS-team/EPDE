"""Post-hoc replay of median/worst reference fits, without changing methods."""
import experiment as E
R,np,ROOT=E.R,E.np,E.ROOT
import json,time


def run(name):
    seed=18;path=ROOT/'results'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'))
    e=E.Experiment(name,seed);ref=next(j for j,c in enumerate(e.cs) if c['label']=='reference' and c['slot']==0);c=e.cs[ref]
    order=np.argsort(a['benchmark_energy'][:,ref]);chosen=[('median_draw',int(order[len(order)//2])),('worst_draw',int(order[-1]))]
    events=[('anchor',e.raw,a['anchor_vectors'],True,None),('clean_center',e.p.data,a['clean_center_vectors'],False,None)]
    for tag,b in chosen:events.append((tag,e.p.noisy(5,d['actual_noise_seeds'][b]),a['benchmark_vectors'][b],True,b))
    rows=[]
    for tag,fields,expected,correct,b in events:
        for g,(idx,st) in enumerate(zip(e.indices,e.states)):
            pp,_=R.reconstruct({v:y[idx] for v,y in fields.items()},e.axes,idx)
            banktag,kind=e.choices[g][ref].rsplit('_',1);bank=st['banks'][banktag];seg=st['segment']
            m=bank.matrix({v:y[seg] for v,y in pp.items()});corr=st['corrections'][banktag]
            gram=E.J.grams(m-corr['bias'])-corr['gram'] if correct and kind=='clsm' else E.J.grams(m)
            coef,info=R.signed_fit(gram,m,c['xi'],c['yi']);np.testing.assert_allclose(coef,-expected[g,c['xi'],ref],rtol=5e-8,atol=2e-8)
            ss=np.linalg.norm(m[:,c['xi']],axis=0);ss=np.where(ss>0,ss,1.)
            gg=gram[np.ix_(c['xi'],c['xi'])]/ss[:,None]/ss[None,:];eig=np.linalg.eigvalsh((gg+gg.T)/2)
            rows.append(dict(event=tag,draw=b,fold=g,selected=banktag+'_'+kind,coefficients=coef,exact=c['exact'],
                eigenvalues=eig,minimum_absolute_eigenvalue=float(np.min(np.abs(eig))),
                normalized_inverse_sensitivity=float(np.max(np.abs(eig))/np.min(np.abs(eig))),**info))
    out=dict(system=name,seed=seed,scope='Post-hoc median/worst reference-energy events; no selection rule changes.',rows=rows,
        anchor_energy=float(a['anchor_energy'][ref]),benchmark_energy_quantiles=np.quantile(a['benchmark_energy'][:,ref],[0,.5,.9,.99,1]),
        source_sha256={__file__:E.S.sha(__file__),E.__file__:E.S.sha(E.__file__)},input_sha256={str(path):E.S.sha(path),str(path.with_suffix('.npz')):E.S.sha(path.with_suffix('.npz'))})
    R.dump(ROOT/'fit_diagnostics'/f'{name}.json',out);print(name,'fit replay PASS',round(time.monotonic()-e.start,1),'s',flush=True)


if __name__=='__main__':
    for name in ('ac','kdv_cossin','ks','wave'):run(name)
