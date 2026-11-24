"""Post-selection controls; saved clean fields never choose fits or test banks."""
import run_operator as E
Q,F,R,A,np=E.Q,E.F,E.R,E.A,E.np
import json,hashlib


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    output=[]
    for name in ('wave','ac','kdv_cossin','ks','pde_compound'):
        p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
        for seed in range(3):
            path=Q.ROOT/'unit_results'/f'{name}_seed{seed}.json'
            saved=json.loads(path.read_text());_,old=F.baseline(name,seed)
            previous=json.loads((F.ROOT/'fit_results'/f'{name}_seed{seed}.json').read_text())
            frozen=F.evaluation_grams(p,old,terms,opaque,labels)
            clean_groups=[]
            for group in old['groups']:
                idx=np.array(group['indices'])
                clean_groups.append(R.reconstruct({v:y[idx] for v,y in p.data.items()},axes,idx)[0])
            folds=[]
            for g in range(3):
                segment,ax,opq,plan=Q.mapped_plan(old,g,previous['plans'][g]['plan'],axes,opaque)
                local=Q.Bank(ax,terms,opq,plan=plan)
                references=[c for c in cs if c['label']=='reference']
                controls={}
                for tag,fields in [('direct_clean',p.data),('split_clean_reconstruction',clean_groups[g])]:
                    matrix=local.matrix({v:y[segment] for v,y in fields.items()})
                    fits=[R.signed_fit(matrix.T@matrix,matrix,c['xi'],c['yi']) for c in references]
                    controls[tag]=F.scores(references,fits,frozen,f'{g}_loo_')
                    for c,row in zip(references,controls[tag]):
                        x=matrix[:,c['xi']];sc=np.linalg.norm(x,axis=0)
                        sv=np.linalg.svd(x/sc,compute_uv=False)
                        row['normalized_design_singular_values']=sv
                        row['normalized_design_condition']=float(sv[0]/sv[-1])
                freq=[]
                for sel in old['groups'][g]['selection']:
                    k=sel['loo']['max_mode'];freq.append(np.r_[np.arange(k+1),-np.arange(1,k+1)]*sel['spacing'])
                bank=Q.Bank(axes,terms,opaque,frequencies=freq)
                truth=bank.matrix(p.data);recon=[bank.matrix(fields) for fields in clean_groups]
                others=[i for i in range(3) if i!=g];exact=[]
                for c in references:
                    coeff=np.zeros(len(terms));coeff[c['yi']]=-1.;coeff[c['xi']]=c['exact']
                    rr=[x@coeff for x in recon];direct=truth@coeff
                    scale=float(np.sum(np.abs(truth[:,c['yi']])**2))
                    exact.append(dict(id=c['id'],direct_clean_energy=float(np.vdot(direct,direct).real/scale),
                        split_clean_cross=float(np.vdot(rr[others[0]],rr[others[1]]).real/scale),
                        split_clean_ordinary=float(sum(np.vdot(rr[i],rr[i]).real for i in others)/(2*scale))))
                folds.append(dict(training=g,local=controls,exact_full_bank=exact))
            result=dict(system=name,seed=seed,folds=folds,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            output.append(result)
            print(name,seed,'direct / split-clean fit energy / clean-reconstruction exact score',
                *[float(np.mean([f['local'][tag][0]['clean_energy'] for f in folds])) for tag in ('direct_clean','split_clean_reconstruction')],
                float(np.mean([f['exact_full_bank'][0]['split_clean_cross'] for f in folds])),flush=True)
    R.dump(Q.ROOT/'clean_reconstruction_controls.json',output)


if __name__=='__main__':main()
