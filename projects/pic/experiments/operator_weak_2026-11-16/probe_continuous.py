import quadrature as Q
F,R,np=Q.F,Q.R,Q.np
import json
import argparse
import time


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['ns','wave','ks','kdv_cossin','pde_compound'])
    args=ap.parse_args();out=[]
    for name in args.systems:
        start=time.monotonic();p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p)
        terms,opaque=R.symbols(p,labels);_,old=F.baseline(name,0)
        previous=json.loads((F.ROOT/'fit_results'/f'{name}_seed0.json').read_text())
        grams=F.evaluation_grams(p,old,terms,opaque,labels)
        rows=[]
        for g in range(3):
            idx=np.array(old['groups'][g]['indices'])
            segment,ax,opq,plan=Q.mapped_plan(old,g,previous['plans'][g]['plan'],axes,opaque)
            bank=Q.Bank(ax,terms,opq,plan=plan)
            for kind,values in [('clean',p.data),('noisy',p.noisy(5,0))]:
                pilot,_=R.reconstruct({v:y[idx] for v,y in values.items()},axes,idx)
                mat=bank.matrix({v:y[segment] for v,y in pilot.items()})
                fits=[R.signed_fit(mat.T@mat,mat,c['xi'],c['yi']) for c in cs]
                scored=F.scores(cs,fits,grams,f'{g}_loo_')
                for c,row in zip(cs,scored):
                    if c['label']=='reference':rows.append(dict(slot=c['slot'],group=g,kind=kind,**row))
        for slot in sorted({r['slot'] for r in rows}):
            print(name,slot,{kind:float(np.mean([r['clean_energy'] for r in rows if r['slot']==slot and r['kind']==kind]))
                            for kind in ('clean','noisy')},flush=True)
        out.append(dict(system=name,rows=rows,seconds=time.monotonic()-start))
        print('seconds',round(time.monotonic()-start,2),flush=True)
    R.dump(Q.ROOT/'continuous_probe.json',out)


if __name__=='__main__':main()
