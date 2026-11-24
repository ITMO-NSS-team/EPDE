from context import *
from scipy.linalg import eigvalsh

def main():
    e,d,path=C.primary('kdv_cossin','iid',51);prior=read(PREVIOUS/'design_screen.json');specs=prior['specs']
    with np.load(PREVIOUS/'design_screen.npz') as z:a={k:z[k] for k in z.files}
    rows=[]
    for bank in specs:
        m=a[bank+'_pilot'];cv=a[bank+'_covariance'];details=[]
        for index,c in enumerate(e.cs):
            st=[j for j in c['xi'] if j in e.active];de=[j for j in c['xi'] if j in e.deterministic]
            if not st:continue
            x=m[:,st]
            if de:x=x-m[:,de]@np.linalg.lstsq(m[:,de],x,rcond=1e-12)[0]
            g=S.gram(x);v=cv[np.ix_(st,st)];sc=np.sqrt(np.diag(v));g=g/sc[:,None]/sc;v=v/sc[:,None]/sc
            vals=eigvalsh((g+g.T)/2,(v+v.T)/2);assert vals[0]>=-1e-7*max(1,abs(vals[-1]))
            details.append(dict(candidate_index=index,stochastic_features=st,deterministic_features=de,minimum=float(vals[0]),log_minimum=float(np.log1p(max(0,vals[0])))))
        q=[r['log_minimum'] for r in details];rows.append(dict(bank=bank,score=float(np.quantile(q,.25)),mean=float(np.mean(q)),supports=details))
    rank=sorted((r for r in rows if r['bank']!='local'),key=lambda r:(-r['score'],-r['mean']));out=dict(status='FROZEN',selected=rank[0]['bank'],ranking=[r['bank'] for r in rank],rows=rows,specs=specs,
        source_sha256=sources('screen.py'),input_sha256={str(PREVIOUS/'design_screen.json'):sha(PREVIOUS/'design_screen.json'),str(PREVIOUS/'design_screen.npz'):sha(PREVIOUS/'design_screen.npz')})
    dump(ROOT/'design_screen.json',out);print('FROZEN',out['selected'],[(r['bank'],r['score']) for r in rows],flush=True)

if __name__=='__main__':main()
