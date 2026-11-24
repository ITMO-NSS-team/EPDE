from common import *
from fast_covariance import Geometry,Covariance
def main():
    errors=[]
    for name in ('ac','ks','ode','ns','pde_compound'):
        p=R.datasets.load(name);axes=R.axes_of(p);_,labels=R.panel(p);terms,opaque=R.symbols(p,labels);geometry=Geometry(axes);base,old=S.E.F.baseline(name,0)
        banks=[N.Bank(axes,terms,opaque,N.spec_from_plan(axes,old['base_plan'])),N.Bank(axes,terms,opaque,N.common_spec([axes],terms,old['base_plan']['q'],k=2))]
        for rho in (-.99999,.99999):
            model={v:[100.,[rho]*len(axes),[3]*len(axes)] for v in p.data};a=O.Covariance(axes,np.eye(len(axes[0])),model);b=Covariance(geometry,model)
            for bank in banks:
                x=O.matrix(bank,p.data,a);y=O.matrix(bank,p.data,b);scale=max(float(np.max(np.abs(x))),np.finfo(float).tiny);np.testing.assert_allclose(x,y,rtol=2e-8,atol=2e-10*scale);errors.append(float(np.max(np.abs(x-y)))/scale)
    dump(ROOT/'fast_tail_checks.json',dict(status='PASS',matrix_pairs=len(errors),max_relative_to_matrix_scale=max(errors),source_sha256=hashes('fast_tail_checks.py','fast_covariance.py')));print('PASS',len(errors),max(errors),flush=True)
if __name__=='__main__':main()
