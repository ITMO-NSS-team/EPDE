from common import *
from fast_covariance import Geometry,Covariance
def main():
    cases=0;maxrel=0.
    for name in SYSTEMS:
        p=R.datasets.load(name);axes=R.axes_of(p);_,labels=R.panel(p);terms,opaque=R.symbols(p,labels);geometry=Geometry(axes)
        base,old=S.E.F.baseline(name,0);banks={'local':N.Bank(axes,terms,opaque,N.spec_from_plan(axes,old['base_plan'])),'broad':N.Bank(axes,terms,opaque,N.common_spec([axes],terms,old['base_plan']['q'],k=2))}
        for rho in (-.4,0.,.4,.98):
            model={v:[.1,[rho]*len(axes),[3]*len(axes)] for v in p.data};original=O.Covariance(axes,np.eye(len(axes[0])),model);fast=Covariance(geometry,model)
            for bank in banks.values():
                a=O.matrix(bank,p.data,original);b=O.matrix(bank,p.data,fast);scale=max(float(np.max(np.abs(a))),np.finfo(float).tiny)
                np.testing.assert_allclose(b,a,rtol=2e-10,atol=2e-12*scale);maxrel=max(maxrel,float(np.max(np.abs(a-b)))/scale);cases+=1
    dump(ROOT/'fast_checks.json',dict(status='PASS',matrix_pairs=cases,max_relative_to_matrix_scale=maxrel,source_sha256=hashes('fast_checks.py','fast_covariance.py')));print('PASS',cases,maxrel,flush=True)
if __name__=='__main__':main()
