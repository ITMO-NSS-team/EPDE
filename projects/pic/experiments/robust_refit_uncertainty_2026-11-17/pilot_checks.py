from common import *
import correlated_pilot as P

def main():
    rows=[];rng=np.random.default_rng(983001)
    for shape in [(31,),(51,32),(41,24,21)]:
        axes=[np.linspace(0,1,n) for n in shape];base=rng.normal(size=shape);y={'u':base};zero={'u':[1.,[0.]*len(shape),[]]};a,ai=R.reconstruct(y,axes,np.arange(shape[0]));b,bi=P.reconstruct(y,axes,zero);err=float(np.max(np.abs(a['u']-b['u'])));np.testing.assert_allclose(a['u'],b['u'],rtol=2e-7,atol=2e-9)
        for rho in (.4,-.4):
            m={'u':[1.,[rho]*len(shape),[]]};c,_=P.reconstruct(y,axes,m);trend=(2+3*axes[0]).reshape((shape[0],)+(1,)*(len(shape)-1));e,_=P.reconstruct({'u':-2*base+trend},axes,{'u':[2.,m['u'][1],[]]});np.testing.assert_allclose(e['u'],-2*c['u']+trend,rtol=1e-5,atol=2e-5)
        rows.append(dict(shape=shape,iid_max_abs_error=err))
    dump(ROOT/'pilot_checks.json',dict(status='PASS',rows=rows,source_sha256=hashes('pilot_checks.py','correlated_pilot.py','pilot_protocol.txt')));print(rows)
if __name__=='__main__':main()
