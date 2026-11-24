import iid_studentization as I
E,np,ROOT=I.E,I.np,I.ROOT
from math import comb

def main():
    rng=np.random.default_rng(171125);n=320;k=3;noise=rng.normal(size=(32768,n))
    def amplitudes(z):return np.sqrt(np.mean(np.diff(z,n=k,axis=-1)**2,axis=-1)/comb(2*k,k))
    sig=amplitudes(noise);assert abs(float(np.mean(sig**2))-1)<.01
    for row,expected in zip(noise[:32],sig[:32]):
        got=I.amplitude({'u':row},{'u':1.},{'u':[k]});np.testing.assert_allclose(got,expected,rtol=2e-15)
        scaled=I.amplitude({'u':-17*row},{'u':17.},{'u':[k]});np.testing.assert_allclose(got,scaled,rtol=2e-15)
    t=np.linspace(0,1,n);basis=np.stack([np.sin(j*np.pi*t) for j in range(1,8)])
    basis/=np.linalg.norm(basis,axis=1,keepdims=True)
    cal=rng.normal(size=(256,n));check=rng.normal(size=(4096,n));actual=noise
    zcal=(cal@basis.T)/amplitudes(cal)[:,None];var=zcal.var(axis=0,ddof=1)
    qb=E.score((check@basis.T)/amplitudes(check)[:,None],var);qa=E.score((actual@basis.T)/sig[:,None],var)
    cov=float(np.mean(qa<=np.quantile(qb,.95)));assert abs(cov-.95)<.015
    E.R.dump(ROOT/'iid_checks.json',dict(status='PASS',noise_variance_mean=float(np.mean(sig**2)),scalar_amplitude_replay=True,
        units_invariant=True,studentized_linear_gaussian_coverage=cov,
        source_sha256={__file__:E.sha(__file__),I.__file__:E.sha(I.__file__)}))
    print('iid/studentization checks PASS',cov)

if __name__=='__main__':main()
