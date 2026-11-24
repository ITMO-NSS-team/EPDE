import experiment as E
R,np,ROOT=E.R,E.np,E.ROOT


def main():
    t=np.linspace(0,4,90);x=np.linspace(-1,1,23);tt,xx=np.meshgrid(t,x,indexing='ij');rng=np.random.default_rng(89731)
    clean={'u':1.3+np.sin(tt)*np.cos(xx),'v':.7+np.cos(1.3*tt)*np.sin(2*xx)}
    raw={v:y+.07*rng.standard_normal(y.shape) for v,y in clean.items()}
    pilot,info=R.reconstruct(raw,[t,x],np.arange(len(t)));smooth=E.Smoother(raw,[t,x],info)
    for v,y in smooth(raw).items():np.testing.assert_allclose(y,pilot[v],rtol=2e-10,atol=2e-11)
    const={v:np.full_like(y,2.69) for v,y in raw.items()}
    for y in smooth(const).values():np.testing.assert_allclose(y,2.69,rtol=2e-10,atol=2e-11)
    sc=smooth(clean);ssc=smooth(sc);res={v:clean[v]-sc[v] for v in clean};sr=smooth(res)
    errors=[]
    for v in clean:
        left=clean[v]-(2*sc[v]-ssc[v]);right=res[v]-sr[v]
        np.testing.assert_allclose(left,right,rtol=2e-10,atol=2e-11);errors.append(float(np.max(np.abs(left-right))))
    R.dump(ROOT/'checks.json',dict(status='PASS',multivariable_spatial_smoother_replays=True,constants_preserved=True,
        squared_fixed_smoother_bias_identity_max_error=max(errors),source_sha256=E.sources()))
    print('Multivariable smoother, constants and squared-bias identity PASS')


if __name__=='__main__':main()
