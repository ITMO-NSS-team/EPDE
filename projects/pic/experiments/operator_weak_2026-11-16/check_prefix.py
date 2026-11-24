"""Independent brute-force check of multidimensional signed-frequency prefixes."""
import run_operator as E
R,np=E.R,E.np


def main():
    rng=np.random.default_rng(32);shape=(5,7);rows=int(np.prod(shape))
    pilot=rng.normal(size=(rows,4))+1j*rng.normal(size=(rows,4))
    variance=rng.uniform(.1,2,size=(rows,4));z=pilot+.2*rng.normal(size=pilot.shape)
    scales=np.array([.3,1.,4.,9.]);training=dict(pilot=pilot,corrected=z,variance=variance)
    mask,meta=E.risk_prefix(training,shape,scales)
    delta=np.sum((2*variance-np.abs(z)**2)/scales,axis=1).reshape(shape)
    ii=np.array([0,1,2,1,2]);jj=np.array([0,1,2,3,1,2,3]);brute=[]
    for i in range(3):
        for j in range(4):
            keep=(ii[:,None]<=i)&(jj[None,:]<=j)
            brute.append((float(delta[keep].sum()),i,j,keep))
    best=min(brute,key=lambda x:x[0]);assert tuple(meta['max_modes'])==best[1:3]
    np.testing.assert_array_equal(mask,best[3].reshape(-1))
    # A physical unit conversion of each operator must preserve its selection.
    change=np.array([1e-3,2.,3e2,4e-2])
    scaled=dict(pilot=pilot*change,corrected=z*change,variance=variance*change**2)
    other,meta2=E.risk_prefix(scaled,shape,scales*change**2)
    np.testing.assert_array_equal(mask,other)
    R.dump(E.Q.ROOT/'prefix_checks.json',dict(brute_force_matches=True,operator_units_invariant=True,
                                           selected_modes=meta['max_modes']))
    print('Brute-force multidimensional prefix and unit invariance passed.')


if __name__=='__main__':main()
