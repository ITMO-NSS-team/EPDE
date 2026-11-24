"""Independent direct-fit and raw-grid separation checks for slab validation."""
from experiment import live, np, full_directory
from assess_swap import read_archive
from slabs import geometry, slab_matrices, refit


if __name__=='__main__':
    p=live.load_data('ac')
    a=read_archive(full_directory('ac',923301,'candidate'))
    plan={k:tuple(v) for k,v in a['plan'].items()}
    axis=1
    edges,small=geometry(p.shape,axis,4,plan)
    terms,opaque=live.R.symbols(p,a['labels'])
    raw=p.noisy(5,923301)
    mats=slab_matrices(p,raw,terms,opaque,axis,edges,small)
    changed={'u':raw['u'].copy()}
    changed['u'][:,edges[1]:edges[2]]+=np.random.default_rng(8001).normal(size=changed['u'][:,edges[1]:edges[2]].shape)
    other=slab_matrices(p,changed,terms,opaque,axis,edges,small)
    for j in (0,2,3):
        np.testing.assert_array_equal(mats[j],other[j])
    assert not np.array_equal(mats[1],other[1])
    rng=np.random.default_rng(101)
    train=rng.normal(size=(80,6))
    train[:,5]=train[:,0]
    targets=np.array([2,3,4])
    xs=np.array([[0,1],[1,2],[0,5]])
    weights,invalid=refit({2:(np.arange(3),targets,xs)},train,train.T@train,3)
    assert invalid.tolist()==[False,False,True]
    for j in range(2):
        direct=np.linalg.lstsq(train[:,xs[j]],train[:,targets[j]],rcond=None)[0]
        np.testing.assert_allclose(weights.toarray()[j,xs[j]],direct,atol=1e-12)
    coarse_edges,coarse=geometry(p.shape,axis,2,plan)
    training=slab_matrices(p,raw,terms,opaque,axis,coarse_edges,coarse)
    changed_training=slab_matrices(p,changed,terms,opaque,axis,coarse_edges,coarse)
    np.testing.assert_array_equal(training[1],changed_training[1])
    print('Direct least squares, singular exclusion, and raw-sample separation at both scales verified.')
