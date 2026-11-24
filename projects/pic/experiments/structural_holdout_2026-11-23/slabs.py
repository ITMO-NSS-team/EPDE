"""Leave one raw slab out, rotating over scales and axes of the observed grid."""
from collections import defaultdict
from experiment import live, np
from epde.operators.common.survival import _block_rows
from scipy.sparse import csr_matrix


def geometry(lengths, axis, count, original):
    boundaries = np.linspace(0,lengths[axis],count+1,dtype=int)
    sizes = np.array(lengths)
    sizes[axis] = np.diff(boundaries).min()
    effective = sizes-2*np.array(original['cut'])
    radius = list(original['ms'])
    radius[axis] = min(radius[axis],(effective[axis]-1)//4)
    if radius[axis]<original['q'][axis]:
        return None
    stride = list(original['ss'])
    stride[axis] = min(stride[axis],max(1,radius[axis]//2))
    floor = _block_rows(7)
    while np.prod(live.R.WF.lattice_shape(effective,radius,stride))<floor:
        j=int(np.argmax(stride))
        if stride[j]==1:
            return None
        stride[j]=max(1,stride[j]//2)
    return boundaries, dict(original,ms=tuple(radius),ss=tuple(stride))


def slab_matrices(problem, raw, terms, opaque, axis, edges, plan):
    axes=live.R.axes_of(problem)
    matrices=[]
    for start,end in zip(edges[:-1],edges[1:]):
        cut=tuple(slice(start,end) if j==axis else slice(None) for j in range(len(axes)))
        subset=[a[s] for a,s in zip(axes,cut)]
        effective=np.array([len(a) for a in subset])-2*np.array(plan['cut'])
        shape=live.R.WF.lattice_shape(effective,plan['ms'],plan['ss'])
        bank=live.R.LocalBank(subset,terms,{k:v[cut] for k,v in opaque.items()},dict(plan,shape=shape))
        mat=bank.matrix({k:v[cut] for k,v in raw.items()})
        np.testing.assert_allclose(mat[:,0],1.,atol=1e-12)
        mat[:,0]=1.
        assert len(mat)>=_block_rows(7)
        matrices.append(mat)
    return matrices


def wape(matrix, coefficients, pivots):
    magnitudes=np.abs(matrix).mean(axis=0)[pivots]
    residuals=np.empty(len(pivots))
    batch=max(1, 4_000_000//len(matrix))
    for start in range(0,len(residuals),batch):
        chunk=slice(start,start+batch)
        residuals[chunk]=np.abs(coefficients[chunk] @ matrix.T).mean(axis=1)
    return np.divide(residuals,magnitudes,out=np.full_like(magnitudes,np.inf),where=magnitudes>0)


def refit(groups, training, gram, design_count):
    rows,cols,vals=[],[],[]
    invalid=np.zeros(design_count,dtype=bool)
    for width,(ids,targets,x) in groups.items():
        rows.extend(ids)
        cols.extend(targets)
        vals.extend([-1.]*len(ids))
        if width==0:
            continue
        g=gram[x[:,:,None],x[:,None,:]]
        scales=np.sqrt(np.maximum(np.diagonal(g,axis1=1,axis2=2),0.))
        scales=np.where(scales>0,scales,1.)
        normalized=g/scales[:,:,None]/scales[:,None,:]
        rhs=gram[x,targets[:,None]]/scales
        good=np.linalg.eigvalsh(normalized)[:,0]>1e-8
        weights=np.zeros_like(rhs)
        weights[good]=np.linalg.solve(normalized[good],rhs[good,:,None])[:,:,0]/scales[good]
        for k in np.flatnonzero(~good):
            features=training[:,x[k]]/scales[k]
            w,_,rank,_=np.linalg.lstsq(features,training[:,targets[k]],rcond=None)
            if rank==width:
                weights[k]=w/scales[k]
            else:
                invalid[ids[k]]=True
        rows.extend(np.repeat(ids,width))
        cols.extend(x.ravel())
        vals.extend(weights.ravel())
    return csr_matrix((vals,(rows,cols)),shape=(design_count,training.shape[1])),invalid


def slab_scores(problem, raw, archive, combos):
    ids=np.unique(combos)
    points=[archive['points'][i] for i in ids]
    pivots=np.array([p[1] for p in points])
    terms,opaque=live.R.symbols(problem,archive['labels'])
    assert not archive['labels'][0]
    designs={}
    index=[]
    rows,cols,values=[],[],[]
    for i,p in enumerate(points):
        key=(p[1],tuple(j for j,w in p[2] if j!=p[1]))
        index.append(designs.setdefault(key,len(designs)))
        for j,w in p[2]:
            rows.append(i)
            cols.append(j)
            values.append(w)
    frozen=csr_matrix((values,(rows,cols)),shape=(len(ids),len(terms)))
    design_pivots=np.array([key[0] for key in designs])
    grouped=defaultdict(list)
    for (target,predictors),i in designs.items():
        grouped[len(predictors)].append((i,target,predictors))
    groups={w:tuple(np.asarray(a,dtype=int) for a in zip(*items)) for w,items in grouped.items()}
    original={k:tuple(v) for k,v in archive['plan'].items()}
    scale_log=[]
    axis_scores=[]
    cross_axes=[]
    for axis in range(len(problem.shape)):
        scale_scores=[]
        cross_levels=[]
        parent_models=[]
        parent_edges=None
        for count in (2,4,8):
            setup=geometry(problem.shape,axis,count,original)
            if setup is None:
                continue
            edges,plan=setup
            mats=slab_matrices(problem,raw,terms,opaque,axis,edges,plan)
            grams=[m.T @ m for m in mats]
            test_scores=[]
            models=[]
            cross_children=[]
            for j,held in enumerate(mats):
                outside=[k for k in range(count) if j!=k]
                train=np.concatenate([mats[k] for k in outside])
                g=np.add.reduce([grams[k] for k in outside])
                weights,invalid=refit(groups,train,g,len(designs))
                score=wape(held,weights,design_pivots)
                score[invalid]=np.inf
                test_scores.append([wape(held,frozen,pivots),score[index]])
                models.append((weights,invalid))
                if count==2*len(parent_models):
                    assert edges[j]>=parent_edges[j//2] and edges[j+1]<=parent_edges[j//2+1]
                    coarse,unresolved=parent_models[j//2]
                    cross=wape(held,coarse,design_pivots)
                    cross[unresolved]=np.inf
                    cross_children.append(cross[index])
            if cross_children:
                cross_levels.append(np.mean(cross_children,axis=0))
            parent_models,parent_edges=models,edges
            scale_scores.append(np.mean(test_scores,axis=0))
            scale_log.append(dict(axis=axis,slabs=count,refits=count*len(designs),min_rows=min(map(len,mats)),
                                  cross_scale_children=len(cross_children)))
        if scale_scores:
            avg=np.mean(scale_scores,axis=0)
            axis_scores.append(avg)
            cross_axes.append(np.mean(cross_levels,axis=0) if cross_levels else avg[1])
    if not axis_scores:
        raise ValueError(f'No supported raw slab partitions for {problem.name}')
    scores=np.full((3,len(archive['points'])),np.inf)
    scores[:2,ids]=np.mean(axis_scores,axis=0)
    scores[2,ids]=np.mean(cross_axes,axis=0)
    return dict(slab_frozen=scores[0],slab_crossfit=scores[1],slab_cross_scale=scores[2]),scale_log
