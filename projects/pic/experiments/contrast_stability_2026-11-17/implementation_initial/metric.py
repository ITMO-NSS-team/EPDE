"""Multiscale sibling coefficient contrasts normalized by measured noise."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent/'uncertainty_refit_2026-11-16'))
import experiment as E
R,np,K,SV=E.R,E.np,E.K,E.R.SV


class Extractor:
    def __init__(self,candidates,shape):
        self.cs,self.shape=candidates,tuple(shape);self.groups=[];self.axis_groups=[];offset=0
        n=int(np.prod(shape));self.layouts=list(SV._axis_layouts(n,shape))
        self.parts=[SV._dyadic_slab_edges(n,moved,SV._block_rows(SV._reference_width(0))) for _,_,moved in self.layouts]
        for j,c in enumerate(candidates):
            axes=[]
            for axis,parts in enumerate(self.parts):
                gids=[]
                for level,edges in enumerate(parts):
                    count=(len(edges)-1)//2*len(c['xi']);assert (len(edges)-1)%2==0
                    gids.append(len(self.groups));self.groups.append(dict(candidate=j,axis=axis,level=level,
                        start=offset,stop=offset+count,slabs=len(edges)-1,features=len(c['xi'])))
                    offset+=count
                axes.append(gids)
            self.axis_groups.append(axes)
        self.width=offset

    def extract(self,matrix):
        output=[];dyad=[];deficient=[];group=0;n=len(matrix);w=np.ones(n)
        for c in self.cs:
            x,y=matrix[:,c['xi']],matrix[:,c['yi']];p=len(c['xi']);axis_scores=[];defect=0
            for (_,order,moved),parts in zip(self.layouts,self.parts):
                if not p:
                    output.extend([np.array([]) for _ in parts]);axis_scores.append(0.);group+=len(parts);continue
                xx,yy=(x,y) if order is None else (x[order],y[order])
                fine=parts[-1];gg,gy=SV._level_gram_blocks(xx,yy,w,moved,fine,append_constant=False);grams=[];rhs=[]
                for edges in parts:
                    pos=np.searchsorted(fine,edges[:-1]);grams.append(np.add.reduceat(gg,pos,axis=0));rhs.append(np.add.reduceat(gy,pos,axis=0))
                ga,ra=np.concatenate(grams),np.concatenate(rhs)
                factors=SV._SlabFactors(lambda:np.column_stack([xx,yy]),w,parts)
                def design(rows):
                    f=factors(rows);return f[:,:,:p],f[:,:,p]
                coefs=SV._solve_min_norm(ga,ra,design if SV._DYAD_SOLVE in SV._DESIGN_SOLVES else None)
                norms=np.sqrt(np.where(np.diagonal(ga,axis1=-2,axis2=-1)>0,np.diagonal(ga,axis1=-2,axis2=-1),1.))
                eig=np.linalg.eigvalsh(ga/norms[:,:,None]/norms[:,None,:])
                defect+=int(np.sum(np.any(np.abs(eig)<=SV._DYAD_RCOND*np.max(np.abs(eig),axis=1,keepdims=True),axis=1)))
                start=0;per_term=np.zeros(p)
                for edges in parts:
                    cc=coefs[start:start+len(edges)-1];start+=len(edges)-1
                    med=np.median(cc,axis=0);mad=np.median(np.abs(cc-med),axis=0)
                    with np.errstate(divide='ignore',invalid='ignore'):rel=mad/np.abs(med)
                    per_term+=np.where(np.isnan(rel),0.,rel)
                    contrast=(cc[::2]-cc[1::2]).ravel();output.append(contrast)
                    assert len(contrast)==self.groups[group]['stop']-self.groups[group]['start'];group+=1
                axis_scores.append(float(np.sum(np.nan_to_num(per_term/len(parts)))))
            dyad.append(float(np.mean(axis_scores)));deficient.append(defect)
        out=np.concatenate(output);assert len(out)==self.width and np.isfinite(out).all()
        return out,np.array(dyad),np.array(deficient)

    def reduce(self,squares):
        values=[]
        for c,axes in zip(self.cs,self.axis_groups):
            if not c['xi']:values.append(np.zeros(squares.shape[:-1]));continue
            values.append(np.mean([np.mean([np.mean(squares[...,self.groups[g]['start']:self.groups[g]['stop']],axis=-1) for g in levels],axis=0)
                                   for levels in axes],axis=0))
        return np.stack(values,axis=-1)


class NoiseModel:
    def __init__(self,extractor,pilot,draws):
        self.extractor=extractor;self.count=len(draws);self.bias=draws.mean(axis=0)-pilot;self.variance=draws.var(axis=0,ddof=1)
        assert np.isfinite(self.bias).all() and np.isfinite(self.variance).all() and np.all(self.variance>=0)

    def score(self,contrasts,bias=True):
        d=contrasts-self.bias if bias else contrasts
        v=self.variance*(1+1/self.count) if bias else self.variance
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):ratio=d*d/v
        ratio=np.where((v==0)&(d==0),0.,ratio)
        assert not np.isnan(ratio).any()
        return self.extractor.reduce(ratio)


def fronts(coords,cs,slot):
    ix=np.array([j for j,c in enumerate(cs) if c['slot']==slot]);v=np.asarray(coords).T[ix]
    dom=np.all(v[:,None,:]<=v[None,:,:],axis=2)&np.any(v[:,None,:]<v[None,:,:],axis=2)
    return ix[~dom.any(axis=0)].tolist()
