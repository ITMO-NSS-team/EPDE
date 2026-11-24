"""Exact diagonal AR covariance contractions for identity observation maps."""
from common import np,O,R
from scipy.sparse import csr_matrix,eye
class Geometry:
    def __init__(self,axes):self.axes=axes;self.maps={};self.coefficients={}
    def linear_map(self,axis,order):
        key=(axis,order)
        if key not in self.maps:
            if order==0:m=eye(len(self.axes[axis]),format='csr')
            else:
                d=csr_matrix(R.WF._central4(np.eye(len(self.axes[axis])),self.axes[axis],0));m=d
                for _ in range(order-1):m=d@m
            self.maps[key]=m
        return self.maps[key]
    def diagonal(self,axis,oa,ob,rho):
        key=(axis,min(oa,ob),max(oa,ob))
        if key not in self.coefficients:
            aa=self.linear_map(axis,oa);bb=self.linear_map(axis,ob);rows=[];maxlag=0
            for i in range(aa.shape[0]):
                aj=aa.indices[aa.indptr[i]:aa.indptr[i+1]];av=aa.data[aa.indptr[i]:aa.indptr[i+1]];bj=bb.indices[bb.indptr[i]:bb.indptr[i+1]];bv=bb.data[bb.indptr[i]:bb.indptr[i+1]]
                lags=np.abs(aj[:,None]-bj[None,:]).ravel();weights=(av[:,None]*bv[None,:]).ravel();coef=np.bincount(lags,weights=weights);rows.append(coef);maxlag=max(maxlag,len(coef))
            coef=np.zeros((aa.shape[0],maxlag))
            for i,row in enumerate(rows):coef[i,:len(row)]=row
            self.coefficients[key]=coef
        coeff=self.coefficients[key];return coeff@(float(rho)**np.arange(coeff.shape[1]))
class Covariance:
    def __init__(self,geometry,model):self.geometry=geometry;self.axes=geometry.axes;self.model=model;self.cache={}
    def pair(self,left,right):
        if left[0]!=right[0]:return 0.
        key=tuple(sorted((left,right)))
        if key not in self.cache:
            sig,rhos,*_=self.model[left[0]];value=sig**2
            for axis,(oa,ob,rho) in enumerate(zip(left[1],right[1],rhos)):
                shape=[1]*len(self.axes);shape[axis]=len(self.axes[axis]);value=value*self.geometry.diagonal(axis,oa,ob,rho).reshape(shape)
            self.cache[key]=value
        return self.cache[key]
