"""Vectorized evaluation of independent fields with the same noise model."""
from common import np,R
from fast_covariance import Covariance
from itertools import combinations
class Fields:
    def __init__(self,values,axes,opaque):
        self.values=values;self.axes=axes;self.shape=next(iter(values.values())).shape;self.ndim=len(axes);self.cache={};self.opaque=opaque
        self.coords=[a.reshape([1]+[-1 if b==j else 1 for b in range(self.ndim)]) for j,a in enumerate(axes)]
    def get(self,var,orders):
        key=var,orders
        if key not in self.cache:
            if sum(orders)==0:self.cache[key]=self.values[var]
            else:
                axis=max(j for j,o in enumerate(orders) if o);lo=list(orders);lo[axis]-=1;self.cache[key]=R.WF._central4(self.get(var,tuple(lo)),self.axes[axis],axis+1)
        return self.cache[key]
    def point(self,data,coords,opaque):
        arr=np.ones(self.shape)
        for var,order in data:arr=arr*self.get(var,order)
        for axis,p in enumerate(coords):
            if p:arr=arr*self.coords[axis]**p
        for key in opaque:arr=arr*self.opaque[key]
        return arr
def integrate(bank,arr,orders):
    for axis in sorted(range(len(bank.axes)),key=lambda a:bank.shape[a]/arr.shape[a+1]):arr=np.moveaxis(np.tensordot(bank.kernels[axis][orders[axis]],arr,axes=(1,axis+1)),0,axis+1)
    return arr
def matrices(e,values,model):
    fields=Fields(values,e.axes,e.opaque);points={};cov=Covariance(e.geometry,model);out={};batch=fields.shape[0]
    for tag,bank in e.banks.items():
        integrals={};columns=[]
        for term,monos in zip(bank.terms,bank.monos):
            column=0.
            for (data,coords,orders),factor in monos.items():
                key=data,coords,term.opaque
                if key not in points:
                    arr=fields.point(data,coords,term.opaque)
                    if len(data)>=2:
                        deterministic=fields.point((),coords,term.opaque)
                        for i,j in combinations(range(len(data)),2):
                            correction=cov.pair(data[i],data[j])
                            for k in range(len(data)):
                                if k not in (i,j):correction=correction*fields.get(*data[k])
                            arr=arr-deterministic*correction
                    points[key]=arr
                ik=key,orders
                if ik not in integrals:integrals[ik]=integrate(bank,points[key],orders)
                column=column+factor*integrals[ik]
            columns.append(np.asarray(column).reshape(batch,-1))
        m=np.stack(columns,axis=-1);out[tag]=e.basis.convert(m) if tag=='broad' else m.real
    return out
def errors(student,t,batch_size=4):
    e=student.e;model=student.decode(t);stacks={b:[] for b in e.banks}
    for start in range(0,32,batch_size):
        values={v:[] for v in student.variables}
        for j in range(start,min(start+batch_size,32)):
            rng=np.random.default_rng(np.random.SeedSequence([981105,e.pid,e.seed,j]))
            for v in student.variables:values[v].append(e.pilot[v]+R.WF.ar_noise(rng,e.raw[v].shape,*model[v][:2]))
        for b,m in matrices(e,{v:np.stack(y) for v,y in values.items()},model).items():stacks[b].append(m-student.pilot[b][None])
    return {b:np.concatenate(x) for b,x in stacks.items()}
