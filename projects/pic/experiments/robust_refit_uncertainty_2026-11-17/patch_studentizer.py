from studentize import Studentizer,TRUST
from common import np
class Patches:
    def __init__(self,e):self.e=e;self.base=Studentizer(e);self.patches=[self.base];self.center=self.base.center;self.cross=self.base.cross
    def encode(self,m):return self.base.encode(m)
    def build(self):self.base.build();self.cross=self.base.cross
    @property
    def exact_calls(self):return sum(p.exact_calls for p in self.patches)
    @property
    def response_calls(self):return sum(p.response_calls for p in self.patches)
    def validate(self,parameters):
        median=np.median(np.stack(parameters),axis=0)
        if np.max(np.abs(median-self.center))>TRUST:
            p=Studentizer(self.e);p.center=median;p.build();self.patches.append(p)
        out=[]
        for i,p in enumerate(self.patches):
            candidates=[t for t in parameters if int(np.argmin([np.max(np.abs(t-pp.center)) for pp in self.patches]))==i and np.max(np.abs(t-p.center))<=TRUST]
            if not candidates:p.force_exact=True;result=dict(status='disabled_no_check_parameters',relative_error=None)
            else:result=p.validate(candidates)
            out.append(dict(center=p.center,**result))
        return dict(status='validated_patches',patches=out)
    def evaluate(self,t):
        valid=[p for p in self.patches if not p.force_exact and np.max(np.abs(t-p.center))<=TRUST]
        if not valid:return self.base.direct(t)
        return min(valid,key=lambda p:np.max(np.abs(t-p.center))).evaluate(t)
