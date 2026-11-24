from common import *
import paired_study_fast as E
from studentize import Studentizer
import batch_moments as B
import time
def main():
    rows=[]
    for name in SYSTEMS:
        e=E.Study(name,41,'iid');rng=np.random.default_rng(981119);fields=[e.draw(rng)[0] for _ in range(4)];values={v:np.stack([f[v] for f in fields]) for v in e.raw};many=B.matrices(e,values,e.model)
        for j,field in enumerate(fields):
            for bank,m in e.matrices(field,e.model).items():
                sc=max(float(np.max(np.abs(m))),np.finfo(float).tiny);np.testing.assert_allclose(many[bank][j],m,rtol=2e-10,atol=2e-12*sc)
        rows.append(dict(system=name,matrix_pairs=8))
    e=E.Study('ks',41,'positive');student=Studentizer(e);start=time.monotonic();single=student.errors(student.center);single_time=time.monotonic()-start;start=time.monotonic();batch=B.errors(student,student.center);batch_time=time.monotonic()-start
    for bank in single:
        sc=max(float(np.max(np.abs(single[bank]))),np.finfo(float).tiny);np.testing.assert_allclose(batch[bank],single[bank],rtol=3e-10,atol=3e-12*sc)
    result=dict(status='PASS',matrix_pairs=120,ks_common_draws=32,single_seconds=single_time,batch_seconds=batch_time,source_sha256=hashes('batch_checks.py','batch_moments.py'));dump(ROOT/'batch_checks.json',result);print(result,flush=True)
if __name__=='__main__':main()
