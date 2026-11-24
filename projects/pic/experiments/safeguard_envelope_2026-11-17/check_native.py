from study_context import *
import native_score as Native
import time

def main():
    rng=np.random.default_rng(985809);configs=[(0.,1e-5,.99,-.8),(.4,1e-3,.95,-.5),(-.4,1e-8,.9,0.),(.99995,1e-6,.995,-.9998),(-.99995,1e-10,.99,-.99),(.8,10.,-.3,.3)]
    configs += [(float(rng.uniform(-.9,.9)),float(np.exp(rng.uniform(-20,3))),float(rng.uniform(-.95,.99)),float(rng.uniform(-.99,.95))) for j in range(20)]
    rows=[];oldtime=newtime=0.
    for n in (7,32,318,1000):
        y=rng.normal(size=(n,1))
        for pars in configs:
            start=time.perf_counter();a=Native.ORIGINAL(y,*pars);oldtime+=time.perf_counter()-start;start=time.perf_counter();b=Native.axis_score(y,*pars);newtime+=time.perf_counter()-start
            va=np.log(a[0]/n)+a[1]/n;vb=np.log(b[0]/n)+b[1]/n;ga=a[2]/a[0]+a[3]/n;gb=b[2]/b[0]+b[3]/n;jac=np.array([1-pars[0]**2,1,1-pars[2]**2,1-pars[3]**2]);error=float(max(abs((ga-gb)*jac)))
            rows.append(dict(n=n,parameters=pars,value_error=abs(va-vb),transformed_gradient_error=error));assert abs(va-vb)<=1e-8,(n,pars,va,vb);assert error<=2e-6,(n,pars,error)
    dump(ROOT/'native_arithmetic_checks.json',dict(status='PASS',rows=rows,python_seconds=oldtime,native_seconds=newtime,speedup=oldtime/newtime,source_sha256=sources('native_score.c','native_score.py','check_native.py','native_score.dll')));print('PASS',len(rows),'speedup',oldtime/newtime,'max gradient',max(r['transformed_gradient_error'] for r in rows),flush=True)

if __name__=='__main__':main()
