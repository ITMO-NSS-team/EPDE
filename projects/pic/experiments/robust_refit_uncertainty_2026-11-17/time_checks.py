from common import *
import time_fit as F
def main():
    rng=np.random.default_rng(981120);shape=(101,128,31);rho=[.3,.4,-.4];noise=R.WF.ar_noise(rng,shape,.2,rho);t=np.linspace(0,1,shape[0]);signal=(t+t*t*.1)[:,None,None]*np.sin(np.linspace(0,20,shape[1]))[None,:,None]
    model,info=F.estimate(noise+signal);assert info['status']=='ok';np.testing.assert_allclose(model[0],.2,rtol=.04);np.testing.assert_allclose(model[1],rho,atol=.04)
    transformed,ti=F.estimate(-3*(noise+signal)+np.linspace(7,11,shape[0])[:,None,None]);assert ti['status']=='ok';np.testing.assert_allclose(transformed[0],3*model[0],rtol=2e-5);np.testing.assert_allclose(transformed[1],model[1],atol=2e-5)
    dump(ROOT/'time_checks.json',dict(status='PASS',synthetic_model=model,true_model=[.2,rho],unit_linear_trend_invariance=True,source_sha256=hashes('time_checks.py','time_fit.py','time_protocol.txt','state_score.py','repair_fit.py')));print('PASS',model,flush=True)
if __name__=='__main__':main()
