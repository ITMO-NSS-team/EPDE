import experiment as E
R,np,ROOT=E.R,E.np,E.ROOT
from pathlib import Path
import json
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Curve:
    def __init__(self,a,f,c,bank):
        ix=[c['yi'],c['xi'][0]];arm=bank+'_studentized'
        cal=(a[arm+'_calibration']-a[bank+'_pilot'])[...,ix]/a['calibration_amplitude'][:,None,None]
        self.check=(a[arm+'_check']-a[bank+'_pilot'])[...,ix]/a['check_amplitude'][:,None,None]
        centered=cal-cal.mean(axis=0);self.cov=np.einsum('bri,brj->rij',centered,centered)/(len(cal)-1)
        self.obs=a[arm+'_observed'][:,ix]/f['amplitude'];self.rows=len(self.obs)
        self.yy=self.check[:,:,0]**2;self.yx=self.check[:,:,0]*self.check[:,:,1];self.xx=self.check[:,:,1]**2
        self.ols=float(self.obs[:,1]@self.obs[:,0]/(self.obs[:,1]@self.obs[:,1]))

    def evaluate(self,angles):
        angles=np.atleast_1d(angles);co=np.cos(angles);sn=np.sin(angles);co=np.where(np.abs(angles)==np.pi/2,0.,co)
        var=co[:,None]**2*self.cov[:,0,0]-2*(co*sn)[:,None]*self.cov[:,0,1]+sn[:,None]**2*self.cov[:,1,1]
        assert np.all(var>0)
        qcheck=(self.yy@(co[:,None]**2/var).T-2*self.yx@((co*sn)[:,None]/var).T+self.xx@(sn[:,None]**2/var).T)/self.rows
        qobs=np.mean((co[:,None]*self.obs[:,0]-sn[:,None]*self.obs[:,1])**2/var,axis=1)
        cutoff=np.quantile(qcheck,.95,axis=0)
        return qobs,cutoff

    def margin(self,theta):
        q,r=self.evaluate(theta);return float(q[0]-r[0])

    def region(self,n):
        angles=np.unique(np.r_[np.linspace(-np.pi/2,np.pi/2,n),np.arctan(self.ols)])
        pieces=[]
        for i in range(0,len(angles),128):
            observed,cutoff=self.evaluate(angles[i:i+128]);pieces.append(observed-cutoff)
        margin=np.concatenate(pieces)
        roots=[]
        for i in range(len(angles)-1):
            if margin[i]==0:roots.append(float(angles[i]))
            if margin[i]*margin[i+1]<0:roots.append(float(brentq(self.margin,angles[i],angles[i+1],xtol=1e-12)))
        roots=sorted(set(roots));edges=[-np.pi/2,*roots,np.pi/2];intervals=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            if self.margin((lo+hi)/2)<=0:
                intervals.append([None if lo==-np.pi/2 else float(np.tan(lo)),None if hi==np.pi/2 else float(np.tan(hi))])
        return roots,intervals


def main():
    rows=[];inputhash={};maximum=0.
    for seed in (24,25,26,27,28,29):
        path=ROOT/'iid'/f'wave_seed{seed}.json';meta=json.loads(path.read_text());fp=path.with_name(path.stem+'_frozen.json');f=json.loads(fp.read_text())
        with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
        e=E.Study('wave',seed);c=next(c for c in e.cs if c['label']=='reference');assert len(c['xi'])==1
        for bank in ('local','broad'):
            curve=Curve(a,f,c,bank)
            # Numerical score audit at data-derived and fixed angles, before truth query.
            for angle in [np.arctan(curve.ols),-.7,0.,.7,np.pi/2]:
                v=np.array([0.,-1.]) if angle==np.pi/2 else np.array([np.cos(angle),-np.sin(angle)])
                var=np.einsum('i,rij,j->r',v,curve.cov,v);qb=E.score(curve.check@v,var);qo=E.score(curve.obs@v,var)
                got,rad=curve.evaluate(angle);np.testing.assert_allclose([got[0],rad[0]],[qo,np.quantile(qb,.95)],rtol=2e-10,atol=2e-11)
                maximum=max(maximum,abs(float(got[0]-qo)),abs(float(rad[0]-np.quantile(qb,.95))))
            roots,intervals=curve.region(8193);fine,other=curve.region(16385)
            assert len(roots)==len(fine) and len(intervals)==len(other)
            np.testing.assert_allclose(roots,fine,rtol=0,atol=2e-10)
            for root in fine:assert abs(curve.margin(root))<1e-7
            row=dict(seed=seed,bank=bank,intervals=other,angle_roots=fine,ols=curve.ols,
                infinite_limit_accepted=curve.margin(np.pi/2)<=0,resolution_agreement=True)
            # Region now fixed; the exact coefficient is a diagnostic query only.
            truth=float(c['exact'][0]);q,rad=curve.evaluate(np.arctan(truth));anchor=bool(q[0]<=rad[0])
            region_contains=any((lo is None or lo<=truth) and (hi is None or truth<=hi) for lo,hi in other);assert anchor==region_contains
            recorded=next(r for r in meta['rows'] if r['arm']==bank+'_studentized' and r['candidate']==c['id'])
            assert anchor==recorded['anchor_acceptance']
            actual=(a[bank+'_studentized_actual'][...,c['yi']]-truth*a[bank+'_studentized_actual'][...,c['xi'][0]])/a['actual_amplitude'][:,None]
            v=np.array([1.,-truth]);variance=np.einsum('i,rij,j->r',v,curve.cov,v)
            count=int(np.sum(E.score(actual,variance)<=recorded['radius']));assert count==recorded['zero_acceptance_count']
            row.update(exact=truth,exact_in_region=anchor,actual_exact_acceptance_count=count,actual_draws=256);rows.append(row)
            print(seed,bank,other,'contains exact',anchor,flush=True)
        inputhash.update({str(p):E.sha(p) for p in [path,path.with_suffix('.npz'),fp]})
    result=dict(status='PASS',scope='One-parameter fixed-support numerical test inversion; not post-search confidence.',rows=rows,
        score_replay_max_error=maximum,source_sha256={str(Path(__file__)):E.sha(__file__),str(ROOT/'inversion_protocol.txt'):E.sha(ROOT/'inversion_protocol.txt')},input_sha256=inputhash)
    R.dump(ROOT/'wave_inversion.json',result)
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,bank in zip(axs,['local','broad']):
        rr=[r for r in rows if r['bank']==bank]
        for r in rr:
            for lo,hi in r['intervals']:
                if lo is not None and hi is not None:ax.plot([lo,hi],[r['seed'],r['seed']],lw=3,color='#087f8c')
            ax.plot(r['ols'],r['seed'],'x',color='#b74929')
        ax.axvline(.04,color='grey',ls='--',label='Exact coefficient');ax.set_yticks([r['seed'] for r in rr]);ax.set_title(bank.title()+' weak moments');ax.set_xlabel('Coefficient of u_xx');ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Fresh anchor seed');fig.suptitle('Wave: inverted95% iid-studentized residual regions; x = ordinary fit')
    fig.tight_layout();fig.savefig(ROOT/'wave_intervals.png',dpi=180);fig.savefig(ROOT/'wave_intervals.svg');plt.close(fig)

if __name__=='__main__':main()
