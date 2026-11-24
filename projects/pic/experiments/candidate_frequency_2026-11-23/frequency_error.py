"""Measure conditional frequency uncertainty from observation noise and mesh resolution."""
import csv
from dataclasses import replace
from functools import lru_cache

from candidate_trial import ROOT, folder, live, np
from assess_swap import read_archive, write_csv
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

REPEATS = 250
FIRST_SEED = 924001
DT = .025


def parent_inputs():
    with (ROOT/'forcing_profiles.csv').open() as source:
        row = next(r for r in csv.DictReader(source) if r['dataset']=='ode' and int(r['seed'])==923301)
    a = read_archive(folder('ode',923301,'candidate'))
    parent = int(row['parent'])
    selected = next(r for r in a['operator']['profiles'] if r['point']==parent)
    p = a['points'][parent]
    labels = [a['labels'][p[1]],*[a['labels'][j] for j,coef in p[2] if j!=p[1]]]
    return labels, selected, {k:tuple(v) for k,v in a['plan'].items()}


class Profile:
    def __init__(self, t, plan, labels, record, original):
        self.t, self.plan, self.record = t, plan, record
        self.problem = replace(original,grids=(t,),data={'u':np.zeros_like(t)})
        fixed = []
        moving = []
        for term in labels:
            (moving if any(n in ('sin','cos') for n,p in term) else fixed).append(term)
        assert fixed[0]==labels[0] and len(moving)==1
        self.moving = moving[0]
        self.fixed = self.bank(fixed)

    def bank(self, labels):
        terms,opaque = live.R.symbols(self.problem,labels)
        return live.R.LocalBank([self.t],terms,opaque,self.plan)

    def estimate(self, u):
        values = {'u':u}
        fixed = self.fixed.matrix(values)
        target = fixed[:,0]
        @lru_cache(maxsize=None)
        def sse(w):
            term = [(n,[(k,w if k=='freq' else v) for k,v in p]) for n,p in self.moving]
            col = self.bank([term]).matrix(values)
            x = np.column_stack((fixed[:,1:],col))
            norms = np.linalg.norm(x,axis=0)
            unit = x/norms
            weights,_,rank,_ = np.linalg.lstsq(unit,target,rcond=None)
            assert rank==x.shape[1]
            residual = target-unit@weights
            return float(residual@residual)
        r = self.record
        lo,hi = r['lower'],r['upper']
        result = minimize_scalar(sse,method='bounded',bounds=(lo,hi),
                                 options=dict(maxiter=24,xatol=(hi-lo)*1.e-4))
        return min([lo,hi,r['initial'],float(result.x)],key=lambda w:(sse(w),abs(w-r['initial'])))


def summarize(group, dt, clean, draws):
    values = np.asarray(draws)
    q = np.quantile(values,[.025,.5,.975])
    return dict(observations=group,dt=dt,clean_frequency=clean,clean_bias=clean-2.,
                noise_draws=len(values),noisy_q025=q[0],noisy_median=q[1],noisy_q975=q[2],
                noisy_sd=np.std(values,ddof=1),inside_one_percent=int(np.sum(np.abs(values-2.)<=.02)))


def plot(summary, measured):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(8,3.6), layout='constrained')
    ax.axvspan(1.98,2.02,color='#e1ebde',label='1% scoring band')
    ax.axvline(2,color='#517052',linestyle='--',linewidth=1)
    for i,r in enumerate(summary):
        ax.plot([r['noisy_q025'],r['noisy_q975']],[i,i],color='#185376',lw=2.5,
                label='Middle 95% of refits' if i==0 else None)
        ax.scatter(r['noisy_median'],i,s=38,color='#185376',zorder=3,label='Median' if i==0 else None)
        ax.scatter(r['clean_frequency'],i-.13,s=28,color='#303030',marker='D',zorder=3,
                   label='Noise-free fit' if i==0 else None)
    ax.scatter(measured,0,marker='x',s=64,color='#ad4827',zorder=4,label='Observed 1.9557')
    names = ['Stored data, Δt = 0.05',*[f"Simulation, Δt = {r['dt']}" for r in summary[1:]]]
    ax.set_yticks(range(len(names)),names)
    ax.set_ylim(len(names)-.6,-.5)
    ax.set_xlabel('Fitted angular frequency (radians per unit time)')
    ax.set_title('Frequency estimates under 5% measurement noise',loc='left')
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.legend(loc='upper center',bbox_to_anchor=(.40,-.21),ncol=3,frameon=False,fontsize=9)
    fig.savefig(ROOT/'frequency_error.png',dpi=180)
    plt.close(fig)


def main():
    original = live.load_data('ode')
    labels,record,plan = parent_inputs()
    t, = live.R.axes_of(original)
    _,(cut,axes),_ = live.split([t])
    original_profile = Profile(axes[0],plan,labels,record,original)
    measured = original_profile.estimate(original.noisy(5,923301)['u'][cut])
    np.testing.assert_allclose(measured,record['proposed'],atol=2.e-7,rtol=0)
    clean = original_profile.estimate(original.data['u'][cut])
    seeds = range(FIRST_SEED,FIRST_SEED+REPEATS)
    draws = [original_profile.estimate(original.noisy(5,n)['u'][cut]) for n in seeds]
    summary = [summarize('stored benchmark',float(t[1]-t[0]),clean,draws)]
    raw = [dict(observations='stored benchmark',dt=.05,seed=n,frequency=w) for n,w in zip(seeds,draws)]
    # Align the interval, physical window width and centres on all meshes.
    fine = np.arange(641)*DT
    def rhs(t,y):
        return y[1],1.5*t-4*y[0]-np.sin(2*t)*y[1]
    def solve(tol):
        sol = solve_ivp(rhs,(0.,16.),(.8,2.),t_eval=fine,method='DOP853',rtol=tol,atol=tol/10)
        assert sol.success
        return sol.y[0]
    u = solve(1.e-11)
    np.testing.assert_allclose(u,solve(1.e-12),rtol=1.e-9,atol=1.e-9)
    noise_sd = .05*np.std(u)
    z = np.stack([np.random.default_rng(n).standard_normal(len(fine)) for n in seeds])
    for step in (4,2,1):
        dt = DT*step
        selection = slice(0,321,step)
        grid = fine[selection]
        p = dict(q=(2,),ms=(round(.8/dt),),ss=(round(.1/dt),),cut=(0,),shape=(65,))
        centers = grid[p['ms'][0]:len(grid)-p['ms'][0]:p['ss'][0]]
        np.testing.assert_allclose(centers,np.arange(65)*.1+.8)
        profile = Profile(grid,p,labels,record,original)
        clean = profile.estimate(u[selection])
        draws = [profile.estimate(u[selection]+noise_sd*e[selection]) for e in z]
        summary.append(summarize('converged simulation',dt,clean,draws))
        raw.extend(dict(observations='converged simulation',dt=dt,seed=n,frequency=w) for n,w in zip(seeds,draws))
    write_csv(ROOT/'frequency_error.csv',summary)
    write_csv(ROOT/'frequency_error_draws.csv',raw)
    plot(summary,measured)
    print(summary,flush=True)


if __name__=='__main__':
    main()
