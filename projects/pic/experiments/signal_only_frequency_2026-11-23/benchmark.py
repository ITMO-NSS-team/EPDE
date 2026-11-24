"""Test frequency proposals without equation-term fitting or held-out samples."""
import ast
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'candidate_frequency_2026-11-23'))
from candidate_trial import live
from scipy.integrate import solve_ivp
from signal_frequency import np, propose

STORED_SEEDS = range(926001,926251)
SIM_SEEDS = range(926501,926551)


def score(x,dt,truth,origin,name,state,duration,seed):
    native,dual,changed = propose(x,dt)
    r = dict(origin=origin,system=name,state=state,duration=duration,seed=seed,samples=len(x),
             spacing=dt,omega=truth,cycles=len(x)*dt*truth/(2*np.pi),replaced=int(changed))
    for key,peaks in [('native',native),('dual',dual)]:
        if not peaks:
            raise ValueError(f'No spectral peaks for {name}/{state}/{duration}/{seed}')
        closest = min(peaks,key=lambda w:abs(w-truth))
        error = abs(closest-truth)/truth
        r.update({key+'_peaks':json.dumps(peaks),key+'_closest':closest,key+'_error':error,
                  key+'_hit':int(error<=.01)})
    return r


def stored(name):
    p = live.load_data(name)
    _,(slices,axes),_ = live.split(live.R.axes_of(p))
    dt = float(np.median(np.diff(axes[0])))
    truth = p.meta['omega'] if name=='duffing' else 2.
    for n in range(923701,923713):
        f = ROOT.parent/'candidate_frequency_2026-11-23'/'results'/name/str(n)/'native'/'search.log'
        line, = [s for s in f.read_text().splitlines() if 'Frequencies of trigonometric found in the data:' in s]
        archived = ast.literal_eval(line.split('data: ')[1])
        native,_,_ = propose(p.noisy(5,n)['u'][slices],dt)
        np.testing.assert_allclose(sorted(native),archived,rtol=1e-8,atol=1e-9)
    return [score(p.noisy(5,n)['u'][slices],dt,truth,'stored',name,'original',1,n) for n in STORED_SEEDS]


def simulate(name,state):
    p = live.load_data(name)
    _,(_,axes),_ = live.split(live.R.axes_of(p))
    n = len(axes[0])
    dt = float(np.median(np.diff(axes[0])))
    truth = p.meta['omega'] if name=='duffing' else 2.
    if state=='shifted':
        truth *= 1.2
    original = (0.,0.) if name=='duffing' else (.8,2.)
    initial = {'original':original,'displacement':(1.,0.),'velocity':(0.,1.),'shifted':(1.,0.)}[state]
    if name=='duffing':
        def derivative(t,y):
            u,v = y
            return v,p.meta['gamma']*np.cos(truth*t)-p.meta['delta']*v-p.meta['alpha']*u-p.meta['beta']*u**3
    else:
        def derivative(t,y):
            u,v = y
            return v,1.5*t-4*u-np.sin(truth*t)*v
    t = np.arange(8*n)*dt
    def solve(tol):
        s = solve_ivp(derivative,(0.,t[-1]),initial,t_eval=t,method='DOP853',atol=tol/10,rtol=tol)
        assert s.success and np.array_equal(s.t,t)
        return s.y[0]
    x = solve(1.e-11)
    np.testing.assert_allclose(x,solve(1.e-12),atol=2e-8,rtol=2e-8)
    sigma = .05*np.std(x[:n])
    rows = []
    for seed in SIM_SEEDS:
        noisy = x+sigma*np.random.default_rng(seed).standard_normal(len(x))
        for duration in (1,4,8):
            rows.append(score(noisy[:n*duration],dt,truth,'simulated',name,state,duration,seed))
    return rows


def evaluate(job):
    name,state = job
    return stored(name) if state=='stored' else simulate(name,state)


def write(path,rows):
    with path.open('w',encoding='utf8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    for r in rows:
        key = r['origin'],r['system'],r['state'],r['duration']
        groups.setdefault(key,[]).append(r)
    out = []
    for (origin,name,state,duration),cases in groups.items():
        entry = dict(origin=origin,system=name,state=state,duration=duration,pairs=len(cases),
                     cycles=cases[0]['cycles'],alternate_proposals=sum(r['replaced'] for r in cases),
                     gains=sum(r['dual_hit']>r['native_hit'] for r in cases),
                     losses=sum(r['dual_hit']<r['native_hit'] for r in cases))
        for method in ('native','dual'):
            est = np.array([r[method+'_closest'] for r in cases])
            err = np.array([r[method+'_error'] for r in cases])
            entry.update({method+'_hits':sum(r[method+'_hit'] for r in cases),
                          method+'_median_error_pct':float(100*np.median(err)),
                          method+'_median_omega':float(np.median(est))})
        out.append(entry)
    return out


def check():
    t = np.arange(1000)*.05
    for w in (.85,1.37,2.4):
        a,b,_ = propose(np.sin(w*t+.4),.05)
        assert abs(a[0]/w-1)<.001 and a[0]==b[0]
    a,b,changed = propose(np.sin(1.17*t)+.8*np.sin(2.69*t),.05)
    assert len(a)==2 and a==b and not changed
    assert propose(np.ones(100),.1)==((),(),False)


def plot(cases,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig,(short,long) = plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    colors = {'Fourier':'#326583','Lag':'#b24d25'}
    short.axvspan(-1,1,color='#e2eddf')
    short.axvline(0,color='#657765',lw=1,ls='--')
    for i,name in enumerate(('ode','duffing')):
        sample = [r for r in cases if r['origin']=='stored' and r['system']==name]
        assert all(r['replaced'] for r in sample)
        for j,method in enumerate(('Fourier','Lag')):
            err = [100*(json.loads(r['dual_peaks'])[j]/r['omega']-1) for r in sample]
            lo,med,hi = np.quantile(err,[.025,.5,.975])
            y = i+(-.11 if j==0 else .11)
            short.plot([lo,hi],[y,y],color=colors[method],lw=3)
            short.scatter(med,y,s=32,color=colors[method],label=method if i==0 else None)
    short.set_yticks([0,1],['Forced oscillator','Duffing'])
    short.set_ylim(1.35,-.4)
    short.set_xlabel('Frequency error (%)')
    short.set_title('Original training records',loc='left')
    short.legend(frameon=False,loc='center right')
    short.text(.02,.02,'Bars span the middle 95% of noise draws.',transform=short.transAxes,fontsize=8)

    for method,label,color,offset in [('native','Fourier only',colors['Fourier'],-.16),
                                      ('dual','Fourier + lag option',colors['Lag'],.16)]:
        percentages = []
        for duration in (1,4,8):
            rows = [r for r in summary if r['origin']=='simulated' and r['system']=='duffing' and r['duration']==duration]
            percentages.append(100*sum(r[method+'_hits'] for r in rows)/sum(r['pairs'] for r in rows))
        long.bar(np.arange(3)+offset,percentages,width=.31,label=label,color=color)
    long.set_xticks(range(3),['10','40','80'])
    long.set_ylim(0,105)
    long.set_ylabel('Lists within 1% of drive (%)')
    long.set_xlabel('Observation time (simulation units)')
    long.set_title('Four simulated Duffing trajectories',loc='left')
    long.legend(frameon=False,loc='upper left',fontsize=9)
    for ax in (short,long):
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'frequency_comparison.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    check()
    jobs = [(name,state) for name in ('duffing','ode')
            for state in ('stored','original','displacement','velocity','shifted')]
    with ProcessPoolExecutor(max_workers=4) as pool:
        cases = [r for group in pool.map(evaluate,jobs) for r in group]
    summary = summarize(cases)
    write(ROOT/'paired_frequencies.csv',cases)
    write(ROOT/'by_record.csv',summary)
    plot(cases,summary)
    print([(r['system'],r['native_hits'],r['dual_hits'],r['pairs']) for r in summary if r['origin']=='stored'])
