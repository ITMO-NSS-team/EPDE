"""Check how allocation of the actual Duffing measurements changes frequency visibility."""
import sys
from statistics import median
from experiment import ROOT, alternate, live, np, whole_record
from assess_swap import write_csv
sys.path.insert(0,str(ROOT.parent/'signal_only_frequency_2026-11-23'))
from signal_frequency import propose


def run():
    problem=live.load_data('duffing')
    axes=live.R.axes_of(problem)
    _,early,late=live.split(axes)
    _,even,odd=alternate(axes)
    _,whole,_=whole_record(axes)
    truth=problem.meta['omega']
    rows=[]
    for seed in [0,*range(926001,926251)]:
        field=problem.data['u'] if seed==0 else problem.noisy(5,seed)['u']
        for name,(cut,grid) in [('early_train',early),('late_held',late),('even_train',even),('odd_held',odd),('full',whole)]:
            times=grid[0]
            dt=float(np.median(np.diff(times)))
            fourier,dual,_=propose(field[cut],dt)
            for rule,values in [('fourier',fourier),('fourier_lag',dual)]:
                closest=min(values,key=lambda w:abs(w-truth),default=None)
                rows.append(dict(seed=seed,partition=name,rule=rule,samples=len(times),
                                 observed_span=times[-1]-times[0],cycles=(times[-1]-times[0])*truth/(2*np.pi),
                                 closest=closest,hit=int(closest is not None and abs(closest/truth-1)<=.01)))
    write_csv(ROOT/'duffing_frequencies.csv',rows)
    summaries=[]
    for part in ('early_train','late_held','even_train','odd_held','full'):
        for rule in ('fourier','fourier_lag'):
            group=[r for r in rows if r['partition']==part and r['rule']==rule]
            clean=group[0]
            assert clean['seed']==0 and len(group)==251
            noisy=group[1:]
            available=[r['closest'] for r in noisy if r['closest'] is not None]
            summaries.append(dict(partition=part,rule=rule,samples=clean['samples'],cycles=clean['cycles'],
                                  clean_closest=clean['closest'],noisy_proposed=len(available),
                                  noisy_closest_median=median(available) if available else None,
                                  noisy_hits=sum(r['hit'] for r in noisy),draws=len(noisy)))
    write_csv(ROOT/'duffing_summary.csv',summaries)


if __name__=='__main__':
    run()
