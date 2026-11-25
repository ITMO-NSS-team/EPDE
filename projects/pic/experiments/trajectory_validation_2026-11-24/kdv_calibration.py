"""Test shared-law KdV calibration with equal sample counts on independent noise draws."""
import csv
from dataclasses import replace
from io import StringIO
from settings import ROOT, limit_threads

if __name__ == '__main__':
    limit_threads()

from noise_suite import SEEDS, atomic_text, directory, live, measurement
from compare import read_archive
import numpy as np


def main():
    forced, whole = live.load_data('kdv_cossin'), live.load_data('kdv')
    indices = np.ix_(20+2*np.arange(81),15+6*np.arange(81))
    unforced = replace(whole,grids=tuple(g[indices] for g in whole.grids),
                       data={k:v[indices] for k,v in whole.data.items()})
    assert forced.shape == unforced.shape == (81,81)
    target, predictors, truth = live.R.parse(forced.truth[0])
    terms, opaque = live.R.symbols(forced,[target,*predictors])
    disp, = [i for i,t in enumerate(terms) if t.data == (('u',(0,3)),)]
    force, = [i for i,t in enumerate(terms) if t.opaque]
    nonlinear, = set(range(1,len(terms))) - {disp,force}
    np.testing.assert_allclose([truth[i-1] for i in (nonlinear,disp,force)], [-6.,-1.,1.])
    intrinsic = [terms[i] for i in (0,nonlinear,disp)]
    plans = [read_archive(directory('kdv_cossin',s,'correlated'))['plan'] for s in SEEDS]
    rows = []
    for i, seed in enumerate(range(936811,936823)):
        plan = {k:tuple(v) for k,v in plans[i%len(plans)].items()}
        forced_bank = live.R.LocalBank(live.R.axes_of(forced),terms,opaque,plan)
        calibration_bank = live.R.LocalBank(live.R.axes_of(unforced),intrinsic,{},plan)
        def matrix(values):
            return forced_bank.matrix(values)[:,[0,nonlinear,disp,force]]
        clean = matrix(forced.data)
        for family in ('gaussian','correlated'):
            def draw(p, offset):
                return p.noisy(5,seed+offset) if family=='gaussian' else measurement(family)(p,5,seed+offset)[0]
            first, repeat = draw(forced,0), draw(forced,10000)
            fa = matrix(first)
            avg = matrix({k:(v+repeat[k])*.5 for k,v in first.items()})
            cal = calibration_bank.matrix(draw(unforced,10000))
            held = matrix(draw(forced,20000))
            assert cal.shape[0] == fa.shape[0]
            shared = np.linalg.lstsq(cal[:,1:],cal[:,0],rcond=None)[0]
            source = np.linalg.lstsq(fa[:,3:],fa[:,0]-fa[:,1:3]@shared,rcond=None)[0]
            stacked = np.vstack((fa/np.linalg.norm(fa[:,0]),
                                np.column_stack((cal,np.zeros(len(cal))))/np.linalg.norm(cal[:,0])))
            fits = dict(average_forced=np.linalg.lstsq(avg[:,1:],avg[:,0],rcond=None)[0],
                        joint=np.linalg.lstsq(stacked[:,1:],stacked[:,0],rcond=None)[0],
                        staged=np.r_[shared,source])
            def wape(bank, beta):
                return np.abs(bank[:,0]-bank[:,1:]@beta).sum()/np.abs(bank[:,0]).sum()
            reference = np.array([-6., -1., 1.])
            for method,beta in fits.items():
                rows.append(dict(seed=seed, family=family, method=method, nonlinearity=beta[0],
                                 dispersion=beta[1], forcing=beta[2], abs_error=abs(beta[1]+1),
                                 clean_wape=wape(clean,beta), held_wape=wape(held,beta),
                                 physical_clean_wape=wape(clean,reference), physical_held_wape=wape(held,reference)))
    out = StringIO(newline='')
    writer = csv.DictWriter(out,fieldnames=rows[0],lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(ROOT/'kdv_calibration.csv',out.getvalue())
    print(f'Validated {len(rows)} fits using equal-size recordings.')


if __name__=='__main__':
    main()
