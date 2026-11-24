"""Coordinate-only repair for float32 Duffing time coordinates.

Historical integer windows are kept. Use their actual center coordinates and
the largest common symmetric radius contained in every intended native span.
The first-step extrapolation in the previous mapper overshoots the last raw
sample by 1.71e-6 in group 2. No extrapolation or tolerance relaxation.
"""
import run_case as C
np=C.np


def spec_from_plan(axes,plan,k=0):
    signed=np.r_[np.arange(k+1),-np.arange(1,k+1)];centers=[];radii=[]
    for a,cut,m,n,s in zip(axes,plan['cut'],plan['ms'],plan['shape'],plan['ss']):
        ids=cut+m+np.arange(n)*s
        assert np.min(ids-m)>=cut and np.max(ids+m)<len(a)-cut
        radius=float(min(np.min(a[ids]-a[ids-m]),np.min(a[ids+m]-a[ids])))
        assert radius>0
        centers.append(a[ids].tolist());radii.append(radius)
    return dict(q=list(plan['q']),centers=centers,radii=radii,interpolation_cut=list(plan['cut']),
                frequencies=[(signed*np.pi/r).tolist() for r in radii])


if __name__=='__main__':
    import argparse
    if C.S.os.name=='nt':C.R.ctypes.windll.kernel32.SetPriorityClass(C.R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(12,13,14),required=True);a=ap.parse_args()
    original_sources=C.sources
    def sources():return {**original_sources(),str(C.Path(__file__)):C.S.sha(__file__)}
    C.sources=sources;C.N.spec_from_plan=spec_from_plan
    if not (C.ROOT/'results'/f'duffing_seed{a.seed}.json').exists():C.run('duffing',a.seed)
