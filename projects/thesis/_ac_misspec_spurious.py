"""Caveat demo: per-point dominance protects against spurious ONLY while OLS
zeros their coef -- which needs the true term present. Here we MISSPECIFY the AC
library (drop the true diffusion u_xx) and inject interface-shaped spurious
(x*u_xx, sin(x)*u_xx, u*u_xx). With the true diffusion gone, the unexplained
interface residual is up for grabs: does a spurious term acquire a coef and
LOCALLY DOMINATE (non-zero dom%)?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from term_contribution import build_pool, analyze_equation

cfg, search, all_vars = build_pool('ac')
# True AC minus the diffusion term: du/dx0 = 5u - 5u^3   (u_xx REMOVED)
eqstr = "-5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} = du/dx0{power: 1.0}"
spurious = ['x{power: 1.0, dim: 1.0} * d^2u/dx1^2{power: 1.0}',
            'sin{power: 1.0, freq: 2.0, dim: 1.0} * d^2u/dx1^2{power: 1.0}',
            'u{power: 1.0} * d^2u/dx1^2{power: 1.0}']
rows, r2, tgt = analyze_equation(eqstr, spurious, search.pool, all_vars)
print(f"\n  MISSPECIFIED AC (true u_xx dropped)   target={tgt}   R^2={r2:.5f}")
print(f"    {'term':<46}{'kind':>6}{'coef':>12}{'L2':>8}{'per-pt':>8}{'dom%':>8}{'tgt%':>9}")
for (lbl, tr, coef, rms, share, share_tgt, expl, pp, relm, dom, dr2) in sorted(rows, key=lambda r: -r[7]):
    print(f"    {lbl:<46}{'TRUE' if tr else 'spur':>6}{coef:>12.4g}"
          f"{share:>8.1%}{pp:>8.1%}{dom:>8.1%}{expl:>9.1%}")
