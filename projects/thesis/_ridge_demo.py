"""Demonstrate how to avoid the enormous cancelling coefficients: the collinear
fit {v,u,du/dx0} -> dw/dx0*u^2 has a non-unique least-squares solution. np.linalg
.solve (what EPDE uses) returns a huge-norm member; min-norm (lstsq/pinv) and
ridge return the SMALL-norm member with the SAME (or barely changed) fit."""
import os, sys
_THIS = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(_THIS, '..', '..')), _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from term_contribution import build_pool
from lorenz_dominator_check import _design, _wape

LINE = ("63037981.63349468 * v{power: 1.0} + -63036502.10721292 * u{power: 1.0} + "
        "-6303806.5546302125 * du/dx0{power: 1.0} + 0.0 = dw/dx0{power: 1.0} * u{power: 2.0}")

cfg, search, all_vars = build_pool('lorenz')
X, y, labels = _design(LINE, search.pool, all_vars)
print(f"features={labels}  target=dw/dx0*u^2   X shape={X.shape}")
print(f"feature collinearity: cond(Xc) = {np.linalg.cond(X - X.mean(0)):.2e}\n")

# How EPDE solves it: normal-equations via solve(G, Gy) on the centered design.
Xc = X - X.mean(0); yc = y - y.mean()
G = Xc.T @ Xc
try:
    b_solve = np.linalg.solve(G, Xc.T @ yc)
except np.linalg.LinAlgError:
    b_solve = np.full(X.shape[1], np.nan)
print(f"{'method':<28}{'max|coef|':>14}{'||coef||':>14}{'WAPE':>10}")
def report(name, b):
    pred = Xc @ b
    print(f"{name:<28}{np.max(np.abs(b)):>14.3e}{np.linalg.norm(b):>14.3e}{_wape(yc, pred):>10.4f}")

report('np.linalg.solve (EPDE)', b_solve)
# Min-norm least squares (pinv / lstsq) -- SAME fit, smallest-norm coefficients.
report('lstsq min-norm (pinv)', np.linalg.lstsq(Xc, yc, rcond=1e-10)[0])
# Ridge / Tikhonov (G + lambda I) -- the VWSR-ridge from the opening task.
for lam in (1e-3, 1.0, 1e3):
    report(f'ridge  lambda={lam:g}', np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ yc))
print("\n-> min-norm & ridge give O(1) coefficients with the SAME poor fit (WAPE ~0.47):")
print("   the huge cancelling coefs were purely a solve()-on-singular artifact.")
