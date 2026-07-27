"""Is the live mixed 5-term form an analytical alternative, or an FD-error overfit?
Refit its structure with ANALYTIC vs FD derivatives of the heat kernel and see if
its extra terms (T_t, T_xx, T_yy beyond the pure-xT T*T_xx, T*T_yy) survive."""
import numpy as np
k, t0, nt, ny, nx = 0.5, 10.0, 30, 25, 25
t = np.arange(nt, dtype=float); x = np.linspace(-10, 10, nx); y = np.linspace(-10, 10, ny)
T_, Y, X = np.meshgrid(t, y, x, indexing='ij')
tau = T_ + t0; r2 = X*X + Y*Y
T   = np.exp(-r2/(4*k*tau)) / (4*np.pi*k*tau)            # 2D heat kernel
# analytic derivatives (exact: T_t == k*(T_xx+T_yy))
Txx = T*(X*X/(4*k*k*tau*tau) - 1/(2*k*tau))
Tyy = T*(Y*Y/(4*k*k*tau*tau) - 1/(2*k*tau))
Tt  = T*(r2/(4*k*tau*tau) - 1/tau)

def fd(F, ax, c):
    return np.gradient(F, c, axis=ax)
Txx_fd = fd(fd(T,2,x),2,x); Tyy_fd = fd(fd(T,1,y),1,y); Tt_fd = fd(T,0,t)

def fit_mixed(T,Tt,Txx,Tyy, tag):
    s = (slice(3,-3),)*3
    T,Tt,Txx,Tyy = (a[s].ravel() for a in (T,Tt,Txx,Tyy))
    # MIXED structure: target = T*Tt ; features = [T_t, T*T_xx, T_xx, T_yy, T*T_yy]
    tgt = T*Tt
    cols = np.column_stack([Tt, T*Txx, Txx, Tyy, T*Tyy])
    names = ['T_t', 'T*T_xx', 'T_xx', 'T_yy', 'T*T_yy']
    c,*_ = np.linalg.lstsq(cols, tgt, rcond=None)
    resid = tgt - cols@c
    # scale-invariant (pointwise cancellation) discrepancy vs the same for TRUTH (pure xT)
    contribs = cols*c
    rho = np.mean(np.abs(resid)/(np.abs(tgt)+np.sum(np.abs(contribs),1)+1e-12))
    print(f"\n[{tag}] mixed structure refit (target = T*T_t):")
    for n,ci in zip(names,c):
        flag = '   <-- EXTRA (should be ~0 if analytical)' if n in ('T_t','T_xx','T_yy') else ''
        print(f"    {n:<8} = {ci:+.6f}{flag}")
    print(f"    pointwise-cancellation discrepancy = {rho:.6f}")
    extra = abs(c[0])+abs(c[2])+abs(c[3]); core = abs(c[1])+abs(c[4])
    print(f"    |extra terms| = {extra:.6f}   |core T*Txx,T*Tyy| = {core:.6f}   ratio extra/core = {extra/core:.4f}")

fit_mixed(T,Tt,Txx,Tyy, "ANALYTIC derivs (zero FD error)")
fit_mixed(T,Tt_fd,Txx_fd,Tyy_fd, "FD derivs (what EPDE sees)")
