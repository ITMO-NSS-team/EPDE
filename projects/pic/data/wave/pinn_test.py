"""
1D Wave Equation PINN -- DeepXDE, trained on a gridded dataset.

Data file: wave_sln_80.csv  (81 x 81)
    rows    = space x (Nx = 81 points)
    columns = time  t (Nt = 81 points)
    U[0, :] = U[-1, :] = 0  -> zero Dirichlet BCs at x = X_MIN, X_MAX

PDE:    u_tt = c^2 * u_xx
Loss:   PDE residual + Dirichlet BC + gridded observations
        (the IC and u_t IC are implicitly enforced by the t = 0 column
         of observations, so no separate IC term is needed)
"""

import deepxde as dde
import numpy as np


# ============================================================ user settings
DATA_PATH    = "./wave_sln_80.csv"   ### EDIT ###
C            = 0.2          # wave speed; PDE is u_tt = 0.04 * u_xx  =>  c^2 = 0.04
X_MIN, X_MAX = 0.0, 1.0     ### EDIT ### spatial extent of the grid
T_MIN, T_MAX = 0.0, 1.0     ### EDIT ### time extent (verify this!)


# ============================================================ load data
U = np.loadtxt(DATA_PATH, delimiter=",")           # shape (Nx, Nt)
Nx, Nt = U.shape

x_grid = np.linspace(X_MIN, X_MAX, Nx)
t_grid = np.linspace(T_MIN, T_MAX, Nt)
XX, TT = np.meshgrid(x_grid, t_grid, indexing="ij")  # both (Nx, Nt)

X_obs = np.stack([XX.ravel(), TT.ravel()], axis=1)   # (Nx*Nt, 2)
u_obs = U.reshape(-1, 1)                             # (Nx*Nt, 1)


# ============================================================ PDE residual
def pde(x, u):
    u_tt = dde.grad.hessian(u, x, i=1, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    return u_tt - C ** 2 * u_xx


# ============================================================ geometry
geom = dde.geometry.Interval(X_MIN, X_MAX)
timedomain = dde.geometry.TimeDomain(T_MIN, T_MAX)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# ============================================================ constraints
bc = dde.icbc.DirichletBC(
    geomtime,
    lambda x: 0.0,
    lambda _, on_boundary: on_boundary,
)

# observations -- this is what makes it "fit my data"
observe_u = dde.icbc.PointSetBC(X_obs, u_obs, component=0)


# ============================================================ training set
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, observe_u],
    num_domain=8000,
    num_boundary=200,
    num_test=10000,
)


# ============================================================ network + train
# tanh MLP, 5 hidden layers x 64 units. Output range is ~[-6.3, 6.3].
net = dde.nn.FNN([2] + [64] * 5 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# loss order: [pde, bc, observations] -- boost data term so the net latches on
LW = [1.0, 100.0, 100.0]

model.compile("adam", lr=1e-3, loss_weights=LW)
model.train(iterations=20000, display_every=1000)

model.compile("L-BFGS", loss_weights=LW)
losshistory, train_state = model.train()


# ============================================================ evaluation
U_pred = model.predict(X_obs).reshape(Nx, Nt)
rel_l2 = np.linalg.norm(U_pred - U) / np.linalg.norm(U)
print(f"Relative L2 error vs. dataset: {rel_l2:.4e}")
print(f"Max abs error               : {np.abs(U_pred - U).max():.4e}")

# residual of the learned solution on the data grid -- should be ~0 if c is right
def residual_grid():
    import torch  # or tf depending on backend; deepxde handles the autograd
    pts = X_obs
    pred = model.predict(pts, operator=pde)
    return pred.reshape(Nx, Nt)

print(f"Mean |PDE residual| on grid : {np.abs(residual_grid()).mean():.4e}")

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
np.savez("wave_pinn_prediction.npz",
         x=x_grid, t=t_grid, u_data=U, u_pred=U_pred)