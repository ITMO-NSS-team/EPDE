import numpy as np

import torch
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def weights(M, N, x0, grid):
    delta = np.zeros((M+1, N+1, N+1))
    delta[0, 0, 0] = 1
    c1 = 1
    for n in range(1, N+1):
        c2 = 1
        for nu in range(n):
            c3 = grid[n]-grid[nu]
            c2 *= c3
            m1 = min([n, M])+1
            for m in range(m1):
                delta[m, n, nu] = (
                    (grid[n]-x0)*delta[m, n-1, nu]-m*delta[m-1, n-1, nu])/c3
        for m in range(m1):
            delta[m, n, n] = c1/c2 * \
                (m*delta[m-1, n-1, n-1]-(grid[n-1]-x0)*delta[m, n-1, n-1])
        c1 = c2
    return delta

def sort_grid(x0,der_grid, return_pos=False):
    x0vect=np.zeros_like(der_grid)+x0
    dist=(x0vect-der_grid)**2
    position=np.argsort(dist)
    sorted_grid=np.take(der_grid,np.argsort(dist))
    if return_pos:
        return sorted_grid, position
    else:
        return sorted_grid


def take_der_grid_function(grid_func,grid,der_grid,der_order=1,acc_order=2):
    der_list=[]
    for x0 in grid:
        sorted_grid,position=sort_grid(x0,der_grid,return_pos=True)
        delta=weights(der_order,len(der_grid)-1,x0,sorted_grid)
        #der_sum=sum(delta[der_order, acc_order, :]*func(der_grid))
        sgf=np.take(grid_func,position)
        der_sum=sum(delta[der_order, acc_order, :]*sgf)
        der_list.append(der_sum)
    return der_list



r0_fix=0.1

plot=False

data_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'optics_data'))

grid_file=os.path.join(data_dir,'grid_{}.csv'.format(r0_fix))

rv_file=os.path.join(data_dir,'R_{}.csv'.format(r0_fix))

grid=np.genfromtxt(grid_file, delimiter=',')


rv=np.genfromtxt(rv_file, delimiter=',')
    

m_grid=grid/np.max(grid)


rv=rv[20:]
m_grid=m_grid[20:]

rv=(rv-np.min(rv))/np.max(rv-np.min(rv))

if plot:
    plt.plot(m_grid, rv)
    plt.show()


device = torch.device('cpu')

grid=torch.from_numpy(np.array(m_grid).reshape(-1,1)).float()
data=torch.from_numpy(np.array(rv).reshape(-1,1)).float()

grid.to(device)
data.to(device)

model = torch.nn.Sequential(
    torch.nn.Linear(1, 256),
    torch.nn.Tanh(),
    # torch.nn.Dropout(0.1),
    # torch.nn.ReLU(),
    torch.nn.Linear(256, 64),
    # # torch.nn.Dropout(0.1),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1024),
    # torch.nn.Dropout(0.1),
    torch.nn.Tanh(),
    torch.nn.Linear(1024, 1)
    # torch.nn.Tanh()
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



batch_size = 64 # or whatever


t=0

loss_mean=1000
min_loss=np.inf


while loss_mean>1e-5 and t<1e4:

    # X is a torch Variable
    permutation = torch.randperm(grid.size()[0])
    
    loss_list=[]
    
    for i in range(0,grid.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = grid[indices], data[indices]

        # in case you wanted a semi-full example
        # outputs = model.forward(batch_x)
        loss = torch.mean((batch_y-model(batch_x))**2)#+0.1*torch.tensor([torch.mean(torch.abs(p)) for p in model.parameters()]).sum().item()

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    loss_mean=np.mean(loss_list)
    if loss_mean<min_loss:
        best_model=model
        min_loss=loss_mean
    print('Surface trainig t={}, loss={}'.format(t,loss_mean))
    t+=1
model=best_model

if plot:
    plt.plot(m_grid, rv)
    plt.plot(m_grid, model(grid).detach().numpy().reshape(-1))
    plt.show()

d0np=model(grid).detach().numpy().reshape(-1)


d1np=take_der_grid_function(d0np,m_grid,m_grid,der_order=1,acc_order=6)

if plot:
    plt.plot(m_grid,d1np)
    plt.show()
    plt.close()


d2np=take_der_grid_function(d0np,m_grid,m_grid,der_order=2,acc_order=6)

if plot:
    plt.plot(m_grid,d2np)
    plt.show()
    plt.close()


d0np=np.array(d0np,dtype=np.float64)
d1np=np.array(d1np,dtype=np.float64)
d2np=np.array(d2np,dtype=np.float64)

def functional(A,lam):
    const=np.zeros(len(d0np),dtype=np.float64)+1
    predictors=[d0np*d1np,d2np,const,d0np,d0np*d2np]
    vect=const-1
    for i,coeff in enumerate(A):
        vect+=float(coeff)*predictors[i]
    disc=d1np-vect
    return np.linalg.norm(disc)+lam*np.linalg.norm(A,ord=1)


print(functional([1.0,1.0,1.0,1.9,1.9],100))

from scipy.optimize import minimize
from tedeous.input_preprocessing import Equation
from tedeous.solver import Solver
from tedeous.device import solver_device

for lam in [10**(-10),10**(-9),10**(-8),10**(-7),10**(-6),10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),10**0,10**1,10**2,10**3,10**4,10**5]:
    print(float(lam))
    res = minimize(functional, [1,1,1,1,1],args=(float(lam)),options={'maxiter':1e6})
    xopt=res.x
    coeffs=np.zeros(6)+1
    coeffs[0:3]=xopt[0:3]
    coeffs[4:]=xopt[3:]
    print(coeffs)


    #coord_list = [m_grid]

    #coord_list=torch.tensor(coord_list)
    #grid=coord_list.reshape(-1,1).float()

    ## point t=0
    #bnd1 = torch.from_numpy(np.array([[float(m_grid[0])]], dtype=np.float64)).float()
    
    
    ##  So u(0)=-1/2
    #bndval1 = torch.from_numpy(np.array([[float(rv[0])]], dtype=np.float64))

    ## point t=0
    #bnd3 = torch.from_numpy(np.array([[float(m_grid[1])]], dtype=np.float64)).float()
    
    
    ##  So u(0)=-1/2
    #bndval3 = torch.from_numpy(np.array([[float(rv[1])]], dtype=np.float64))


    ## point t=0
    #bnd2 = torch.from_numpy(np.array([[float(m_grid[-1])]], dtype=np.float64)).float()
    
    
    ##  So u(0)=-1/2
    #bndval2 = torch.from_numpy(np.array([[float(rv[-1])]], dtype=np.float64))    



    #    # Putting all bconds together
    #bconds = [[bnd1, bndval1, 'dirichlet'],
    #            [bnd2, bndval2, 'dirichlet'],
    #            [bnd3, bndval3, 'dirichlet']
    #            ]

    #solver_device('gpu')

    #eq = {
    #    'u*du/dt':
    #        {
    #            'coeff': float(coeffs[0]),
    #            'u*du/dt': [[None], [0]],
    #            'pow': [1,1],
    #            'var':[0,0]
    #        },
    #    'd2u/dt2':
    #        {
    #            'coeff': float(coeffs[1]),
    #            'd2u/dt2': [0, 0],
    #            'pow': 1,
    #            'var':0
    #        },
    #    'const':
    #        {
    #            'coeff': float(coeffs[2]),
    #            'u': [None],
    #            'pow': 0,
    #            'var':0
    #        },
    #    'du/dt':
    #        {
    #            'coeff': float(coeffs[3]),
    #            'd2u/dt2': [0],
    #            'pow': 1,
    #            'var':0
    #        },
    #    'u':
    #        {
    #            'coeff': float(coeffs[4]),
    #            'u': [None],
    #            'pow': 1,
    #            'var':0
    #        },
    #    'u*d2u/dt2':
    #        {
    #            'coeff': float(coeffs[5]),
    #            'u*du/dt': [[None], [0,0]],
    #            'pow': [1,1],
    #            'var':[0,0]
    #        },
    #}

    #equation = Equation(grid, eq, bconds).set_strategy('autograd')

    #img_dir=os.path.join(os.path.dirname( __file__ ), 'sparse_regression')

    #model = torch.nn.Sequential(
    #torch.nn.Linear(1, 100),
    #torch.nn.Tanh(),
    #torch.nn.Linear(100, 100),
    #torch.nn.Tanh(),
    #torch.nn.Linear(100, 100),
    #torch.nn.Tanh(),
    #torch.nn.Linear(100, 1)
    #)

    #model = Solver(grid, equation, model, 'autograd').solve(lambda_bound=1000,verbose=1, learning_rate=1e-3,
    #                                        eps=1e-6, tmin=1000, tmax=1e6,use_cache=True,cache_verbose=True,
    #                                        save_always=True,print_every=None,model_randomize_parameter=1e-4,
    #                                        optimizer_mode='LBFGS',no_improvement_patience=1000,patience=5,step_plot_print=False,step_plot_save=True,image_save_dir=img_dir)

            
    #plt.plot(m_grid, rv, '+', label = 'test data')
    #plt.plot(m_grid, model(grid.cpu()).reshape(-1).detach().numpy(), color = 'r', label='solution of the discovered ODE')
    #plt.show()






    #plt.legend(loc='upper right')
    #img_filename=os.path.join(results_dir,'sln_{}_{}.png'.format(r0_fix,i))
    #plt.savefig(img_filename)
    #plt.close()
    #txt_filename=os.path.join(results_dir,'eqn_{}_{}.txt'.format(r0_fix,i))
    #with open(txt_filename, 'w') as the_file:
    #    the_file.write(text_eq[i])
