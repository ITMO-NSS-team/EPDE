import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from numpy.polynomial import Polynomial

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



def nn_autograd_simple(model, points, order,axis=0):
    points.requires_grad=True
    f = model(points).sum()
    for i in range(order):
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:,axis].sum()
    return grads[:,axis]


def nn_autograd_mixed(model, points,axis=[0]):
    points.requires_grad=True
    f = model(points).sum()
    for ax in axis:
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:,ax].sum()
    return grads[:,axis[-1]]



def nn_autograd(*args,axis=0):
    model=args[0]
    points=args[1]
    if len(args)==3:
        order=args[2]
        grads=nn_autograd_simple(model, points, order,axis=axis)
    else:
        grads=nn_autograd_mixed(model, points,axis=axis)
    return grads.reshape(-1,1)


def NN_interpolate(data,grid,batch_size=64):
    
    device = torch.device('cpu')

    grid=torch.from_numpy(np.array(grid).reshape(-1,1)).float()
    data=torch.from_numpy(np.array(data).reshape(-1,1)).float()

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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    t=0

    loss_mean=1000
    min_loss=np.inf

    print('Starting NN interpolation')
    while loss_mean>1e-5 and t<1e5:

        # X is a torch Variable
        permutation = torch.randperm(grid.size()[0])
    
        loss_list=[]
    
        for i in range(0,grid.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = grid[indices], data[indices]

            # in case you wanted a semi-full example
            # outputs = model.forward(batch_x)
            loss = torch.mean(torch.abs(batch_y-model(batch_x)))#+0.1*torch.tensor([torch.mean(torch.abs(p)) for p in model.parameters()]).sum().item()

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_mean=np.mean(loss_list)
        if loss_mean<min_loss:
            best_model=model
            min_loss=loss_mean
        #print('Surface trainig t={}, loss={}'.format(t,loss_mean))
        t+=1
    print('NN interpolation done')
    return best_model



def compute_derivs(data,grid,interp_mode='NN',diff_mode='FD', plot=False,save_derivs=False):
 
    

    if interp_mode=='NN':
        nn_grid=torch.from_numpy(np.array(grid).reshape(-1,1)).float()
        model=NN_interpolate(data,grid)
        d0np=model(nn_grid).detach().numpy().reshape(-1)
    elif 'poly' in interp_mode:
        
        deg=int(interp_mode.split('_')[-1])
        
        p=Polynomial

        pfit=p.fit(grid, data, deg=deg)

        d0np=pfit(grid)






    if diff_mode=='FD':
        d1np=take_der_grid_function(d0np,grid,grid,der_order=1,acc_order=6)
        d2np=take_der_grid_function(d0np,grid,grid,der_order=2,acc_order=6)
    elif diff_mode=='autograd' and interp_mode=='NN':
        d1=nn_autograd_simple(model, nn_grid, 1)
        d1np=d1.detach().numpy().reshape(-1)
        d2=nn_autograd_simple(model, nn_grid, 2)
        d2np=d2.detach().numpy().reshape(-1)
    elif diff_mode=='poly' and 'poly' in interp_mode:
        d1np=pfit.deriv(m=1)(grid)

        d2np=pfit.deriv(m=2)(grid)
    else:
        raise RuntimeError('differentiation_mode not implemented')


    if plot:

        plt.plot(grid, data)
        plt.show()
        plt.close()

        plt.plot(grid, data)
        plt.plot(grid, d0np)
        plt.show()
        plt.close()


        plt.plot(grid,d1np)
        plt.show()
        plt.close()

        plt.plot(grid,d2np)
        plt.show()
        plt.close()

    if save_derivs:
        data_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'optics_data'))

        np.savetxt(os.path.join(data_dir,'{}_d0.csv'.format(r0_fix)).format(r0_fix),d0np[:len(d2np)])
        np.savetxt(os.path.join(data_dir,'{}_d1.csv'.format(r0_fix)).format(r0_fix),d1np[:len(d2np)])
        np.savetxt(os.path.join(data_dir,'{}_d2.csv'.format(r0_fix)).format(r0_fix),d2np[:len(d2np)])

    return d0np,d1np,d2np
