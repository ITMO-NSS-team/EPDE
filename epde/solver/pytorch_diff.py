# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from mpl_toolkits.mplot3d import Axes3D
from epde.solver.solver import solver
import time


x=torch.from_numpy(np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
t=torch.from_numpy(np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))

grid=torch.cartesian_prod(x,t).float()

model = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 100),
    torch.nn.Tanh(),
    torch.nn.Linear(100, 1),
    torch.nn.Tanh()
)



bnd1=torch.cartesian_prod(x,torch.from_numpy(np.array([0],dtype=np.float64))).float()

bndval1=torch.sin(np.pi*bnd1[:,0])

bnd2=torch.cartesian_prod(x,torch.from_numpy(np.array([1],dtype=np.float64))).float()

bndval2=torch.sin(np.pi*bnd2[:,0])

bnd3=torch.cartesian_prod(torch.from_numpy(np.array([0],dtype=np.float64)),t).float()

bndval3=torch.from_numpy(np.zeros(len(bnd3),dtype=np.float64))

bnd4=torch.cartesian_prod(torch.from_numpy(np.array([1],dtype=np.float64)),t).float()

bndval4=torch.from_numpy(np.zeros(len(bnd4),dtype=np.float64))

bconds=[[bnd1,bndval1],[bnd2,bndval2],[bnd3,bndval3],[bnd4,bndval4]]

operator=[[4,[0,0],1],[-1,[1,1],1]] # Для произвольной ф-ции: [fun_torch_tensor, None, 0]


start=time.time()
model=solver(grid,model,operator,bconds,lambda_bound=10,verbose=False,learning_rate=0.001)
end=time.time()

print('Time taken= ',end-start)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(grid[:,0].reshape(-1), grid[:,1].reshape(-1), model(grid).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2)
plt.show()

