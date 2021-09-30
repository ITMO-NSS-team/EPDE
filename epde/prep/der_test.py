# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:55:46 2021

@author: user
"""
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

# mat=pd.read_csv('Data_32_points_.dat',index_col=None,header=None,sep=' ')

# t_grid=torch.from_numpy(mat[0].values)

# r_grid=torch.from_numpy(np.arange(0.5,16.5,0.5))

# data=torch.from_numpy(mat[range(1,33)].values.reshape(-1,1))

# grid = torch.cartesian_prod(t_grid, r_grid).float()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.set_title('Iteration = ' + str(t))
# ax.plot_trisurf(grid[:, 0].reshape(-1), grid[:, 1].reshape(-1),
#                 data.reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
# ax.set_xlabel("t")
# ax.set_ylabel("r")
# plt.show()

mat1=pd.read_csv('Data_32_points_.dat',index_col=None,header=None,sep=' ')
mat1=mat1[range(1,33)]

rename_dict={}
for i in range(1,33):
    rename_dict[i]='r'+str(5*i)

mat1=mat1.rename(columns=rename_dict)

true_grid=[]
true_val=[]

for time in range(3001):
    for coord in range(1,10):
        true_grid.append([time*0.05,coord*0.5])
        true_val.append(mat1['r'+str(5*coord)][time])

true_grid=np.array(true_grid)
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
                true_val, cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()


device = torch.device('cpu')

grid=torch.from_numpy(np.array(true_grid)).float()
data=torch.from_numpy(np.array(true_val).reshape(-1,1)).float()

data_norm=(data-torch.min(data))/torch.max(data-torch.min(data))

# data_norm=data

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(true_grid[:, 0], true_grid[:, 1],
                data_norm.numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()



grid.to(device)
data_norm.to(device)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 1024),
    # torch.nn.Dropout(0.1),
    torch.nn.Sigmoid(),#Tanh(),
    torch.nn.Linear(1024, 1024),
    # torch.nn.Dropout(0.1),
    torch.nn.Sigmoid(),#Tanh(),
    torch.nn.Linear(1024, 1024),
    torch.nn.Sigmoid(),#Tanh(), # added
    torch.nn.Linear(1024, 64), # added
    
    # torch.nn.Dropout(0.1),
    torch.nn.Sigmoid(),#Tanh(),
    torch.nn.Linear(64, 1)
    # torch.nn.Tanh()
)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.05)

l1_lambda = 0.001
l1_norm =sum(p.abs().sum() for p in model.parameters())


loss = torch.mean(torch.abs(data_norm-model(grid)))+ l1_lambda * l1_norm

def closure():
    optimizer.zero_grad()
    l1_lambda = 0.001
    l1_norm =sum(p.abs().sum() for p in model.parameters())
    loss = torch.mean(torch.abs(data_norm-model(grid)))+ l1_lambda * l1_norm
    loss.backward()
    return loss

t=1
loss_prev=loss.item()
while loss>1e-5 and t<2*1e3:
    optimizer.step(closure)
    l1_lambda = 0.001
    l1_norm =sum(p.abs().sum() for p in model.parameters())
    loss = torch.mean(torch.abs(data_norm-model(grid)))+l1_lambda * l1_norm
    t+=1
    print('Surface trainig t={}, loss={}'.format(t,loss))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Iteration = ' + str(t))
ax.plot_trisurf(grid[:, 0].reshape(-1), grid[:, 1].reshape(-1),
                model(grid).detach().numpy().reshape(-1), cmap=cm.jet, linewidth=0.2, alpha=1)
ax.set_xlabel("t")
ax.set_ylabel("r")
plt.show()
