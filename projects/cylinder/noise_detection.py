#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:03:01 2021

@author: maslyaev
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def Heatmap(Matrix, interval = None, area = ((0, 1), (0, 1)), xlabel = '', ylabel = '', figsize=(8,6), filename = None, title = ''):
    y, x = np.meshgrid(np.linspace(area[0][0], area[0][1], Matrix.shape[0]), np.linspace(area[1][0], area[1][1], Matrix.shape[1]))
    fig, ax = plt.subplots(figsize = figsize)
    plt.xlabel(xlabel)
    ax.set(ylabel=ylabel)
    ax.xaxis.labelpad = -10
    if interval:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=interval[0], vmax=interval[1])    
    else:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=min(-abs(np.max(Matrix)), -abs(np.min(Matrix))),
                          vmax=max(abs(np.max(Matrix)), abs(np.min(Matrix)))) 
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.title(title)
    plt.show()
    if type(filename) != type(None): plt.savefig(filename + '.eps', format='eps')

if __name__ == '__main__':
    mat1=pd.read_csv('/home/maslyaev/epde/EPDE/tests/cylinder/data/Data_32_points_.dat',index_col=None,header=None,sep=' ')
    mat1_np = mat1.dropna(axis = 1).to_numpy()
    data_clear = mat1_np[:, 1:]
    files = {}
    directory = '/home/maslyaev/epde/EPDE/tests/cylinder/data/Noise_1'
    for file in os.listdir(directory):
        # print(file, type(file))
        filename = directory + '/' + file
        # print(filename)
        files[file] = np.loadtxt(filename, delimiter=' ', skiprows = 3, usecols=range(21))
        
    for filename in files.keys():
        std = np.std(files[filename][:3668, 1:] - data_clear[:3668, :20])
        print(filename, std)
        Heatmap(files[filename][:3668, 1:], title = filename + ' ' + str(std),
                interval = (np.min(files[filename][:3668, 1:]), np.max(files[filename][:3668, 1:])))