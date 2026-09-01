#import packages needed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#go through each trial and plot data
for trial in ['Trial1', 'Trial2', 'Trial3']: 
    DPmean_data_RB0 = np.load(trial+'//DPmean_data_RB0.npy') #mean value of upper link angle
    DPmean_data_RB1 = np.load(trial+'//DPmean_data_RB1.npy') #mean value of lower link angle
    DPstd_data_RB0 = np.load(trial+'//DPstd_data_RB0.npy') #standard deviation of upper link angle
    DPstd_data_RB1 = np.load(trial+'//DPstd_data_RB1.npy') #standard deviation of lower link angle
    
    
    
    #------------------------------Plotting------------------------------------
    gs = gridspec.GridSpec(4,1) 
    TextSize = 15
    plt.figure(0) 
    plt.figure(figsize=(10,10))  
    
    ax = plt.subplot(gs[0, 0])
    plt.xlabel('$t$',size = TextSize)
    plt.ylabel(r'$\phi_1$',size = TextSize)
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.plot(DPmean_data_RB0[0], DPmean_data_RB0[1],'b')
    
    ax = plt.subplot(gs[1, 0])
    plt.xlabel('$t$',size = TextSize)
    plt.ylabel(r'$\sigma_{\phi_1}$',size = TextSize)
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.plot(DPstd_data_RB0[0], DPstd_data_RB0[1],'b', alpha = 0.7)
    
    ax = plt.subplot(gs[2, 0])
    plt.xlabel('$t$',size = TextSize)
    plt.ylabel(r'$\phi_1$',size = TextSize)
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.plot(DPmean_data_RB1[0], DPmean_data_RB1[1],'g')
    
    ax = plt.subplot(gs[3, 0])
    plt.xlabel('$t$',size = TextSize)
    plt.ylabel(r'$\sigma_{\phi_2}$',size = TextSize)
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.plot(DPstd_data_RB1[0], DPstd_data_RB1[1],'g', alpha = 0.7)
    
    plt.show()