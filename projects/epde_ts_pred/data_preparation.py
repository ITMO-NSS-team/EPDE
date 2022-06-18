# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:01:59 2022

@author: user
"""

import pandas as pd
import numpy as np

import os
if not(os.path.isdir('prepared_data')):
    os.mkdir('prepared_data')


#file_number='8202026016'
# file_number='8202026032'
# file_number='8202026034'
file_number='8202026000' 


df=pd.read_csv("{}.dat".format(file_number),sep='	',skiprows=(1),na_values='--',decimal=',')
# df['deg']=df['deg']-int(np.nan_to_num(df['deg'])[np.nan_to_num(df['deg'])>0][0])
df['deg']=df['deg']%360


df=df[df['ms']>20]

angle_set=[]
angle_bundle=[]
for t,angle in enumerate(df['deg']):
    if angle!=angle:
        if len(angle_set)>0:
            angle_bundle.append(angle_set)
        angle_set=[]
    else:
        # if angle>180:
        #     angle=angle-360
        angle_set.append(angle)
            
        
    

        
angle_set_norm=[]

for some_set in angle_bundle:
    angle_set_norm.append(np.array(some_set)-some_set[0])
    

        
for i,arc in enumerate(angle_set_norm):
    df_halfper=pd.DataFrame(columns=['deg','t_arc'])
    for t,angle in enumerate(arc):
        angle0=arc[0]
        df_halfper=df_halfper.append({"deg":angle-angle0,"t_arc":t},ignore_index=True)
    df_halfper.to_csv('prepared_data/{}_halfper_{}.csv'.format(file_number, i))


df_complete=pd.DataFrame(columns=['deg','t_arc'])

for i,arc in enumerate(angle_set_norm):
    for t,angle in enumerate(arc):
        angle0=arc[0]
        df_complete=df_complete.append({"deg":angle-angle0,"t_arc":t},ignore_index=True)

electroduce_length_all=df_complete.groupby('t_arc').count().values.reshape(-1)
electroduce_length_sig=len(electroduce_length_all[electroduce_length_all>4])
        
median_arc=[np.quantile(df_complete[df_complete['t_arc']==time]['deg'],0.5) for time in range(electroduce_length_sig)]

# time_arc=[(0+0.25*i)*1e-3 for i in range(len(mean_arc))]    

time_arc=[i for i in range(len(median_arc))]

df_median=pd.DataFrame({"median_arc_time":time_arc,'median_arc':median_arc})    

df_median.to_csv('prepared_data/median_arc_{}.csv'.format(file_number))


mean_arc=[np.mean(df_complete[df_complete['t_arc']==time]['deg']) for time in df_complete['t_arc'].unique()]

# time_arc=[(0+0.25*i)*1e-3 for i in range(len(mean_arc))]    

time_arc=[i for i in range(len(mean_arc))]

df_mean=pd.DataFrame({"mean_arc_time":time_arc,'mean_arc':mean_arc})    

df_mean.to_csv('prepared_data/mean_arc_{}.csv'.format(file_number))


