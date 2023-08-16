#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import os
import sys
import pandas as pd
import re




def read_eqn(results_dir,eqn_id):

    file_path=os.path.join(results_dir,eqn_id)

    file=open(file_path)

    lines=file.readlines()

    terms=lines[0].replace("\n","").replace(" ","").replace("{power:1.0}","").replace("=","=1.0*").replace('du/dx1*u','u*du/dx1').replace('d^2u/dx1^2*u','u*d^2u/dx1^2')

    terms=re.split("\+|\=",terms)

    eqn={}

    for term in terms:
        term1=term.split('*')
        if float(term1[0])!=0:
            if len(term1)==1:
                eqn["C"]=float(term1[0])
            elif len(term1)==2:
                eqn[str(term1[1])]=float(term1[0])
            else:
                s1="*"
                eqn[s1.join(term1[1:])]=float(term1[0])
        
    return eqn



sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

os.chdir(os.path.abspath(os.path.join(os.path.dirname( __file__ ))))


results_dir=os.path.join(os.path.dirname( __file__ ), 'results_oprics')


eqn_list=['eqn_0.1_0.txt','eqn_0.1_1.txt','eqn_0.1_2.txt','eqn_0.1_3.txt','eqn_0.1_4.txt','eqn_0.2_2.txt','eqn_0.3_0.txt','eqn_0.3_1.txt','eqn_0.3_2.txt'
          ,'eqn_0.3_3.txt','eqn_0.4_0.txt','eqn_0.4_1.txt','eqn_0.4_3.txt','eqn_0.5_0.txt','eqn_0.5_1.txt','eqn_0.5_2.txt','eqn_0.5_4.txt']



read_eq_dict={}

for eq in eqn_list:
    read_eq_dict[eq]=read_eqn(results_dir,eq)

df=pd.DataFrame(read_eq_dict).transpose().fillna(0)

print(df)

divisor=df['d^2u/dx1^2'].values

for col in df.columns:
    df[col]=df[col].values/divisor


df.to_csv('total_results_optics.csv')




#d={}

#for eqn in read_eq_list:
#    for term in eqn:
#        if term in d.keys():
#            d[term]+=1
#        else:
#            d[term]=1



#values=np.array(list(d.values()))

#tokens=np.array(list(d.keys()))

#sorted_indices=np.argsort(values)

#sorted_tokens=tokens[sorted_indices][::-1]

#sorted_values=values[sorted_indices][::-1]

#print(sorted_tokens)
#print(sorted_values)


#for eq in eqn_list:
#    read_eq_list.append(read_eqn_coeffs(results_dir,eq))
