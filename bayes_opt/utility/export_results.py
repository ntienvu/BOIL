# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 21:37:03 2016

@author: Vu
"""

import sys
sys.path.insert(0,'../..')
sys.path.insert(0,'../')
import numpy as np
import pickle
import os
import sys

out_dir="pickle_storage"

def print_result(bo,myfunction,Score,mybatch_type,acq_type):

    print_result_sequential(bo,myfunction,Score,mybatch_type,acq_type)

def print_result_sequential(bo,myfunction,Score,method_type,acq_type):
    
    if 'ystars' in acq_type:
        acq_type['ystars']=[]
    if 'xstars' in acq_type:
        acq_type['xstars']=[]
        
    #Regret=Score["Regret"]
    ybest=Score["ybest"]
    MyTime=Score["MyTime"]
    
    print('{:s} {:d}'.format(myfunction.name,myfunction.input_dim))
  
    MaxFx=[val.max() for idx,val in enumerate(ybest)]

    
    if myfunction.ismax==1:
        print('MaxBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx)))    
    else:
        print('MinBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx)))
        
    
    if 'MyOptTime' in Score:
        MyOptTime=Score["MyOptTime"]

        print('OptTime/Iter={:.1f}({:.1f})'.format(np.mean(MyOptTime),np.std(MyOptTime)))
        
    try:
        strFile="{:s}_{:s}_{:d}_{:s}_{:s}.pickle".format(myfunction.name,myfunction.alg_name,myfunction.input_dim,method_type['name'],acq_type['name'])
    except:
        strFile="{:s}_{:d}_{:s}_{:s}.pickle".format(myfunction.name,myfunction.input_dim,method_type['name'],acq_type['name'])
    if sys.version_info[0] < 3:
        version=2
    else:
        version=3
        
    path=os.path.join(out_dir,strFile)
    
    if version==2:
        with open(path, 'wb') as f:
            pickle.dump([ybest, MyTime,bo[-1].bounds,MyOptTime], f)
    else:
        pickle.dump( [ybest, MyTime,bo,MyOptTime], open( path, "wb" ) )


       
def yBest_Iteration(YY,BatchSzArray,step=3):
    
    nRepeat=len(YY)
    
    result=[0]*nRepeat

    for ii,yy in enumerate(YY):
        result[ii]=[np.max(yy[:uu+1]) for uu in range(len(yy))]
        
    result=np.asarray(result)
    
    result_mean=np.mean(result,axis=0)
    result_mean=result_mean[BatchSzArray[0]-1:]
    result_std=np.std(result,axis=0)
    result_std=result_std[BatchSzArray[0]-1:]
    
    return result_mean[::step], result_std[::step], None, None


    std_cum_TT=np.std(mean_cum_simple_regret_TT,axis=0)
    std_cum_TT=np.array(std_cum_TT).ravel()
    mean_cum_simple_regret_TT=np.mean(mean_cum_simple_regret_TT,axis=0)
   
    #return mean_TT[::step],std_TT[::step]#,mean_cum_TT[::5],std_cum_TT[::5]
    #return mean_TT,std_TT,np.mean(mean_cum_simple_regret_TT),np.mea(std_cum_TT)
    
    #half_list_index=np.int(len(mean_cum_simple_regret_TT)*0.5)
    #return np.mean(mean_cum_simple_regret_TT[half_list_index:]),np.mean(std_cum_TT[half_list_index:])
    return np.mean(mean_cum_simple_regret_TT),np.mean(std_cum_TT)