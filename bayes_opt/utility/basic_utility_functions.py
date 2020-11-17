# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 21:37:03 2016

@author: Vu
"""

import itertools
import numpy as np
import tensorflow as tf
import random
from bayes_opt.curve_compression import transform_logistic_marginal
from tqdm import tqdm

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)
    
def generate_random_points(bounds,size=1):
    x_max = [np.random.uniform(x[0], x[1], size=size) for x in bounds]
    x_max=np.asarray(x_max)
    x_max=x_max.T
    return x_max


def best_output_to_best_input(output,X_input):
    # giving a list of output
    # return the corresponding input
    
    best_X=[0]*len(output)
    for ii in range(len(output)):
        best_X[ii]=[0]*len(output[ii])
        for jj in range(len(output[ii])):
            idxBest=np.argmax(output[ii][0:jj+1])
            best_X[ii][jj]=X_input[ii][idxBest]
            
    return best_X
    
def evaluating_the_final_utility(bo):
    # for a fair comparison, the final score is evaluated by marginalizing 
    # across different choices in the preference function (Logistic function)
    
    # ignore the intermediate curves
    idxReal=[idx for idx,val in enumerate(bo.markVirtualObs) if val==0]
    Y_curves=[val for idx,val in enumerate(bo.Y_curves) if idx in idxReal]  
            
    y_score=[0]*len(Y_curves)
    for ii,curve in enumerate(Y_curves):
        y_score[ii]=transform_logistic_marginal([curve],bo.SearchSpace[-1,1])
        
    return np.asarray(y_score)
        
def evaluating_final_util_MaxEpoch(bo):
    # for a fair comparison, the final score is evaluated by (1) training using MaxEpoch and 
    # (2) marginalizing across different choices in the preference function (Logistic function)
    
    # ignore the intermediate curves
    idxReal=[idx for idx,val in enumerate(bo.markVirtualObs) if val==0]
    Y_curves=[val for idx,val in enumerate(bo.Y_curves) if idx in idxReal]  
    T_original=[val for idx,val in enumerate(bo.T_original) if idx in idxReal]   
    Y_original=[val for idx,val in enumerate(bo.Y_original) if idx in idxReal]   
    X_original=[val for idx,val in enumerate(bo.X_original) if idx in idxReal]  
            
            
    y_score=[0]*len(Y_curves)
    for ii,curve in enumerate(Y_curves):
        y_score[ii]=transform_logistic_marginal([curve],bo.SearchSpace[-1,1])
        
    # identify the best input x for each run, then run it until the end

    # run these input until the max episode
    #T_max=50
    best_X=None
    Y_best_max_T=np.copy(y_score)
    T_max=bo.SearchSpace[-1,1]

    for jj in tqdm(range(1,len(y_score))):
        
        idxBest=np.argmax(y_score[0:jj+1])
        
        old_best_X=best_X
        best_X=X_original[idxBest]

        if T_max==T_original[idxBest]:
            Y_best_max_T[jj]=y_score[idxBest]
            continue

        if jj>0 and (old_best_X==best_X).all():
            Y_best_max_T[jj]=Y_best_max_T[jj-1]
        else:
            input_test=best_X.tolist()+[T_max]
            curve,time=bo.f(input_test)
            Y_best_max_T[jj]=transform_logistic_marginal(curve,T_max)
            #print(input_test,y_score[idxBest],Y_best_max_T[jj])
            
                
    return np.asarray(y_score)


        
