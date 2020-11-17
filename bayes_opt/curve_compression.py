# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:18:39 2020

@author: Vu
"""


import itertools
import numpy as np

def apply_one_transform_average(curve, midpoint=3, growth=1,MaxEpisode=1000):
    # averaging the reward curve into numeric score
            
    if isinstance(curve, (list,)):
        curve=curve[0]
 
    def linear_func(x): # this is a straightline
        if len(x)==1:
            return 1
        else:
            return [1 for u in x]
	
    my_xrange_scaled=np.linspace(0.01,5, MaxEpisode)
    my_logistic_value_scaled=linear_func(my_xrange_scaled)
    my_logistic_value_scaled=my_logistic_value_scaled[:len(curve)] # this is for visualisation purpose

    average=np.mean(curve)
    
    return average,my_logistic_value_scaled    


def return_logistic_curve(midpoint, growth, MaxEpoch=1000):
    # given the growth, midpoint and npoint, return the Logistic curve for visualization
    
    def logistic_func(x):
        #alpha=32
        if len(x)==1:
            return 1.0/(1+np.exp(-growth*(x-midpoint)))
        else:
            return [1.0/(1+np.exp(-growth*(u-midpoint))) for u in x]
        
    my_xrange_scaled=np.linspace(-6,6, MaxEpoch)
    my_logistic_value_scaled=logistic_func(my_xrange_scaled)
    
    return my_logistic_value_scaled



#def apply_one_transform_decreasing(curve, midpoint=3, growth=1,MaxEpisode=1000):
#            
#    if isinstance(curve, (list,)):
#        curve=curve[0]
#
# 
#    def ln_func(x):
#        #alpha=32
#        if len(x)==1:
#            return 1/x
#        else:
#            return [1/u for u in x]
#	
#    #my_xrange_scaled=np.linspace(-6,6, len(curve))
#    my_xrange_scaled=np.linspace(0.01,5, MaxEpisode)
#    my_logistic_value_scaled=ln_func(my_xrange_scaled)
#    my_logistic_value_scaled=my_logistic_value_scaled[:len(curve)]
#
#
#    # if curve is negative, add a constant to make it positive
#    if np.max(curve)<=0 and np.min(curve)<=0:
#        curve=curve+500
#    
#    threshold=(midpoint+6-2)*len(curve)/(12)
#    threshold=np.int(threshold)
#    #print(threshold)
#    
#    
#    prod_func=curve*my_logistic_value_scaled
#    
#    #print(len(prod_func))
#    average=[np.mean(prod_func[threshold:pos]) for pos in range(threshold,len(prod_func))]
#
#    #print(average)
#    if np.isnan(average[-1]):
#        print('bug [curve]')
#    return average[-1],my_logistic_value_scaled    


#def apply_one_transform_linear(curve, midpoint=3, growth=1,MaxEpisode=1000):
#            
#    if isinstance(curve, (list,)):
#        curve=curve[0]
# 
#    def ln_func(x):
#        if len(x)==1:
#            return x
#        else:
#            return [u for u in x]
#	
#    my_xrange_scaled=np.linspace(0.01,5, MaxEpisode)
#    my_logistic_value_scaled=ln_func(my_xrange_scaled)
#    my_logistic_value_scaled=my_logistic_value_scaled[:len(curve)]
#
#    # if curve is negative, add a constant to make it positive
#    if np.max(curve)<=0 and np.min(curve)<=0:
#        curve=curve+500
#     
#    threshold=(midpoint+6-2)*len(curve)/(12)
#    threshold=np.int(threshold)
#    
#    prod_func=curve*my_logistic_value_scaled
#    
#    average=[np.mean(prod_func[threshold:pos]) for pos in range(threshold,len(prod_func))]
#
#    return average[-1],my_logistic_value_scaled    

def apply_one_transform_ln(curve, midpoint=3, growth=1,MaxEpisode=1000):
    # this is the log transformation, used in the ablation study
    if isinstance(curve, (list,)):
        curve=curve[0]
 
    def ln_func(x):
        if len(x)==1:
            return 20+np.log(x)
        else:
            return [np.log(u) for u in x]
	
    my_xrange_scaled=np.linspace(0.01,5, MaxEpisode)
    my_logistic_value_scaled=ln_func(my_xrange_scaled)
    my_logistic_value_scaled=my_logistic_value_scaled[:len(curve)]

    # if curve is negative, add a constant to make it positive
    if np.max(curve)<=0 and np.min(curve)<=0:
        curve=curve+500
    
    threshold=(midpoint+6-2)*len(curve)/(12)
    threshold=np.int(threshold)    
    
    prod_func=curve*my_logistic_value_scaled
    
    average=[np.mean(prod_func[threshold:pos]) for pos in range(threshold,len(prod_func))]

    if np.isnan(average[-1]):
        print('bug [curve]')
    return average[-1],my_logistic_value_scaled


def apply_one_transform_logistic(curve, midpoint=-2, growth=1,MaxEpisode=1000,IsReturnCurve=False):
    # this is the Logistic transformation, used in the paper
    if isinstance(curve, (list,)):
        curve=curve[0]
        
    def logistic_func(x):
        return 1.0/(1+np.exp(-growth*(x-midpoint)))
	
    my_xrange_scaled=np.linspace(-6,6, MaxEpisode)

    my_logistic_value_scaled=logistic_func(my_xrange_scaled)

    my_logistic_value_scaled=my_logistic_value_scaled[:len(curve)]

    # if curve is negative, add a constant to make it positive
    if np.max(curve)<=0 and np.min(curve)<=0:
        curve=curve+500
    
    threshold=(midpoint+6-2)*len(curve)/(12)
    threshold=np.int(threshold)
    
    prod_func=curve*my_logistic_value_scaled
    
    average=[np.mean(prod_func[threshold:pos+1]) for pos in range(threshold,len(prod_func))]

    if IsReturnCurve==True:
        return average[-1],my_logistic_value_scaled
    else:
        return average[-1]


#def return_logistic_curve(midpoint, growth, npoint):
#           
#    def logistic_func(x):
#        #alpha=32
#        if len(x)==1:
#            return 1.0/(1+np.exp(-growth*(x-midpoint)))
#        else:
#            return [1.0/(1+np.exp(-growth*(u-midpoint))) for u in x]
#        
#    my_xrange_scaled=np.linspace(-6,6, npoint)
#    my_logistic_value_scaled=logistic_func(my_xrange_scaled)
#    
#    return my_logistic_value_scaled




    
def transform_logistic_marginal(curves,MaxEpisode=1000):
    # curve is a matrix [nParameter x MaxIter]
    # or curve is a vector [1 x MaxIter]

    def transform_one_logistic_marginal(curves,MaxEpisode):
        # curve is a vector [1 x MaxIter]
    
        midpoint_list=[-3,-2,-1,0,1]
        growth_list=[0.1,1,2,3]
        
        temp_Y_value=[0]*(len(midpoint_list)*len(growth_list))
        for idx, (val1, val2) in enumerate(itertools.product(midpoint_list,growth_list)):
            temp_Y_value[idx]=apply_one_transform_logistic(curves,val1, val2,MaxEpisode)
                
        temp_Y_value=np.asarray(temp_Y_value)
        
        Y=np.mean(temp_Y_value,axis=0)
        return Y
    if len(curves)==1:
        output=transform_one_logistic_marginal(curves[0],MaxEpisode)
    else:
        output=[0]*len(curves)
        for idx, curve in enumerate(curves):
            output[idx]=transform_one_logistic_marginal(curve,MaxEpisode)
    return output    


def transform_logistic(curves, midpoint=0, growth=1,MaxEpisode=1000):
    # curve is a matrix [nParameter x MaxIter]
    # or curve is a vector [1 x MaxIter]

    if len(curves)==1:
        output=apply_one_transform_logistic(curves[0], midpoint, growth,MaxEpisode)
    else:
        output=[0]*len(curves)
        for idx, curve in enumerate(curves):
            output[idx]=apply_one_transform_logistic(curve, midpoint, growth,MaxEpisode)
    return output
    

        
    