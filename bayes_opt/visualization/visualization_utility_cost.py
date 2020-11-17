# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:22:32 2016

@author: Vu
"""
from __future__ import division

import sys
sys.path.insert(0,'..')
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt.acquisition_maximization import acq_max_with_name



from bayes_opt.acquisition_functions import AcquisitionFunction
import os

cdict = {'red': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 0.7),
                  (1.0, 1.0, 1.0)),
          'green': ((0.0, 0.0, 0.0),
                    (0.5, 1.0, 0.0),
                    (1.0, 1.0, 1.0)),
          'blue': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.5, 1.0))}

my_cmap = plt.get_cmap('Blues')

        
counter = 0

out_dir="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\pickle_storage"
out_dir=""



    
def show_optimization_progress(bo):

    fig=plt.figure(figsize=(6, 3))
    myYbest=[bo.Y_original[:idx+1].max() for idx,val in enumerate(bo.Y_original)]
    plt.plot(range(len(myYbest)),myYbest,linewidth=2,color='m',linestyle='-',marker='o')
    plt.title('Best Found Value')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
       
    
def plot_bo_2d(bo):
    
    x1 = np.linspace(bo.scaleSearchSpace[0,0], bo.scaleSearchSpace[0,1], 70)
    x2 = np.linspace(bo.scaleSearchSpace[1,0], bo.scaleSearchSpace[1,1], 70)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.SearchSpace[0,0], bo.SearchSpace[0,1], 70)
    x2_ori = np.linspace(bo.SearchSpace[1,0], bo.SearchSpace[1,1], 70)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
  
    fig = plt.figure()
    
    #axis2d = fig.add_subplot(1, 2, 1)
    acq2d = fig.add_subplot(1, 1, 1)
    

    utility = bo.acq_func.acq_kind(X, bo.gp)
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=140,label='Selected')
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=30,label='Peak')

    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    acq2d.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1])
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq2d.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

 
def plot_2d_GPmean_var(bo,strFolderOut=None):
    
    global counter
    counter=counter+1
    
    x1 = np.linspace(bo.scaleSearchSpace[0,0], bo.scaleSearchSpace[0,1], 30)
    x2 = np.linspace(bo.scaleSearchSpace[1,0], bo.scaleSearchSpace[1,1], 30)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    T=X[:,1]
    T=np.atleast_2d(T)
    T=T.T
    T-np.reshape(T,(900,-1))
    
    x1_ori = np.linspace(bo.SearchSpace[0,0], bo.SearchSpace[0,1], 30)
    x2_ori = np.linspace(bo.SearchSpace[1,0], bo.SearchSpace[1,1], 30)   
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
  
    fig = plt.figure(figsize=(8,2.5))
    
    axis2d_GPmean = fig.add_subplot(1, 2, 1)
    axis2d_GPvariance = fig.add_subplot(1, 2, 2)
    #axis2d_util = fig.add_subplot(1, 5, 3)
    #axis2d_cost = fig.add_subplot(1, 5, 4)

    #axis2d_acq = fig.add_subplot(1, 5, 5)

    
    gp_mean, sigma = bo.gp.predict(X,eval_MSE=True)
    gp_mean_original=gp_mean*np.std(bo.Y_original)+np.mean(bo.Y_original)
    # plot the utility

    CS_acq=axis2d_GPmean.contourf(x1g_ori,x2g_ori,gp_mean.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    try:
        idxVirtual=[idx for idx,val in enumerate(bo.markVirtualObs) if val==1]
        idxReal=[idx for idx,val in enumerate(bo.markVirtualObs) if val==0]
    
        nLastVirtual=len(bo.markVirtualObs)-idxReal[-1]
        idxVirtual_ExcludeLast=idxVirtual[:-nLastVirtual]
        idxReal_ExcludeLast=idxReal[:-1]
    except:
        idxVirtual=[]
        idxReal=list(range(len(bo.Y)))
        idxVirtual_ExcludeLast=[]
        idxReal_ExcludeLast=idxReal[:-1]
    #x_stars=np.asarray(bo.x_stars_original)   
    
    axis2d_GPmean.scatter(bo.X_original[idxVirtual_ExcludeLast,0],bo.T_original[idxVirtual_ExcludeLast,0],color='r',label='Augmented Obs')  
    axis2d_GPmean.scatter(bo.X_original[idxReal_ExcludeLast,0],bo.T_original[idxReal_ExcludeLast,0],marker='s',color='g',label='Obs')  
    

    axis2d_GPmean.set_title('GP mean',fontsize=16)
    axis2d_GPmean.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis2d_GPmean.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1]+5)
    axis2d_GPmean.set_ylabel('#Episode',fontsize=16)

    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    axis2d_GPmean.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq, ax=axis2d_GPmean, shrink=0.9)
    
    
    gp_var_original=sigma
    #gp_var_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)

    CS_acq=axis2d_GPvariance.contourf(x1g_ori,x2g_ori,gp_var_original.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d_GPvariance.scatter(bo.X_original[idxVirtual_ExcludeLast,0],bo.T_original[idxVirtual_ExcludeLast,0],color='r',label='Augmented Obs')  
    axis2d_GPvariance.scatter(bo.X_original[idxReal_ExcludeLast,0],bo.T_original[idxReal_ExcludeLast,0],marker='s',color='g',label='Obs')  

    
    axis2d_GPvariance.set_yticks([])
    axis2d_GPvariance.set_title('GP variance',fontsize=16)
    axis2d_GPvariance.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis2d_GPvariance.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1]+5)
    axis2d_GPvariance.set_xlabel(r'$x$',fontsize=16)

    fig.colorbar(CS_acq, ax=axis2d_GPvariance, shrink=0.9)
    

    if strFolderOut is not None:
        strFileName="{:d}_GP2d.pdf".format(counter)
        strFinalPath=os.path.join(strFolderOut,strFileName)
        fig.savefig(strFinalPath, bbox_inches='tight')
    
    
    
def plot_bo_2d_cost_utility_AF(bo,strFolderOut=None):
    
    global counter
    counter=counter+1
    
    x1 = np.linspace(bo.scaleSearchSpace[0,0], bo.scaleSearchSpace[0,1], 30)
    x2 = np.linspace(bo.scaleSearchSpace[1,0], bo.scaleSearchSpace[1,1], 30)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    T=X[:,1]
    T=np.atleast_2d(T)
    T=T.T
    T-np.reshape(T,(900,-1))
    
    x1_ori = np.linspace(bo.SearchSpace[0,0], bo.SearchSpace[0,1], 30)
    x2_ori = np.linspace(bo.SearchSpace[1,0], bo.SearchSpace[1,1], 30)   
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
  
    fig = plt.figure(figsize=(18,3))
    
    axis2d_GPmean = fig.add_subplot(1, 5, 1)
    axis2d_GPvariance = fig.add_subplot(1, 5, 2)
    axis2d_util = fig.add_subplot(1, 5, 3)
    axis2d_cost = fig.add_subplot(1, 5, 4)

    axis2d_acq = fig.add_subplot(1, 5, 5)

    
    gp_mean, sigma = bo.gp.predict(X,eval_MSE=True)
    gp_mean_original=gp_mean*np.std(bo.Y_original)+np.mean(bo.Y_original)
    # plot the utility

    CS_acq=axis2d_GPmean.contourf(x1g_ori,x2g_ori,gp_mean.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    try:
        idxVirtual=[idx for idx,val in enumerate(bo.markVirtualObs) if val==1]
        idxReal=[idx for idx,val in enumerate(bo.markVirtualObs) if val==0]
    
        nLastVirtual=len(bo.markVirtualObs)-idxReal[-1]
        idxVirtual_ExcludeLast=idxVirtual[:-nLastVirtual]
        idxReal_ExcludeLast=idxReal[:-1]
    except:
        idxVirtual=[]
        idxReal=list(range(len(bo.Y)))
        idxVirtual_ExcludeLast=[]
        idxReal_ExcludeLast=idxReal[:-1]
    #x_stars=np.asarray(bo.x_stars_original)   
    
    axis2d_GPmean.scatter(bo.X_original[idxVirtual_ExcludeLast,0],bo.T_original[idxVirtual_ExcludeLast,0],color='r',label='Augmented Obs')  
    axis2d_GPmean.scatter(bo.X_original[idxReal_ExcludeLast,0],bo.T_original[idxReal_ExcludeLast,0],marker='s',color='g',label='Obs')  
    

    axis2d_GPmean.set_title('GP mean',fontsize=16)
    axis2d_GPmean.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis2d_GPmean.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1]+5)
    axis2d_GPmean.set_ylabel('#Episode',fontsize=16)

    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    axis2d_GPmean.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq, ax=axis2d_GPmean, shrink=0.9)
    
    
    gp_var_original=sigma
    #gp_var_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)

    CS_acq=axis2d_GPvariance.contourf(x1g_ori,x2g_ori,gp_var_original.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d_GPvariance.scatter(bo.X_original[idxVirtual_ExcludeLast,0],bo.T_original[idxVirtual_ExcludeLast,0],color='r',label='Augmented Obs')  
    axis2d_GPvariance.scatter(bo.X_original[idxReal_ExcludeLast,0],bo.T_original[idxReal_ExcludeLast,0],marker='s',color='g',label='Obs')  

    
    axis2d_GPvariance.set_yticks([])
    axis2d_GPvariance.set_title('GP variance',fontsize=16)
    axis2d_GPvariance.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis2d_GPvariance.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1]+5)
    axis2d_GPvariance.set_xlabel('gamma',fontsize=16)

    
    fig.colorbar(CS_acq, ax=axis2d_GPvariance, shrink=0.9)
    
    
    # plot the cost
    #gp_mean_cost, sigma = bo.gp_cost.predict(X)
    
    try:
        mean_cost=bo.linear_regression.predict(X)
    except:
        mean_cost=bo.linear_regression.predict(T)
        
    mean_cost=np.reshape(mean_cost,(-1,1))
    mean_cost[mean_cost<0]=0.001
    mean_cost=np.log(1+np.exp(mean_cost))
    #mean_cost[mean_cost<0]=0.001

    #gp_mean_cost=mean_cost
    gp_mean_cost_original=mean_cost*np.std(bo.Y_cost_original)+np.mean(bo.Y_cost_original)
    
    CS_acq_cost=axis2d_cost.contourf(x1g_ori,x2g_ori,mean_cost.reshape(x1g.shape),cmap=my_cmap,origin='lower')
   
    axis2d_cost.scatter(bo.X_original[idxVirtual_ExcludeLast,0],bo.T_original[idxVirtual_ExcludeLast,0],color='r',label='Augmented Obs')  

    axis2d_cost.scatter(bo.X_original[idxReal_ExcludeLast,0],bo.T_original[idxReal_ExcludeLast,0],marker='s',color='g',label='Obs')  


    axis2d_cost.set_title(r'Cost $\mu_c$',fontsize=16)
    #axis2d_cost.set_xlabel('gamma',fontsize=16)
    #axis2d_cost.set_ylabel('#Episode',fontsize=16)

    axis2d_cost.set_yticks([])
    axis2d_cost.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis2d_cost.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1]+5)
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    axis2d_cost.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq_cost, ax=axis2d_cost, shrink=0.9)
    
    
  

    # optimie the GP predictive mean function to find the max of mu
    x_mu_max,mu_max_val=acq_max_with_name(gp=bo.gp,scaleSearchSpace=bo.scaleSearchSpace,acq_name='mu',IsReturnY=True)

    # plot acquisition function
    acq_func={}
    acq_func['name']=bo.acq_name
    acq_func['dim']=1
    acq_func['scaleSearchSpace']=bo.scaleSearchSpace
    acq_func['mu_max']=  mu_max_val


    myacq=AcquisitionFunction(acq_func)
    util_value = myacq.acq_kind(X, bo.gp)
    util_value=np.log(1+np.exp(util_value))
    util_value=np.reshape(util_value,(-1,1))
    
    
    CS_acq=axis2d_util.contourf(x1g_ori,x2g_ori,util_value.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    #axis2d_util.scatter(bo.X_original[idxReal_ExcludeLast,0],bo.T_original[idxReal_ExcludeLast,0],marker='s',color='g',label='Data')
    #axis2d_util.scatter(bo.X_original[idxReal[-1],0],bo.T_original[idxReal[-1],0],marker='s', color='yellow',s=80,label='Selected')
    axis2d_util.scatter(bo.X_original[idxVirtual_ExcludeLast,0],bo.T_original[idxVirtual_ExcludeLast,0],color='r',label='Augmented Obs')  

    axis2d_util.scatter(bo.X_original[idxReal_ExcludeLast,0],bo.T_original[idxReal_ExcludeLast,0],marker='s',color='g',label='Obs')  


    axis2d_util.set_yticks([])

    axis2d_util.set_title(r'Acquisition $\alpha$',fontsize=16)
    axis2d_util.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis2d_util.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1]+5)
    axis2d_util.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))

    
    fig.colorbar(CS_acq, ax=axis2d_util, shrink=0.9)


    # find the max of utility and max of cost to normalize
#    x_max_acq, utility_max=acq_max_with_name(gp=bo.gp,scaleSearchSpace=bo.scaleSearchSpace,
#                             acq_name=bo.acq_name,IsReturnY=True)
#    x_max_acq, utility_min=acq_max_with_name(gp=bo.gp,scaleSearchSpace=bo.scaleSearchSpace,
#                             acq_name=bo.acq_name,IsReturnY=True,IsMax=False)      
    
 
    
    util_min=np.min(util_value)
    cost_min=np.min(mean_cost)
    if util_min<0 or cost_min<0:
        print("bug")
    acq_value=util_value/(mean_cost)
    
    val=np.min(acq_value)
    if val<0:
        print("val<0")
    acq_value=np.reshape(acq_value,x1g.shape)
    idxMax=np.argmax(acq_value)
        
    CS_acq=axis2d_acq.contourf(x1g_ori,x2g_ori,acq_value.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    axis2d_acq.scatter(bo.X_original[idxVirtual_ExcludeLast,0],bo.T_original[idxVirtual_ExcludeLast,0],color='r')  
    axis2d_acq.scatter(bo.X_original[idxReal,0],bo.T_original[idxReal,0],marker='s',color='g')  
    axis2d_acq.scatter(bo.X_original[idxReal[-1],0],bo.T_original[idxReal[-1],0],marker='s', color='k',s=80,label='Selected')
    #axis2d_acq.scatter(X_ori[idxMax,0],X_ori[idxMax,1],marker='^',s=100,color='k',label='Maximum')

    
    axis2d_acq.set_yticks([])

    axis2d_acq.set_title(r'Decision $\alpha$ / $\mu_c$',fontsize=16)
    axis2d_acq.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis2d_acq.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1]+5)
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    axis2d_acq.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq, ax=axis2d_acq, shrink=0.9)
    

    if strFolderOut is not None:
        strFileName="{:d}_GP2d_AF_5.pdf".format(counter)
        strFinalPath=os.path.join(strFolderOut,strFileName)
        fig.savefig(strFinalPath, bbox_inches='tight')
    
def plot_2d_Acq_Cost(bo,strFolderOut=None):
    
    global counter
    counter=counter+1
    
    x1 = np.linspace(bo.scaleSearchSpace[0,0], bo.scaleSearchSpace[0,1], 30)
    x2 = np.linspace(bo.scaleSearchSpace[1,0], bo.scaleSearchSpace[1,1], 30)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    T=X[:,1]
    T=np.atleast_2d(T)
    T=T.T
    T-np.reshape(T,(900,-1))
    
    x1_ori = np.linspace(bo.SearchSpace[0,0], bo.SearchSpace[0,1], 30)
    x2_ori = np.linspace(bo.SearchSpace[1,0], bo.SearchSpace[1,1], 30)   
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
  
    fig = plt.figure(figsize=(14,2.5))
    
    
    axis2d_util = fig.add_subplot(1, 3, 1)
    axis2d_cost = fig.add_subplot(1, 3, 2)
    axis2d_acq = fig.add_subplot(1, 3, 3)

    
    gp_mean, sigma = bo.gp.predict(X,eval_MSE=True)
    # plot the utility

    try:
        idxVirtual=[idx for idx,val in enumerate(bo.markVirtualObs) if val==1]
        idxReal=[idx for idx,val in enumerate(bo.markVirtualObs) if val==0]
    
        nLastVirtual=len(bo.markVirtualObs)-idxReal[-1]
        idxVirtual_ExcludeLast=idxVirtual[:-nLastVirtual]
        idxReal_ExcludeLast=idxReal[:-1]
    except:
        idxVirtual=[]
        idxReal=list(range(len(bo.Y)))
        idxVirtual_ExcludeLast=[]
        idxReal_ExcludeLast=idxReal[:-1]
    #x_stars=np.asarray(bo.x_stars_original)   
    
    try:
        mean_cost=bo.linear_regression.predict(X)
    except:
        mean_cost=bo.linear_regression.predict(T)
        
    mean_cost=np.reshape(mean_cost,(-1,1))
    mean_cost[mean_cost<0]=0.001
    mean_cost=np.log(1+np.exp(mean_cost))
    #mean_cost[mean_cost<0]=0.001

    #gp_mean_cost=mean_cost
    gp_mean_cost_original=mean_cost*np.std(bo.Y_cost_original)+np.mean(bo.Y_cost_original)
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq_cost=axis2d_cost.contourf(x1g_ori,x2g_ori,mean_cost.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    #idxBest=np.argmax(gp_mean_cost)
    axis2d_cost.scatter(bo.X_original[idxVirtual_ExcludeLast,0],bo.T_original[idxVirtual_ExcludeLast,0],color='r',label='Augmented Obs')  

    axis2d_cost.scatter(bo.X_original[idxReal_ExcludeLast,0],bo.T_original[idxReal_ExcludeLast,0],marker='s',color='g',label='Obs')  


    axis2d_cost.set_title(r'Cost $\mu_c$',fontsize=16)
    axis2d_cost.set_xlabel('$\gamma$',fontsize=16)
    #axis2d_cost.set_ylabel('#Episode',fontsize=16)

    axis2d_cost.set_yticks([])
    axis2d_cost.set_xlim(bo.SearchSpace[0,0]-0.003, bo.SearchSpace[0,1]+0.003)
    axis2d_cost.set_ylim(bo.SearchSpace[1,0]-5, bo.SearchSpace[1,1]+5)
    
    fig.colorbar(CS_acq_cost, ax=axis2d_cost, shrink=0.9)
    
    
    # plot acquisition function
    acq_func={}
    acq_func['name']=bo.acq['name']
    acq_func['dim']=1
    acq_func['scaleSearchSpace']=bo.scaleSearchSpace

    myacq=AcquisitionFunction(acq_func)
    util_value = myacq.acq_kind(X, bo.gp)
    util_value=np.log(1+np.exp(util_value))
    util_value=np.reshape(util_value,(-1,1))
    
    
    CS_acq=axis2d_util.contourf(x1g_ori,x2g_ori,util_value.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    #axis2d_util.scatter(bo.X_original[idxReal_ExcludeLast,0],bo.T_original[idxReal_ExcludeLast,0],marker='s',color='g',label='Data')
    #axis2d_util.scatter(bo.X_original[idxReal[-1],0],bo.T_original[idxReal[-1],0],marker='s', color='yellow',s=80,label='Selected')
    axis2d_util.scatter(bo.X_original[idxVirtual_ExcludeLast,0],bo.T_original[idxVirtual_ExcludeLast,0],color='r',label='Augmented Obs')  
    axis2d_util.scatter(bo.X_original[idxReal_ExcludeLast,0],bo.T_original[idxReal_ExcludeLast,0],marker='s',color='g',label='Obs')  


    #axis2d_util.set_yticks([])
    axis2d_util.set_ylabel('#Episode',fontsize=16)


    axis2d_util.set_title(r'Acquisition $\alpha$',fontsize=16)
    axis2d_util.set_xlim(bo.SearchSpace[0,0]-0.003, bo.SearchSpace[0,1]+0.003)
    axis2d_util.set_ylim(bo.SearchSpace[1,0]-5, bo.SearchSpace[1,1]+5)
    axis2d_util.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))

    
    fig.colorbar(CS_acq, ax=axis2d_util, shrink=0.9)


    # find the max of utility and max of cost to normalize
    x_max_acq, utility_max=acq_max_with_name(gp=bo.gp,scaleSearchSpace=bo.scaleSearchSpace,
                             acq_name=bo.acq['name'],IsReturnY=True)
    x_max_acq, utility_min=acq_max_with_name(gp=bo.gp,scaleSearchSpace=bo.scaleSearchSpace,
                             acq_name=bo.acq['name'],IsReturnY=True,IsMax=False)      
    
   
    util_min=np.min(util_value)
    cost_min=np.min(mean_cost)
    if util_min<0 or cost_min<0:
        print("bug")
    acq_value=util_value/(mean_cost)
    
    val=np.min(acq_value)
    if val<0:
        print("val<0")
    acq_value=np.reshape(acq_value,x1g.shape)
    idxMax=np.argmax(acq_value)
        
    CS_acq=axis2d_acq.contourf(x1g_ori,x2g_ori,acq_value.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    axis2d_acq.scatter(bo.X_original[idxVirtual_ExcludeLast,0],bo.T_original[idxVirtual_ExcludeLast,0],color='r')  

    axis2d_acq.scatter(bo.X_original[idxReal,0],bo.T_original[idxReal,0],marker='s',color='g')  
    axis2d_acq.scatter(bo.X_original[idxReal[-1],0],bo.T_original[idxReal[-1],0],marker='s', color='k',s=80,label='Next Point')
    #axis2d_acq.scatter(X_ori[idxMax,0],X_ori[idxMax,1],marker='^',s=100,color='k',label='Maximum')

    
    axis2d_acq.set_yticks([])

    axis2d_acq.set_title(r'Decision $\alpha$ / $\mu_c$',fontsize=16)
    axis2d_acq.set_xlim(bo.SearchSpace[0,0]-0.003, bo.SearchSpace[0,1]+0.003)
    axis2d_acq.set_ylim(bo.SearchSpace[1,0]-5, bo.SearchSpace[1,1]+5)
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    axis2d_acq.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq, ax=axis2d_acq, shrink=0.9)
    

    if strFolderOut is not None:
        strFileName="{:d}_GP2d_AF.pdf".format(counter)
        strFinalPath=os.path.join(strFolderOut,strFileName)
        fig.savefig(strFinalPath, bbox_inches='tight')
    
        
def plot_bo_2d_cost_utility(bo,strFolderOut=None):
    
    global counter
    counter=counter+1
    
    x1 = np.linspace(bo.scaleSearchSpace[0,0], bo.scaleSearchSpace[0,1], 30)
    x2 = np.linspace(bo.scaleSearchSpace[1,0], bo.scaleSearchSpace[1,1], 30)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    T=X[:,1]
    T=np.atleast_2d(T)
    T=T.T
    T-np.reshape(T,(900,-1))
    
    x1_ori = np.linspace(bo.SearchSpace[0,0], bo.SearchSpace[0,1], 30)
    x2_ori = np.linspace(bo.SearchSpace[1,0], bo.SearchSpace[1,1], 30)   
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
  
    fig = plt.figure(figsize=(18,3.5))
    
    axis2d_utility = fig.add_subplot(1, 4, 1)
    axis2d_pvrs_variance = fig.add_subplot(1, 4, 2)
    axis2d_cost = fig.add_subplot(1, 4, 3)
    axis2d_acq = fig.add_subplot(1, 4, 4)
    
    gp_mean, sigma = bo.gp.predict(X,eval_MSE=True)
    gp_mean_original=gp_mean*np.std(bo.Y_original)+np.mean(bo.Y_original)
    # plot the utility

    #utility = bo.acq_func.acq_kind(X, bo.gp)
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=axis2d_utility.contourf(x1g_ori,x2g_ori,gp_mean_original.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(gp_mean)
    
    x_stars=np.asarray(bo.x_stars_original)
    
    axis2d_utility.scatter(bo.X_original[:,0],bo.T_original[:,0],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    axis2d_utility.scatter(bo.X_original[-1,0],bo.T_original[-1,0],marker='s', color='yellow',s=80,label='Selected')
    #axis2d_utility.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=30,label='Peak')
#    axis2d_utility.scatter(x_stars[:,0],x_stars[:,1],label='x*',marker='*',color='r',s=50)

    
    axis2d_utility.set_title('Utility Function',fontsize=16)
    axis2d_utility.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis2d_utility.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1]+5)
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    axis2d_utility.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq, ax=axis2d_utility, shrink=0.9)
    

    
    # plot the cost
    #gp_mean_cost, sigma = bo.gp_cost.predict(X)
    mean_cost=bo.linear_regression.predict(T)
    gp_mean_cost=mean_cost
    #gp_mean_cost_original=gp_mean_cost*np.std(bo.Y_cost_original)+np.mean(bo.Y_cost_original)
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq_cost=axis2d_cost.contourf(x1g_ori,x2g_ori,mean_cost.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(gp_mean_cost)
    
    axis2d_cost.scatter(bo.X_original[:,0],bo.T_original[:,0],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    axis2d_cost.scatter(bo.X_original[-1,0],bo.T_original[-1,0],marker='s', color='yellow',s=80,label='Selected')
    #axis2d_cost.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=30,label='Peak')
    #axis2d_cost.scatter(x_stars[:,0],x_stars[:,1],label='x*',marker='*',color='r',s=50)

    axis2d_cost.set_title('Cost Function',fontsize=16)
    axis2d_cost.set_xlabel('gamma',fontsize=16)
    axis2d_cost.set_ylabel('#Episode',fontsize=16)


    axis2d_cost.set_xlim(bo.SearchSpace[0,0], bo.SearchSpace[0,1])
    axis2d_cost.set_ylim(bo.SearchSpace[1,0], bo.SearchSpace[1,1]+10)
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    axis2d_cost.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq_cost, ax=axis2d_cost, shrink=0.9)
    
    
   
    if strFolderOut is not None:
        strFileName="{:d}_GP2d_AF.pdf".format(counter)
        strFinalPath=os.path.join(strFolderOut,strFileName)
        fig.savefig(strFinalPath, bbox_inches='tight')
    
    
def plot_original_function(myfunction):
    
    origin = 'lower'

    func=myfunction.func

    if myfunction.input_dim>2:
        print("Cannot plot function which dimension is >2")
        return

    if myfunction.input_dim==1:    
        x = np.linspace(myfunction.SearchSpace['x'][0], myfunction.SearchSpace['x'][1], 100)
        y = func(x)
    
        fig=plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        strTitle="{:s}".format(myfunction.name)

        plt.title(strTitle)
    
    if myfunction.input_dim==2:    
        
        # Create an array with parameters SearchSpace
        if isinstance(myfunction.SearchSpace,dict):
            # Get the name of the parameters        
            SearchSpace = []
            for key in myfunction.SearchSpace.keys():
                SearchSpace.append(myfunction.SearchSpace[key])
            SearchSpace = np.asarray(SearchSpace)
        else:
            SearchSpace=np.asarray(myfunction.SearchSpace)
            
        x1 = np.linspace(SearchSpace[0][0], SearchSpace[0][1], 50)
        x2 = np.linspace(SearchSpace[1][0], SearchSpace[1][1], 50)
        x1g,x2g=np.meshgrid(x1,x2)
        X_plot=np.c_[x1g.flatten(), x2g.flatten()]
        Y = func(X_plot)
    
        #fig=plt.figure(figsize=(8, 5))
        
        #fig = plt.figure(figsize=(12, 3.5))
        fig = plt.figure(figsize=(14, 4))
        
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax2d = fig.add_subplot(1, 2, 2)
        
        alpha = 0.7
        ax3d.plot_surface(x1g,x2g,Y.reshape(x1g.shape),cmap=my_cmap,alpha=alpha) 
        
        
        idxBest=np.argmax(Y)
        #idxBest=np.argmin(Y)
    
        ax3d.scatter(X_plot[idxBest,0],X_plot[idxBest,1],Y[idxBest],marker='*',color='r',s=200,label='Peak')
    

        strTitle="{:s}".format(myfunction.name)
        #print strTitle
        ax3d.set_title(strTitle)
        #ax3d.view_init(40, 130)

        
        idxBest=np.argmax(Y)
        CS=ax2d.contourf(x1g,x2g,Y.reshape(x1g.shape),cmap=my_cmap,origin=origin)   
       
        #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin=origin,hold='on')
        ax2d.scatter(X_plot[idxBest,0],X_plot[idxBest,1],marker='*',color='r',s=300,label='Peak')
        plt.colorbar(CS, ax=ax2d, shrink=0.9)

        ax2d.set_title(strTitle)

        
    strFileName="{:s}.eps".format(myfunction.name)
    strPath=os.path.join(out_dir,strFileName)
    fig.savefig(strPath, bbox_inches='tight')
    