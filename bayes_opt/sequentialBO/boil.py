# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""


import numpy as np
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from bayes_opt import GaussianProcess
from bayes_opt import ProductGaussianProcess

from bayes_opt.acquisition_maximization import acq_max_with_name,acq_min_scipy_kwargs
import time
from sklearn import linear_model
import copy
from bayes_opt.curve_compression import transform_logistic
from sklearn.preprocessing import MinMaxScaler


#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class BOIL(object):

    #def __init__(self, gp_params, func_params, acq_params, verbose=True):
    def __init__(self, func, SearchSpace,acq_name="ei_mu_max",verbose=1):

        """      
        Input parameters
        ----------
        
        gp_params:                  GP parameters
        gp_params.theta:            to compute the kernel
        gp_params.delta:            to compute the kernel
        
        func_params:                function to optimize
        func_params.init bound:     initial SearchSpace for parameters
        func_params.SearchSpace:        SearchSpace on parameters        
        func_params.func:           a function to be optimized
        
        
        acq_params:            acquisition function, 
        acq_params.acq_func['name']=['ei','ucb','poi']
        acq_params.opt_toolbox:     optimization toolbox 'nlopt','direct','scipy'
                            
        Returns
        -------
        dim:            dimension
        SearchSpace:         SearchSpace on original scale
        scaleSearchSpace:    SearchSpace on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """
        
        self.method='boil'
        self.verbose=verbose
        if isinstance(SearchSpace,dict):
            # Get the name of the parameters
            self.keys = list(SearchSpace.keys())
            
            self.SearchSpace = []
            for key in list(SearchSpace.keys()):
                self.SearchSpace.append(SearchSpace[key])
            self.SearchSpace = np.asarray(self.SearchSpace)
        else:
            self.SearchSpace=np.asarray(SearchSpace)
            
            
        self.dim = len(SearchSpace)

        scaler = MinMaxScaler()
        scaler.fit(self.SearchSpace[:-1,:].T)
        
        scalerT = MinMaxScaler()
        SearchSpace_T=np.atleast_2d(self.SearchSpace[-1,:]).T
        scalerT.fit(SearchSpace_T)

        self.Xscaler=scaler
        self.Tscaler=scalerT

        # create a scaleSearchSpace 0-1
        self.scaleSearchSpace=np.array([np.zeros(self.dim), np.ones(self.dim)]).T
                
        # function to be optimised
        self.f = func
    
        # store X in original scale
        self.X_ori= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_ori = None
        
        # store the number of episode
        self.T=None
        self.T_original=None
        
        # store the cost original scale
        self.Y_cost_original=None
        
        self.time_opt=0
         
        self.max_min_gap=self.SearchSpace[:,1]-self.SearchSpace[:,0]


        # acquisition function
        self.acq_name = acq_name
        self.logmarginal=0

        self.gp=ProductGaussianProcess(self.scaleSearchSpace,verbose=verbose)

        # store the curves of performances
        self.Y_curves=[]
        
        # store the cost original scale
        self.Y_cost_original=None
        
        self.time_opt=0
        
        # acquisition function
        self.acq_func = None
   
        self.logmarginal=0
        
        self.markVirtualObs=[]
        
        self.countVirtual=[]

        self.linear_regression = linear_model.LinearRegression()

        self.condition_number=[]
        
        # maximum number of augmentations
        self.max_n_augmentation=10
        self.threshold_cond=15
        
    def init(self, n_init_points=3, seed=1):
        """      
        Input parameters
        ----------
        n_init_points:        # init points
        """
        np.random.seed(seed)

        # Generate random points
        SearchSpace=np.copy(self.SearchSpace)
        SearchSpace[-1,0]=SearchSpace[-1,1] # last dimension, set it to MaxIter

        l = [np.random.uniform(x[0], x[1]) for _ in range(n_init_points) for x in SearchSpace] 

        # Concatenate new random points to possible existing
        # points from self.explore method.
        temp=np.asarray(l)
        temp=temp.T
        init_X=list(temp.reshape((n_init_points,-1)))
        
        self.X_original = np.asarray(init_X)
        self.T_original=self.X_original[:,-1]
        self.T_original=np.reshape(self.T_original,(n_init_points,-1))
        
        self.X_original=self.X_original[:,:-1] # remove the last dimension of MaxEpisode
        self.X_original=np.reshape(self.X_original,(n_init_points,-1))

        # Evaluate target function at all initialization           
        y_init_curves, y_init_cost=self.f(init_X)

        y_init_cost=np.atleast_2d(np.asarray(y_init_cost))#.astype('Float64')

        self.Y_curves+=y_init_curves

        # we transform the y_init_curves as the average of [ curves * logistic ]
        y_init=transform_logistic(y_init_curves,self.gp.logistic_hyper['midpoint'],\
                                  self.gp.logistic_hyper['growth'], self.SearchSpace[-1,1])
        #y_init=y_init_curves
        y_init=np.reshape(y_init,(n_init_points,1))
        
        # record keeping ========================================================
        self.Y_original = np.asarray(y_init)      
        self.Y_cost_original=np.reshape(y_init_cost,(-1,1))

        # convert it to scaleX
        self.X = self.Xscaler.transform(np.asarray(init_X)[:,:-1])#remove the last dimension of MaxEpisode
        #self.X=self.X[:,:-1]
        self.X=np.reshape(self.X,(n_init_points,-1))

        self.T = self.Tscaler.transform(self.T_original)

        self.markVirtualObs+=[0]*n_init_points

        # generating virtual observations for each initial point
        for ii in range(n_init_points):
            self.generating_virtual_observations(self.X[ii,:],
                         self.T[ii],[y_init_curves[ii]],y_init_cost[0][ii],IsRandom=False)

        self.Y_cost=(self.Y_cost_original-np.min(self.Y_cost_original))/(np.max(self.Y_cost_original)-np.min(self.Y_cost_original))

        if np.std(self.Y_original)==0:
            self.Y=(self.Y_original-np.mean(self.Y_original))
        else:
            self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)

       
    def utility_cost_evaluation(self,x,acq_func,isDebug=False):
        # this is a wrapper function to evaluate at multiple x(s)
        
        
        def utility_cost_evaluation_single(x,acq_func,isDebug=False):
            # given a location x, we will evaluate the utility and cost
            
            utility=acq_func.acq_kind(x,gp=self.gp)
            
            try:
                mean_cost=self.linear_regression.predict(np.reshape(x,(1,-1)))
                
            except:
                print(x)
                print("bug")
    
            mean_cost=max(0,mean_cost)+0.1 # to avoid <=0 cost
            
            #acquisition_function_value= utility_normalized/cost_normalized
            if 'ei' in acq_func.acq_name:
                acquisition_function_value= np.log(utility)-np.log(mean_cost)
            else:
                acquisition_function_value= np.log(1+np.exp(utility))/np.log(1+np.exp(mean_cost))
    
            if isDebug==True:
                print("acq_func at the selected point \t utility:",np.round(utility,decimals=4),"\t cost:",mean_cost)
                if utility==0:
                    print("utility =0===============================================================================")
       
            return acquisition_function_value*(-1) # since we will minimize this acquisition function
        
        
        if len(x)==self.dim: # one observation
            temp=utility_cost_evaluation_single(x,acq_func,isDebug=isDebug)
            if isDebug==True:
                return temp
            else:
                utility=np.mean(temp)
        
        else: # multiple observations
            utility=[0]*len(x)
            for idx,val in enumerate(x):
                temp=utility_cost_evaluation_single(x=val,acq_func=acq_func,isDebug=isDebug)
                                                     
                utility[idx]=np.mean(temp)
                
            utility=np.asarray(utility)    				               
        return utility   
    
        
    def acq_utility_cost(self):
        
        # generate a set of x* at T=MaxIter
        # instead of running optimization on the whole space, we will only operate on the region of interest
        # the region of interest in DRL is where the MaxEpisode
    
        # we find maximum of EI

        acq={}
        acq['name']=self.acq_name
        acq['dim']=self.scaleSearchSpace.shape[0]
        acq['scaleSearchSpace']=self.scaleSearchSpace   
    
        if self.acq_name=='ei_mu_max':# using max of mean(x) as the incumbent
            
            # optimie the GP predictive mean function to find the max of mu
            x_mu_max,mu_max_val=acq_max_with_name(gp=self.gp,scaleSearchSpace=self.scaleSearchSpace,acq_name='mu',IsReturnY=True)
            acq['mu_max']=  mu_max_val

        myacq=AcquisitionFunction(acq)
        
        x_min = acq_min_scipy_kwargs(myfunc=self.utility_cost_evaluation,SearchSpace=self.scaleSearchSpace,
                        acq_func=myacq, isDebug=False)
        
        if self.verbose==True:
            acq_val=self.utility_cost_evaluation(x_min,myacq,isDebug=False)
            print("selected point from acq func:",np.round(x_min,decimals=4),"acq val=log(Utility/Cost)=",(-1)*np.round(acq_val,decimals=4)) # since we minimize the acq func
            if np.round(acq_val,decimals=4)==0:
                print("acq value =0")
            
        return x_min
    
    
    def select_informative_location_by_uncertainty(self,n_virtual_obs,x_max,t_max):
        # this function will select a list of informative locations to place a virtual obs
        # x_max is the selected hyperparameter
        # t_max is the selected number of epochs to train
        
        
        SearchSpace=np.copy(self.scaleSearchSpace)
        for dd in range(self.dim-1):
            SearchSpace[dd,0],SearchSpace[dd,1]=x_max[dd],x_max[dd]
            
        SearchSpace[-1,1]=t_max
        
        temp_X,temp_T=self.X.copy(),self.T.copy()
        temp_gp=copy.deepcopy(self.gp )
        
        temp_Y=np.random.random(size=(len(temp_T),1))
        
        temp_gp.fit(temp_X,temp_T,temp_Y,self.Y_curves)
        
        new_batch_T=None

        pred_var_value=[0]*n_virtual_obs
        for ii in range(n_virtual_obs):
            x_max_pred_variance, pred_var_value[ii]=acq_max_with_name(gp=temp_gp,
                              scaleSearchSpace=SearchSpace,acq_name='pure_exploration',IsReturnY=True)
            
            # stop augmenting if the uncertainty is smaller than a threshold
            # or stop augmenting if the uncertainty is smaller than a threshold

            log_cond=np.log( temp_gp.compute_condition_number() )
            if log_cond>self.threshold_cond or pred_var_value[ii]<(self.gp.noise_delta+1e-3):
                break
          
            if x_max_pred_variance[-1] in temp_T[-ii:]: # if repetition, stop augmenting
                break
            
            temp_X = np.vstack((temp_X, x_max.reshape((1, -1)))) # append new x
            temp_T = np.vstack((temp_T, x_max_pred_variance[-1].reshape((1, -1)))) # append new t
            temp_gp.X,temp_gp.T=temp_X,temp_T
            temp_Y=np.random.random(size=(len(temp_T),1))
            
            temp_gp.fit(temp_X,temp_T,temp_Y,self.Y_curves)

            if new_batch_T is None:
                new_batch_T=x_max_pred_variance[-1].reshape((1, -1))
            else:
                new_batch_T= np.vstack((new_batch_T, x_max_pred_variance[-1].reshape((1, -1))))
        
#        if self.verbose:
#            print("pred_var_value at the augmented points:",np.round( pred_var_value,decimals=4))

        if new_batch_T is None:
            return [],0

        else:
            output=np.sort(new_batch_T.ravel()).tolist()
            return output, len(output)

    
    def generating_virtual_observations(self,x_max,t_max,y_original_curves,y_cost_original,IsRandom=False):
        
        #temp_X_new_original=x_max*self.max_min_gap[:-1]+self.SearchSpace[:-1,0]
        temp_X_new_original=self.Xscaler.inverse_transform(np.reshape(x_max,(-1,self.dim-1)))

        # selecting MAX number of virtual observations, e.g., we dont want to augment more than 10 points
        max_n_virtual_obs=np.int(t_max*self.max_n_augmentation)
        if max_n_virtual_obs==0:
            self.countVirtual.append(0)
            return
        
        if IsRandom==True:# select informative locations by random uniform   
            l = [np.random.uniform(0, t_max) for _ in range(max_n_virtual_obs)]
        else:
            # select informative locations by uncertainty as in the paper
            l,n_virtual_obs=self.select_informative_location_by_uncertainty(max_n_virtual_obs,x_max,t_max)        
            
        self.countVirtual.append(n_virtual_obs)
        
        if self.verbose:
            np.set_printoptions(suppress=True)
            print("Max #augmented points",max_n_virtual_obs, "\t #augmented points ",len(l),
                  "\t Augmented points: ",np.round(l,decimals=3))
            
        l_original=[self.SearchSpace[-1,0]+val*self.max_min_gap[-1] for val in l]
        #l_original=[self.Tscaler.inverse_transform(val) for val in l]
                           
        virtual_obs_t_original=np.asarray(l_original).T
        virtual_obs_t=np.asarray(l).T
        
        # compute y_original for the virtual observations
        y_virtual_original=[0]*n_virtual_obs
        for ii in range(n_virtual_obs):
            
            idx=np.int(virtual_obs_t_original[ii])
            
            temp_curve=y_original_curves[0][:idx+1]
            self.markVirtualObs.append(1)

            y_virtual_original[ii]=transform_logistic([temp_curve],\
                      self.gp.logistic_hyper['midpoint'],self.gp.logistic_hyper['growth'],self.SearchSpace[-1,1])
           
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
            self.T = np.vstack((self.T, virtual_obs_t[ii].reshape((1, -1))))
            temp=np.asarray(virtual_obs_t_original[ii])
            self.T_original=np.vstack((self.T_original, temp.reshape((1, -1))))


            self.Y_original = np.append(self.Y_original,[y_virtual_original[ii]])
            self.Y_curves.append(temp_curve)
            
            # interpolating the cost for augmented observation
            y_cost_estimate=y_cost_original*virtual_obs_t[ii]
            self.Y_cost_original = np.append(self.Y_cost_original,[y_cost_estimate])
            
        
#        if self.verbose:
#            temp_y_original_whole_curve=transform_logistic(y_original_curves,\
#                               self.gp.logistic_hyper['midpoint'],self.gp.logistic_hyper['growth'],self.SearchSpace[-1,1])
#            print(np.round(temp_y_original_whole_curve,decimals=4), np.round(y_virtual_original,decimals=4))
#            
        
    def suggest_nextpoint(self): # logistic, time-cost, virtual
        """
        Main optimization method.


        Returns
        -------
        x: recommented point for evaluation
        """
 
        # init a new Gaussian Process============================================
        self.gp=ProductGaussianProcess(self.scaleSearchSpace,self.gp.hyper,self.gp.logistic_hyper)
        self.gp.fit(self.X, self.T,self.Y,self.Y_curves)
            
        # we store the condition number here=====================================
        self.condition_number.append(self.gp.cond_num)
        if self.verbose:
            print("ln of conditioning number of GP covariance matrix", np.round(np.log(self.gp.cond_num),decimals=1))

        # count number of real observations
        count=len(self.markVirtualObs)-np.sum(self.markVirtualObs)
        count=np.int(count)

        # optimize GP hyperparameters and Logistic hyper after 3*d iterations
        if  len(self.Y)%(2*self.dim)==0:

            hyper=[self.gp.hyper['lengthscale_x'],self.gp.hyper['lengthscale_t'], \
                   self.gp.logistic_hyper['midpoint'],self.gp.logistic_hyper['growth']]
            newlengthscale_x,newlengthscale_t,new_midpoint, new_growth = self.gp.optimize_lengthscale_logistic_hyper(hyper,self.gp.noise_delta)
            
            self.gp.hyper['lengthscale_x']=newlengthscale_x
            self.gp.hyper['lengthscale_t']=self.gp.hyper['lengthscale_t']
            self.gp.logistic_hyper['midpoint']=new_midpoint
            self.gp.logistic_hyper['growth']=new_growth
          
            if self.verbose:
                print("==estimated lengthscale_x={:.4f}   lengthscale_t={:.3f}   Logistic_m0={:.1f}   Logistic_g0={:.1f}".format(
                    newlengthscale_x,newlengthscale_t,new_midpoint,new_growth))
                
        # Set acquisition function
        start_opt=time.time()

        # linear regression is used to fit the cost
        # fit X and T
        combine_input=np.hstack((self.X,self.T))
        self.linear_regression.fit(combine_input,self.Y_cost)
        
        # maximize the acquisition function to select the next point =================================
        x_max_temp=self.acq_utility_cost()
        x_max=x_max_temp[:-1]
        t_max=x_max_temp[-1]       
            
        # record keeping stuffs ====================================================
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))

        # this is for house keeping stuff        
        self.markVirtualObs.append(0)

        self.X = np.vstack((self.X, x_max.reshape((1, -1))))
        self.T = np.vstack((self.T, t_max.reshape((1, -1))))

        # compute X in original scale
        temp_X_new_original=self.Xscaler.inverse_transform(np.reshape(x_max,(-1,self.dim-1)))
        #temp_X_new_original=x_max*self.max_min_gap[:-1]+self.SearchSpace[:-1,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        #temp_T_new_original=t_max*self.max_min_gap[-1]+self.SearchSpace[-1,0]
        temp_T_new_original=self.Tscaler.inverse_transform(np.reshape(t_max,(-1,1)))
        self.T_original=np.vstack((self.T_original, temp_T_new_original))

        # evaluate Y using original X
        x_original_to_test=x_max_temp*self.max_min_gap+self.SearchSpace[:,0]

        # evaluate the black-box function=================================================
        y_original_curves, y_cost_original= self.f(x_original_to_test)
        
        # compute the utility score by transformation
        y_original=transform_logistic(y_original_curves,\
              self.gp.logistic_hyper['midpoint'],self.gp.logistic_hyper['growth'],self.SearchSpace[-1,1])
        
        if len(y_original_curves)==1: # list
            self.Y_curves.append(y_original_curves[0])
        else:
            self.Y_curves.append(y_original_curves)

        
        self.Y_original = np.append(self.Y_original,y_original)
        self.Y_cost_original = np.append(self.Y_cost_original,y_cost_original)

        # augmenting virtual observations =====================================================
        self.generating_virtual_observations(x_max,t_max,y_original_curves,y_cost_original[0])
        
        # update Y after change Y_original        
        if np.std(self.Y_original)==0:
            self.Y=(self.Y_original-np.mean(self.Y_original))
        else:
            self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
            
        self.Y_cost=(self.Y_cost_original-np.min(self.Y_cost_original))/(np.max(self.Y_cost_original)-np.min(self.Y_cost_original))
                    
        #if self.verbose:
        np.set_printoptions(suppress=True)

        print("[original scale] x={} t={:.0f} current y={:.2f}, ybest={:.2f}".format( np.round(self.X_original[-1],decimals=4),\
              np.asscalar(self.T_original[-1]),np.asscalar(self.Y_original[-1]), np.asscalar(self.Y_original.max())))


