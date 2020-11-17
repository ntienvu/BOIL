# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""


import numpy as np
from bayes_opt.acquisition_functions import AcquisitionFunction
from bayes_opt import GaussianProcess
from bayes_opt import ProductGaussianProcess

from bayes_opt.acquisition_maximization import acq_max_with_name,acq_min_scipy_kwargs, acq_max
import time
from bayes_opt.utility.basic_utility_functions import transform_logistic
from sklearn.preprocessing import MinMaxScaler


#@author: Vu

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class BO_L(object):

    def __init__(self, func, SearchSpace,acq_name="ei_mu_max",verbose=1):
        """      
        BO_L: we perform Bayes Opt using Logistic transformation at the MaxEpoch
        ----------
        
        func:                       a function to be optimized
        SearchSpace:                bounds on parameters        
        acq_name:                   acquisition function name, such as [ei, gp_ucb]
                           
        Returns
        -------
        dim:            dimension
        SearchSpace:         SearchSpace on original scale
        scaleSearchSpace:    SearchSpace on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        self.method='bo_l'
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
            
        # we will performa BO at the Max Iteration, thus we set the SearchSpace of Epoch to the Max Value
        self.SearchSpace[-1,0]=self.SearchSpace[-1,1]
            
        self.dim = len(SearchSpace)

        scaler = MinMaxScaler()
        scaler.fit(self.SearchSpace.T)
        
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

        self.gp=ProductGaussianProcess(self.scaleSearchSpace,verbose=self.verbose)


#        # store the curves of performances
        self.Y_curves=[]

       
    # will be later used for visualization
    def posterior(self, Xnew):
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
        
    def init(self, n_init_points=3, seed=1):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """
        np.random.seed(seed)

        # Generate random points
        SearchSpace=np.copy(self.SearchSpace)
        SearchSpace[-1,0]=SearchSpace[-1,1] # last dimension, set it to MaxIter


        l = [np.random.uniform(x[0], x[1]) for _ in range(n_init_points) for x in SearchSpace]        #l=[np.linspace(x[0],x[1],num=n_init_points) for x in self.init_SearchSpace]

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
        #y_init_curves=self.f(init_X)
        #y_init_cost=y_init_curves
        
        
        self.Y_curves+=y_init_curves

        # we transform the y_init_curves as the average of [ curves * logistic ]
        y_init=transform_logistic(y_init_curves,self.gp.logistic_hyper['midpoint'],\
                                  self.gp.logistic_hyper['growth'], self.SearchSpace[-1,1])
        #y_init=y_init_curves
        y_init=np.reshape(y_init,(n_init_points,1))
        
     
        self.Y_original = np.asarray(y_init)      
        self.Y_cost_original=np.reshape(y_init_cost,(-1,1))

        self.Y_original_maxGP=np.asarray(y_init)      

        # convert it to scaleX
        self.X = self.Xscaler.transform(init_X)
        self.X=self.X[:,:-1]#remove the last dimension of MaxEpisode
        self.X=np.reshape(self.X,(n_init_points,-1))

        #temp=(self.T_original-self.SearchSpace[-1,0])/self.max_min_gap[-1]
        #self.T = np.asarray(temp)
        self.T = self.Tscaler.transform(self.T_original)

        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)



        
        
    def suggest_nextpoint(self): # logistic, time-cost, virtual
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """
     
        
        if self.acq_name=='random':
            x_max = [np.random.uniform(x[0], x[1], size=1) for x in self.SearchSpace]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
            
            self.time_opt=np.hstack((self.time_opt,0))
            return

        # init a new Gaussian Process
        #self.gp=ProductGaussianProcess(self.gp_params)
        self.gp=ProductGaussianProcess(self.scaleSearchSpace,verbose=self.verbose)

        self.gp.fit(self.X, self.T,self.Y,self.Y_curves)

        # optimize GP parameters after 3*d iterations
        if  len(self.Y)%(3*self.dim)==0:

            hyper=[self.gp.hyper['lengthscale_x'],self.gp.hyper['lengthscale_t'], \
                   self.gp.logistic_hyper['midpoint'],self.gp.logistic_hyper['growth']]
            newlengthscale_x,newlengthscale_t,new_midpoint, new_growth = self.gp.optimize_lengthscale_logistic_hyper(hyper,self.gp.noise_delta)
            
            self.gp.hyper['lengthscale_x']=newlengthscale_x
            self.gp.hyper['lengthscale_t']=self.gp.hyper['lengthscale_t']
            self.gp.logistic_hyper['midpoint']=new_midpoint
            self.gp.logistic_hyper['growth']=new_growth
            if self.verbose:
                print("estimated lengthscale_x={}, estimated lengthscale_t={}".format(
                    newlengthscale_x,newlengthscale_t))
   
        # Set acquisition function
        start_opt=time.time()

        # linear regression is used to fit the cost - alternatively we can use GP
        

        acq={}
        acq['name']=self.acq_name
        acq['dim']=self.scaleSearchSpace.shape[0]
        acq['scaleSearchSpace']=self.scaleSearchSpace   
        
        if self.acq_name=='ei_mu_max':# using max of mean(x) as the incumbent
            
            # optimie the GP predictive mean function to find the max of mu
            x_mu_max,mu_max_val=acq_max_with_name(gp=self.gp,scaleSearchSpace=self.scaleSearchSpace,acq_name='mu',IsReturnY=True)
            acq['mu_max']=  mu_max_val
        
        scaleSearchSpace=np.copy(self.scaleSearchSpace)
        scaleSearchSpace[-1,0]=scaleSearchSpace[-1,1] # last dimension, set it to MaxIter
        
        #x_max_temp=acq_max_with_name(gp=self.gp,scaleSearchSpace=scaleSearchSpace,acq_name=self.acq_name)
        
        
        myacq=AcquisitionFunction(acq)
        #x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=self.scaleSearchSpace,opt_toolbox=self.opt_toolbox,seeds=self.xstars)
        x_max_temp = acq_max(ac=myacq.acq_kind,gp=self.gp,bounds=self.scaleSearchSpace)

#        x_max_temp = acq_min_scipy_kwargs(myfunc=myacq.acq_kind,bounds=self.scaleSearchSpace,
#                        acq_func=myacq, isDebug=False)
        
        
        #x_max_temp=self.acq_utility_cost()
        x_max=x_max_temp[:-1]
        x_max_t=x_max_temp[-1]       
      
        
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))

        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))
        self.T = np.vstack((self.T, x_max_t.reshape((1, -1))))


        # compute X in original scale
        temp_X_new_original=x_max*self.max_min_gap[:-1]+self.SearchSpace[:-1,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        temp_T_new_original=x_max_t*self.max_min_gap[-1]+self.SearchSpace[-1,0]
        self.T_original=np.vstack((self.T_original, temp_T_new_original))

        # evaluate Y using original X
        x_original_to_test=x_max_temp*self.max_min_gap+self.SearchSpace[:,0]

        y_original_curves, y_cost_original= self.f(x_original_to_test)
        
        #y_original_curves= self.f(x_original_to_test)
        #y_cost_original=y_original_curves
        
        y_original=transform_logistic(y_original_curves,self.gp.logistic_hyper['midpoint'],\
                                      self.gp.logistic_hyper['growth'],self.SearchSpace[-1,1])
        #y_original=y_original_curves
        self.Y_curves.append(y_original_curves)
      
        
        self.Y_original = np.append(self.Y_original,y_original)
        self.Y_cost_original = np.append(self.Y_cost_original,y_cost_original)

        # update Y after change Y_original        
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
            
        #self.Y_cost=(self.Y_cost_original-np.mean(self.Y_cost_original))/np.std(self.Y_cost_original)
        
        if self.verbose:
            print("x={} t={} current y={:.4f}, ybest={:.4f}".format(self.X_original[-1],self.T_original[-1],self.Y_original[-1],self.Y_original.max()))


