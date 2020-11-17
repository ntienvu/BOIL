# -*- coding: utf-8 -*-
# define Gaussian Process class


import numpy as np
from bayes_opt.acquisition_functions import unique_rows
from sklearn.preprocessing import MinMaxScaler

from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import scipy.linalg as spla
from bayes_opt.curve_compression import apply_one_transform_logistic, transform_logistic


class ProductGaussianProcess(object):
    # in this class of Gaussian process, we define k( {x,t}, {x',t'} )= k(x,x')*k(t,t')
    
    
    #def __init__ (self,param):
    def __init__ (self,SearchSpace,gp_hyper=None,logistic_hyper=None,verbose=0):
        self.noise_delta=1e-4
        self.noise_upperbound=1e-2
        self.mycov=self.cov_RBF_time
        self.SearchSpace=SearchSpace
        scaler = MinMaxScaler()
        scaler.fit(SearchSpace.T)
        self.Xscaler=scaler
        self.verbose=verbose
        self.dim=SearchSpace.shape[0]
        
        if gp_hyper is None:
            self.hyper={}
            self.hyper['var']=1 # standardise the data
            self.hyper['lengthscale_x']=0.02 #to be optimised
            self.hyper['lengthscale_t']=0.2 #to be optimised
        else:
            self.hyper=gp_hyper

        
        if logistic_hyper is None:
            self.logistic_hyper={}
            self.logistic_hyper['midpoint']=0.0
            self.logistic_hyper['growth']=1.0   
        else:
            self.logistic_hyper=logistic_hyper

        self.X=[]
        self.T=[]
        self.Y=[]
        self.Y_curves=None
#        self.hyper['lengthscale_x']_old=self.hyper['lengthscale_x']
#        self.hyper['lengthscale_x']_old_t=self.hyper['lengthscale_x']_t
        
        self.alpha=[] # for Cholesky update
        self.L=[] # for Cholesky update LL'=A
        
        self.MaxEpisode=0
        
        return None
       

    def cov_RBF_time(self, x1,t1,x2,t2,lengthscale,lengthscale_t):
        
        Euc_dist=euclidean_distances(x1,x2)
        exp_dist_x=np.exp(-np.square(Euc_dist)/lengthscale)
        
        Euc_dist=euclidean_distances(t1,t2)
        exp_dist_t=np.exp(-np.square(Euc_dist)/lengthscale_t)
        
        return exp_dist_x*exp_dist_t
                
    def fit(self,X,T,Y,Y_curves):
        """
        Fit Gaussian Process model

        Input Parameters
        ----------
        x: the observed points 
        t: time or number of episode
        y: the outcome y=f(x)
        
        """ 
        temp=np.hstack((X,T))
        ur = unique_rows(temp)
        
        T=T[ur]
        X=X[ur]
        Y=Y[ur]
        
        self.X=X
        self.Y=Y
        self.T=T
        self.Y_curves=[val for idx,val in enumerate(Y_curves) if ur[idx]==True]
        
        for curves in self.Y_curves:
            self.MaxEpisode=max(len(curves),self.MaxEpisode)
        #self.Y_curves=Y_curves[myidx]
            
        Euc_dist_x=euclidean_distances(X,X)
        #exp_dist_x=np.exp(-np.square(Euc_dist)/self.hyper['lengthscale_x'])+np.eye(len(X))*self.noise_delta
    
        Euc_dist_t=euclidean_distances(T,T)
        #exp_dist_t=np.exp(-np.square(Euc_dist)/self.hyper['lengthscale_x']_t)+np.eye(len(X))*self.noise_delta       
    
        self.KK_x_x=np.exp(-np.square(Euc_dist_x)/self.hyper['lengthscale_x']\
                           -np.square(Euc_dist_t)/self.hyper['lengthscale_t'])+np.eye(len(X))*self.noise_delta
          
        if np.isnan(self.KK_x_x).any(): #NaN
            print("nan in KK_x_x")
        
        #self.KK_x_x_inv=np.linalg.pinv(self.KK_x_x)
        self.L=np.linalg.cholesky(self.KK_x_x)
        temp=np.linalg.solve(self.L,self.Y)
        self.alpha=np.linalg.solve(self.L.T,temp)
        self.cond_num=self.compute_condition_number()
        
    def compute_condition_number(self):
        cond_num=np.linalg.cond(self.KK_x_x)
        return cond_num
    

    def log_marginal_lengthscale_logistic_hyper(self,hyper,noise_delta):
        """
        Compute Log Marginal likelihood of the GP model w.r.t. the provided lengthscale, noise_delta and Logistic hyperparameter
        """

        def compute_log_marginal_with_logistic_hyper(lengthscale, lengthscale_t,midpoint,growth,noise_delta):
            # compute K
            temp=np.hstack((self.X,self.T))
            ur = unique_rows(temp)
            myX=self.X[ur]
            myT=self.T[ur]
            
            # transform Y_curve to Y_original, then to Y
            Y_original=transform_logistic(self.Y_curves,midpoint,growth,self.MaxEpisode)
            myY=(Y_original-np.mean(Y_original))/np.std(Y_original)
            
            myY=myY[ur]
          
            self.Euc_dist_x=euclidean_distances(myX,myX)
            self.Euc_dist_t=euclidean_distances(myT,myT)
        
            KK=np.exp(-np.square(self.Euc_dist_x)/lengthscale-np.square(self.Euc_dist_t)/lengthscale_t)\
                +np.eye(len(myX))*noise_delta
                    
            
            try:
                temp_inv=np.linalg.solve(KK,myY)
            except: # singular
                return -np.inf
            
            try:
                #logmarginal=-0.5*np.dot(self.Y.T,temp_inv)-0.5*np.log(np.linalg.det(KK+noise_delta))-0.5*len(X)*np.log(2*3.14)
                first_term=-0.5*np.dot(myY.T,temp_inv)
                
                # if the matrix is too large, we randomly select a part of the data for fast computation
                if KK.shape[0]>200:
                    idx=np.random.permutation(KK.shape[0])
                    idx=idx[:200]
                    KK=KK[np.ix_(idx,idx)]
                #Wi, LW, LWi, W_logdet = pdinv(KK)
                #sign,W_logdet2=np.linalg.slogdet(KK)
                chol  = spla.cholesky(KK, lower=True)
                W_logdet=np.sum(np.log(np.diag(chol)))
                # Uses the identity that log det A = log prod diag chol A = sum log diag chol A
    
                #second_term=-0.5*W_logdet2
                second_term=-W_logdet
            except: # singular
                return -np.inf
            

            logmarginal=first_term+second_term-0.5*len(myY)*np.log(2*3.14)
                
            if np.isnan(np.asscalar(logmarginal))==True:
                print("lengthscale_x={:f} lengthscale_t={:f} first term ={:.4f} second  term ={:.4f}".format(
                        lengthscale,lengthscale_t,np.asscalar(first_term),np.asscalar(second_term)))

            #print(lengthscale, lengthscale_t,midpoint,growth,"logmarginal:",logmarginal)
            return np.asscalar(logmarginal)
        
        logmarginal=0

        if not isinstance(hyper,list) and len(hyper.shape)==2:
            logmarginal=[0]*hyper.shape[0]
            growth=hyper[:,3]
            midpoint=hyper[:,2]
            lengthscale_t=hyper[:,1]
            lengthscale_x=hyper[:,0]
            for idx in range(hyper.shape[0]):
                logmarginal[idx]=compute_log_marginal_with_logistic_hyper(lengthscale_x[idx],\
                           lengthscale_t[idx],midpoint[idx],growth[idx],noise_delta)
        else:
            lengthscale_x,lengthscale_t,midpoint,growth=hyper
            logmarginal=compute_log_marginal_with_logistic_hyper(lengthscale_x,lengthscale_t,\
                                                                 midpoint,growth,noise_delta)
        return logmarginal

#    def optimize_lengthscale_SE_maximizing(self,previous_theta,noise_delta):
#        """
#        Optimize to select the optimal lengthscale parameter
#        """
#                
#        # define a bound on the lengthscale
#        SearchSpace_lengthscale_min=0.01
#        SearchSpace_lengthscale_max=0.5
#        #mySearchSpace=[np.asarray([SearchSpace_lengthscale_min,SearchSpace_lengthscale_max]).T]
#        
#        mySearchSpace=np.asarray([[SearchSpace_lengthscale_min,SearchSpace_lengthscale_max],\
#                             [10*SearchSpace_lengthscale_min,2*SearchSpace_lengthscale_max]])
#        
#        # Concatenate new random points to possible existing
#        # points from self.explore method.
#        lengthscale_tries = np.random.uniform(mySearchSpace[:, 0], mySearchSpace[:, 1],size=(20, mySearchSpace.shape[0]))
#
#        #print lengthscale_tries
#
#        # evaluate
#        self.flagOptimizeHyperFirst=0 # for efficiency
#
#        logmarginal_tries=self.log_marginal_lengthscale(lengthscale_tries,noise_delta)
#        #print logmarginal_tries
#
#        #find x optimal for init
#        idx_max=np.argmax(logmarginal_tries)
#        lengthscale_init_max=lengthscale_tries[idx_max]
#        #print lengthscale_init_max
#        
#        myopts ={'maxiter':20*self.dim,'maxfun':20*self.dim}
#
#        x_max=[]
#        max_log_marginal=None
#        
#        res = minimize(lambda x: -self.log_marginal_lengthscale(x,noise_delta),lengthscale_init_max,
#                       SearchSpace=mySearchSpace,method="L-BFGS-B",options=myopts)#L-BFGS-B
#        if 'x' not in res:
#            val=self.log_marginal_lengthscale(res,noise_delta)    
#        else:
#            val=self.log_marginal_lengthscale(res.x,noise_delta)  
#        
#        # Store it if better than previous minimum(maximum).
#        if max_log_marginal is None or val >= max_log_marginal:
#            if 'x' not in res:
#                x_max = res
#            else:
#                x_max = res.x
#            max_log_marginal = val
#            #print res.x
#
#        return x_max
    
    def optimize_lengthscale_SE_logistic_hyper(self,previous_hyper,noise_delta):
        """
        Optimize to select the optimal lengthscale parameter
        """
        
        # define a bound on the lengthscale
        SearchSpace_l_min=0.005
        SearchSpace_l_max=0.3
        
        SearchSpace_midpoint_min=-3
        SearchSpace_midpoint_max=2
        
        SearchSpace_growth_min=0.5
        SearchSpace_growth_max=2
        #mySearchSpace=[np.asarray([SearchSpace_lengthscale_min,SearchSpace_lengthscale_max]).T]
        
        mySearchSpace=np.asarray([[SearchSpace_l_min,SearchSpace_l_max],[10*SearchSpace_l_min,2*SearchSpace_l_max],
                             [SearchSpace_midpoint_min,SearchSpace_midpoint_max],[SearchSpace_growth_min,SearchSpace_growth_max]])
        
        lengthscale_tries = np.random.uniform(mySearchSpace[:, 0], mySearchSpace[:, 1],size=(20, 4))

        # evaluate
        self.flagOptimizeHyperFirst=0 # for efficiency

        logmarginal_tries=self.log_marginal_lengthscale_logistic_hyper(lengthscale_tries,noise_delta)

        #find x optimal for init
        idx_max=np.argmax(logmarginal_tries)
        lengthscale_init_max=lengthscale_tries[idx_max]
        #print lengthscale_init_max
        
        myopts ={'maxiter':30*self.dim,'maxfun':30*self.dim}

        x_max=[]
        max_log_marginal=None
        
        res = minimize(lambda x: -self.log_marginal_lengthscale_logistic_hyper(x,noise_delta),lengthscale_init_max,
                       bounds=mySearchSpace,method="L-BFGS-B",options=myopts)#L-BFGS-B
        if 'x' not in res:
            val=self.log_marginal_lengthscale_logistic_hyper(res,noise_delta)    
        else:
            val=self.log_marginal_lengthscale_logistic_hyper(res.x,noise_delta)  
        
        # Store it if better than previous minimum(maximum).
        if max_log_marginal is None or val >= max_log_marginal:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_log_marginal = val
            #print res.x

        return x_max


#    def optimize_lengthscale(self,previous_theta_x, previous_theta_t,noise_delta):
#
#        prev_theta=[previous_theta_x,previous_theta_t]
#        newlengthscale,newlengthscale_t=self.optimize_lengthscale_SE_maximizing(prev_theta,noise_delta)
#        self.hyper['lengthscale_x']=newlengthscale
#        self.hyper['lengthscale_t']=newlengthscale_t
#        
#        # refit the model
#        temp=np.hstack((self.X,self.T))
#        ur = unique_rows(temp)
#        
#        self.fit(self.X[ur],self.T[ur],self.Y[ur],self.Y_curves)
#        
#        return newlengthscale,newlengthscale_t
            
    def optimize_lengthscale_logistic_hyper(self,prev_hyper,noise_delta):
        # optimize both GP lengthscale and logistic hyperparameter

            
        #prev_theta=[prev_theta_x,prev_theta_t,prev_midpoint,prev_growth]
        newlengthscale,newlengthscale_t,newmidpoint,newgrowth=self.optimize_lengthscale_SE_logistic_hyper(prev_hyper,noise_delta)
        self.hyper['lengthscale_x']=newlengthscale
        self.hyper['lengthscale_t']=newlengthscale_t
        
        # refit the model
        temp=np.hstack((self.X,self.T))
        ur = unique_rows(temp)

        # update Y here
        Y_original=transform_logistic(self.Y_curves,newmidpoint,newgrowth,self.SearchSpace[-1,1])
        Y=(Y_original-np.mean(Y_original))/np.std(Y_original)
        self.Y=Y
        #
        self.fit(self.X[ur],self.T[ur],self.Y[ur],self.Y_curves)
        
        return newlengthscale,newlengthscale_t,newmidpoint,newgrowth


    def compute_var(self,X,T,xTest,tTest):
        """
        compute variance given X and xTest
        
        Input Parameters
        ----------
        X: the observed points
        xTest: the testing points 
        
        Returns
        -------
        diag(var)
        """ 
        
        xTest=np.asarray(xTest)
        xTest=np.atleast_2d(xTest)
        
        tTest=np.asarray(tTest)
        tTest=np.atleast_2d(tTest)
        tTest=np.reshape(tTest,(-1,1))
        
        if self.kernel_name=='SE':
            #Euc_dist=euclidean_distances(xTest,xTest)
            #KK_xTest_xTest=np.exp(-np.square(Euc_dist)/self.hyper['lengthscale_x'])+np.eye(xTest.shape[0])*self.noise_delta
            #ur = unique_rows(X)
            myX=X
            myT=T
            
            Euc_dist_x=euclidean_distances(myX,myX)
            #exp_dist_x=np.exp(-np.square(self.Euc_dist_x)/lengthscale)+np.eye(len(myX))*noise_delta
        
            Euc_dist_t=euclidean_distances(myT,myT)
            #exp_dist_t=np.exp(-np.square(self.Euc_dist_t)/lengthscale_t)+np.eye(len(myX))*noise_delta      
        
            KK=np.exp(-np.square(Euc_dist_x)/self.hyper['lengthscale_x']-np.square(Euc_dist_t)/self.hyper['lengthscale_t'])\
                +np.eye(len(myX))*self.noise_delta
                    
                 
            Euc_dist_test_train_x=euclidean_distances(xTest,X)
            #Exp_dist_test_train_x=np.exp(-np.square(Euc_dist_test_train_x)/self.hyper['lengthscale_x'])
            
            Euc_dist_test_train_t=euclidean_distances(tTest,T)
            #Exp_dist_test_train_t=np.exp(-np.square(Euc_dist_test_train_t)/self.hyper['lengthscale_t'])
            
            KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train_x)/self.hyper['lengthscale_x']-np.square(Euc_dist_test_train_t)/self.hyper['lengthscale_t'])
                
        try:
            temp=np.linalg.solve(KK,KK_xTest_xTrain.T)
        except:
            temp=np.linalg.lstsq(KK,KK_xTest_xTrain.T, rcond=-1)
            temp=temp[0]
            
        #var=KK_xTest_xTest-np.dot(temp.T,KK_xTest_xTrain.T)
        var=np.eye(xTest.shape[0])-np.dot(temp.T,KK_xTest_xTrain.T)
        var=np.diag(var)
        var.flags['WRITEABLE']=True
        var[var<1e-100]=0
        return var 

    
        
    def predict(self,xTest, eval_MSE=True):
        """
        compute predictive mean and variance
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """    

        if len(xTest.shape)==1: # 1d
            xTest=xTest.reshape((-1,self.X.shape[1]+1))
            
        tTest=xTest[:,-1]
        tTest=np.atleast_2d(tTest)
        tTest=np.reshape(tTest,(xTest.shape[0],-1))
        
        xTest=xTest[:,:-1]
        
        # prevent singular matrix
        temp=np.hstack((self.X,self.T))
        ur = unique_rows(temp)
        
        X=self.X[ur]
        T=self.T[ur]
                
        Euc_dist_x=euclidean_distances(xTest,xTest)
        Euc_dist_t=euclidean_distances(tTest,tTest)

        KK_xTest_xTest=np.exp(-np.square(Euc_dist_x)/self.hyper['lengthscale_x']-np.square(Euc_dist_t)/self.hyper['lengthscale_t'])\
            +np.eye(xTest.shape[0])*self.noise_delta
        
        Euc_dist_test_train_x=euclidean_distances(xTest,X)
        
        Euc_dist_test_train_t=euclidean_distances(tTest,T)
        
        KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train_x)/self.hyper['lengthscale_x']-np.square(Euc_dist_test_train_t)/self.hyper['lengthscale_t'])
            
        #Exp_dist_test_train_x*Exp_dist_test_train_t
  
        # using Cholesky update
        mean=np.dot(KK_xTest_xTrain,self.alpha)
        v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        var=KK_xTest_xTest-np.dot(v.T,v)
        

        return mean.ravel(),np.diag(var)  

    def posterior(self,x):
        # compute mean function and covariance function
        return self.predict(self,x)
        
    
