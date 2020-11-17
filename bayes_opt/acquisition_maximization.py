# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from bayes_opt.acquisition_functions import AcquisitionFunction
import sobol_seq

__author__ = 'Vu'


def acq_max_with_name(gp,scaleSearchSpace,acq_name="ei",IsReturnY=False,IsMax=True,fstar_scaled=None):
    acq={}
    acq['name']=acq_name
    acq['dim']=scaleSearchSpace.shape[0]
    acq['scaleSearchSpace']=scaleSearchSpace   
    if fstar_scaled:
        acq['fstar_scaled']=fstar_scaled   

    myacq=AcquisitionFunction(acq)
    if IsMax:
        x_max = acq_max(ac=myacq.acq_kind,gp=gp,bounds=scaleSearchSpace,opt_toolbox='scipy')
    else:
        x_max = acq_min_scipy(ac=myacq.acq_kind,gp=gp,bounds=scaleSearchSpace)
    if IsReturnY==True:
        y_max=myacq.acq_kind(x_max,gp=gp)
        return x_max,y_max
    return x_max

def acq_max(ac, gp, bounds, opt_toolbox='scipy',seeds=[],IsMax=True):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    y_max=np.max(gp.Y)
  
    x_max = acq_max_scipy(ac=ac,gp=gp,y_max=y_max,bounds=bounds)

    return x_max

def generate_sobol_seq(dim,nSobol):
    mysobol_seq = sobol_seq.i4_sobol_generate(dim, nSobol)
    return mysobol_seq
    

def acq_min_scipy_kwargs(myfunc, SearchSpace, **kwargs):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    dim=SearchSpace.shape[0]
    # Start with the lower bound as the argmax
    x_max = SearchSpace[:, 0]
    min_acq = None

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    myopts ={'maxiter':10*dim,'maxfun':20*dim}
    #myopts ={'maxiter':5*dim}

    #sobol_sequence=generate_sobol_seq(dim=dim,nSobol=500*dim)

    # multi start
    for i in range(3*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(100*dim, dim))
        
        #x_tries=sobol_sequence
    
        # evaluate
        y_tries=myfunc(x_tries,**kwargs)
        
        #find x optimal for init
        idx_min=np.argmin(y_tries)

        x_init_min=x_tries[idx_min]
    
        res = minimize(lambda x: myfunc(x.reshape(1, -1), **kwargs),x_init_min.reshape(1, -1),bounds=SearchSpace,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B

        if 'x' not in res:
            val=myfunc(res,**kwargs)        
        else:
            val=myfunc(res.x,**kwargs) 
        
        # Store it if better than previous minimum(maximum).
        if min_acq is None or val <= min_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            min_acq = val
            #print max_acq

    return np.clip(x_max, SearchSpace[:, 0], SearchSpace[:, 1])

    
def acq_min_scipy(ac, gp, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    min_acq = None

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    myopts ={'maxiter':10*dim,'maxfun':20*dim}
    #myopts ={'maxiter':5*dim}

    # multi start
    for i in range(3*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(50*dim, dim))
    
        # evaluate
        y_tries=ac(x_tries,gp=gp)
        
        #find x optimal for init
        idx_max=np.argmin(y_tries)

        x_init_max=x_tries[idx_max]
        
    
        res = minimize(lambda x: ac(x.reshape(1, -1), gp=gp),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B
      
        if 'x' not in res:
            val=ac(res,gp)        
        else:
            val=ac(res.x,gp) 
        
        # Store it if better than previous minimum(maximum).
        if min_acq is None or val <= min_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            min_acq = val
            #print max_acq

    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    
def acq_max_scipy(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    myopts ={'maxiter':10*dim,'maxfun':20*dim}
    #myopts ={'maxiter':5*dim}


    # multi start
    for i in range(1*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(50*dim, dim))
    
        # evaluate
        y_tries=ac(x_tries,gp=gp)
        #print "elapse evaluate={:.5f}".format(end_eval-start_eval)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B


        
        if 'x' not in res:
            val=ac(res,gp)        
        else:
            val=ac(res.x,gp) 

        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max

    
def acq_max_with_init(ac, gp, y_max, bounds, init_location=[]):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    myopts ={'maxiter':5*dim,'maxfun':10*dim}
    #myopts ={'maxiter':5*dim}


    # multi start
    #for i in xrange(5*dim):
    #for i in xrange(1*dim):
    for i in range(2*dim):
        # Find the minimum of minus the acquisition function 
        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20*dim, dim))
        
        if init_location!=[]:
            x_tries=np.vstack((x_tries,init_location))
        
            
        y_tries=ac(x_tries,gp=gp)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
        start_opt=time.time()
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B


        #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
        # value at the estimated point
        #val=ac(res.x,gp,y_max)        
        
        if 'x' not in res:
            val=ac(res,gp)        
        else:
            val=ac(res.x,gp) 

        
        end_opt=time.time()
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])

