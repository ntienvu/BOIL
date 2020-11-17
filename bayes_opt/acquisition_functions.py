import numpy as np
from scipy.stats import norm


counter = 0


class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq):

        self.acq=acq
        acq_name=acq['name']
        
        if 'mu_max' in acq:
            self.mu_max=acq['mu_max'] # this is for ei_mu acquisition function
        
        ListAcq=['bucb','ucb', 'ei','poi','random','ucb_pe',
                 'pure_exploration','mu','lcb','ei_mu_max'                          ]
        
        # check valid acquisition function
        IsTrue=[val for idx,val in enumerate(ListAcq) if val in acq_name]
        #if  not in acq_name:
        if  IsTrue == []:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(acq_name)
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name
            
        self.dim=acq['dim']
        
        if 'scalebounds' not in acq:
            self.scalebounds=[0,1]*self.dim
            
        else:
            self.scalebounds=acq['scalebounds']
               

    def acq_kind(self, x, gp):
        
        #if type(meta) is dict and 'y_max' in meta.keys():
        #   y_max=meta['y_max']
        y_max=np.max(gp.Y)
        #print self.kind
        if np.any(np.isnan(x)):
            return 0
       
        if self.acq_name == 'ucb':
            return self._ucb(x, gp)
        if self.acq_name == 'lcb':
            return self._lcb(x, gp)
        if self.acq_name == 'ei':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'ei_mu_max': # using max mu(x) as incumbent
            return self._ei(x, gp, self.mu_max)
        if self.acq_name == 'poi':
            return self._poi(x, gp, y_max)
        
        if self.acq_name == 'pure_exploration':
            return self._pure_exploration(x, gp) 
      
        if self.acq_name == 'mu':
            return self._mu(x, gp)
        
        if self.acq_name == 'ucb_pe':
            return self._ucb_pe(x, gp,self.acq['kappa'],self.acq['maxlcb'])
       
            
    def utility_plot(self, x, gp, y_max):
        if np.any(np.isnan(x)):
            return 0
        if self.acq_name == 'ei':
            return self._ei_plot(x, gp, y_max)
  
   
    @staticmethod
    def _mu(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        mean=np.atleast_2d(mean).T
        return mean
                
    @staticmethod
    def _lcb(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = 2 * np.log(len(gp.Y));

        return mean - np.sqrt(beta_t) * np.sqrt(var) 
        
    
    @staticmethod
    def _ucb(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T                
        
        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = 2 * np.log(len(gp.Y));
  
        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        return mean + np.sqrt(beta_t) * np.sqrt(var) 
    
    
    @staticmethod
    def _ucb_pe(x, gp, kappa, maxlcb):
        mean, var = gp.predict_bucb(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T

        value=mean + kappa * np.sqrt(var)        
        myidx=[idx for idx,val in enumerate(value) if val<maxlcb]
        var[myidx]=0        
        return var
    
   
    @staticmethod
    def _pure_exploration(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
        return np.sqrt(var)
        
   
    @staticmethod
    def _ei(x, gp, y_max):
        y_max=np.asscalar(y_max)
        mean, var = gp.predict(x, eval_MSE=True)
        var2 = np.maximum(var, 1e-10 + 0 * var)
        z = (mean - y_max)/np.sqrt(var2)        
        out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-10]=0
        return out
 
 
    @staticmethod      
    def _poi(x, gp,y_max): # run Predictive Entropy Search using Spearmint
        mean, var = gp.predict(x, eval_MSE=True)    
        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)
        z = (mean - y_max)/np.sqrt(var)        
        return norm.cdf(z)

   
def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]



class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'
