import numpy as np
import itertools
#import time
#import bayes_opt.utils
from bayes_opt.utility.basic_utility_functions import transform_logistic_marginal, transform_logistic

counter = 0

class RandomSearch(object):
    def __init__(self, func_params, verbose=True):
        """      
        Input parameters
        ----------
        
        
        func_params:                function to optimize
        func_params.init bound:     initial bounds for parameters
        func_params.bounds:         bounds on parameters        
        func_params.func:           a function to be optimized
        
        Returns
        -------
        dim:            dimension
        """

        # Find number of parameters
        self.verbose=verbose

        try:
            bounds=func_params['function']['bounds']
        except:
            bounds=func_params['function'].bounds

        self.dim = len(bounds)

        # Create an array with parameters bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            self.keys = list(bounds.keys())
        
            self.bounds = []
            for key in self.keys:
                self.bounds.append(bounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(bounds)
 
        # create a scalebounds 0-1
        self.scalebounds = np.array([np.zeros(self.dim), np.ones(self.dim)]).T
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        
        
        # Some function to be optimized
        try:
            self.f = func_params['function']['func']
        except:
            self.f = func_params['function'].func

            
        # store X in original scale
        self.X_original = None

        # store X in 0-1 scale
        self.X = None
        
        self.Y_curves=[]
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_original = None

        self.time_opt = 0
    
    """
    def init(self, n_init_points=3,seed=1):
        utils.set_seed(seed)
        """
        
    def suggest_nextpoint(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """
        
        
        x_max = [np.random.uniform(x[0], x[1], size=1) for x in self.bounds]
        x_max = np.asarray(x_max).T
        
        if self.X_original is None:
            self.X_original = x_max[:,:-1]
            self.T_original = x_max[:,-1]

        else:
            self.X_original=np.vstack((self.X_original, x_max[:,:-1]))
            self.T_original=np.vstack((self.T_original, x_max[:,-1]))

       
        y_curves, y_cost=self.f(x_max)
        y_max=transform_logistic_marginal(y_curves,self.bounds[-1,1])

        self.Y_curves+=y_curves

        # evaluate Y using original X
        if self.Y_original is None:
            #self.Y_original = np.array([self.f(x_max)])
            self.Y_original = np.array([y_max])
            y_cost=np.atleast_2d(np.asarray(y_cost)).astype('Float64')
            self.Y_cost_original=y_cost
            self.Y_cost_original=np.reshape(self.Y_cost_original,(-1,1))
            
        
        else:
            #self.Y_original = np.append(self.Y_original, self.f(x_max))
            self.Y_original = np.append(self.Y_original, y_max)
            self.Y_cost_original = np.append(self.Y_cost_original, y_cost)

        # update Y after change Y_original
        self.Y=(self.Y_original - np.mean(self.Y_original))/np.std(self.Y_original)
        if self.verbose:
            print("x={} current y={:.4f}, ybest={:.4f}".format(self.X_original[-1],self.Y_original[-1], self.Y_original.max()))

        return

class GridSearch(object):
    def __init__(self, func_params, acq_params, verbose=True):
        """      
        Input parameters
        ----------
        
        
        func_params:                function to optimize
        func_params.init bound:     initial bounds for parameters
        func_params.bounds:         bounds on parameters        
        func_params.func:           a function to be optimized
        
        
        acq_params:                 acquisition function, 
        acq_params.steps:           number of gridsearch sweeps to be run
        acq_params.points:          either list of points per parameter or integer of number of points
        acq_params.opt_toolbox:     optimization toolbox 'nlopt','direct','scipy'
                            
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scalebounds:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on grid search
        gp:             Gaussian Process object
        """

        # Find number of parameters
        self.verbose=verbose

        try:
            bounds=func_params['function']['bounds']
        except:
            bounds=func_params['function'].bounds

        self.dim = len(bounds)

        # Create an array with parameters bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            self.keys = list(bounds.keys())
        
            self.bounds = []
            for key in list(bounds.keys()):
                self.bounds.append(bounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(bounds)
        
        self.grid = []
        self.grid_params = acq_params
        for i, param in enumerate(self.grid_params["points"]):
            if isinstance(param, int):
                self.grid.append([self.bounds[i, 0] + s * (self.bounds[i, 1] - self.bounds[i, 0]) / (param-1) for s in range(param)])
            else:
                if len(param.shape) > 1:
                    self.grid.append(params[0, :])
                else:
                    self.grid.append([self.bounds[i, 0] + s * (self.bounds[i, 1] - self.bounds[i, 0]) / (int(param[0])-1) for s in range(param[0])])
        self.points = list(itertools.product(*self.grid))
        self.grid_lvl = [0,0]
        # create a scalebounds 0-1
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        
        
        # Some function to be optimized
        try:
            self.f = func_params['function']['func']
        except:
            self.f = func_params['function'].func
    
        # store X in original scale
        self.X_original= None

        self.Y_curves=[]
        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_original = None

        self.time_opt=0    
    
    def init(self, seed=1):

        np.random.seed(seed)
        
    def suggest_nextpoint(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.grid_lvl[0] >= len(self.points):
            if self.grid_lvl[1] == self.grid_params["steps"]:
                return
            self.grid_lvl[0] = 0
            self.grid_lvl[1] += 1
            
            cur_max = self.X_original[self.Y_original.argmax()]
            for i, param in enumerate(self.grid_params["points"]):
                max_i = self.grid[i].index(cur_max[i])
                b_min = self.grid[i][0] if max_i == 0 else (self.grid[i][max_i] + self.grid[i][max_i - 1]) / 2
                b_max = self.grid[i][-1] if max_i == len(self.grid[i]) - 1 else (self.grid[i][max_i] + self.grid[i][max_i + 1]) / 2
                if isinstance(param, int):
                    self.grid[i] = [b_min + s * (b_max - b_min) / (param-1) for s in range(param)]
                else:
                    if len(param.shape) > 1:
                        self.grid[i] = param[self.grid_lvl[1], :]
                    else:
                        intervals = param[self.grid_lvl[1]]
                        self.grid[i] = [b_min + s * (b_max - b_min) / (intervals-1) for s in range(intervals)]
            self.points = list(itertools.product(*self.grid))       
            print("\n")

        x_max = self.points[self.grid_lvl[0]]
        self.grid_lvl[0] += 1
        x_max = np.asarray(x_max).T
        if self.X_original is None:
            self.X_original = np.array([x_max])
        else:
            self.X_original=np.vstack((self.X_original, x_max))
        # evaluate Y using original X
        if self.Y_original is None:
            self.Y_original = np.array([self.f(x_max)])
        else:
            self.Y_original = np.append(self.Y_original, self.f(x_max))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
       
        if self.verbose:
            print("x={} current y={:.4f}, ybest={:.4f}".format(self.X_original[-1],self.Y_original[-1], self.Y_original.max()))

class HyperBand(object):
    def __init__(self, func_params, verbose=True):
        """      
        Input parameters
        ----------
        
        
        func_params:                function to optimize
        func_params.init bound:     initial bounds for parameters
        func_params.bounds:         bounds on parameters        
        func_params.func:           a function to be optimized
        
        Returns
        -------
        dim:            dimension
        """

        # Find number of parameters
        self.verbose=verbose

        try:
            bounds=func_params['function']['bounds']
        except:
            bounds=func_params['function'].bounds

        self.dim = len(bounds)

        # Create an array with parameters bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            self.keys = list(bounds.keys())
        
            self.bounds = []
            for key in self.keys:
                self.bounds.append(bounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(bounds)
 
        # create a scalebounds 0-1
        self.scalebounds = np.array([np.zeros(self.dim), np.ones(self.dim)]).T
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        
        
        # Some function to be optimized
        try:
            self.f = func_params['function']['func']
        except:
            self.f = func_params['function'].func

            
        # store X in original scale
        self.X_original = None

        # store X in 0-1 scale
        self.X = None
        
        self.Y_curves=[]
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_original = None

        self.time_opt = 0
        
        #hyperBanc setup
        self.max_iter = func_params['R']
        self.thinning = func_params['n']
        self.s_max = int(np.log(self.max_iter) / np.log(self.thinning))
        self.stepsPerBracket = (self.s_max + 1) * self.max_iter

        self.s = self.s_max
        self.setup_configs(self.s)
        
        self.done = False

    def setup_configs(self, s):
        # starts a new iteration of successive halving
        n = int(np.ceil(int(self.stepsPerBracket/self.max_iter/(s+1))*self.thinning**s))
        self.run_times = [int(self.max_iter*self.thinning**(-s)) + self.bounds[-1][0] for s in range(s, -1, -1)]
        self.configs = [[np.random.uniform(x[0], x[1], size=1) for x in self.bounds] for _ in range(n)]
        self.next_conf = 0

    def suggest_nextpoint(self):
        """
            Executes next simulation for HyperBand
        """
        x_max = self.configs[self.next_conf]
        x_max[-1] = np.array([self.run_times[0]])
        x_max = np.asarray(x_max).T

        if self.X_original is None:
            self.X_original = x_max[:,:-1]
            self.T_original = x_max[:,-1]

        else:
            self.X_original=np.vstack((self.X_original, x_max[:,:-1]))
            self.T_original=np.vstack((self.T_original, x_max[:,-1]))

       
        y_curves, y_cost=self.f(x_max)
        y_max=transform_logistic_marginal(y_curves,self.bounds[-1,1])

        self.Y_curves+=y_curves

        # evaluate Y using original X
        if self.Y_original is None:
            #self.Y_original = np.array([self.f(x_max)])
            self.Y_original = np.array([y_max])
            y_cost=np.atleast_2d(np.asarray(y_cost)).astype('Float64')
            self.Y_cost_original=y_cost
            self.Y_cost_original=np.reshape(self.Y_cost_original,(-1,1))
        else:
            #self.Y_original = np.append(self.Y_original, self.f(x_max))
            self.Y_original = np.append(self.Y_original, y_max)
            self.Y_cost_original = np.append(self.Y_cost_original, y_cost)

        # update Y after change Y_original
        self.Y=(self.Y_original - np.mean(self.Y_original))/np.std(self.Y_original)
        if self.verbose:
            print("x={} current y={:.4f}, ybest={:.4f}".format(self.X_original[-1],self.Y_original[-1], self.Y_original.max()))

        self.next_conf += 1
        if self.next_conf >= len(self.configs):
            n = len(self.configs)
            self.configs =  [ self.configs[i] for i in np.argsort(self.Y_original[-n:])[0:int( n/self.thinning )] ]
            self.next_conf = 0
            self.run_times.pop(0)
            if len(self.run_times) == 0:
                self.s -= 1
                if self.s >= 0:
                    self.setup_configs(self.s)
                else:
                    self.done = True
        return
