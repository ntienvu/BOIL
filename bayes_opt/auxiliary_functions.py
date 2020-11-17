# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'..')


import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm


def run_seq_BO(func,func_input_bounds,acq='ei',n_init=3,niter=10,kernel='SE',stopping_condition=0,verbose=True,isPlot=False):
    # create an empty object for BO
        
    if isinstance(func_input_bounds,dict):
        # Get the name of the parameters
    
        bounds = []
        for key in list(func_input_bounds.keys()):
            #print(key)
            bounds.append(func_input_bounds[key])
            
        bounds = np.asarray(bounds)
    else:
        bounds=np.asarray(func_input_bounds)
        
    
    # create an empty object for BO
    acq_func={}
    #acq_func['name']='thompson'
    dim=len(bounds)
    
    acq_func['name']=acq # ei, ucb, random
    acq_func['dim']=dim
    
    #print(myfunction.bounds)
    
    myfunction={}
    myfunction['func']=func
    myfunction['bounds']=bounds
    
    func_params={}
    func_params['function']=myfunction
    
    
    acq_params={}
    acq_params['acq_func']=acq_func
    #acq_params['optimize_gp']=['marginal','loo','maximize']
    acq_params['optimize_gp']='maximize'
    acq_params['stopping']=stopping_condition

    #gp_params = {'kernel':'SE','lengthscale':0.1*dim,'noise_delta':1e-10,'flagIncremental':0}
    
    if kernel=='SE':
        gp_params = {'kernel':'SE','dim':dim,'lengthscale':0.1*dim,'noise_delta':1e-10}
    elif kernel=='ARD':
        gp_params = {'kernel':'ARD','dim':dim,'noise_delta':1e-10}
    else:
        print('Please select SE kernel or ARD kernel')
        return

    acq_params['opt_toolbox']='scipy'
    
    bo=CAI_Seq_BO(gp_params,func_params,acq_params,verbose)
    bo.init(gp_params,n_init_points=n_init)
    
    for index in range(0,niter):
        bo.suggest_nextpoint()
            
        if isPlot and dim<=2 and acq_func['name']!="random":
            visualization.plot_bo(bo)
        
    
    maxIdx=np.argmax(bo.Y_original)
    
    return bo.X_original[maxIdx],bo.Y_original[maxIdx],bo
       

def run_experiment(bo,gp_params,yoptimal=0,n_init=3,NN=10,runid=1):
    # create an empty object for BO
    
    start_time = time.time()
    bo.init(gp_params,n_init_points=n_init,seed=runid)
    
    # number of recommended parameters
    for index in tqdm(range(0,NN-1)):
        #print index
        bo.suggest_nextpoint()

    fxoptimal=bo.Y_original
    elapsed_time = time.time() - start_time

    return fxoptimal, elapsed_time

  
            
def yBest_Iteration(YY,BatchSzArray,IsPradaBO=1,Y_optimal=0,step=3,IsMax=-1):
    
    
    nRepeat=len(YY)
    YY=np.asarray(YY)
    #YY=np.vstack(YY.T)
    #YY=YY.T
    print(YY.shape, step)
    ##YY_mean=np.mean(YY,axis=0)
    #YY_std=np.std(YY,axis=0)
    
    mean_TT=[]
    
    mean_cum_TT=[]
    
    for idxtt,tt in enumerate(range(0,nRepeat)): # TT run
    
        if IsPradaBO==1:
            temp_mean=YY[idxtt,0:BatchSzArray[0]].max()
        else:
            temp_mean=YY[idxtt,0:BatchSzArray[0]].min()
        
        temp_mean_cum=YY[idxtt,0:BatchSzArray[0]].mean()

        start_point=0
        for idx,bz in enumerate(BatchSzArray): # batch
            if idx==len(BatchSzArray)-1:
                break
            bz=np.int(bz)

            #    get the average in this batch
            temp_mean_cum=np.vstack((temp_mean_cum,YY[idxtt,start_point:start_point+bz].mean()))
            
            # find maximum in each batch            
            if IsPradaBO==1:
                temp_mean=np.vstack((temp_mean,YY[idxtt,start_point:start_point+bz].max()))
            else:
                temp_mean=np.vstack((temp_mean,YY[idxtt,start_point:start_point+bz].min()))

            start_point=start_point+bz

        if IsPradaBO==1:
            myYbest=[temp_mean[:idx+1].max()*IsMax for idx,val in enumerate(temp_mean)]
            temp_mean_cum=temp_mean_cum*IsMax
            temp_mean=temp_mean*IsMax
        else:
            myYbest=[temp_mean[:idx+1].min() for idx,val in enumerate(temp_mean)]

        
        temp_regret=np.abs(temp_mean-Y_optimal)
        myYbest_cum=[np.mean(temp_regret[:idx+1]) for idx,val in enumerate(temp_regret)]


        if len(mean_TT)==0:
            mean_TT=myYbest
            mean_cum_TT=myYbest_cum
        else:
            #mean_TT.append(temp_mean)
            mean_TT=np.vstack((mean_TT,myYbest))
            mean_cum_TT=np.vstack((mean_cum_TT,myYbest_cum))
            
    mean_TT    =np.array(mean_TT)
    std_TT=np.std(mean_TT,axis=0)
    std_TT=np.array(std_TT).ravel()
    mean_TT=np.mean(mean_TT,axis=0)

    
    mean_cum_TT=np.array(mean_cum_TT)   
    std_cum_TT=np.std(mean_cum_TT,axis=0)
    std_cum_TT=np.array(std_cum_TT).ravel()
    mean_cum_TT=np.mean(mean_cum_TT,axis=0)
   
    #return mean_TT[::step],std_TT[::step]#,mean_cum_TT[::5],std_cum_TT[::5]
    
    return mean_TT[::step],std_TT[::step],mean_cum_TT[::step],std_cum_TT[::step]
    
