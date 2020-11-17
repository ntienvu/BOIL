# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score,accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from bayes_opt.test_functions.cnn.cnn_tf_housenumber_blackbox import run_cnn_evaluation

from bayes_opt.test_functions.cnn.cnn_tf_cifar10_blackbox import run_cnn_evaluation_cifar

import os
#import matlab.engine
#import matlab
#eng = matlab.engine.start_matlab()
from sklearn.metrics import f1_score 
        
def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x
  
class CNN_HouseNumber:
    '''
    Tuning Convolutional Neural Network on Housing Number dataset
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 7
        
        if bounds == None: 
            self.bounds = OrderedDict([('filter_sz',(1,8)),('pool_sz',(1,5)),('batch_sz',(16,1000)),
                                       ('lr',(1e-6,1e-2)),
            ('momentum',(0.82,0.999)),('decay',(0.92,0.99)),('MaxIter',(60,130))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='CNN_HouseNumber'


        path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(path,'cnn/house_number_data.pkl'),'rb') as f:
            temp = pickle.load(f)
        
        [self.Xtrain,self.Ytrain,self.Xtest,self.Ytest]=temp

    def func_returnAcc(self,X):
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            output=run_cnn_evaluation(X,self.Xtrain,self.Ytrain,self.Xtest,self.Ytest,isReturnAcc=True)
            acc_curve=[output[0]]
            elapse=[output[1]]
        else:
            output=np.apply_along_axis(run_cnn_evaluation,1,X,self.Xtrain,self.Ytrain,self.Xtest,self.Ytest,isReturnAcc=True)
            acc_curve=output[:,0].tolist()
            elapse=output[:,1].tolist()

        # we return -1* error_curve
        return acc_curve,elapse
    
    def func(self,X):
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            output=run_cnn_evaluation(X,self.Xtrain,self.Ytrain,self.Xtest,self.Ytest)
            error_curve=[output[0]]
            elapse=[output[1]]
        else:
            output=np.apply_along_axis(run_cnn_evaluation,1,X,self.Xtrain,self.Ytrain,self.Xtest,self.Ytest)
            error_curve=output[:,0].tolist()
            elapse=output[:,1].tolist()

        # we return -1* error_curve
        return error_curve,elapse

    
class CNN_Cifar10:
    '''
    Tuning Convolutional Neural Network on Housing Number dataset
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 7
        
        if bounds == None: 
            self.bounds = OrderedDict([('filter_sz',(1,8)),('pool_sz',(1,5)),('batch_sz',(16,1000)),
                                       ('lr',(1e-6,1e-3)),
            ('momentum',(0.82,0.999)),('decay',(0.92,0.99)),('MaxIter',(60,130))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='CNN_Cifar10'


        path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(path,'cnn/cifar10.pickle'),'rb') as f:
            temp = pickle.load(f)
        
        [self.Xtrain,self.Ytrain,self.Xtest,self.Ytest]=temp

    def func_returnAcc(self,X):
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            output=run_cnn_evaluation_cifar(X,self.Xtrain,self.Ytrain,self.Xtest,self.Ytest,isReturnAcc=True)
            acc_curve=[output[0]]
            elapse=[output[1]]
        else:
            output=np.apply_along_axis(run_cnn_evaluation_cifar,1,X,self.Xtrain,self.Ytrain,self.Xtest,self.Ytest,isReturnAcc=True)
            acc_curve=output[:,0].tolist()
            elapse=output[:,1].tolist()

        # we return -1* error_curve
        return acc_curve,elapse
    
    def func(self,X):
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            output=run_cnn_evaluation_cifar(X,self.Xtrain,self.Ytrain,self.Xtest,self.Ytest)
            error_curve=[output[0]]
            elapse=[output[1]]
        else:
            output=np.apply_along_axis(run_cnn_evaluation_cifar,1,X,self.Xtrain,self.Ytrain,self.Xtest,self.Ytest)
            error_curve=output[:,0].tolist()
            elapse=output[:,1].tolist()

        # we return -1* error_curve
        return error_curve,elapse