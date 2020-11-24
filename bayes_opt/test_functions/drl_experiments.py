# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 21:52:22 2018

@author: VuNguyen
"""
import time, sys
import numpy as np
import tensorflow as tf
import gym
import tensorflow.compat.v1 as v1
# tf.disable_v2_behavior()

import copy
from bayes_opt.test_functions.drl import agents
from bayes_opt.test_functions.drl.wrapper import TaxiWrapper


def reshape(x, input_dim):
    """
    Reshapes x into a matrix with input_dim columns
    """
    x = np.array(x)
    if x.size == input_dim:
        x = x.reshape((1, input_dim))
    return x


algs = {"A2C": agents.A2C,
        "DQN": agents.DQN}


class DRL_experiment:
    def __init__(self, alg_name, env_name, varParams, fixParams={}, bounds=None, gpu_id=None):
        self.env = env_name
        self.alg_name = alg_name  # for printing purpose
        self.agent = algs[alg_name]
        if bounds is None:
            self.bounds = self.agent.bounds
        else:
            self.bounds = bounds
        self.varParams = varParams
        self.fixParams = fixParams
        self.input_dim = len(varParams)
        self.name = env_name
        self.ismax = 1
        self.gpu_id = gpu_id  # for multiple GPU machine: None, 0, 1

    def func(self, X):
        X = np.asarray(X)
        
        if len(X.shape) == 1:  # 1 data point
            output = self.evaluate(X)
            Reward = [output[0]]
            elapse = [output[1]]
        else:
            output = np.apply_along_axis(self.evaluate, 1, X)
            Reward = output[:, 0].tolist()
            elapse = output[:, 1].tolist()
        return Reward, elapse

    def evaluate(self, X, unwrap=False, display=False):
        # tf.reset_default_graph()
        tf.compat.v1.reset_default_graph()
        ag = self.agent()
        for i, val in enumerate(X):
            ag.params[self.varParams[i]] = val
        for k, v in self.fixParams.items():
            ag.params[k] = v

        start_time = time.time()
        run_seed = np.random.randint(0, 100000)
        # tf.reset_default_graph()
        env = gym.make(self.env)
        env.seed(run_seed)
        if unwrap:
            env = env.unwrapped
        if self.env == "Taxi-v2":
            env = TaxiWrapper(env)
        if display:
            from gym.wrappers import Monitor
            env = Monitor(env, './video', force=True)
        tf.random.set_seed(run_seed)
        ag.initialise(copy.copy(env.observation_space), copy.copy(env.action_space))
        # tf.set_random_seed(run_seed)
        tf.random.set_seed(run_seed)

        init = v1.global_variables_initializer()
        
        if self.gpu_id is None:
            session = v1.InteractiveSession()
        else:
            gpu_options = tf.GPUOptions(visible_device_list=str(self.gpu_id))   # set a GPU ID for multiple GPU cards
            session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

            # config = tf.ConfigProto(device_count = {'GPU': self.gpu_id})#gpu_id=0 or 1
            # session = tf.InteractiveSession(config=config)

        session.run(init)
        ag.set_session(session)
            
        N = int(ag.params['maxEpisodes'])
        totalrewards = np.empty(N)
        
        t = 0
        for e in range(N):
            totalrewards[e], t = ag.nextEpisode(env, t, display=display)
        end_time = time.time()
        elapse = end_time-start_time
        env.close()
        session.close()
        return totalrewards, elapse
