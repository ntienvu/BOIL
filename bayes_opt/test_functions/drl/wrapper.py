import numpy as np
import os
from collections import deque
import gym
from gym import spaces
import pickle as pickle
from multiprocessing import Process, Pipe
import copy

class TaxiWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        done = (reward == 20)
        return ob, reward, done, info

class ParallelEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among batch_size processes and
    executed in parallel.

    Args:
        env (meta_policy_search.envs.base.MetaEnv): meta environment object
        batch_size (int): number of parallel environments
    """

    def __init__(self, env, batch_size):
        self.n_envs = batch_size
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(batch_size)])
        seeds = np.random.choice(range(10**6), size=batch_size, replace=False)

        self.ps = [
            Process(target=worker, args=(work_remote, remote, pickle.dumps(env), seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        """
        Executes actions on each env

        Args:
            actions (list): list of actions of length batch_size

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length batch_size
        """
        assert len(actions) == self.num_envs

        # step remote environments
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, env_infos = zip(*results)
        return obs, rewards, dones, env_infos

    def reset(self):
        """
        Resets the environments of each worker

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self.n_envs


def worker(remote, parent_remote, env_pickle, seed):
    """
    Instantiation of a parallel worker for collecting samples. It loops continually checking the task that the remote
    sends to it.

    Args:
        remote (multiprocessing.Connection):
        parent_remote (multiprocessing.Connection):
        env_pickle (pkl): pickled environment
        seed (int): random seed for the worker
    """
    parent_remote.close()

    env = pickle.loads(env_pickle)
    np.random.seed(seed)
    while True:
        # receive command and data from the remote
        cmd, data = remote.recv()

        # do a step in each of the environment of the worker
        if cmd == 'step':
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            remote.send((obs, reward, done, info))

        # reset all the environments of the worker
        elif cmd == 'reset':
            obs = env.reset()
            remote.send(obs)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError