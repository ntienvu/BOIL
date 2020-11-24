import numpy as np
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf
import time as time_mod

import gym.spaces
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import namedtuple

from bayes_opt.test_functions.drl import replay_buffer as bf
from bayes_opt.test_functions.drl.models import make_net, make_dueling_dqn, make_policy_net
tf.disable_v2_behavior()

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class RL_alg:
    params = OrderedDict()
    bounds = []

    def initialise(self, state_space, action_space):
        # setups the place holders for state, action, reward and termination variables
        # in discrete settings we use a 1-hot encoding
        # in continuous settings, we normalise onto the interval [-1,1]
        if isinstance(state_space, gym.spaces.Discrete):
            self.in_shape = [state_space.n]
            self.norm_state = (lambda x: np.squeeze(np.eye(state_space.n)[x], axis=0))
        else:
            self.in_shape = state_space.shape
            upper = np.array(state_space.high)
            upper[upper > 100000] = 1.0
            lower = np.array(state_space.low)
            lower[lower < -100000] = -1.0
            self.state_scale = [(upper + lower) / 2, (upper - lower) / 2]
            self.norm_state = (lambda s:
                               (np.reshape(s, (1, *state_space.shape)) - self.state_scale[0]) / self.state_scale[1])
        self.state_ph = tf.placeholder(tf.float32, shape=(None, *self.in_shape), name='obs')
        
        if isinstance(action_space, gym.spaces.Discrete):
            self.A = action_space.n
            self.action_ph = tf.placeholder(tf.int32, (None, 1), name='a')
            self.norm_act = (lambda x: x)
            self.denorm_act = (lambda x: x)
        elif isinstance(action_space, gym.spaces.Box):
            self.A = int(np.prod(action_space.shape))
            self.action_ph = tf.placeholder(tf.float32, shape=(None, self.A), name='obs')
            self.action_scale = [(action_space.high + action_space.low) / 2, (action_space.high - action_space.low) / 2]
            self.norm_act = (lambda a: (a - self.action_scale[0]) / self.action_scale[1])
            self.denorm_act = (lambda a: np.clip(a, -1, 1) * self.action_scale[1] + self.action_scale[0])
           
        self.reward_ph = tf.placeholder(tf.float32, (None, 1), name='r')
        self.done_ph = tf.placeholder(tf.float32, (None, 1), name='done')

    def set_session(self, session):
        self.session = session

    def act(self, state, time):
        action = self._act(self.norm_state(np.atleast_2d(state)), time)
        return self.denorm_act(action)

    def _act(self, state, time):
        pass

    def train(self, experience, time):
        prev_obs, action, reward, observation, done = experience
        prev_obs = self.norm_state(np.atleast_2d(prev_obs))
        observation = self.norm_state(np.atleast_2d(observation))
        action = self.norm_act(np.atleast_2d(action))
        self._train(prev_obs, action, reward, observation, done, time)

    def _train(self, prev_obs, action, reward, observation, done, time):
        pass

    def nextEpisode(self, env, time, display=False):
        observation = env.reset()
        """ if display and not:
            plt.clf()
            plt.imshow(env.render(mode='rgb_array'))
            plt.pause(0.001)  # pause for plots to update
        if display:
            from IPython import display as dsp
            plt.figure(100)
            img = plt.imshow(env.render(mode='rgb_array'))"""
        if display:
            from gym.wrappers import Monitor
            env = Monitor(env, './video', force=True)
        totalreward = 0
        last_step = time_mod.time()
        for t in range(time, time + self.params["maxSteps"]):
            action = self.act(observation, t)
            prev_obs = observation
            if time_mod.time() - last_step < 0.01:          # to avoid Mujoco error
                time_mod.sleep(0.01)
            observation, reward, done, _ = env.step(action)
            last_step = time_mod.time()
            totalreward += reward
            """if display:
                plt.figure(100)
                plt.clf()
                plt.imshow(env.render(mode='rgb_array'))
                plt.pause(0.001)  # pause for plots to update 
            if display:
                img.set_data(env.render(mode='rgb_array'))  # just update the data
                dsp.display(plt.gcf())
                dsp.clear_output(wait=True) """
            experience = (prev_obs, action, reward, observation, done)
            self.train(experience, t)
            if done:
                break
        return totalreward, t


def schedule(maxT, t, start, end):
    if maxT == 0:
        return start
    elif t >= maxT:
        return end
    else:
        return start + (end - start) * t/maxT


class DQN(RL_alg):
    """
        DQN and subclass implementations
    """
    params = OrderedDict([
        ("DDQN", True),                 # Enables double DQN computation
        ("Duelling", True),             # Enables duelling architecture
        ("PER", 1),                     # Enables prioritisation for experience replay
                                        # 0 - Uniform, 1 - PER (SegmentTree), 2 - PER (SegmentSumTree)
        ("gamma", 0.99),                # discount factor
        ("lr", 5e-4),                   # learning rate for gradient steps
        ("adam_beta1", 0.9),
        ("adam_beta2", 0.999),
        ("adam_epsilon", 1e-08),

        ("update_rate", 100),           # interval for copying parameters to target network
        ("maxSteps", 200),              # maximum length of a  single episode
        ("buffer_size", 50000),         # 
        ("batch_sz", 64),               #
        ("architecture", [32, 32]),     # 
        ("maxExp", 10000),              # exploration duration over which epsilon decays linearly
        ("eps_0", 1.0),                 # start epsilon
        ("eps_final", 0.05),            # final epsilon
        ("alpha", 0.4),                 # "amount" of prioritisation
        ("beta_time", 10000),           # linearly annealing of importance sampling correction for prioritisation
        ("beta_0", 0.6),                #
        ("beta_final", 1.0)])           #

    bounds = OrderedDict([
        ("gamma", (0.0, 1.0)),
        ("lr", (1e-5, 1e-2)),
        ("adam_beta1", (0.8, 1.0)),
        ("adam_beta2", (0.9, 1.0)),
        ("adam_epsilon", (0.0, 1.0)),
        ("buffer_size", (100, 100000)),
        ("batch_sz", (1, 512)),
        ("maxSteps", (50, 2000)),
        ("alpha", (0.0, 1.0)),
        ("update_rate", (1, 1000)),
        ("maxExp", (0, 1000000)),
        ("eps_0", (0.0, 1.0)),
        ("eps_final", (0.0, 1.0)),
        ("beta_time", (0, 1000000)),
        ("beta_0", (0.0, 1.0)),
        ("beta_final", (0.0, 1.0))])

    def initialise(self, state_space, action_space):
        assert isinstance(action_space, gym.spaces.Discrete), "DQN does not support continuous action spaces"
        super(DQN, self).initialise(state_space, action_space)
        if self.params["PER"] == 0:
            self.memory = bf.ReplayBuffer(self.params["buffer_size"], self.params["batch_sz"])
        elif self.params["PER"] == 1:
            self.memory = bf.SumTreePERBuffer(self.params["buffer_size"], self.params["batch_sz"], self.params["alpha"])
        else:
            self.memory = bf.SumSegmentTreePERBuffer(self.params["buffer_size"],
                                                     self.params["batch_sz"], self.params["alpha"])
        
        self.optimiser = tf.train.AdamOptimizer(self.params["lr"], beta1=self.params["adam_beta1"],
                                                beta2=self.params["adam_beta2"], epsilon=self.params["adam_epsilon"])

        self.next_state_ph = tf.placeholder(tf.float32, shape=(self.params["batch_sz"], *self.in_shape), name='obs')
        self.next_a_ph = tf.placeholder(tf.int32, [self.params["batch_sz"]], name='next_a')
        self.weight_ph = tf.placeholder(tf.float32, [self.params["batch_sz"]], name='loss_weight_ph')
        
        if self.params["Duelling"]:
            self.dqn_out, dqn_params = make_dueling_dqn(self.state_ph, self.params["architecture"], self.A)
            self.target_out, target_params = make_dueling_dqn(self.next_state_ph, self.params["architecture"], self.A)
        else:
            self.dqn_out, dqn_params = make_net(self.state_ph, self.params["architecture"], final=self.A)
            self.target_out, target_params = make_net(self.next_state_ph, self.params["architecture"], final=self.A)
        self.update_target_params_ops = [t.assign(s) for s, t in zip(dqn_params, target_params)]
        
        if self.params["DDQN"]:
            self.max_q = tf.reshape(tf.gather_nd(
                self.target_out, tf.stack((tf.range(self.params["batch_sz"]), self.next_a_ph), -1)), (-1, 1))
        else:
            self.max_q = tf.reshape(tf.reduce_max(self.target_out, axis=1), (-1, 1))
        
        self.target = self.reward_ph + (1.0 - self.done_ph) * self.params["gamma"] * self.max_q
        self.gathered_outputs = tf.gather_nd(
            self.dqn_out, tf.stack((tf.reshape(tf.range(self.params["batch_sz"]), (-1, 1)), self.action_ph), -1),
            name='gathered_outputs')
        # compute huber loss
        self.new_weights = tf.abs(self.gathered_outputs - self.target, name='abs_error')
        self.loss = tf.reduce_sum(
            self.weight_ph * tf.where(self.new_weights <= 1.0, x=0.5 * tf.square(self.new_weights),
                                      y=(self.new_weights - 0.5)))
        self.train_op = self.optimiser.minimize(self.loss, var_list=dqn_params)

    def correct_params(self):
        for p in ["update_rate", "maxSteps", "buffer_size", "batch_sz", "maxExp", "beta_time"]:
            self.params[p] = int(self.params[p])
        
    def _train(self, prev_obs, action, reward, observation, done, t):
        self.memory.add(prev_obs, action, reward, observation, done)
        idxes, weights, batch = self.memory.sample(
                beta=schedule(self.params["beta_time"], t, self.params["beta_0"], self.params["beta_final"]))
    
        batch = Transition(*zip(*batch))
        next_a = np.argmax(self.session.run(self.dqn_out,
                                            feed_dict={self.state_ph: np.vstack(batch.next_state)}), axis=1)

        _, new_weights, loss = self.session.run([self.train_op, self.new_weights, self.loss],
                                                feed_dict={
                                                self.state_ph: np.vstack(batch.state),
                                                self.next_state_ph: np.vstack(batch.next_state),
                                                self.action_ph: np.vstack(batch.action),
                                                self.next_a_ph: np.array(next_a),
                                                self.reward_ph: np.vstack(batch.reward),
                                                self.done_ph: np.vstack(batch.done),
                                                self.weight_ph: weights})
        self.memory.update_priorities(idxes, np.reshape(new_weights, (-1,)))
        if t % self.params["update_rate"] == 0:
            self.session.run(self.update_target_params_ops, feed_dict={})
    
    def _act(self, state, time):
        if np.random.random() < schedule(self.params["maxExp"], time, self.params["eps_0"], self.params["eps_final"]):
            return np.random.choice(self.A)
        else:
            return np.argmax(self.session.run(self.dqn_out, feed_dict={self.state_ph: state}))


class A2C(RL_alg):
    """
        A2C implementation and subclasses
    """
    params = OrderedDict([
        ("gamma", 0.99),                    # discount factor
        ("lr_actor", 5e-4),                 # learning rate for actor gradient steps
        ("actor_beta1", 0.9),
        ("actor_beta2", 0.999),
        ("actor_epsilon", 1e-08),
        ("lr_critic", 5e-4),                # learning rate for critic gradient steps
        ("critic_beta1", 0.9),
        ("critic_beta2", 0.999),
        ("critic_epsilon", 1e-08),
        ("maxSteps", 200),                  # maximum length of a  single episode
        ("batch_sz", 1),                    # dummy for now in case we want to add multistep updates
        ("val_architecture", [32, 32]),     # architecture for the value network
        ("pol_architecture", [32, 32]),     # architecture for the policy network
        ("ent_coef", 0.01)])                  # coefficient weighting the entropy penalisation

    bounds = OrderedDict([
        ("gamma", (0.0, 1.0)),
        ("lr_actor", (1e-5, 1e-2)),
        ("actor_beta1", (0.8, 1.0)),
        ("actor_beta2", (0.9, 1.0)),
        ("actor_epsilon", (0.0, 1.0)),
        ("lr_critic", (1e-5, 1e-2)),
        ("critic_beta1", (0.8, 1.0)),
        ("critic_beta2", (0.9, 1.0)),
        ("critic_epsilon", (0.0, 1.0)),
        ("ent_coef", (0, 0.1)),
        ("maxSteps", (50, 2000))])

    def initialise(self, state_space, action_space):
        super(A2C, self).initialise(state_space, action_space)
        self.training_steps = 0
        self.R = tf.placeholder(tf.float32, (None,), name='r')
        self.Adv = tf.placeholder(tf.float32, (None,), name='adv')
        # inputs and targets
        self.val_net, val_params = make_net(self.state_ph, self.params["val_architecture"], final=1)
 
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor_net, actor_params = make_policy_net(self.state_ph, self.params["pol_architecture"], self.A)
            self.policy = tf.distributions.Categorical(tf.nn.softmax(self.actor_net))
            self.action = tf.squeeze(self.policy.sample(1))
        elif isinstance(action_space, gym.spaces.Box):
            self.actor_net, self.pol_var, actor_params = \
                make_policy_net(self.state_ph, self.params["pol_architecture"], self.A, continuous=True)
            self.policy = tfp.distributions.MultivariateNormalDiag(loc=self.actor_net, scale_diag=self.pol_var)
            self.action = self.policy.sample()
        else:
            raise(Exception("This type of actionspace is not supported"))

        entropy = self.policy.entropy()
        logprobs = self.policy.log_prob(self.action_ph)
        pg_loss = tf.reduce_mean(-logprobs * self.Adv - self.params["ent_coef"]*entropy)
        vf_loss = tf.reduce_mean(tf.squared_difference(self.val_net, self.R))

        policy_opt = tf.train.AdamOptimizer(learning_rate=self.params["lr_actor"],
                                            beta1=self.params["actor_beta1"],
                                            beta2=self.params["actor_beta2"],
                                            epsilon=self.params["actor_epsilon"])
        value_opt = tf.train.AdamOptimizer(learning_rate=self.params["lr_critic"],
                                           beta1=self.params["critic_beta1"],
                                           beta2=self.params["critic_beta2"],
                                           epsilon=self.params["critic_epsilon"])
        self.vtrain_op = value_opt.minimize(vf_loss, var_list=val_params)
        self.ptrain_op = policy_opt.minimize(pg_loss, var_list=actor_params)

    def _train(self, prev_obs, action, reward, observation, done, t):
        V = self.session.run(self.val_net, feed_dict={self.state_ph: prev_obs})[0]
        R = [reward] if done else reward + self.params["gamma"] *\
                                  self.session.run(self.val_net, feed_dict={self.state_ph: observation})[0]
        advs = R-V
        self.session.run([self.vtrain_op, self.ptrain_op],
                         feed_dict={self.state_ph: prev_obs, self.action_ph: action, self.Adv: advs, self.R: R})
        
    def correct_params(self):
        for p in ["maxSteps", "batch_sz"]:
            self.params[p] = int(self.params[p])

    def _act(self, state, t):
        return self.session.run(self.action, feed_dict={self.state_ph: state})
