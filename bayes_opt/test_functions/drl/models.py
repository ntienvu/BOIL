import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

def make_net( in_layer, architecture, final = None, run_seed = None):
        params = []
        last_layer = in_layer
        #initializer = tf.initializers.glorot_uniform()
        for desc in architecture:
            if isinstance(desc, int) or isinstance(desc[0], int):
                #new_layer = tf.layers.Dense(desc, activation = tf.nn.tanh, kernel_initializer = tf.contrib.layers.xavier_initializer(seed=run_seed) )
                new_layer = tf.layers.Dense(desc, activation = tf.nn.tanh, kernel_initializer = tf.initializers.glorot_uniform(seed=run_seed) )
            else:
                new_layer = tf.layers.Conv2D(desc[1], desc[2], strides = desc[3], activation = tf.nn.tanh, kernel_initializer = tf.contrib.layers.xavier_initializer(seed=run_seed))
            last_layer = new_layer(last_layer)
            params.extend(new_layer.weights)

        if final is not None:
            if isinstance(final, int) or isinstance(final[0], int):
                #new_layer = tf.layers.Dense(final, kernel_initializer = tf.contrib.layers.xavier_initializer(seed=run_seed) )
                new_layer = tf.layers.Dense(final, kernel_initializer = tf.initializers.glorot_uniform(seed=run_seed) )

            else:
                new_layer = tf.layers.Conv2D(final[1], final[2], strides = final[3], kernel_initializer = tf.initializers.glorot_uniform(seed=run_seed))
            last_layer = new_layer(last_layer)
            params.extend(new_layer.weights)
        return last_layer, params


def make_dueling_dqn( in_layer, architecture, actions, run_seed = None):
        # calculate output and cost
        out, params = make_net(in_layer, architecture)
        out_shape = out.get_shape().as_list()[1]
        initializer = tf.initializers.glorot_uniform(seed=0)

        fcV_W = tf.Variable(initializer(shape = [out_shape, 512]))
        fcV_b = tf.Variable(tf.zeros([512], dtype=tf.float32), dtype=tf.float32)
        val_a = tf.nn.elu(tf.matmul(out, fcV_W) + fcV_b)

        fcV2_W = tf.Variable(initializer(shape = [512, 1]))
        fcV2_b = tf.Variable(tf.zeros([1], dtype=tf.float32))
        value_out = tf.matmul(val_a, fcV2_W) + fcV2_b


        fcA_W = tf.Variable(initializer(shape = [out_shape, 512]))
        fcA_b = tf.Variable(tf.zeros([512], dtype=tf.float32))
        adv_a = tf.nn.elu(tf.matmul(out, fcA_W) + fcA_b)

        fcA2_W = tf.Variable(initializer(shape = [512, actions]))
        fcA2_b = tf.Variable(tf.zeros([actions], dtype=tf.float32))
        adv_out = tf.matmul(adv_a, fcA2_W) + fcA2_b

        params += [fcV_W, fcV_b, fcV2_W, fcV2_b, fcA_W, fcA_b, fcA2_W, fcA2_b]
        output = value_out + adv_out - tf.reduce_mean(adv_out)
        return output, params

def make_policy_net( in_layer, architecture, actions, continuous = False, run_seed = None):
        # calculate output and cost
        out, params = make_net(in_layer, architecture)
        out_shape = out.get_shape().as_list()[1]
        initializer = tf.initializers.glorot_uniform(seed=0)

        fcPol_W = tf.Variable(initializer(shape = [out_shape, actions]))
        fcPol_b = tf.Variable(tf.zeros([actions], dtype=tf.float32))
        pol_out = tf.tanh(tf.matmul(out, fcPol_W) + fcPol_b)
        
        params += [fcPol_W, fcPol_b]

        if continuous:
            fcPol_W_var = tf.Variable(initializer(shape = [out_shape, actions]))
            fcPol_b_var = tf.Variable(tf.zeros([actions], dtype=tf.float32))
            pol_out_var = tf.nn.softplus(tf.matmul(out, fcPol_W_var) + fcPol_b_var)
            params += [fcPol_W_var, fcPol_b_var]      
            return pol_out, pol_out_var, params
        return pol_out, params


def make_ddpg_net(state_in, act_in, architecture, state_layer = 50, run_seed = None):
    initializer = tf.contrib.layers.xavier_initializer(seed=0)
    obs_dim = state_in.get_shape().as_list()[1]
    act_dim = act_in.get_shape().as_list()[1]

    if state_layer is not None:
        if isinstance(state_layer, int):
            inter_W = tf.Variable(initializer(shape = [obs_dim, state_layer]))
            inter_b = tf.Variable(tf.zeros([state_layer], dtype=tf.float32))
            intermediate = tf.nn.tanh(tf.matmul(state_in, inter_W) + inter_b)
            inter_dim = state_layer
            params = [inter_W, inter_b]
        else:
            raise(Exception("Not supported yet"))
    else: 
        intermediate = state_in
        inter_dim = obs_dim
        params = []

    state_W = tf.Variable(initializer(shape = [inter_dim, architecture[0]]))
    action_W = tf.Variable(initializer(shape = [act_dim, architecture[0]]))
    joint_b = tf.Variable(tf.zeros([architecture[0]], dtype=tf.float32))
    joint = tf.nn.tanh(tf.matmul(intermediate, state_W) + tf.matmul(act_in, action_W) + joint_b) 
    params += [state_W, action_W, joint_b]

    out, shared_params = make_net(joint, architecture[1:], final = 1)
    params += shared_params
    return out, params