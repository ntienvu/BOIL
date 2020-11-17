# -*- coding: utf-8 -*-

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#from datetime import datetime
#from scipy.signal import convolve2d
#from scipy.io import loadmat
from sklearn.utils import shuffle
import time
#from benchmark import get_data, error_rate
#import pickle


def error_rate(p, t):
    return np.mean(p != t)

def convpool(X, W, b):
    # just assume pool size is (2,2) because we need to augment it with 1s
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)


def init_filter(shape, poolsz):
    # w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2]) / np.prod(poolsz))
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
    return w.astype(np.float32)


def rearrange(X):
    # input is (32, 32, 3, N)
    # output is (N, 32, 32, 3)
    # N = X.shape[-1]
    # out = np.zeros((N, 32, 32, 3), dtype=np.float32)
    # for i in xrange(N):
    #     for j in xrange(3):
    #         out[i, :, :, j] = X[:, :, j, i]
    # return out / 255
    return (X.transpose(3, 0, 1, 2) / 255).astype(np.float32)


def run_cnn_evaluation(x,Xtrain,Ytrain,Xtest,Ytest,isReturnAcc=False):
    
    tf.reset_default_graph()
    starttime = time.time()

    """
    with open('house_number_data.pkl','rb') as f:
        temp = pickle.load(f)
        
    [Xtrain,Ytrain,Xtest,Ytest]=temp
    """
    
    filter_sz,pool_sz,batch_sz,lr,momentum,decay,max_iter=x
    filter_sz=np.int(filter_sz)
    pool_sz=np.int(pool_sz)
    batch_sz=np.int(batch_sz)
    max_iter=np.int(max_iter)
    #lr=0.0001
    #momentum=0.9
    #decay=0.99


    # gradient descent params
    #max_iter = 200
    print_period = 50
    N = Xtrain.shape[0]
    #batch_sz = 1000
    n_batches = N // batch_sz


    
    #with open('house_number_data.pkl','wb') as f:
        #pickle.dump([Xtrain,Ytrain,Xtest,Ytest], f)
    # print "Xtest.shape:", Xtest.shape
    # print "Ytest.shape:", Ytest.shape

    # initial weights
    M = 500
    K = 10
    #filter_sz=5
    #pool_sz=2
    poolsz=(pool_sz,pool_sz)
    #poolsz = (2, 2)

    W1_shape = (filter_sz, filter_sz, 3, 20) # (filter_width, filter_height, num_color_channels, num_feature_maps)
    W1_init = init_filter(W1_shape, poolsz)
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32) # one bias per output feature map

    W2_shape = (filter_sz, filter_sz, 20, 50) # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
    W2_init = init_filter(W2_shape, poolsz)
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

    # vanilla ANN weights
    W3_init = np.random.randn(W2_shape[-1]*8*8, M) / np.sqrt(W2_shape[-1]*8*8 + M)
    b3_init = np.zeros(M, dtype=np.float32)
    W4_init = np.random.randn(M, K) / np.sqrt(M + K)
    b4_init = np.zeros(K, dtype=np.float32)


    # define variables and expressions
    # using None as the first shape element takes up too much RAM unfortunately
    X = tf.placeholder(tf.float32, shape=(batch_sz, 32, 32, 3), name='X')
    T = tf.placeholder(tf.int32, shape=(batch_sz,), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))

    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    Z2_shape = Z2.get_shape().as_list()
    Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
    Z3 = tf.nn.relu( tf.matmul(Z2r, W3) + b3 )

    #Z3_dropout = tf.layers.dropout(
      #inputs=Z3, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

    Yish = tf.matmul(Z3, W4) + b4

    
    cost = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=Yish,
            labels=T
        )
    )

    train_op = tf.train.RMSPropOptimizer(lr, decay, momentum).minimize(cost)

    # we'll use this to calculate the error rate
    predict_op = tf.argmax(Yish, 1)

    Accuracy_Curve,Cost_Curve=[],[]

    init = tf.global_variables_initializer()
        
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            myacc,mycost=[],[]
            for j in range(n_batches):
			
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz),]

                if len(Xbatch) == batch_sz:
                    session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                    if j % print_period == 0:
                        # due to RAM limitations we need to have a fixed size input
                        # so as a result, we have this ugly total cost and prediction computation
                        test_cost = 0
                        prediction = np.zeros(len(Xtest))
                        for k in range(len(Xtest) // batch_sz):
                            Xtestbatch = Xtest[k*batch_sz:(k*batch_sz + batch_sz),]
                            Ytestbatch = Ytest[k*batch_sz:(k*batch_sz + batch_sz),]
                            test_cost += session.run(cost, feed_dict={X: Xtestbatch, T: Ytestbatch})
                            prediction[k*batch_sz:(k*batch_sz + batch_sz)] = session.run(
                                predict_op, feed_dict={X: Xtestbatch})
                        err = error_rate(prediction, Ytest)
                        #print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                        #LL.append(test_cost)
                        myacc.append(1-err)
                        mycost.append((-1)*test_cost)
			
            Accuracy_Curve.append(np.mean(myacc))
            Cost_Curve.append(np.mean(mycost))


    endtime=time.time()
    elapse=endtime-starttime

    if isReturnAcc==False:
        return np.asarray(Cost_Curve),elapse
    else:
        return np.asarray(Accuracy_Curve),elapse
            
        


if __name__ == '__main__':
    
	#filter_sz,pool_sz,batch_sz,lr,momentum,decay,max_iter=x

    x=[5,2,500,1e-4,0.9,0.9,100]
    curve,elapse=run_cnn_evaluation(x)
    print(elapse)
    plt.plot(curve)
