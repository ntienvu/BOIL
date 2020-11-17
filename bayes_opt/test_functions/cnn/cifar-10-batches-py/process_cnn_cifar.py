# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:40:31 2020

@author: Lenovo
"""
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


label=[]
data=[]

vu=unpickle("data_batch_1")
label=label+vu[b'labels']
data=vu[b'data']

"""
vu=unpickle("data_batch_2")
label=label+vu[b'labels']
data=np.vstack((data,vu[b'data']))

vu=unpickle("data_batch_3")
label=label+vu[b'labels']
data=np.vstack((data,vu[b'data']))

vu=unpickle("data_batch_4")
label=label+vu[b'labels']
data=np.vstack((data,vu[b'data']))

vu=unpickle("data_batch_5")
label=label+vu[b'labels']
data=np.vstack((data,vu[b'data']))
"""


indices = np.arange(len(label))
np.random.shuffle(indices)

data=np.reshape(data,(len(label),32,32,3))

label=np.asarray(label)
#label=np.reshape(label,(-1,1))


label=label[indices]
data=data[indices]

#label=label[:1000]
#data=data[:1000]

vu_test=unpickle("test_batch")

label_test=vu_test[b'labels']
data_test=vu_test[b'data']

data_test=np.reshape(data_test,(10000,32,32,3))


label_test=np.asarray(label_test)
#label_test=np.reshape(label_test,(-1,1))


with open('cifar10.pickle', 'wb') as f:
    pickle.dump([data, label,data_test,label_test], f)
    
with open('cifar10.pickle', 'rb') as f:
    X_train, y_train, X_test,y_test = pickle.load(f)