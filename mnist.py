# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 16:02:01 2018

@author: Administrator
"""

import mnist_data
import numpy as np
import network 
import matplotlib.pyplot as plt

train = mnist_data.fetch_traingset()
test = mnist_data.fetch_testingset()

x_train = train['images']
y_train = train['labels']

x_test = test['images']
y_test = test['labels']


def vectorized(i):
    e = np.zeros((10, 1))
    e[i] = 1.0
    return e

y_train = [ vectorized(i) for i in y_train ]
y_test = [ vectorized(i) for i in y_test ] 


train_data = zip(x_train,y_train)
test_data = zip(x_test,y_test)

net = network.Network([784,30,30,10])

training_error,testing_accuracy = net.SGD(train_data, epochs=300,
                                          mini_batch_size=10000, eta=5.0,
                                          test_data=test_data,
                                          monitor_training_error=True,
                                          monitor_testing_accuracy=True)

plt.plot(training_error,label='training_error')
plt.plot(testing_accuracy,label='testing_accuracy')

plt.xlabel('Epochs')
plt.ylabel('Evaluation')

plt.legend()
plt.show()