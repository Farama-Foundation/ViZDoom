#!/usr/bin/python


import lasagne
import theano.tensor as T
from theano.tensor import tanh
import theano
import numpy as np

input1 = T.matrix("one")
input2 = T.matrix("two")
targets = T.matrix("targets")
input1_len = 5
input2_len = 3

network = lasagne.layers.InputLayer(shape =(None,input1_len),input_var = input1 )
input_layer2 = lasagne.layers.InputLayer(shape =(None, input2_len),input_var = input2)

network = lasagne.layers.DenseLayer(network, 1000,nonlinearity = lasagne.nonlinearities.tanh)

network = lasagne.layers.ConcatLayer([network,input_layer2])

network = lasagne.layers.DenseLayer(network,1,nonlinearity = None)

output = lasagne.layers.get_output(network)

loss = lasagne.objectives.squared_error(output,targets).mean()
params = lasagne.layers.get_all_params(network,trainable = True)
updates = lasagne.updates.sgd(loss,params,learning_rate = 0.01)

f = theano.function([input1,input2],output)
f_learn = theano.function([input1,input2,targets],loss,updates = updates)
x = 20

a = np.float32(np.random.random([x,input1_len]))
b = np.float32(np.random.random([x,input2_len]))

c = a.sum(1) +b.sum(1)
c = c.reshape(x, 1)
iters = 10000
log_freq =100
for i in range(iters):
	l = f_learn(a,b,c)
	if i%log_freq==0:
		print l

#print f(a,b)