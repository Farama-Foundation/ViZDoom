#!/usr/bin/python

import numpy as np
import theano
import theano.tensor as T
import lasagne


dtype = np.float32
states = np.eye(3, dtype = dtype).reshape(3,1,1,3)
values = np.float32(np.random.rand(3,4)) -1

#values = np.array([[0.1, 148, 135,147],[147,147,149,148],[148,147,147,147]], dtype = dtype)
output_dim = values.shape[1]
hidden_units = 50

#Network setup
inputs = T.tensor4('inputs' )
targets = T.matrix('targets')

network = lasagne.layers.InputLayer(shape=( None, 1, 1, 3 ), input_var = inputs)
network = lasagne.layers.DenseLayer(network,50, nonlinearity = lasagne.nonlinearities.tanh)
network = lasagne.layers.DenseLayer(network, output_dim,nonlinearity = lasagne.nonlinearities.tanh)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(prediction, targets).mean()
params = lasagne.layers.get_all_params(network, trainable = True)
updates = lasagne.updates.sgd(loss, params, learning_rate = 0.01)

f_learn = theano.function([inputs, targets],  loss, updates = updates)
f_test = theano.function([inputs], prediction)


#Training
it = 5000
for i in range(it):
	l = f_learn(states, values)
print "Loss: " +str(l)
print "Expected:"
print values
print "Learned:"
print f_test(states)


